"""推薦投注戰績追蹤（均注）。

記錄模型對每場的推薦（預設 1X2 取機率最高那邊），賽後用真實賽果結算，
以「每場固定 1 注」計算命中率、損益、ROI。

賠率：預設用模型公平賠率(1/機率)當基準——此時長期 ROI 應趨近 0（這其實是
校準/紀律檢驗，不是獲利）。若要追蹤「真實獲利」，把 odds 欄換成實際盤口賠率即可。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import markets

LEDGER_COLS = ["date", "match_num", "home", "away", "market", "selection",
               "odds", "stake", "result", "pnl"]


def _pick_1x2(model, home, away, neutral=True):
    """回傳模型 1X2 推薦：(selection, 公平賠率, 機率)。"""
    o = markets.outcome_1x2(model.score_matrix(home, away, neutral=neutral))
    sel = max(o, key=o.get)
    p = o[sel]
    return sel, (1.0 / p if p > 0 else 99.0), p


def load_ledger(path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=LEDGER_COLS)


def save_ledger(df: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def log_upcoming(matches, model, ledger_path, neutral=True, min_prob=0.0):
    """把尚未記錄、未開賽、雙方已知的比賽，記入模型 1X2 推薦（均注 1）。

    min_prob：只記錄推薦機率 >= 此門檻者（例如 0.5 只記較有把握的）。
    回傳新增筆數。
    """
    df = load_ledger(ledger_path)
    seen = set(zip(df["match_num"], df["market"])) if len(df) else set()
    rows = []
    for m in matches:
        if m.played or (m.num, "1X2") in seen:
            continue
        if m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        sel, odds, p = _pick_1x2(model, m.team1, m.team2, neutral=neutral)
        if p < min_prob:
            continue
        rows.append(dict(date=m.date, match_num=m.num, home=m.team1, away=m.team2,
                         market="1X2", selection=sel, odds=round(odds, 3),
                         stake=1.0, result="pending", pnl=0.0))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, ledger_path)
    return len(rows)


def settle(ledger_path, results: dict[int, tuple[int, int]]):
    """用真實賽果（{match_num: (home_goals, away_goals)}）結算 pending 注單。

    回傳結算筆數。
    """
    df = load_ledger(ledger_path)
    n = 0
    for i, r in df[df["result"] == "pending"].iterrows():
        res = results.get(int(r["match_num"]))
        if res is None:
            continue
        hg, ag = res
        actual = "home" if hg > ag else "away" if hg < ag else "draw"
        won = r["selection"] == actual
        df.at[i, "result"] = "win" if won else "loss"
        df.at[i, "pnl"] = (r["odds"] - 1.0) * r["stake"] if won else -r["stake"]
        n += 1
    if n:
        save_ledger(df, ledger_path)
    return n


@dataclass
class TrackSummary:
    n: int
    settled: int
    pending: int
    wins: int
    staked: float
    pnl: float

    @property
    def hit_rate(self):
        return self.wins / self.settled if self.settled else 0.0

    @property
    def roi(self):
        return self.pnl / self.staked if self.staked else 0.0

    def text(self) -> str:
        return (f"推薦戰績（均注 1）｜已結算 {self.settled} 注（待結 {self.pending}）\n"
                f"  命中率 {self.hit_rate:.1%}｜總下注 {self.staked:.0f}｜"
                f"損益 {self.pnl:+.2f}｜ROI {self.roi:+.1%}")


def summary(ledger_path) -> TrackSummary:
    df = load_ledger(ledger_path)
    settled = df[df["result"].isin(["win", "loss"])]
    return TrackSummary(
        n=len(df), settled=len(settled),
        pending=int((df["result"] == "pending").sum()),
        wins=int((df["result"] == "win").sum()),
        staked=float(settled["stake"].sum()),
        pnl=float(settled["pnl"].sum()),
    )
