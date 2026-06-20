"""推薦戰績追蹤：每個推薦項目分別記錄「過/沒過」，統整總勝率與勝敗。

對每場記錄模型的多個推薦項目：1X2、大小2.5、兩隊進球(BTTS)、亞盤讓球。
賽後用真實比分判定每項 win/loss/push，彙整總勝率與各盤口別的勝敗。
只記「事前」推薦（未開賽前），不馬後砲。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import markets

LEDGER_COLS = ["date", "match_num", "home", "away", "market", "selection",
               "line", "result"]
_MARKET_ZH = {"1X2": "勝平負", "OU": "大小2.5", "BTTS": "兩隊進球", "AH": "亞盤"}
_1X2_ZH = {"home": "主勝", "draw": "和", "away": "客勝"}


def _recommendations(model, home, away, neutral=True, min_lean=0.03):
    """回傳該場推薦項目清單 [(market, selection, line), ...]。"""
    mat = model.score_matrix(home, away, neutral=neutral)
    recs = []
    o = markets.outcome_1x2(mat)
    recs.append(("1X2", max(o, key=o.get), ""))      # 1X2 一定有最高那邊
    ou = markets.over_under(mat, 2.5)
    if abs(ou["over_win"] - 0.5) >= min_lean:          # 有明顯方向才推薦
        recs.append(("OU", "大" if ou["over_win"] >= 0.5 else "小", 2.5))
    bt = markets.btts(mat)
    if abs(bt["yes"] - 0.5) >= min_lean:
        recs.append(("BTTS", "是" if bt["yes"] >= 0.5 else "否", ""))
    lam, mu = model.expected_goals(home, away, neutral=neutral)
    hl = markets.main_handicap_line(lam - mu)
    ah_h = markets.asian_handicap(mat, hl, "home")
    cover_h = ah_h.p_win + ah_h.p_half_win + 0.5 * ah_h.p_push
    recs.append(("AH", "主" if cover_h >= 0.5 else "客", hl))
    return recs


def _settle_one(market, selection, line, hg, ag) -> str:
    """單一推薦項目用真實比分判定 win/loss/push。"""
    total = hg + ag
    if market == "1X2":
        actual = "home" if hg > ag else "away" if hg < ag else "draw"
        return "win" if selection == actual else "loss"
    if market == "OU":
        line = float(line)
        if line.is_integer() and total == line:
            return "push"
        over = total > line
        return "win" if (selection == "大") == over else "loss"
    if market == "BTTS":
        both = hg > 0 and ag > 0
        return "win" if (selection == "是") == both else "loss"
    if market == "AH":
        side = "home" if selection == "主" else "away"
        n = max(hg, ag) + 2
        mat = np.zeros((n, n))
        mat[hg, ag] = 1.0
        ah = markets.asian_handicap(mat, float(line), side)
        if ah.p_push >= 0.999:
            return "push"
        win = ah.p_win + 0.5 * ah.p_half_win
        loss = ah.p_loss + 0.5 * ah.p_half_loss
        if abs(win - loss) < 1e-9:
            return "push"
        return "win" if win > loss else "loss"
    return "push"


def load_ledger(path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        for c in LEDGER_COLS:
            if c not in df.columns:
                df[c] = ""
        return df[LEDGER_COLS]
    return pd.DataFrame(columns=LEDGER_COLS)


def save_ledger(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def log_upcoming(matches, model, ledger_path, neutral=True):
    """把未開賽、雙方已知的比賽各推薦項目記入帳本（已存在的不重複）。回傳新增筆數。"""
    df = load_ledger(ledger_path)
    seen = set(zip(df["match_num"].astype(str), df["market"], df["selection"].astype(str)))
    rows = []
    for m in matches:
        if m.played or m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        for market, sel, line in _recommendations(model, m.team1, m.team2, neutral):
            if (str(m.num), market, str(sel)) in seen:
                continue
            rows.append(dict(date=m.date, match_num=m.num, home=m.team1, away=m.team2,
                             market=market, selection=sel, line=line, result="pending"))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, ledger_path)
    return len(rows)


def settle(ledger_path, results: dict):
    """用真實賽果 {match_num: (hg, ag)} 結算所有 pending 項目。回傳結算筆數。"""
    df = load_ledger(ledger_path)
    n = 0
    for i, r in df[df["result"] == "pending"].iterrows():
        res = results.get(int(r["match_num"]))
        if res is None:
            continue
        df.at[i, "result"] = _settle_one(r["market"], r["selection"], r["line"],
                                         int(res[0]), int(res[1]))
        n += 1
    if n:
        save_ledger(df, ledger_path)
    return n


@dataclass
class TrackSummary:
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    pending: int = 0
    by_market: dict = field(default_factory=dict)  # market -> (w, l, p)

    @property
    def settled(self):
        return self.wins + self.losses + self.pushes

    @property
    def win_rate(self):
        denom = self.wins + self.losses
        return self.wins / denom if denom else 0.0

    def text(self) -> str:
        parts = []
        for mk, (w, l, p) in self.by_market.items():
            d = w + l
            rt = f"{w/d:.0%}" if d else "—"
            parts.append(f"{_MARKET_ZH.get(mk, mk)} {w}–{l}（{rt}）")
        head = (f"推薦戰績｜總計 {self.wins} 勝 {self.losses} 敗"
                f"{f' {self.pushes} 走盤' if self.pushes else ''}"
                f"｜勝率 {self.win_rate:.1%}（待結 {self.pending}）")
        return head + ("\n  " + "　".join(parts) if parts else "")


def summary(ledger_path) -> TrackSummary:
    df = load_ledger(ledger_path)
    s = TrackSummary()
    s.pending = int((df["result"] == "pending").sum())
    s.wins = int((df["result"] == "win").sum())
    s.losses = int((df["result"] == "loss").sum())
    s.pushes = int((df["result"] == "push").sum())
    for mk in ["1X2", "OU", "BTTS", "AH"]:
        sub = df[df["market"] == mk]
        w = int((sub["result"] == "win").sum())
        l = int((sub["result"] == "loss").sum())
        p = int((sub["result"] == "push").sum())
        if w + l + p:
            s.by_market[mk] = (w, l, p)
    return s
