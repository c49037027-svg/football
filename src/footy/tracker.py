"""推薦戰績追蹤：記錄每個推薦項目的賠率、過/沒過、實際損益(ROI)與 CLV。

兩種模式：
1. 有真實盤口（odds_index，需 ODDS_API_KEY）：只記模型相對市場有正期望值(+EV)的
   推薦，存下注時賠率；賽後用真實比分結算，算實際 ROI、勝率，並比較收盤賠率算 CLV。
   ——這才是判斷「正收益」的硬指標（勝率高 ≠ 賺錢）。
2. 無盤口（離線/免費方案）：退回記模型最看好的選項，只統計勝率（自我校驗）。

只記「事前」推薦（未開賽前），不馬後砲。均注：每注 1 單位。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .models import markets
from .value.odds import blend_probs, remove_vig_proportional

LEDGER_COLS = ["date", "match_num", "home", "away", "market", "selection",
               "line", "odds", "edge", "source", "close_odds", "result", "pl"]
_MARKET_ZH = {"1X2": "勝平負", "OU": "大小2.5", "BTTS": "兩隊進球", "AH": "亞盤"}
_1X2_ZH = {"home": "主勝", "draw": "和", "away": "客勝"}
STAKE = 1.0  # 均注：每注 1 單位
# 市場融合權重（模型佔比）：1=純模型、0=純市場。可用環境變數覆寫。
# 實證上純模型贏不過效率市場，故預設拉一半向市場，只賭真正的分歧。
BLEND_WEIGHT = float(os.environ.get("BLEND_WEIGHT", "0.5"))


# ------------- 模型 vs 市場：找正期望值(+EV)推薦 -------------
def _ev_no_push(p: float, odds: float) -> float:
    """單位期望值 = 機率·賠率 − 1（決策用估計）。"""
    return p * odds - 1.0


def _group_quotes(quotes, market):
    """把某盤口的報價依 line 分組：{line: {selection: odds}}。1X2 的 line 用 ""。"""
    out: dict = {}
    for q in quotes:
        if q.market != market or not q.odds or q.odds <= 1.0:
            continue
        line = "" if market in ("1X2", "BTTS") else q.line
        out.setdefault(line, {})[q.selection] = float(q.odds)
    return out


def _blended(model_ps: dict, sides: dict, order: list, weight: float) -> dict:
    """sides={selection:odds}。兩邊齊全 → 去 vig 後與模型融合；否則退回純模型。"""
    if all(s in sides for s in order):
        fair = remove_vig_proportional([sides[s] for s in order])
        bl = blend_probs([model_ps[s] for s in order], fair, weight)
        return dict(zip(order, bl))
    return {s: model_ps[s] for s in order}


def _market_edges(model, home, away, quotes, neutral=True, min_edge=0.0,
                  weight=None):
    """市場去 vig + 模型融合後，挑出每個盤口期望值最高且 > min_edge 的推薦。

    quotes：MarketQuote 清單（.market/.selection/.odds/.line）。
    回傳 [dict(market, selection, line, odds, edge, p)]，p 為融合後機率。
    """
    w = BLEND_WEIGHT if weight is None else weight
    mat = model.score_matrix(home, away, neutral=neutral)
    best: dict[str, dict] = {}

    def offer(market, sel, line, odds, p):
        edge = _ev_no_push(p, odds)
        if edge <= min_edge:
            return
        cur = best.get(market)
        if cur is None or edge > cur["edge"]:
            best[market] = dict(market=market, selection=sel, line=line,
                                odds=float(odds), edge=float(edge), p=float(p))

    # 1X2
    h2h = _group_quotes(quotes, "1X2").get("", {})
    if h2h:
        o = markets.outcome_1x2(mat)
        bl = _blended(o, h2h, ["home", "draw", "away"], w)
        for s, odds in h2h.items():
            if s in bl:
                offer("1X2", s, "", odds, bl[s])

    # 大小（逐線）
    for line, sides in _group_quotes(quotes, "OU").items():
        ou = markets.over_under(mat, line)
        model_ps = {"over": ou["over_win"], "under": ou["under_win"]}
        bl = _blended(model_ps, sides, ["over", "under"], w)
        for s, odds in sides.items():
            offer("OU", "大" if s == "over" else "小", line, odds, bl[s])

    # 亞盤（逐線；用覆蓋率近似 edge，結算仍精算半輸半贏）
    for line, sides in _group_quotes(quotes, "AH").items():
        ah_h = markets.asian_handicap(mat, float(line), "home")
        cov_h = ah_h.p_win + ah_h.p_half_win + 0.5 * ah_h.p_push
        model_ps = {"home": cov_h, "away": 1.0 - cov_h}
        bl = _blended(model_ps, sides, ["home", "away"], w)
        for s, odds in sides.items():
            offer("AH", "主" if s == "home" else "客", line, odds, bl[s])

    # 兩隊進球
    bt_sides = _group_quotes(quotes, "BTTS").get("", {})
    if bt_sides:
        bt = markets.btts(mat)
        model_ps = {"yes": bt["yes"], "no": bt["no"]}
        bl = _blended(model_ps, bt_sides, ["yes", "no"], w)
        for s, odds in bt_sides.items():
            offer("BTTS", "是" if s == "yes" else "否", "", odds, bl[s])

    return list(best.values())


def _recommendations(model, home, away, neutral=True, min_lean=0.03):
    """無盤口模式：回傳該場模型推薦項目 [(market, selection, line), ...]。"""
    mat = model.score_matrix(home, away, neutral=neutral)
    recs = []
    o = markets.outcome_1x2(mat)
    recs.append(("1X2", max(o, key=o.get), ""))
    ou = markets.over_under(mat, 2.5)
    if abs(ou["over_win"] - 0.5) >= min_lean:
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


# ------------- 結算 -------------
def _settle_outcome(market, selection, line, hg, ag):
    """回傳結果權重 (win, half_win, push, half_loss, loss)，總和為 1。"""
    total = hg + ag
    if market == "1X2":
        actual = "home" if hg > ag else "away" if hg < ag else "draw"
        return (1, 0, 0, 0, 0) if selection == actual else (0, 0, 0, 0, 1)
    if market == "OU":
        line = float(line)
        if float(line).is_integer() and total == line:
            return (0, 0, 1, 0, 0)
        win = (selection == "大") == (total > line)
        return (1, 0, 0, 0, 0) if win else (0, 0, 0, 0, 1)
    if market == "BTTS":
        both = hg > 0 and ag > 0
        win = (selection in ("是", "yes")) == both
        return (1, 0, 0, 0, 0) if win else (0, 0, 0, 0, 1)
    if market == "AH":
        side = "home" if selection in ("主", "home") else "away"
        n = max(hg, ag) + 2
        mat = np.zeros((n, n))
        mat[hg, ag] = 1.0
        ah = markets.asian_handicap(mat, float(line), side)
        return (ah.p_win, ah.p_half_win, ah.p_push, ah.p_half_loss, ah.p_loss)
    return (0, 0, 1, 0, 0)


def _label(w) -> str:
    net = (w[0] + 0.5 * w[1]) - (w[4] + 0.5 * w[3])
    if abs(net) < 1e-9:
        return "push"
    return "win" if net > 0 else "loss"


def _settle_one(market, selection, line, hg, ag) -> str:
    """單一推薦項目用真實比分判定 win/loss/push（相容舊介面）。"""
    return _label(_settle_outcome(market, selection, line, hg, ag))


def _pl(w, odds, stake=STAKE) -> float:
    """由結果權重 + 賠率算單位損益（含半輸半贏）。"""
    win, hw, pu, hl, ls = w
    return stake * (win * (odds - 1.0) + hw * (odds - 1.0) / 2.0
                    - hl * 0.5 - ls * 1.0)


# ------------- 帳本讀寫 -------------
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


def _to_float(x):
    try:
        if x in ("", None) or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def log_upcoming(matches, model, ledger_path, neutral=True,
                 odds_index=None, min_edge=0.0, weight=None):
    """記錄未開賽推薦（已存在不重複）。回傳新增筆數。

    odds_index：{match_num: [MarketQuote...]}。給定時走「+EV 真實盤口」模式
    （市場去 vig 後與模型融合，只記仍有正期望值的推薦）；None 則走勝率模式。
    weight：模型融合權重（None=用 BLEND_WEIGHT）。
    """
    df = load_ledger(ledger_path)
    seen = set(zip(df["match_num"].astype(str), df["market"], df["selection"].astype(str)))
    rows = []
    for m in matches:
        if m.played or m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        if odds_index is not None:
            quotes = odds_index.get(m.num)
            if not quotes:
                continue  # 沒盤口 → 不下注
            for r in _market_edges(model, m.team1, m.team2, quotes, neutral,
                                   min_edge, weight):
                if (str(m.num), r["market"], str(r["selection"])) in seen:
                    continue
                rows.append(dict(date=m.date, match_num=m.num, home=m.team1,
                                 away=m.team2, market=r["market"],
                                 selection=r["selection"], line=r["line"],
                                 odds=round(r["odds"], 3), edge=round(r["edge"], 4),
                                 source="market", close_odds=round(r["odds"], 3),
                                 result="pending", pl=""))
        else:
            for market, sel, line in _recommendations(model, m.team1, m.team2, neutral):
                if (str(m.num), market, str(sel)) in seen:
                    continue
                rows.append(dict(date=m.date, match_num=m.num, home=m.team1,
                                 away=m.team2, market=market, selection=sel,
                                 line=line, odds="", edge="", source="model",
                                 close_odds="", result="pending", pl=""))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, ledger_path)
    return len(rows)


def refresh_close(ledger_path, odds_index):
    """把未結算 market 推薦的 close_odds 更新成目前最新賠率（逼近收盤）。"""
    df = load_ledger(ledger_path)
    n = 0
    for i, r in df[(df["result"] == "pending") & (df["source"] == "market")].iterrows():
        quotes = odds_index.get(int(r["match_num"])) if odds_index else None
        if not quotes:
            continue
        odds = _lookup_odds(quotes, r["market"], r["selection"], r["line"])
        if odds:
            df.at[i, "close_odds"] = round(odds, 3)
            n += 1
    if n:
        save_ledger(df, ledger_path)
    return n


def _lookup_odds(quotes, market, selection, line):
    """從 quotes 找對應 (market, selection, line) 的賠率。"""
    sel_map = {"大": "over", "小": "under", "主": "home", "客": "away",
               "是": "yes", "否": "no"}
    target = sel_map.get(selection, selection)
    for q in quotes:
        if q.market != market:
            continue
        qsel = q.selection
        if market in ("OU", "AH", "BTTS"):
            if qsel != target:
                continue
            if market in ("OU", "AH") and _to_float(q.line) != _to_float(line):
                continue
            return q.odds
        if market == "1X2" and qsel == target:
            return q.odds
    return None


def settle(ledger_path, results: dict):
    """用真實賽果 {match_num: (hg, ag)} 結算所有 pending 項目。回傳結算筆數。"""
    df = load_ledger(ledger_path)
    n = 0
    for i, r in df[df["result"] == "pending"].iterrows():
        res = results.get(int(r["match_num"]))
        if res is None:
            continue
        w = _settle_outcome(r["market"], r["selection"], r["line"],
                            int(res[0]), int(res[1]))
        df.at[i, "result"] = _label(w)
        odds = _to_float(r["odds"])
        if odds:
            df.at[i, "pl"] = round(_pl(w, odds), 4)
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
    by_market: dict = field(default_factory=dict)   # market -> (w, l, p)
    market_bets: int = 0        # 有真實盤口、已結算的注數
    pl_units: float = 0.0       # 實際損益（單位）
    clv_n: int = 0
    clv_sum: float = 0.0

    @property
    def settled(self):
        return self.wins + self.losses + self.pushes

    @property
    def win_rate(self):
        denom = self.wins + self.losses
        return self.wins / denom if denom else 0.0

    @property
    def roi(self):
        return self.pl_units / self.market_bets if self.market_bets else 0.0

    @property
    def clv(self):
        return self.clv_sum / self.clv_n if self.clv_n else 0.0

    def text(self) -> str:
        parts = []
        for mk, (w, l, p) in self.by_market.items():
            d = w + l
            rt = f"{w/d:.0%}" if d else "—"
            parts.append(f"{_MARKET_ZH.get(mk, mk)} {w}–{l}（{rt}）")
        head = (f"推薦戰績｜總計 {self.wins} 勝 {self.losses} 敗"
                f"{f' {self.pushes} 走盤' if self.pushes else ''}"
                f"｜勝率 {self.win_rate:.1%}（待結 {self.pending}）")
        lines = [head]
        if parts:
            lines.append("　".join(parts))
        if self.market_bets:
            roi_line = (f"實際損益｜{self.pl_units:+.2f}u / {self.market_bets} 注"
                        f"｜ROI {self.roi:+.1%}")
            if self.clv_n:
                roi_line += f"｜CLV {self.clv:+.1%}"
            lines.append(roi_line)
        return "\n".join(lines)


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
    # 真實盤口損益 + CLV（只看 source=market 且已結算）
    mkt = df[(df["source"] == "market") & (df["result"].isin(["win", "loss", "push"]))]
    for _, r in mkt.iterrows():
        pl = _to_float(r["pl"])
        if pl is None:
            continue
        s.market_bets += 1
        s.pl_units += pl
        taken, close = _to_float(r["odds"]), _to_float(r["close_odds"])
        if taken and close and close > 1.0:
            s.clv_sum += taken / close - 1.0
            s.clv_n += 1
    s.pl_units = round(s.pl_units, 4)
    return s


def prepare(matches, model, ledger_path, odds_index=None, min_edge=0.0, weight=None):
    """一站式：記推薦 → 更新收盤賠率 → 用已踢賽果結算 → 回傳 summary。"""
    log_upcoming(matches, model, ledger_path, odds_index=odds_index,
                 min_edge=min_edge, weight=weight)
    if odds_index:
        refresh_close(ledger_path, odds_index)
    settle(ledger_path, {m.num: (m.hg, m.ag) for m in matches if m.played})
    return summary(ledger_path)
