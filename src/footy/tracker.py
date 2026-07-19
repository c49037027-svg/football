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
# 盤口/期望值健全性過濾：擋掉劣質資料與「模型高估冷門」的假 value
MAX_ODDS = 41.0   # 賠率 > 此值（隱含 < ~2.4%）視為髒資料，不採用
# edge 上限（2026-07-18 由 25% 收緊）：世界盃 2,985 筆 odds_log 實證——三個市場的
# 原始模型分歧 >15%（≈融合後 >10%）桶全部大虧（1X2 尾桶 14% 勝率/每注 −42%），
# 現行 25% 上限下的模擬選注每注 −8.4%，收到 10% 轉 +4.2%。MLB 錢線/讓分不受影響
# （edge>10% 僅 7 筆非 OU）。可用環境變數 FOOTY_MAX_EDGE 覆寫（研究用）。
MAX_EDGE = float(os.environ.get("FOOTY_MAX_EDGE", "0.10"))
MIN_PROB = 0.08   # 融合後機率 < 8% 的極端冷門不下注（模型在尾端不可靠）
# 市場融合權重（模型佔比）：1=純模型、0=純市場。可用環境變數覆寫。
# 實證上純模型贏不過效率市場，故預設拉一半向市場，只賭真正的分歧。
BLEND_WEIGHT = float(os.environ.get("BLEND_WEIGHT", "0.5"))


# ------------- 模型 vs 市場：找正期望值(+EV)推薦 -------------
def _ev_no_push(p: float, odds: float) -> float:
    """單位期望值 = 機率·賠率 − 1（決策用估計）。"""
    return p * odds - 1.0


def main_ah_line(quotes):
    """從報價挑「主盤」亞盤讓球線：主客賠率最接近(最均衡)的那條。回傳 line 或 None。"""
    lines = _group_quotes(quotes, "AH")
    best, best_gap = None, 1e9
    for line, sides in lines.items():
        if "home" in sides and "away" in sides:
            gap = abs(sides["home"] - sides["away"])
            if gap < best_gap:
                best, best_gap = line, gap
        elif best is None:
            best = line  # 只有單邊時退而求其次
    return _to_float(best) if best is not None else None


def _group_quotes(quotes, market):
    """把某盤口的報價依 line 分組：{line: {selection: odds}}。1X2 的 line 用 ""。"""
    out: dict = {}
    for q in quotes:
        if q.market != market or not q.odds or q.odds <= 1.0 or q.odds > MAX_ODDS:
            continue
        line = "" if market in ("1X2", "BTTS") else q.line
        out.setdefault(line, {})[q.selection] = float(q.odds)
    return out


def _blended(model_ps: dict, sides: dict, order: list, weight: float) -> dict:
    """sides={selection:odds}。各邊齊全（含模型機率）→ 去 vig 後與模型融合；
    否則退回純模型（只回有資料的選項，缺項略過，不丟例外）。"""
    if all(s in sides for s in order) and all(s in model_ps for s in order):
        fair = remove_vig_proportional([sides[s] for s in order])
        bl = blend_probs([model_ps[s] for s in order], fair, weight)
        return dict(zip(order, bl))
    return {s: model_ps[s] for s in order if s in model_ps}


_SEL_ZH = {"home": "主", "away": "客", "over": "大", "under": "小",
           "yes": "是", "no": "否"}
_MARKET_ORDER = {"1X2": ["home", "draw", "away"], "OU": ["over", "under"],
                 "AH": ["home", "away"], "BTTS": ["yes", "no"]}


def _model_market_probs(model, home, away, quotes, neutral=True):
    """對 quotes 出現的每個盤口/線，算模型機率。

    回傳 {(market, line): {selection: p}}，selection 用原生英文鍵
    （home/draw/away、over/under、home/away、yes/no）。
    """
    mat = model.score_matrix(home, away, neutral=neutral)
    out: dict = {}
    if _group_quotes(quotes, "1X2").get("", {}):
        out[("1X2", "")] = markets.outcome_1x2(mat)
    for line in _group_quotes(quotes, "OU"):
        ou = markets.over_under(mat, line)
        out[("OU", line)] = {"over": ou["over_win"], "under": ou["under_win"]}
    for line in _group_quotes(quotes, "AH"):
        ah = markets.asian_handicap(mat, float(line), "home")
        cov = ah.p_win + ah.p_half_win + 0.5 * ah.p_push
        out[("AH", line)] = {"home": cov, "away": 1.0 - cov}
    if _group_quotes(quotes, "BTTS").get("", {}):
        bt = markets.btts(mat)
        out[("BTTS", "")] = {"yes": bt["yes"], "no": bt["no"]}
    return out


def _pick_bet(model_ps, sides, order, weight, min_edge):
    """某盤口去 vig + 融合後，回傳 +EV 最高的 (selection, odds, edge, p) 或 None。"""
    bl = _blended(model_ps, sides, order, weight)
    best = None
    for s, odds in sides.items():
        if s not in bl:
            continue
        edge = _ev_no_push(bl[s], odds)
        # 過濾：min_edge < edge <= MAX_EDGE（擋假高 edge）且機率 >= MIN_PROB（擋極端冷門）
        if (min_edge < edge <= MAX_EDGE and bl[s] >= MIN_PROB
                and (best is None or edge > best[2])):
            best = (s, odds, edge, bl[s])
    return best


def _market_edges(model, home, away, quotes, neutral=True, min_edge=0.0,
                  weight=None):
    """市場去 vig + 模型融合後，挑出每個盤口期望值最高且 > min_edge 的推薦。

    quotes：MarketQuote 清單（.market/.selection/.odds/.line）。
    回傳 [dict(market, selection, line, odds, edge, p)]，p 為融合後機率。
    """
    w = BLEND_WEIGHT if weight is None else weight
    mprobs = _model_market_probs(model, home, away, quotes, neutral)
    out = []
    for (market, line), model_ps in mprobs.items():
        sides = _group_quotes(quotes, market).get(line, {})
        r = _pick_bet(model_ps, sides, _MARKET_ORDER[market], w, min_edge)
        if r:
            sel = r[0] if market == "1X2" else _SEL_ZH[r[0]]
            out.append(dict(market=market, selection=sel, line=line,
                            odds=r[1], edge=r[2], p=r[3]))
    return out


def _recommendations(model, home, away, neutral=True, min_lean=0.03, quotes=None):
    """模型對該場的推薦（它最看好的一邊，非 +EV 冷門）。

    回傳 [(market, selection, line, odds_or_None), ...]。
    quotes 給定時附上該選項的真實賠率（亞盤用市場主盤線）；劣質賠率(>MAX_ODDS)當無賠率。
    """
    mat = model.score_matrix(home, away, neutral=neutral)

    def odds_for(market, sel, line):
        if not quotes:
            return None
        od = _lookup_odds(quotes, market, sel, line)
        return od if (od and 1.0 < od <= MAX_ODDS) else None

    recs = []
    o = markets.outcome_1x2(mat)
    s = max(o, key=o.get)                         # 1X2：模型機率最高（通常是熱門）
    recs.append(("1X2", s, "", odds_for("1X2", s, "")))
    ou = markets.over_under(mat, 2.5)
    if abs(ou["over_win"] - 0.5) >= min_lean:     # 有明顯方向才推
        sel = "大" if ou["over_win"] >= 0.5 else "小"
        recs.append(("OU", sel, 2.5, odds_for("OU", sel, 2.5)))
    bt = markets.btts(mat)
    if abs(bt["yes"] - 0.5) >= min_lean:
        sel = "是" if bt["yes"] >= 0.5 else "否"
        recs.append(("BTTS", sel, "", odds_for("BTTS", sel, "")))
    # 亞盤：有盤口用市場主盤線，否則用模型自開線；取模型覆蓋的一邊
    line = main_ah_line(quotes) if quotes else None
    if line is None:
        lam, mu = model.expected_goals(home, away, neutral=neutral)
        line = markets.main_handicap_line(lam - mu)
    ah_h = markets.asian_handicap(mat, float(line), "home")
    cover_h = ah_h.p_win + ah_h.p_half_win + 0.5 * ah_h.p_push
    side = "主" if cover_h >= 0.5 else "客"
    recs.append(("AH", side, line, odds_for("AH", side, line)))
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
        win = (selection in ("大", "over")) == (total > line)
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
    """記錄未開賽的模型推薦（它最看好的一邊；已存在不重複）。回傳新增筆數。

    記的是模型的真實推薦（1X2 取機率最高、OU/BTTS 取明顯偏向、AH 取覆蓋側），
    用來追蹤「推薦準確度 + 收益」。odds_index 給定時附真實賠率→可算損益/CLV
    （source=market）；否則只計勝率（source=model）。
    """
    df = load_ledger(ledger_path)
    seen = set(zip(df["match_num"].astype(str), df["market"], df["selection"].astype(str)))
    rows = []
    for m in matches:
        if m.played or m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        quotes = odds_index.get(m.num) if odds_index else None
        for market, sel, line, odds in _recommendations(model, m.team1, m.team2,
                                                         neutral, quotes=quotes):
            if (str(m.num), market, str(sel)) in seen:
                continue
            od = round(odds, 3) if odds else ""
            rows.append(dict(date=m.date, match_num=m.num, home=m.team1, away=m.team2,
                             market=market, selection=sel, line=line, odds=od, edge="",
                             source="market" if odds else "model",
                             close_odds=od, result="pending", pl=""))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, ledger_path)
    return len(rows)


def backfill_played(matches, model, ledger_path, neutral=True):
    """整屆回填：把已開賽、但帳本沒記過的比賽補記模型推薦並立即用真實比分結算。

    僅 source='model'（無賠率→只計勝率，不進 ROI/CLV）。供「開賽至今」回顧用，
    非真實盤口下注。注意：每日重訓的模型可能已把這些賽果學進去，屬事後回顧、偏樂觀。
    回傳新增筆數。
    """
    df = load_ledger(ledger_path)
    seen = set(zip(df["match_num"].astype(str), df["market"], df["selection"].astype(str)))
    rows = []
    for m in matches:
        if not m.played or m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        for market, sel, line, _odds in _recommendations(model, m.team1, m.team2, neutral):
            if (str(m.num), market, str(sel)) in seen:
                continue
            w = _settle_outcome(market, sel, line, int(m.hg), int(m.ag))
            rows.append(dict(date=m.date, match_num=m.num, home=m.team1, away=m.team2,
                             market=market, selection=sel, line=line, odds="", edge="",
                             source="model", close_odds="", result=_label(w), pl=""))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, ledger_path)
    return len(rows)


def _picks_from_snapshot_rows(rows):
    """從某場的快照列（含賽前 model_p + 賠率），重建模型『最看好的一邊』。

    用快照當時的 model_p（賽前機率，無 look-ahead）挑選；回傳 [(market, sel, line, odds)]。
    """
    from collections import defaultdict
    by_mkt = defaultdict(list)
    for r in rows:
        od, mp = _to_float(r["odds"]), _to_float(r["model_p"])
        if od is None or mp is None or not (1.0 < od <= MAX_ODDS):
            continue
        cl = _to_float(r["close_odds"]) or od
        line = _to_float(r["line"])
        by_mkt[r["market"]].append((r["selection"], line, od, mp, cl))
    out = []  # (market, sel, line, odds, close_odds)
    if by_mkt.get("1X2"):
        s, _l, od, _mp, cl = max(by_mkt["1X2"], key=lambda x: x[3])
        out.append(("1X2", s, "", od, cl))
    ou = [x for x in by_mkt.get("OU", []) if x[1] == 2.5]
    if ou:
        s, _l, od, _mp, cl = max(ou, key=lambda x: x[3])
        out.append(("OU", "大" if s == "over" else "小", 2.5, od, cl))
    if by_mkt.get("BTTS"):
        s, _l, od, _mp, cl = max(by_mkt["BTTS"], key=lambda x: x[3])
        out.append(("BTTS", "是" if s == "yes" else "否", "", od, cl))
    ah = by_mkt.get("AH", [])
    if ah:
        lines = defaultdict(dict)
        for s, line, od, mp, cl in ah:
            lines[line][s] = (od, mp, cl)
        main, gap = None, 1e9
        for line, sd in lines.items():
            if "home" in sd and "away" in sd:
                g = abs(sd["home"][0] - sd["away"][0])
                if g < gap:
                    gap, main = g, line
        if main is None:
            main = next(iter(lines))
        sd = lines[main]
        side = max(sd, key=lambda s: sd[s][1])  # 覆蓋機率較高側
        out.append(("AH", "主" if side == "home" else "客", main, sd[side][0], sd[side][2]))
    return out


def rebuild_market_from_snapshots(ledger_path, snap_path):
    """用賠率快照重建『真實盤口』戰績：丟掉舊 market 列，依模型推薦+快照賠率重記，
    已踢者用快照賽果結算。回傳重建筆數。source=model（無快照場次）保留。"""
    snaps = load_snapshots(snap_path)
    if snaps.empty:
        return 0
    df = load_ledger(ledger_path)
    df = df[df["source"] != "market"].copy()  # 移除舊的（冷門）市場列
    new_rows, covered = [], set()
    for num, g in snaps.groupby("match_num"):
        rows = g.to_dict("records")
        r0 = rows[0]
        played = str(r0.get("played")) in ("1", "1.0")
        hg, ag = _to_float(r0.get("hg")), _to_float(r0.get("ag"))
        for market, sel, line, odds, close in _picks_from_snapshot_rows(rows):
            result, pl = "pending", ""
            if played and hg is not None and ag is not None:
                w = _settle_outcome(market, sel, line, int(hg), int(ag))
                result, pl = _label(w), round(_pl(w, odds), 4)
            new_rows.append(dict(date=r0["date"], match_num=int(num),
                                 home=r0["home"], away=r0["away"], market=market,
                                 selection=sel, line=line, odds=round(odds, 3), edge="",
                                 source="market", close_odds=round(close, 3),
                                 result=result, pl=pl))
            covered.add((int(num), market))
    # 移除被市場列取代的 model 列（同場同盤口）
    if covered and not df.empty:
        keep = ~df.apply(lambda r: (int(r["match_num"]), r["market"]) in covered
                         if str(r["match_num"]).strip() not in ("", "nan") else False, axis=1)
        df = df[keep]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_ledger(df, ledger_path)
    return len(new_rows)


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


def prepare(matches, model, ledger_path, odds_index=None, min_edge=0.0,
            weight=None, snap_path=None):
    """一站式：記推薦 → 更新收盤賠率 → 用已踢賽果結算 → 回傳 summary。

    snap_path：若給定且有 odds_index，另存「全選項快照」供日後權重校準
    （見 tune_weight）。預設取 ledger 同目錄的 odds_log.csv。
    """
    log_upcoming(matches, model, ledger_path, odds_index=odds_index,
                 min_edge=min_edge, weight=weight)
    results = {m.num: (m.hg, m.ag) for m in matches if m.played}
    if odds_index:
        refresh_close(ledger_path, odds_index)
        snap = snap_path or str(Path(ledger_path).with_name("odds_log.csv"))
        try:
            log_snapshots(matches, model, odds_index, snap)
            settle_snapshots(snap, results)
            settle(ledger_path, results)
            # 由快照重建真實盤口戰績（模型推薦+快照賠率），自我修復、一致
            rebuild_market_from_snapshots(ledger_path, snap)
            return summary(ledger_path)
        except Exception:  # noqa: BLE001 - 快照/重建失敗不影響主流程
            pass
    settle(ledger_path, results)
    return summary(ledger_path)


# ================= CLV 歷史 + 權重校準 =================
SNAP_COLS = ["date", "match_num", "home", "away", "market", "selection",
             "line", "model_p", "odds", "close_odds", "hg", "ag", "played"]


def load_snapshots(path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        for c in SNAP_COLS:
            if c not in df.columns:
                df[c] = ""
        return df[SNAP_COLS]
    return pd.DataFrame(columns=SNAP_COLS)


def log_snapshots(matches, model, odds_index, snap_path, neutral=True):
    """把每場每盤口的「全部選項」模型機率 + 市場賠率存成快照（不過濾 +EV）。

    這是權重校準的原料：之後可用任意融合權重重放，看哪個權重 ROI/CLV 最佳。
    """
    df = load_snapshots(snap_path)
    seen = set(zip(df["match_num"].astype(str), df["market"],
                   df["selection"].astype(str), df["line"].astype(str)))
    rows = []
    for m in matches:
        if m.played or m.team1 not in model.attack or m.team2 not in model.attack:
            continue
        quotes = odds_index.get(m.num)
        if not quotes:
            continue
        mprobs = _model_market_probs(model, m.team1, m.team2, quotes, neutral)
        for (market, line), model_ps in mprobs.items():
            sides = _group_quotes(quotes, market).get(line, {})
            for sel, odds in sides.items():
                if sel not in model_ps:
                    continue
                key = (str(m.num), market, sel, str(line))
                if key in seen:
                    continue
                rows.append(dict(date=m.date, match_num=m.num, home=m.team1,
                                 away=m.team2, market=market, selection=sel,
                                 line=line, model_p=round(model_ps[sel], 5),
                                 odds=round(odds, 3), close_odds=round(odds, 3),
                                 hg="", ag="", played=0))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        save_ledger(df, snap_path)
    # 更新未結算快照的收盤賠率
    refresh = False
    for i, r in df[df["played"].astype(str).isin(["0", "0.0", "", "nan"])].iterrows():
        quotes = odds_index.get(int(r["match_num"])) if odds_index else None
        if not quotes:
            continue
        od = _lookup_odds(quotes, r["market"],
                          _SEL_ZH.get(r["selection"], r["selection"]), r["line"])
        if od:
            df.at[i, "close_odds"] = round(od, 3)
            refresh = True
    if refresh:
        save_ledger(df, snap_path)
    return len(rows)


def settle_snapshots(snap_path, results: dict):
    """把已踢賽果填入快照（hg/ag/played）。"""
    df = load_snapshots(snap_path)
    n = 0
    for i, r in df[~df["played"].astype(str).isin(["1", "1.0"])].iterrows():
        res = results.get(int(r["match_num"]))
        if res is None:
            continue
        df.at[i, "hg"], df.at[i, "ag"], df.at[i, "played"] = int(res[0]), int(res[1]), 1
        n += 1
    if n:
        save_ledger(df, snap_path)
    return n


def tune_weight(snap_path, grid=None, min_edge=0.0):
    """用快照重放不同融合權重，回傳各權重的 (注數, ROI, CLV) 與建議權重。

    對每個權重 w：重做「去 vig + 融合 + 挑 +EV」決策，用真實賽果結算，
    累計 ROI（總損益/注數）與 CLV。回傳 dict（含 'rows' 與 'best_roi'）。
    """
    import numpy as _np
    df = load_snapshots(snap_path)
    df = df[df["played"].astype(str).isin(["1", "1.0"])]
    if df.empty:
        return {"rows": [], "best_roi": None, "n_matches": 0}
    grid = grid if grid is not None else [round(x, 2) for x in _np.arange(0, 1.01, 0.1)]
    # 依 (match, market, line) 聚成決策單位
    groups: dict = {}
    for _, r in df.iterrows():
        key = (int(r["match_num"]), r["market"], str(r["line"]))
        g = groups.setdefault(key, {"sides": {}, "model": {}, "line": r["line"],
                                    "market": r["market"],
                                    "hg": int(r["hg"]), "ag": int(r["ag"])})
        g["sides"][r["selection"]] = _to_float(r["odds"])
        g["model"][r["selection"]] = _to_float(r["model_p"])
        g["close"] = g.get("close", {})
        g["close"][r["selection"]] = _to_float(r["close_odds"])

    rows = []
    for w in grid:
        pl, clv_sum, clv_n, nbet = 0.0, 0.0, 0, 0
        for g in groups.values():
            order = _MARKET_ORDER[g["market"]]
            pick = _pick_bet(g["model"], g["sides"], order, w, min_edge)
            if not pick:
                continue
            sel, odds, _edge, _p = pick
            nbet += 1
            zsel = sel if g["market"] == "1X2" else _SEL_ZH[sel]
            ow = _settle_outcome(g["market"], zsel, g["line"], g["hg"], g["ag"])
            pl += _pl(ow, odds)
            close = g.get("close", {}).get(sel)
            if close and close > 1.0:
                clv_sum += odds / close - 1.0
                clv_n += 1
        roi = pl / nbet if nbet else 0.0
        clv = clv_sum / clv_n if clv_n else 0.0
        rows.append({"weight": w, "n_bets": nbet, "pl": round(pl, 3),
                     "roi": roi, "clv": clv})
    scored = [r for r in rows if r["n_bets"] > 0]
    best = max(scored, key=lambda r: r["roi"]) if scored else None
    return {"rows": rows, "best_roi": best, "n_matches": len(groups)}


def history(ledger_path):
    """已結算的真實盤口下注，依日期排序的累積績效序列（給績效頁畫圖/列表）。"""
    df = load_ledger(ledger_path)
    df = df[(df["source"] == "market") & df["result"].isin(["win", "loss", "push"])]
    if df.empty:
        return []
    df = df.copy()
    df["_pl"] = df["pl"].map(_to_float)
    df = df[df["_pl"].notna()].sort_values(["date", "match_num"])
    out, cum_pl, n, clv_sum, clv_n, beat = [], 0.0, 0, 0.0, 0, 0
    for _, r in df.iterrows():
        n += 1
        cum_pl += r["_pl"]
        taken, close = _to_float(r["odds"]), _to_float(r["close_odds"])
        clv = None
        if taken and close and close > 1.0:
            clv = taken / close - 1.0
            clv_sum += clv
            clv_n += 1
            beat += clv > 0
        out.append({
            "date": r["date"], "match_num": int(r["match_num"]),
            "home": r["home"], "away": r["away"], "market": r["market"],
            "selection": r["selection"], "line": r["line"],
            "odds": taken, "close_odds": close, "clv": clv,
            "result": r["result"], "pl": r["_pl"],
            "cum_pl": round(cum_pl, 3), "n": n,
            "cum_roi": cum_pl / n, "cum_clv": (clv_sum / clv_n) if clv_n else 0.0,
            "beat_rate": beat / clv_n if clv_n else 0.0,
        })
    return out
