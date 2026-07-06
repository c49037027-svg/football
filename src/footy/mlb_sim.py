"""MLB 建構式（constructive）比賽模擬引擎 —— Phase 0。

有別於 mlb.py 的隊級 Dixon–Coles + 負二項（把「得分分布」直接參數化），
本引擎「從事件長出得分」：

  log5（勝算比法）  →  每個打席用「打者事件率 × 投手被打率 ÷ 聯盟均值」
                       算出這組對戰的 BB/1B/2B/3B/HR/出局機率。
  規則引擎/狀態轉移  →  24 個壘-出局狀態，依打擊結果推進跑者、計分。
  蒙地卡羅          →  逐打席模擬整場 N 次，得分分布自然浮現（免假設負二項）。

為什麼值得：log5 吃得到「這位打者 vs 這位投手 / 打序」的資訊，是隊級模型
根本無法表達的粒度；得分的偏態、離散、序列性也都由規則引擎自動正確產生。

Phase 0 誠實範圍與簡化（皆已註明，待後續逐項升級）：
  - 打席結果只分 6 類（BB+HBP / 1B / 2B / 3B / HR / 出局）；出局不含犧牲、
    雙殺、生產性出局；不含盜壘/失誤。
  - 跑者推進採教科書「固定進 n 個壘」模型（1B 進1、2B 進2、3B 全回、HR 清壘），
    會略低估「二壘跑者靠一安回來」等情形——保守方向、可日後改機率化推進。
  - 先發整場（無牛棚切換）；無再見安打截斷；平手沿用 mlb.moneyline 對角線各半。
  - 事件率來自 statsapi 打者/投手季數據；沙箱擋此站，抓取在 Actions/Render 上跑。

輸出對齊 mlb.MLBMarkets 的比分矩陣，直接沿用既有錢線/大小/讓分與 picks/tracker。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 打席結果類別（順序固定，供 log5 與抽樣共用）
CATS = ("bb", "1b", "2b", "3b", "hr", "out")

# 聯盟每打席基準率（近年 MLB 近似值；league_rates() 可由真實資料覆蓋）
LEAGUE_RATES: dict[str, float] = {
    "bb": 0.090, "1b": 0.140, "2b": 0.045, "3b": 0.004, "hr": 0.033, "out": 0.688,
}


# ---------------- 事件率（由 statsapi 數據換算成每打席機率） ----------------
def _norm(rates: dict) -> dict:
    s = sum(max(rates.get(e, 0.0), 0.0) for e in CATS)
    if s <= 0:
        return dict(LEAGUE_RATES)
    return {e: max(rates.get(e, 0.0), 0.0) / s for e in CATS}


def rates_from_batting(line: dict) -> dict:
    """打者季數據 → 每打席事件率。line 需含 pa/h/2b/3b/hr/bb/hbp（缺者當 0）。

    1B = H − 2B − 3B − HR；BB 類 = BB + HBP；出局 = PA − H − BB − HBP（含 K、
    場內出局、犧牲）。PA 太小回聯盟均值（避免小樣本噪音）。
    """
    pa = float(line.get("pa") or 0)
    if pa < 1:
        return dict(LEAGUE_RATES)
    h = float(line.get("h") or 0)
    d2 = float(line.get("2b") or 0)
    d3 = float(line.get("3b") or 0)
    hr = float(line.get("hr") or 0)
    bb = float(line.get("bb") or 0) + float(line.get("hbp") or 0)
    b1 = max(h - d2 - d3 - hr, 0.0)
    out = max(pa - h - bb, 0.0)
    return _norm({"bb": bb / pa, "1b": b1 / pa, "2b": d2 / pa,
                  "3b": d3 / pa, "hr": hr / pa, "out": out / pa})


def rates_from_pitching(line: dict) -> dict:
    """投手季「被打」數據 → 每打席被打事件率。line 需含 bf/h/2b/3b/hr/bb/hbp。

    bf（面對打者數）缺時用 IP·4.3 近似。與打者同 6 類，供 log5 對戰。
    """
    bf = float(line.get("bf") or 0)
    if bf < 1:
        ip = float(line.get("ip") or 0)
        bf = ip * 4.3
    if bf < 1:
        return dict(LEAGUE_RATES)
    h = float(line.get("h") or 0)
    d2 = float(line.get("2b") or 0)
    d3 = float(line.get("3b") or 0)
    hr = float(line.get("hr") or 0)
    bb = float(line.get("bb") or 0) + float(line.get("hbp") or 0)
    b1 = max(h - d2 - d3 - hr, 0.0)
    out = max(bf - h - bb, 0.0)
    return _norm({"bb": bb / bf, "1b": b1 / bf, "2b": d2 / bf,
                  "3b": d3 / bf, "hr": hr / bf, "out": out / bf})


def league_rates(batting_lines: list[dict]) -> dict:
    """由全體打者數據彙整聯盟每打席均值（log5 正規化用）。空 → 內建近似值。"""
    tot = {e: 0.0 for e in CATS}
    pa_sum = 0.0
    for ln in batting_lines:
        pa = float(ln.get("pa") or 0)
        if pa < 1:
            continue
        r = rates_from_batting(ln)
        for e in CATS:
            tot[e] += r[e] * pa
        pa_sum += pa
    if pa_sum <= 0:
        return dict(LEAGUE_RATES)
    return _norm({e: tot[e] / pa_sum for e in CATS})


# ---------------- log5（勝算比法，多結果推廣） ----------------
def log5_matchup(batter: dict, pitcher: dict, league: dict | None = None,
                 park: float = 1.0) -> dict:
    """打者×投手×聯盟 → 這組對戰的每打席事件機率（勝算比法後正規化）。

    matchup[e] ∝ 打者率[e] · 投手率[e] / 聯盟率[e]（Bill James log5 對兩結果的
    多結果推廣）。park>1 的打者天堂把安打類（1B/2B/3B/HR）以 park^0.5 放大後再正規化。
    """
    lg = league or LEAGUE_RATES
    raw = {}
    for e in CATS:
        l = lg.get(e, 0.0)
        raw[e] = (batter.get(e, 0.0) * pitcher.get(e, 0.0) / l) if l > 0 else 0.0
    if park and park != 1.0:
        f = park ** 0.5
        for e in ("1b", "2b", "3b", "hr"):
            raw[e] *= f
    return _norm(raw)


# ---------------- 規則引擎：壘-出局狀態轉移 ----------------
# 跑者額外進壘機率（run-expectancy 文獻近似值）；rng=None 時退回保守固定進壘。
P_2ND_SCORES_ON_1B = 0.60   # 一安時二壘跑者奔回本壘
P_1ST_TO_3RD_ON_1B = 0.28   # 一安時一壘跑者搶上三壘
P_1ST_SCORES_ON_2B = 0.42   # 二安時一壘跑者奔回本壘


def advance(b1: bool, b2: bool, b3: bool, outs: int, cat: str, rng=None):
    """套用一個打席結果，回傳 (b1, b2, b3, outs, runs)。b* 為各壘是否有人。

    保送只推進被迫跑者；長打清壘。單/二壘安打的跑者額外進壘：給 rng 時依上方
    機率隨機（較貼近真實得分環境），rng=None 時退回保守「固定進 n 個壘」。
    """
    if cat == "out":
        return b1, b2, b3, outs + 1, 0
    if cat == "bb":
        if b1 and b2 and b3:
            return True, True, True, outs, 1          # 滿壘保送擠回一分
        if b1 and b2:
            return True, True, True, outs, 0
        if b1:
            return True, True, b3, outs, 0            # 三壘跑者未被迫
        return True, b2, b3, outs, 0                  # 一壘空，無人被迫
    if cat == "hr":
        return False, False, False, outs, int(b1) + int(b2) + int(b3) + 1
    if cat == "3b":
        return False, False, True, outs, int(b1) + int(b2) + int(b3)
    if cat == "1b":
        runs = 0
        n3 = False
        if b3:
            runs += 1                                 # 三壘跑者回本壘
        if b2:
            if rng is not None and rng.random() < P_2ND_SCORES_ON_1B:
                runs += 1
            else:
                n3 = True
        n2 = False
        if b1:
            if rng is not None and not n3 and rng.random() < P_1ST_TO_3RD_ON_1B:
                n3 = True
            else:
                n2 = True
        return True, n2, n3, outs, runs               # 打者上一壘
    if cat == "2b":
        runs = int(b3) + int(b2)                       # 二、三壘跑者皆回
        n3 = False
        if b1:
            if rng is not None and rng.random() < P_1ST_SCORES_ON_2B:
                runs += 1
            else:
                n3 = True
        return False, True, n3, outs, runs             # 打者上二壘
    raise ValueError(f"unknown outcome: {cat}")


# ---------------- 蒙地卡羅：逐打席模擬 ----------------
def _cdf(probs: dict) -> np.ndarray:
    return np.cumsum([probs[e] for e in CATS])


def _lineup_cdfs(lineup: list[dict], pitcher: dict, league: dict,
                 park: float) -> list[np.ndarray]:
    """先把 9 名打者對該投手的對戰機率算好、轉累積分布（整場不變 → 大幅加速）。"""
    return [_cdf(log5_matchup(b, pitcher, league, park)) for b in lineup]


def _sim_half(cdfs: list[np.ndarray], idx: int, rng) -> tuple[int, int]:
    """模擬半局：回傳 (得分, 下一位打序 idx)。cdfs 為各打序的累積事件分布。"""
    b1 = b2 = b3 = False
    outs = runs = 0
    n = len(cdfs)
    while outs < 3:
        cdf = cdfs[idx % n]
        cat = CATS[int(np.searchsorted(cdf, rng.random()))]
        b1, b2, b3, outs, r = advance(b1, b2, b3, outs, cat, rng)
        runs += r
        idx += 1
    return runs, idx


def simulate_matrix(home_lineup: list[dict], away_lineup: list[dict],
                    home_pitcher: dict, away_pitcher: dict,
                    league: dict | None = None, n_sims: int = 5000,
                    innings: int = 9, park: float = 1.0,
                    max_runs: int = 20, seed: int | None = None) -> np.ndarray:
    """蒙地卡羅整場 → 正規化比分矩陣 mat[h, a]（可直接餵 mlb.moneyline 等）。

    客隊打主隊先發、主隊打客隊先發；park 只作用在主場（兩隊都在此球場打）。
    回傳 (max_runs+1)² 矩陣，行=主隊得分、列=客隊得分。
    """
    lg = league or LEAGUE_RATES
    rng = np.random.default_rng(seed)
    home_cdfs = _lineup_cdfs(home_lineup, away_pitcher, lg, park)   # 主隊打客投
    away_cdfs = _lineup_cdfs(away_lineup, home_pitcher, lg, park)   # 客隊打主投
    mat = np.zeros((max_runs + 1, max_runs + 1))
    hi_idx = ai_idx = 0
    for _ in range(n_sims):
        hr = ar = 0
        for _ in range(innings):
            r_a, ai_idx = _sim_half(away_cdfs, ai_idx, rng)
            ar += r_a
            r_h, hi_idx = _sim_half(home_cdfs, hi_idx, rng)
            hr += r_h
        mat[min(hr, max_runs), min(ar, max_runs)] += 1
    s = mat.sum()
    return mat / s if s > 0 else mat


def analyze_game_sim(home_lineup: list[dict], away_lineup: list[dict],
                     home_pitcher: dict, away_pitcher: dict,
                     league: dict | None = None, total_line: float = 8.5,
                     run_line: float = -1.5, park: float = 1.0,
                     n_sims: int = 5000, top_n: int = 4, seed: int | None = None):
    """用模擬矩陣算錢線/大小/讓分，回傳 mlb.MLBMarkets（與 NB 版介面相同）。"""
    from . import mlb
    from .models import markets
    mat = simulate_matrix(home_lineup, away_lineup, home_pitcher, away_pitcher,
                          league=league, n_sims=n_sims, park=park, seed=seed)
    p_h, p_a = mlb.moneyline(mat)
    ou = markets.over_under(mat, total_line)
    ah = markets.asian_handicap(mat, run_line, "home")
    cover = ah.p_win + ah.p_half_win + 0.5 * ah.p_push
    ks = np.arange(mat.shape[0])
    exp_h = float((mat.sum(axis=1) * ks).sum())
    exp_a = float((mat.sum(axis=0) * ks).sum())
    flat = [((h, a), float(mat[h, a]))
            for h in range(mat.shape[0]) for a in range(mat.shape[1])]
    flat.sort(key=lambda x: -x[1])
    return mlb.MLBMarkets(
        p_home=p_h, p_away=p_a,
        ml_home_odds=round(1.0 / p_h, 2) if p_h else 0.0,
        ml_away_odds=round(1.0 / p_a, 2) if p_a else 0.0,
        total_line=total_line, p_over=ou["over_win"], p_under=ou["under_win"],
        run_line=run_line, p_cover_home=float(cover),
        exp_home=exp_h, exp_away=exp_a, top_scores=flat[:top_n])


# ---------------- 資料層（statsapi；沙箱擋，抓取在部署環境） ----------------
def parse_batting_stats(payload: dict) -> list[dict]:
    """解析 /stats?group=hitting 回應成打者事件計數列（純函式、可測試）。"""
    rows = []
    for block in payload.get("stats", []):
        for sp in block.get("splits", []):
            st = sp.get("stat") or {}
            pl = sp.get("player") or {}
            if pl.get("id") is None:
                continue
            rows.append({
                "id": pl["id"], "name": pl.get("fullName", ""),
                "team": (sp.get("team") or {}).get("name", ""),
                "pa": int(st.get("plateAppearances") or 0),
                "h": int(st.get("hits") or 0),
                "2b": int(st.get("doubles") or 0),
                "3b": int(st.get("triples") or 0),
                "hr": int(st.get("homeRuns") or 0),
                "bb": int(st.get("baseOnBalls") or 0),
                "hbp": int(st.get("hitByPitch") or 0),
            })
    return rows


def parse_pitching_events(payload: dict) -> list[dict]:
    """解析 /stats?group=pitching 成投手「被打」事件列（供 rates_from_pitching）。"""
    rows = []
    for block in payload.get("stats", []):
        for sp in block.get("splits", []):
            st = sp.get("stat") or {}
            pl = sp.get("player") or {}
            if pl.get("id") is None:
                continue
            rows.append({
                "id": pl["id"], "name": pl.get("fullName", ""),
                "team": (sp.get("team") or {}).get("name", ""),
                "bf": int(st.get("battersFaced") or 0),
                "h": int(st.get("hits") or 0),
                "2b": int(st.get("doubles") or 0),
                "3b": int(st.get("triples") or 0),
                "hr": int(st.get("homeRuns") or 0),
                "bb": int(st.get("baseOnBalls") or 0),
                "hbp": int(st.get("hitByPitch") or 0),
            })
    return rows


def parse_lineups(box_payload: dict) -> dict:
    """從單場 boxscore 解析雙方打序（player id 依打順排列）。

    回傳 {"home": [id...], "away": [id...]}；未公布時對應清單為空。
    """
    out = {"home": [], "away": []}
    teams = box_payload.get("teams", {})
    for side in ("home", "away"):
        info = teams.get(side, {})
        order = info.get("battingOrder") or []
        out[side] = [int(x) for x in order] if order else []
    return out


def fetch_batting(season: int, timeout: float = 30.0) -> list[dict]:
    """抓整季全聯盟打者事件數據（部署環境用）。"""
    import requests

    from .mlb import STATSAPI
    r = requests.get(f"{STATSAPI}/stats",
                     params={"stats": "season", "group": "hitting",
                             "season": season, "sportIds": 1, "gameType": "R",
                             "playerPool": "ALL", "limit": 3000},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_batting_stats(r.json())


def fetch_lineups(game_pk: int, timeout: float = 30.0) -> dict:
    """抓單場 boxscore 的雙方打序（賽前數小時才公布；未公布回空）。"""
    import requests

    from .mlb import STATSAPI
    r = requests.get(f"{STATSAPI.replace('/v1', '/v1')}/game/{game_pk}/boxscore",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_lineups(r.json())
