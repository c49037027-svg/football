"""MLB 建構式（constructive）比賽模擬引擎 —— Phase 0。

有別於 mlb.py 的隊級 Dixon–Coles + 負二項（把「得分分布」直接參數化），
本引擎「從事件長出得分」：

  log5（勝算比法）  →  每個打席用「打者事件率 × 投手被打率 ÷ 聯盟均值」
                       算出這組對戰的 BB/1B/2B/3B/HR/出局機率。
  規則引擎/狀態轉移  →  24 個壘-出局狀態，依打擊結果推進跑者、計分。
  蒙地卡羅          →  逐打席模擬整場 N 次，得分分布自然浮現（免假設負二項）。

為什麼值得：log5 吃得到「這位打者 vs 這位投手 / 打序」的資訊，是隊級模型
根本無法表達的粒度；得分的偏態、離散、序列性也都由規則引擎自動正確產生。

⚠️ 回測結論（2025-06→07、200 場，見 docs/FINDINGS.md）：本引擎樣本外**未贏過**
現行負二項(NB)——連「當季全季率」的洩漏上限診斷都輸（錢線 0.698 vs NB 0.690）。
故**研究用、不接生產**；網站預測仍走 mlb.py 的 NB。日後補左右分裂+牛棚可重評估。

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


# ---------------- 打線攻擊係數（把打者資訊接進隊級 NB 的外科手術層） ----------------
# 線性加權：每事件的近似得分價值（wOBA 尺度，出局取 0，供「相對」比較用）。
_LINEAR_WEIGHTS = {"bb": 0.33, "1b": 0.47, "2b": 0.77, "3b": 1.04, "hr": 1.40, "out": 0.0}


def offensive_rate(rates: dict) -> float:
    """由每打席事件率算「每打席得分產能」（線性加權）。用於打線間的相對比較。"""
    return sum(_LINEAR_WEIGHTS[e] * rates.get(e, 0.0) for e in CATS)


class LineupBook:
    """打線攻擊係數簿：今日 9 人相對「該隊季平均打線」的得分乘數。

    與 PitcherBook 對稱設計（避免與隊級 NB 重複計算）：隊級模型已含全隊平均攻擊，
    這裡只修正「今晚這幾位相對隊內平均」的差；係數夾在 [0.85,1.15]，混 0.7 權重
    （單場打線只佔比賽一部分、且有變異）。查無球員（新秀/未登錄）→ 從均值中略過。

    ⚠️ 回測結論（400 場，見 docs/FINDINGS.md）：接上此層的 NB+打線 樣本外**未贏過** NB
    （連當季率洩漏上限都只打平、大小盤還變差）。故**研究用、不接生產**；analyze_game 的
    home_bat_factor/away_bat_factor 預設 1.0（關閉）。
    """

    CLIP = (0.85, 1.15)
    WEIGHT = 0.7

    def __init__(self, bat_rows: list[dict], league: dict | None = None):
        """bat_rows：fetch_batting 產物（{id,name,team,pa,h,2b,3b,hr,bb,hbp}）。"""
        self.league = league or LEAGUE_RATES
        self.by_id: dict[int, float] = {}
        team_sum: dict[str, float] = {}
        team_pa: dict[str, float] = {}
        for r in bat_rows:
            pa = float(r.get("pa") or 0)
            if pa < 1:
                continue
            rate = offensive_rate(rates_from_batting(r))
            self.by_id[int(r["id"])] = rate
            t = r.get("team", "")
            if t:
                team_sum[t] = team_sum.get(t, 0.0) + rate * pa
                team_pa[t] = team_pa.get(t, 0.0) + pa
        self.team_avg = {t: team_sum[t] / team_pa[t]
                         for t in team_pa if team_pa[t] > 0}
        self.league_avg = offensive_rate(self.league) or 1.0

    def factor(self, batter_ids, team: str | None = None) -> float:
        """今日打序 → 得分乘數。基準用該隊季平均打線（查無隊則用聯盟均值）。"""
        ids = [int(i) for i in (batter_ids or []) if int(i) in self.by_id]
        if not ids:
            return 1.0
        today = sum(self.by_id[i] for i in ids) / len(ids)
        base = self.team_avg.get(team) if team else None
        if not base or base <= 0:
            base = self.league_avg
        raw = today / base if base > 0 else 1.0
        raw = min(max(raw, self.CLIP[0]), self.CLIP[1])
        return self.WEIGHT * raw + (1.0 - self.WEIGHT)


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
    out = {"home": [], "away": [], "home_sp": None, "away_sp": None}
    teams = box_payload.get("teams", {})
    for side in ("home", "away"):
        info = teams.get(side, {})
        order = info.get("battingOrder") or []
        out[side] = [int(x) for x in order] if order else []
        pitchers = info.get("pitchers") or []      # 依上場順序，[0] 為先發
        out[f"{side}_sp"] = int(pitchers[0]) if pitchers else None
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


def fetch_pitching_events(season: int, timeout: float = 30.0) -> list[dict]:
    """抓整季全聯盟投手「被打」事件數據（部署環境用）。"""
    import requests

    from .mlb import STATSAPI
    r = requests.get(f"{STATSAPI}/stats",
                     params={"stats": "season", "group": "pitching",
                             "season": season, "sportIds": 1, "gameType": "R",
                             "playerPool": "ALL", "limit": 3000},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_pitching_events(r.json())


def fetch_lineups(game_pk: int, timeout: float = 30.0) -> dict:
    """抓單場 boxscore 的雙方打序（賽前數小時才公布；未公布回空）。"""
    import requests

    from .mlb import STATSAPI
    r = requests.get(f"{STATSAPI}/game/{game_pk}/boxscore",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_lineups(r.json())


# ================= Phase 3：回測骨架（event-sim vs 負二項，樣本外比對） =================
#
# 誠實原則（見 docs/FINDINGS.md）：新方法要 walk-forward 樣本外贏過現行 NB 才採用。
# 難點是「時間點正確」的球員率——本骨架用「前一季」事件率當零洩漏近似（測試季不含
# 當季資訊；缺點：忽略當季成長與新秀，故偏保守、對強弱差略鈍）。真正上線前可換成
# 「截至當日的逐場累積率」進一步精確。純評分/組裝為可離線測試的函式；連線資料組裝
# 走部署環境的 CLI（footy mlb backtest-sim）。


def _logloss(p: float, y: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def score_market(rows: list[tuple]) -> dict:
    """rows: [(p, y)...]，y∈{0,1}。回傳 {logloss, brier, n, mean_p, base_rate}。"""
    n = len(rows)
    if n == 0:
        return {"logloss": None, "brier": None, "n": 0}
    ll = sum(_logloss(p, y) for p, y in rows) / n
    br = sum((p - y) ** 2 for p, y in rows) / n
    return {"logloss": float(ll), "brier": float(br), "n": n,
            "mean_p": sum(p for p, _ in rows) / n,
            "base_rate": sum(y for _, y in rows) / n}


def build_game_record(result: dict, lineups: dict, bat_rates: dict,
                      pit_rates: dict, league: dict | None = None,
                      total_line: float = 8.5, run_line: float = -1.5,
                      park: float = 1.0) -> dict:
    """組一場回測記錄（純函式）。缺打序 → home_lineup 為空 → sim 該場略過（誠實覆蓋缺口）。

    result: {home, away, home_goals, away_goals, home_pitcher_id, away_pitcher_id}
    lineups: {"home":[id...], "away":[id...]}（parse_lineups 產物）
    bat_rates/pit_rates: {player_id: 每打席事件率 dict}
    缺球員 → 退回聯盟率（新秀/查無）。
    """
    lg = league or LEAGUE_RATES

    def lineup_rates(ids):
        return [bat_rates.get(int(i), dict(lg)) for i in ids] if ids else []

    def pit(pid):
        return pit_rates.get(int(pid), dict(lg)) if pid is not None else dict(lg)

    return {
        "home": result.get("home"), "away": result.get("away"),
        "date": result.get("date"),
        "home_score": int(result["home_goals"]), "away_score": int(result["away_goals"]),
        "home_lineup": lineup_rates(lineups.get("home") or []),
        "away_lineup": lineup_rates(lineups.get("away") or []),
        "home_lineup_ids": [int(i) for i in (lineups.get("home") or [])],
        "away_lineup_ids": [int(i) for i in (lineups.get("away") or [])],
        # 先發投手 id：優先用 boxscore 實際先發，退回賽程預告
        "home_sp": lineups.get("home_sp") or result.get("home_pitcher_id"),
        "away_sp": lineups.get("away_sp") or result.get("away_pitcher_id"),
        "home_pitcher": pit(result.get("home_pitcher_id")),
        "away_pitcher": pit(result.get("away_pitcher_id")),
        "league": lg, "park": park, "total_line": total_line, "run_line": run_line,
    }


def sim_predictor(n_sims: int = 3000, seed: int = 0):
    """回測用 event-sim 預測器：game → {p_home, p_over, p_cover_home} 或 None（無打序）。"""
    def f(g):
        if not g.get("home_lineup") or not g.get("away_lineup"):
            return None
        m = analyze_game_sim(g["home_lineup"], g["away_lineup"],
                             g["home_pitcher"], g["away_pitcher"], g.get("league"),
                             total_line=g["_total_line"], run_line=g.get("run_line", -1.5),
                             park=g.get("park", 1.0), n_sims=n_sims, seed=seed)
        return {"p_home": m.p_home, "p_over": m.p_over, "p_cover_home": m.p_cover_home}
    return f


def nb_predictor(model, dispersion: float | None = None):
    """回測用負二項預測器（現行隊級模型）：game → {p_home, p_over, p_cover_home}。"""
    from . import mlb

    def f(g):
        if g["home"] not in model.attack or g["away"] not in model.attack:
            return None
        m = mlb.analyze_game(model, g["home"], g["away"], total_line=g["_total_line"],
                             run_line=g.get("run_line", -1.5), park_factor=g.get("park", 1.0),
                             dispersion=dispersion)
        return {"p_home": m.p_home, "p_over": m.p_over, "p_cover_home": m.p_cover_home}
    return f


def nb_lineup_predictor(model, book: "LineupBook", dispersion: float | None = None):
    """NB + 打線係數：在現行 NB 上，用今日打序相對隊平均的偏差微調 λ（外科手術層）。

    這是「把打者資訊接進生產 NB」的正確做法——保留 NB，只加今日打線的偏差修正。
    """
    from . import mlb

    def f(g):
        if g["home"] not in model.attack or g["away"] not in model.attack:
            return None
        hbf = book.factor(g.get("home_lineup_ids"), team=g["home"])
        abf = book.factor(g.get("away_lineup_ids"), team=g["away"])
        m = mlb.analyze_game(model, g["home"], g["away"], total_line=g["_total_line"],
                             run_line=g.get("run_line", -1.5), park_factor=g.get("park", 1.0),
                             dispersion=dispersion, home_bat_factor=hbf, away_bat_factor=abf)
        return {"p_home": m.p_home, "p_over": m.p_over, "p_cover_home": m.p_cover_home}
    return f


def nb_pitcher_predictor(model, form_book, halflife: float, dispersion: float | None = None):
    """NB + 先發投手（近況加權）：用 PitcherFormBook 依比賽日 point-in-time 算先發係數。

    halflife 小 → 重近況；極大（如 1e9）→ 等於季至今平均（可與近況版對比）。
    主隊先發壓客隊得分、客隊先發壓主隊得分（乘在對手 λ）。
    """
    from . import mlb

    def f(g):
        if g["home"] not in model.attack or g["away"] not in model.attack:
            return None
        as_of = g.get("date")
        hpf = form_book.factor(g.get("home_sp"), as_of=as_of, halflife=halflife)
        apf = form_book.factor(g.get("away_sp"), as_of=as_of, halflife=halflife)
        m = mlb.analyze_game(model, g["home"], g["away"], total_line=g["_total_line"],
                             run_line=g.get("run_line", -1.5), park_factor=g.get("park", 1.0),
                             dispersion=dispersion,
                             home_pitcher_factor=hpf, away_pitcher_factor=apf)
        return {"p_home": m.p_home, "p_over": m.p_over, "p_cover_home": m.p_cover_home}
    return f


def compare_backtest(games: list[dict], predictors: dict,
                     total_line: float = 8.5) -> dict:
    """對同一批賽果，逐一比較各預測器的錢線與大小 log-loss/Brier（樣本外）。

    predictors: {name: fn(game)->{p_home,p_over,...} 或 None}。回 None 的場次該預測器
    略過（覆蓋數 n 會顯示差異——sim 只涵蓋有打序的場次，這是誠實的覆蓋缺口）。
    回傳 {name: {"ml": score, "ou": score}}，並含各自實際涵蓋場數。
    """
    acc = {name: {"ml": [], "ou": []} for name in predictors}
    for g in games:
        g = {**g, "_total_line": total_line}
        hs, as_ = g["home_score"], g["away_score"]
        y_home = 1.0 if hs > as_ else 0.0
        y_over = 1.0 if hs + as_ > total_line else 0.0
        tie = hs == as_
        for name, fn in predictors.items():
            pr = fn(g)
            if pr is None:
                continue
            if not tie:                       # 錢線無和局，平手（極罕見補賽）跳過
                acc[name]["ml"].append((pr["p_home"], y_home))
            acc[name]["ou"].append((pr["p_over"], y_over))
    return {name: {"ml": score_market(d["ml"]), "ou": score_market(d["ou"])}
            for name, d in acc.items()}


def format_backtest(res: dict) -> str:
    """把 compare_backtest 結果印成對照表。"""
    lines = ["模型            | 錢線 n | 錢線 LL | 大小 n | 大小 LL | 大小 Brier",
             "----------------|-------:|--------:|-------:|--------:|----------:"]
    for name, d in res.items():
        ml, ou = d["ml"], d["ou"]
        ll = f"{ml['logloss']:.4f}" if ml["logloss"] is not None else "  —  "
        oll = f"{ou['logloss']:.4f}" if ou["logloss"] is not None else "  —  "
        obr = f"{ou['brier']:.4f}" if ou["brier"] is not None else "  —  "
        lines.append(f"{name:<15} | {ml['n']:>6} | {ll:>7} | "
                     f"{ou['n']:>6} | {oll:>7} | {obr:>10}")
    return "\n".join(lines)
