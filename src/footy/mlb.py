"""MLB（美國職棒）預測：資料抓取 + 盤口計算。

資料源：MLB 官方免費公開 Stats API（statsapi.mlb.com，無需金鑰）。
沙箱網路會擋此站——實際抓取在 Render / GitHub Actions 上跑（同 the-odds-api 模式）。

模型：重用 Dixon–Coles 攻防架構擬合「得分（runs）」——
  max_goals=20（棒球得分高）、rho=0（低比分修正是足球特有）、不用 Elo。
棒球沒有和局（延長賽分勝負）：錢線把比分矩陣對角線（9 局打平）各半分給兩隊。

v1 誠實限制：
  - 未建模「先發投手」——這是 MLB 最大單一因子，隊級模型對盤口會偏鈍；
    today 會顯示預告先發供人工判斷。
  - Poisson 對 MLB 得分略低估變異（實際 var/mean≈1.1~1.4），大小盤位於
    極端線時會略偏保守。屬可接受的基線，之後可換負二項分布。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import markets

STATSAPI = "https://statsapi.mlb.com/api/v1"

# 30 隊中文名（台灣慣用譯名）
MLB_ZH = {
    "Arizona Diamondbacks": "響尾蛇", "Atlanta Braves": "勇士",
    "Baltimore Orioles": "金鶯", "Boston Red Sox": "紅襪",
    "Chicago Cubs": "小熊", "Chicago White Sox": "白襪",
    "Cincinnati Reds": "紅人", "Cleveland Guardians": "守護者",
    "Colorado Rockies": "洛磯", "Detroit Tigers": "老虎",
    "Houston Astros": "太空人", "Kansas City Royals": "皇家",
    "Los Angeles Angels": "天使", "Los Angeles Dodgers": "道奇",
    "Miami Marlins": "馬林魚", "Milwaukee Brewers": "釀酒人",
    "Minnesota Twins": "雙城", "New York Mets": "大都會",
    "New York Yankees": "洋基", "Athletics": "運動家",
    "Oakland Athletics": "運動家", "Philadelphia Phillies": "費城人",
    "Pittsburgh Pirates": "海盜", "San Diego Padres": "教士",
    "San Francisco Giants": "巨人", "Seattle Mariners": "水手",
    "St. Louis Cardinals": "紅雀", "Tampa Bay Rays": "光芒",
    "Texas Rangers": "遊騎兵", "Toronto Blue Jays": "藍鳥",
    "Washington Nationals": "國民",
}


def zh_mlb(name: str) -> str:
    return MLB_ZH.get(name, name)


# ---------------- 資料抓取（statsapi.mlb.com） ----------------
def parse_schedule(payload: dict, finals_only: bool = True) -> list[dict]:
    """解析 /schedule 回應成賽果列。純函式、可測試。

    回傳 [{date, home, away, home_goals, away_goals, status,
           home_pitcher, away_pitcher}]；finals_only 時只留已完賽且有比分者。
    """
    rows = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") not in ("R", "F", "D", "L", "W"):
                continue  # 只要例行賽與季後賽，略過熱身/明星賽
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            status = (g.get("status") or {}).get("abstractGameState", "")
            hs, as_ = home.get("score"), away.get("score")
            if finals_only and (status != "Final" or hs is None or as_ is None):
                continue
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "date": g.get("officialDate") or day.get("date"),
                "game_date_iso": g.get("gameDate"),   # UTC ISO 開賽時刻（含未開打）
                "home": (home.get("team") or {}).get("name", ""),
                "away": (away.get("team") or {}).get("name", ""),
                "home_goals": hs, "away_goals": as_,
                "status": status,
                "home_pitcher": hp.get("fullName", ""),
                "away_pitcher": ap.get("fullName", ""),
                "home_pitcher_id": hp.get("id"),
                "away_pitcher_id": ap.get("id"),
            })
    return rows


def fetch_games(start: str, end: str, timeout: float = 30.0) -> list[dict]:
    """抓一段日期的已完賽賽果（例行賽+季後賽）。需要可連 statsapi.mlb.com。"""
    import requests
    r = requests.get(f"{STATSAPI}/schedule",
                     params={"sportId": 1, "startDate": start, "endDate": end},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_schedule(r.json(), finals_only=True)


def fetch_seasons(years: list[int], out_path: str | Path) -> int:
    """抓多個球季（3~11 月，按月分段抓避免單次過大）寫成訓練 CSV。回傳筆數。"""
    rows: list[dict] = []
    for y in years:
        for m in range(3, 12):
            start = f"{y}-{m:02d}-01"
            end = f"{y}-{m:02d}-28" if m == 2 else f"{y}-{m:02d}-30"
            if m in (3, 5, 7, 8, 10):
                end = f"{y}-{m:02d}-31"
            try:
                rows.extend(fetch_games(start, end))
            except Exception:  # noqa: BLE001 - 空月份/網路暫時失敗略過
                continue
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "home", "away",
                                          "home_goals", "away_goals"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    return len(rows)


def fetch_today(date: str, timeout: float = 30.0) -> list[dict]:
    """抓某日賽程（含未開打，帶預告先發）。"""
    import requests
    r = requests.get(f"{STATSAPI}/schedule",
                     params={"sportId": 1, "date": date,
                             "hydrate": "probablePitcher"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_schedule(r.json(), finals_only=False)


# ---------------- 負二項比分分布（棒球得分過度離散） ----------------
def nb_score_matrix(lam: float, mu: float, k: float | None,
                    max_goals: int = 20) -> np.ndarray:
    """負二項聯合比分矩陣（兩邊獨立）。k 為離散度：var = m + m²/k。

    棒球得分遠比 Poisson 離散（得分常一局爆量），Poisson 會高估強隊錢線、
    低估大小盤尾端。k=None 或極大 → 退回 Poisson。
    """
    from scipy import stats
    ks = np.arange(max_goals + 1)
    if k is None or k <= 0 or k > 1e6:
        h = stats.poisson.pmf(ks, lam)
        a = stats.poisson.pmf(ks, mu)
    else:
        # NB 參數化：n=k, p=k/(k+m) → 平均=m、變異=m+m²/k
        h = stats.nbinom.pmf(ks, k, k / (k + lam))
        a = stats.nbinom.pmf(ks, k, k / (k + mu))
    mat = np.outer(h, a)
    s = mat.sum()
    return mat / s if s > 0 else mat


def dispersion_from_df(df) -> float | None:
    """由賽果估離散度 k（動差法）：k = m²/(v−m)。v≤m（無過度離散）回 None。

    註：原始變異含「對戰強弱差」成分，會略高估離散（k 略低、分布略寬），
    屬保守方向；回測驗證見 docs/FINDINGS.md。
    """
    runs = np.concatenate([df["home_goals"].to_numpy(float),
                           df["away_goals"].to_numpy(float)])
    m, v = float(runs.mean()), float(runs.var())
    if v <= m or m <= 0:
        return None
    return m * m / (v - m)


def dispersion_from_csv(path: str | Path) -> float | None:
    import pandas as pd
    p = Path(path)
    if not p.exists():
        return None
    try:
        return dispersion_from_df(pd.read_csv(p))
    except Exception:  # noqa: BLE001
        return None


# ---------------- 球場因子（park factors） ----------------
def park_factors(df, prior_games: float = 80.0,
                 clip: tuple = (0.90, 1.12)) -> dict[str, float]:
    """由訓練資料估各主場的得分環境：PF = 該隊主場總得分 ÷ 聯盟平均（收縮+夾限）。

    Coors Field（洛磯）這類打者天堂 PF>1。套用時每邊乘 PF^0.5（總分約放大 PF 倍）；
    隊級評分已吸收部分球場效應，此為文獻常用的近似修正，故夾限保守。
    df 需含 home/home_goals/away_goals 欄。
    """
    total = df["home_goals"] + df["away_goals"]
    league = float(total.mean())
    if not league:
        return {}
    out = {}
    for team, grp in df.groupby("home"):
        s = float((grp["home_goals"] + grp["away_goals"]).sum())
        n = len(grp)
        pf = (s + prior_games * league) / ((n + prior_games) * league)
        out[str(team)] = min(max(pf, clip[0]), clip[1])
    return out


def park_factors_from_csv(path: str | Path) -> dict[str, float]:
    """從訓練 CSV 算球場因子；檔案不存在回空 dict（不調整）。"""
    import pandas as pd
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return park_factors(pd.read_csv(p))
    except Exception:  # noqa: BLE001
        return {}


# ---------------- 球隊戰力表 ----------------
def team_power(model) -> list[dict]:
    """每隊對聯盟平均對手的期望得分/失分（主客各半），依淨值排序。"""
    teams = sorted(model.attack)
    rows = []
    for t in teams:
        rf = ra = 0.0
        n = 0
        for u in teams:
            if u == t:
                continue
            lam_h, mu_h = model.expected_goals(t, u)   # t 在主場
            lam_a, mu_a = model.expected_goals(u, t)   # t 作客
            rf += lam_h + mu_a
            ra += mu_h + lam_a
            n += 2
        rows.append({"team": t, "rf": rf / n, "ra": ra / n,
                     "diff": (rf - ra) / n})
    rows.sort(key=lambda r: -r["diff"])
    return rows


# ---------------- 先發投手評分（RA/9 + 收縮估計） ----------------
def ip_to_float(ip: str | float | None) -> float:
    """statsapi 的局數是字串，小數位是「出局數」：'123.2' = 123 又 2/3 局。"""
    if ip in (None, ""):
        return 0.0
    s = str(ip)
    if "." in s:
        whole, outs = s.split(".", 1)
        return int(whole or 0) + int(outs[:1] or 0) / 3.0
    return float(s)


def parse_pitcher_stats(payload: dict) -> list[dict]:
    """解析 /stats?stats=season&group=pitching 回應。純函式、可測試。

    回傳 [{id, name, team, ip, runs, gs}]（ip 已轉為浮點局數）。
    """
    rows = []
    for block in payload.get("stats", []):
        for sp in block.get("splits", []):
            st = sp.get("stat") or {}
            pl = sp.get("player") or {}
            if pl.get("id") is None:
                continue
            rows.append({
                "id": pl["id"],
                "name": pl.get("fullName", ""),
                "team": (sp.get("team") or {}).get("name", ""),
                "ip": ip_to_float(st.get("inningsPitched")),
                "runs": int(st.get("runs") or 0),
                "gs": int(st.get("gamesStarted") or 0),
                # FIP 用（缺欄時評分自動退回純 RA/9）
                "so": int(st.get("strikeOuts") or 0),
                "bb": int(st.get("baseOnBalls") or 0),
                "hr": int(st.get("homeRuns") or 0),
            })
    return rows


def fetch_pitchers(season: int, out_path: str | Path | None = None,
                   timeout: float = 30.0) -> list[dict]:
    """抓整季全聯盟投手數據（一次呼叫），選擇性存 CSV。"""
    import requests
    r = requests.get(f"{STATSAPI}/stats",
                     params={"stats": "season", "group": "pitching",
                             "season": season, "sportIds": 1, "gameType": "R",
                             "playerPool": "ALL", "limit": 2000},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    rows = parse_pitcher_stats(r.json())
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "name", "team", "ip", "runs",
                                              "gs", "so", "bb", "hr"])
            w.writeheader()
            w.writerows(rows)
    return rows


class PitcherBook:
    """先發投手評分簿：RA/9 與 FIP 各半混合 + 收縮估計 → 相對該隊平均先發的係數。

    設計（避免與隊級模型重複計算）：
      - RA/9 含守備與運氣雜訊；FIP =(13*HR+3*BB−2*K)/IP+常數 只看投手可控
        三事件，對未來失分更有預測力。兩者各以 prior_ip 局的聯盟平均收縮後
        各半混合（缺 K/BB/HR 欄的舊資料自動退回純 RA/9）。
      - 係數 = 該投手混合評分 ÷ 其球隊「先發群」IP 加權平均——隊防守評分
        已含全隊投手平均，這裡只修正「今晚這位相對隊內平均」的差。
      - 係數夾在 [0.7, 1.35]；先發約投 6 成局數，套用時混 0.6 權重。
    """

    STARTER_SHARE = 0.6   # 先發平均吃掉的比賽比重
    CLIP = (0.7, 1.35)
    FIP_WEIGHT = 0.5      # FIP 佔混合評分的比重（缺欄時為 0）

    def __init__(self, rows: list[dict], prior_ip: float = 60.0):
        self.rows = [r for r in rows if float(r.get("ip") or 0) > 0]
        total_runs = sum(r["runs"] for r in self.rows)
        total_ip = sum(float(r["ip"]) for r in self.rows)
        self.league_ra9 = 9.0 * total_runs / total_ip if total_ip else 4.5
        self.prior_ip = prior_ip
        # FIP 常數：把聯盟 IP 加權 FIP 核心對齊到聯盟 RA/9（失分尺度，可與 RA/9 混合）
        core_ip = sum(float(r["ip"]) for r in self.rows if self._has_fip(r))
        if core_ip > 0:
            core_sum = sum(self._fip_core(r) * float(r["ip"])
                           for r in self.rows if self._has_fip(r))
            self.fip_const = self.league_ra9 - core_sum / core_ip
        else:
            self.fip_const = None
        self.by_id = {int(r["id"]): r for r in self.rows}
        self.by_name = {r["name"]: r for r in self.rows}
        # 各隊先發群（gs>=1）的 IP 加權平均混合評分
        self.team_avg: dict[str, float] = {}
        agg: dict[str, list] = {}
        for r in self.rows:
            if int(r.get("gs") or 0) >= 1 and r.get("team"):
                agg.setdefault(r["team"], []).append(r)
        for team, rs in agg.items():
            w = sum(float(x["ip"]) for x in rs)
            if w > 0:
                self.team_avg[team] = sum(
                    self._rating(x) * float(x["ip"]) for x in rs) / w

    @staticmethod
    def _has_fip(r: dict) -> bool:
        return any(int(r.get(k) or 0) > 0 for k in ("so", "bb", "hr"))

    @staticmethod
    def _fip_core(r: dict) -> float:
        ip = float(r["ip"])
        return (13 * int(r.get("hr") or 0) + 3 * int(r.get("bb") or 0)
                - 2 * int(r.get("so") or 0)) / ip

    def _shrunk_ra9(self, r: dict) -> float:
        ip, runs = float(r["ip"]), r["runs"]
        prior_runs = self.league_ra9 / 9.0 * self.prior_ip
        return 9.0 * (runs + prior_runs) / (ip + self.prior_ip)

    def _shrunk_fip(self, r: dict) -> float | None:
        if self.fip_const is None or not self._has_fip(r):
            return None
        ip = float(r["ip"])
        fip = self._fip_core(r) + self.fip_const
        return (fip * ip + self.league_ra9 * self.prior_ip) / (ip + self.prior_ip)

    def _rating(self, r: dict) -> float:
        """混合評分（失分尺度）：0.5*RA/9 + 0.5*FIP；無 FIP 資料退回 RA/9。"""
        ra9 = self._shrunk_ra9(r)
        fip = self._shrunk_fip(r)
        if fip is None:
            return ra9
        return (1 - self.FIP_WEIGHT) * ra9 + self.FIP_WEIGHT * fip

    def factor(self, pitcher: "int | str | None") -> tuple[float, str]:
        """回傳 (對手得分乘數已混權重, 說明)。查無此人 → (1.0, 說明)。"""
        r = None
        if isinstance(pitcher, int):
            r = self.by_id.get(pitcher)
        elif pitcher:
            r = self.by_name.get(str(pitcher))
        if r is None:
            return 1.0, "無評分（查無此投手，用隊平均）"
        rating = self._rating(r)
        base = self.team_avg.get(r.get("team", ""), self.league_ra9)
        raw = rating / base if base > 0 else 1.0
        raw = min(max(raw, self.CLIP[0]), self.CLIP[1])
        blended = self.STARTER_SHARE * raw + (1.0 - self.STARTER_SHARE)
        note = f"評分 {rating:.2f}（RA/9+FIP，隊平均 {base:.2f}，係數 {blended:.2f}）"
        return blended, note

    @staticmethod
    def load_csv(path: str | Path) -> "PitcherBook":
        rows = []
        with Path(path).open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"id": int(r["id"]), "name": r["name"],
                             "team": r["team"], "ip": float(r["ip"]),
                             "runs": int(r["runs"]), "gs": int(r["gs"]),
                             "so": int(r.get("so") or 0),
                             "bb": int(r.get("bb") or 0),
                             "hr": int(r.get("hr") or 0)})
        return PitcherBook(rows)


# ---------------- 盤口（棒球版） ----------------
@dataclass
class MLBMarkets:
    p_home: float          # 錢線：主勝機率（延長賽對角線各半分）
    p_away: float
    ml_home_odds: float    # 公平賠率（無水位）
    ml_away_odds: float
    total_line: float
    p_over: float
    p_under: float
    run_line: float        # 主隊視角（-1.5 = 主讓 1.5 分）
    p_cover_home: float    # 主隊過盤機率
    exp_home: float        # 期望得分
    exp_away: float
    top_scores: list       # [((h,a), p), ...]


def moneyline(mat: np.ndarray) -> tuple[float, float]:
    """錢線機率：mat[h,a] 為比分聯合分布。9 局打平（對角線）延長賽視為五五波。"""
    p_h = float(np.tril(mat, -1).sum())   # h > a
    p_a = float(np.triu(mat, 1).sum())    # a > h
    tie = float(np.trace(mat))
    p_h += tie * 0.5
    p_a += tie * 0.5
    s = p_h + p_a
    return p_h / s, p_a / s


def analyze_game(model, home: str, away: str, total_line: float = 8.5,
                 run_line: float = -1.5, top_n: int = 4,
                 home_pitcher_factor: float = 1.0,
                 away_pitcher_factor: float = 1.0,
                 park_factor: float = 1.0,
                 dispersion: float | None = None) -> MLBMarkets:
    """對一場比賽算錢線/大小/讓分。model 為以 runs 訓練的 DixonColesModel。

    pitcher_factor 來自 PitcherBook.factor()：主隊先發的係數乘在「客隊得分」上，
    客隊先發的係數乘在「主隊得分」上（好投手 <1 → 壓低對手得分）。
    park_factor 為主場球場因子：兩邊各乘 PF^0.5（總分約放大 PF 倍）。
    dispersion 為負二項離散度 k（dispersion_from_csv 取得）；None 退回 Poisson。
    """
    lam, mu = model.expected_goals(home, away)
    lam *= away_pitcher_factor   # 客隊先發壓制主隊打線
    mu *= home_pitcher_factor    # 主隊先發壓制客隊打線
    if park_factor and park_factor != 1.0:
        pf = park_factor ** 0.5
        lam *= pf
        mu *= pf
    mat = nb_score_matrix(lam, mu, dispersion, model.max_goals)
    p_h, p_a = moneyline(mat)
    ou = markets.over_under(mat, total_line)
    ah = markets.asian_handicap(mat, run_line, "home")
    cover = ah.p_win + ah.p_half_win + 0.5 * ah.p_push
    flat = [((h, a), float(mat[h, a]))
            for h in range(mat.shape[0]) for a in range(mat.shape[1])]
    flat.sort(key=lambda x: -x[1])
    return MLBMarkets(
        p_home=p_h, p_away=p_a,
        ml_home_odds=round(1.0 / p_h, 2), ml_away_odds=round(1.0 / p_a, 2),
        total_line=total_line, p_over=ou["over_win"], p_under=ou["under_win"],
        run_line=run_line, p_cover_home=float(cover),
        exp_home=float(lam), exp_away=float(mu), top_scores=flat[:top_n])


# ---------------- 盤口比對（the-odds-api，sport=baseball_mlb） ----------------
@dataclass
class _Game:
    num: int
    team1: str
    team2: str
    played: bool = False


def fetch_mlb_odds(games: list, **kw) -> dict:
    """抓 MLB 盤口，回 {num: [MarketQuote...]}。重用足球那套抓取/配對。"""
    from .live.providers import fetch_wc_odds
    return fetch_wc_odds(games, sport="baseball_mlb", **kw)


# ---------------- MLB 戰績追蹤（重用足球 tracker 的結算機制） ----------------
MLB_LEDGER = "data/mlb_bets.csv"
_MLB_MARKET_ZH = {"1X2": "錢線", "OU": "大小", "AH": "讓分"}


def picks_for_game(m: MLBMarkets, quotes=None, min_lean: float = 0.03) -> list[dict]:
    """一場比賽的模型推薦：錢線（看好側）、大小（有明顯方向才推）、讓分。

    quotes 給定時附真實賠率（market key 沿用 tracker：1X2/OU/AH，結算直接相容——
    棒球無和局，1X2 只會出 home/away）。
    """
    from . import tracker

    def odds_for(market, sel, line):
        if not quotes:
            return None
        od = tracker._lookup_odds(quotes, market, sel, line)
        return od if (od and 1.0 < od <= tracker.MAX_ODDS) else None

    picks = []
    ml = "home" if m.p_home >= m.p_away else "away"
    picks.append(("1X2", ml, "", odds_for("1X2", ml, "")))
    if abs(m.p_over - 0.5) >= min_lean:
        sel = "大" if m.p_over >= 0.5 else "小"
        picks.append(("OU", sel, m.total_line, odds_for("OU", sel, m.total_line)))
    rl_sel = "主" if m.p_cover_home >= 0.5 else "客"
    picks.append(("AH", rl_sel, m.run_line, odds_for("AH", rl_sel, m.run_line)))
    return [dict(market=mk, selection=sel, line=ln, odds=od)
            for mk, sel, ln, od in picks]


def taipei_time(iso: str | None) -> str:
    """把 statsapi 的 UTC 開賽 ISO（'2026-07-05T23:05:00Z'）轉台北時間顯示。

    回傳 '週六 07/06 07:05'（台灣 UTC+8）。缺值或解析失敗回空字串。
    """
    if not iso:
        return ""
    import datetime as _dt
    try:
        s = str(iso).replace("Z", "+00:00")
        dt = _dt.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        tpe = dt.astimezone(_dt.timezone(_dt.timedelta(hours=8)))
        wd = "一二三四五六日"[tpe.weekday()]
        return f"週{wd} {tpe.month:02d}/{tpe.day:02d} {tpe.hour:02d}:{tpe.minute:02d}"
    except Exception:  # noqa: BLE001
        return ""


def bet_signals(m: MLBMarkets, quotes=None, weight=None) -> dict:
    """三個盤口的「買/觀望」提示：市場去 vig + 模型融合後算 edge（與足球同一套風控）。

    重用 tracker 的 `_blended`（去 vig + BLEND_WEIGHT 融合）與門檻
    （MAX_ODDS/MAX_EDGE/MIN_PROB）。回傳 {market: {side, odds, edge, verdict, p}}，
    market ∈ {"1X2","OU","AH"}，side 為模型看好側（與頁面顯示一致）。
    有真實賠率且融合後 edge>0 且過風控 → verdict="買"，否則 "觀望"；無賠率 → None。
    """
    from . import tracker
    w = tracker.BLEND_WEIGHT if weight is None else weight

    def signal(market, model_ps, order, fav, line):
        sides = tracker._group_quotes(quotes, market).get(line, {}) if quotes else {}
        bl = tracker._blended(model_ps, sides, order, w)
        p = bl.get(fav, model_ps[fav])
        odds = sides.get(fav)
        if not (odds and 1.0 < odds <= tracker.MAX_ODDS):
            return {"side": fav, "odds": None, "edge": None, "verdict": None, "p": p}
        edge = p * odds - 1.0
        buy = (0.0 < edge <= tracker.MAX_EDGE and p >= tracker.MIN_PROB)
        return {"side": fav, "odds": float(odds), "edge": edge,
                "verdict": "買" if buy else "觀望", "p": p}

    out = {}
    fav_ml = "home" if m.p_home >= m.p_away else "away"
    out["1X2"] = signal("1X2", {"home": m.p_home, "away": m.p_away},
                        ["home", "away"], fav_ml, "")
    fav_ou = "over" if m.p_over >= m.p_under else "under"
    out["OU"] = signal("OU", {"over": m.p_over, "under": m.p_under},
                       ["over", "under"], fav_ou, m.total_line)
    fav_ah = "home" if m.p_cover_home >= 0.5 else "away"
    out["AH"] = signal("AH", {"home": m.p_cover_home, "away": 1.0 - m.p_cover_home},
                       ["home", "away"], fav_ah, m.run_line)
    return out


def best_edge(signals: dict) -> float | None:
    """該場最強「買」訊號的 edge（給 TOP 排序）；無買訊號回 None。"""
    es = [s["edge"] for s in signals.values()
          if s.get("verdict") == "買" and s.get("edge") is not None]
    return max(es) if es else None


def log_picks(ledger_path, date: str, game: dict, picks: list[dict]) -> int:
    """把一場的推薦記入帳本（以 game_pk 當 match_num，不重複）。回傳新增筆數。"""
    import pandas as pd

    from . import tracker
    if not game.get("game_pk"):
        return 0
    df = tracker.load_ledger(ledger_path)
    seen = set(zip(df["match_num"].astype(str), df["market"], df["selection"].astype(str)))
    rows = []
    for p in picks:
        if (str(game["game_pk"]), p["market"], str(p["selection"])) in seen:
            continue
        od = round(p["odds"], 3) if p.get("odds") else ""
        eg = round(p["edge"], 4) if isinstance(p.get("edge"), (int, float)) else ""
        rows.append(dict(date=date, match_num=game["game_pk"], home=game["home"],
                         away=game["away"], market=p["market"], selection=p["selection"],
                         line=p["line"], odds=od, edge=eg,
                         source="market" if p.get("odds") else "model",
                         close_odds=od, result="pending", pl=""))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        tracker.save_ledger(df, ledger_path)
    return len(rows)


def settle_ledger(ledger_path, days_back: int = 5) -> int:
    """抓近幾天賽果結算帳本（match_num=game_pk）。需可連 statsapi。"""
    import datetime as _dt

    from . import tracker
    end = _dt.datetime.utcnow().date()
    start = end - _dt.timedelta(days=days_back)
    finals = fetch_games(start.isoformat(), end.isoformat())
    results = {int(g["game_pk"]): (int(g["home_goals"]), int(g["away_goals"]))
               for g in finals if g.get("game_pk") is not None}
    return tracker.settle(ledger_path, results)


def summary_text(ledger_path) -> str | None:
    """MLB 戰績摘要（沿用 tracker 統計，改棒球盤口名）。無紀錄回 None。"""
    from pathlib import Path as _P

    from . import tracker
    if not _P(ledger_path).exists():
        return None
    s = tracker.summary(ledger_path)
    if not (s.settled or s.pending):
        return None
    parts = []
    for mk, (w, l, p) in s.by_market.items():
        d = w + l
        rt = f"{w/d:.0%}" if d else "—"
        parts.append(f"{_MLB_MARKET_ZH.get(mk, mk)} {w}–{l}（{rt}）")
    head = (f"MLB 推薦戰績｜{s.wins} 勝 {s.losses} 敗"
            f"{f' {s.pushes} 走盤' if s.pushes else ''}"
            f"｜勝率 {s.win_rate:.1%}（待結 {s.pending}）")
    lines = [head]
    if parts:
        lines.append("　".join(parts))
    if s.market_bets:
        roi = f"實際損益｜{s.pl_units:+.2f}u / {s.market_bets} 注｜ROI {s.roi:+.1%}"
        lines.append(roi)
    return "\n".join(lines)


# ---------------- 回測（驗證分布假設，無前視） ----------------
def evaluate(df, cut: str, ks: list | None = None,
             half_life: float = 120.0, reg: float = 0.3) -> dict:
    """walk-forward 回測：cut 前訓練、cut 後評估。比較不同離散度 k 的
    錢線 log loss 與大小 8.5 的 log loss/Brier。k=None 代表 Poisson。
    """
    import pandas as pd

    from .models import dixon_coles as dc
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cut_ts = pd.Timestamp(cut)
    train = df[df["date"] < cut_ts]
    test = df[df["date"] >= cut_ts]
    model = dc.fit(train, half_life_days=half_life, max_goals=20,
                   rho_init=0.0, reg=reg, reference_date=cut_ts)
    k_mom = dispersion_from_df(train)
    if ks is None:
        ks = [None, k_mom, 2.0, 3.0, 5.0, 8.0, 15.0]
    rows = []
    games = [(r.home, r.away, int(r.home_goals), int(r.away_goals))
             for r in test.itertuples()
             if r.home in model.attack and r.away in model.attack]
    eps = 1e-12
    for k in ks:
        ml_ll = ou_ll = ou_br = 0.0
        n = 0
        cal_p = cal_y = 0.0
        for home, away, hg, ag in games:
            if hg == ag:
                continue  # 極罕見（暫停補賽），跳過
            lam, mu = model.expected_goals(home, away)
            mat = nb_score_matrix(lam, mu, k, 20)
            p_h, _ = moneyline(mat)
            y = 1.0 if hg > ag else 0.0
            ml_ll -= y * np.log(max(p_h, eps)) + (1 - y) * np.log(max(1 - p_h, eps))
            ou = markets.over_under(mat, 8.5)
            po = ou["over_win"] / (ou["over_win"] + ou["under_win"])
            yo = 1.0 if hg + ag > 8.5 else 0.0
            ou_ll -= yo * np.log(max(po, eps)) + (1 - yo) * np.log(max(1 - po, eps))
            ou_br += (po - yo) ** 2
            cal_p += p_h
            cal_y += y
            n += 1
        rows.append({"k": k, "n": n, "ml_logloss": ml_ll / n,
                     "ou_logloss": ou_ll / n, "ou_brier": ou_br / n,
                     "mean_p_home": cal_p / n, "home_win_rate": cal_y / n})
    return {"rows": rows, "k_mom": k_mom, "n_train": len(train), "n_test": len(games)}
def us_today() -> str:
    """MLB 賽程日（美東日期，UTC-5 近似）。"""
    import datetime as _dt
    return (_dt.datetime.utcnow() - _dt.timedelta(hours=5)).date().isoformat()


def build_site_page(model_path: str = "models/mlb.pkl",
                    data_path: str = "data/mlb.csv",
                    pitchers_path: str = "data/mlb_pitchers.csv",
                    ledger_path: str = MLB_LEDGER,
                    date: str | None = None, with_odds: bool = True) -> str:
    """建 MLB 分頁 HTML：今日各場預測卡 + 戰力表 + 戰績。任何一步失敗都優雅降級。"""
    from pathlib import Path as _P

    from . import report
    from .models.dixon_coles import DixonColesModel
    date = date or us_today()
    if not _P(model_path).exists():
        return report.render_mlb_page([], date=date, power=None, track_text=None,
                                      note="尚未訓練 MLB 模型——在部署環境跑 "
                                           "footy mlb fetch + fetch-pitchers + train 後，"
                                           "每日建站會自動更新本頁。")
    model = DixonColesModel.load(model_path)
    pf_map = park_factors_from_csv(data_path)
    disp = dispersion_from_csv(data_path)   # 負二項離散度（回測驗證優於 Poisson）
    book = None
    if _P(pitchers_path).exists():
        try:
            book = PitcherBook.load_csv(pitchers_path)
        except Exception:  # noqa: BLE001
            book = None
    # 先結算舊推薦（抓近幾天賽果；網路失敗略過）
    try:
        settle_ledger(ledger_path)
    except Exception:  # noqa: BLE001
        pass
    note = ""
    try:
        games = fetch_today(date)
    except Exception as e:  # noqa: BLE001
        games = []
        note = f"抓不到今日賽程（{e}）。"
    games = [g for g in games if g["home"] in model.attack and g["away"] in model.attack]
    odds_index = {}
    if with_odds and games:
        try:
            gobjs = [_Game(i + 1, g["home"], g["away"]) for i, g in enumerate(games)]
            odds_index = fetch_mlb_odds(gobjs)
        except Exception:  # noqa: BLE001
            odds_index = {}
    rows = []
    for i, g in enumerate(games):
        quotes = odds_index.get(i + 1)
        hf = af = 1.0
        hn = an = ""
        if book:
            hf, hn = book.factor(g.get("home_pitcher_id") or g.get("home_pitcher") or None)
            af, an = book.factor(g.get("away_pitcher_id") or g.get("away_pitcher") or None)
        pf = pf_map.get(g["home"], 1.0)
        # 大小分線/讓分線：有市場主線用市場，否則 8.5 / -1.5
        total_line = 8.5
        run_line = -1.5
        if quotes:
            from . import tracker
            ous = tracker._group_quotes(quotes, "OU")
            if ous:
                total_line = sorted(ous, key=lambda ln: abs(float(ln) - 8.5))[0]
            mkt_rl = tracker.main_ah_line(quotes)   # 主客賠率最均衡的讓分線
            if mkt_rl is not None:
                run_line = float(mkt_rl)
        m = analyze_game(model, g["home"], g["away"], total_line=float(total_line),
                         run_line=run_line,
                         home_pitcher_factor=hf, away_pitcher_factor=af,
                         park_factor=pf, dispersion=disp)
        sig = bet_signals(m, quotes)
        picks = picks_for_game(m, quotes)
        for p in picks:  # 把融合後 edge 帶進帳本（待結算頁依此排序）
            s = sig.get(p["market"])
            if s and s.get("edge") is not None:
                p["edge"] = s["edge"]
        try:
            log_picks(ledger_path, date, g, picks)
        except Exception:  # noqa: BLE001
            pass
        rows.append({"game": g, "m": m, "pf": pf,
                     "hp_note": hn, "ap_note": an,
                     "signals": sig, "best_edge": best_edge(sig),
                     "time": taipei_time(g.get("game_date_iso")),
                     "status": g.get("status", "")})
    power = None
    try:
        power = team_power(model)
    except Exception:  # noqa: BLE001
        power = None
    return report.render_mlb_page(rows, date=date, power=power,
                                  track_text=summary_text(ledger_path), note=note)
