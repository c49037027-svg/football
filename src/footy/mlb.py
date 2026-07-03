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
            rows.append({
                "date": g.get("officialDate") or day.get("date"),
                "home": (home.get("team") or {}).get("name", ""),
                "away": (away.get("team") or {}).get("name", ""),
                "home_goals": hs, "away_goals": as_,
                "status": status,
                "home_pitcher": ((home.get("probablePitcher") or {}).get("fullName", "")),
                "away_pitcher": ((away.get("probablePitcher") or {}).get("fullName", "")),
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
                 run_line: float = -1.5, top_n: int = 4) -> MLBMarkets:
    """對一場比賽算錢線/大小/讓分。model 為以 runs 訓練的 DixonColesModel。"""
    mat = model.score_matrix(home, away)
    p_h, p_a = moneyline(mat)
    ou = markets.over_under(mat, total_line)
    ah = markets.asian_handicap(mat, run_line, "home")
    cover = ah.p_win + ah.p_half_win + 0.5 * ah.p_push
    lam, mu = model.expected_goals(home, away)
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
