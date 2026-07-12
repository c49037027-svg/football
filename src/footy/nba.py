"""NBA 預測：加權嶺回歸攻防評分 + 常態分布市場機率。

與 MLB 的差異（為何不用負二項）：籃球得分 ~112 分/隊，比分近似常態分布，
用「攻防評分 → 預期得分差/總分 → 常態機率」是文獻標準做法（等同 power rating
+ margin model）。盤口機率：
  - 錢線 P(主勝) = Φ(預期分差 / σ_margin)（NBA 有延長無和局）
  - 大小 P(over)  = 1 − Φ((盤口線 − 預期總分) / σ_total)
  - 讓分 P(主過盤) = 1 − Φ((−讓分線 − 預期分差) / σ_margin)
σ 由訓練殘差估計（加權）。買/觀望、去 vig、市場融合、帳本、TOP5 全部
重用 MLB 那一套（bet_signals/picks_for_game/log_picks/market_confidence）。

資料來源：
  - cdn.nba.com scheduleLeagueV2.json：本季完整賽程+比分（免金鑰，一次呼叫）
  - stats.nba.com leaguegamelog：歷史賽季（需瀏覽器 headers；沙箱擋、部署可用）
gameId 前綴分類：001 熱身 / 002 例行 / 003 明星 / 004 季後 / 005 附加。
訓練與結算用 002/004/005。
"""
from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
GAMELOG_URL = "https://stats.nba.com/stats/leaguegamelog"
NBA_LEDGER = "data/nba_bets.csv"
_GTYPE_OK = ("002", "004", "005")   # 例行/季後/附加賽

# 30 隊中文名（台灣慣用譯名）
NBA_ZH = {
    "Atlanta Hawks": "老鷹", "Boston Celtics": "塞爾提克",
    "Brooklyn Nets": "籃網", "Charlotte Hornets": "黃蜂",
    "Chicago Bulls": "公牛", "Cleveland Cavaliers": "騎士",
    "Dallas Mavericks": "獨行俠", "Denver Nuggets": "金塊",
    "Detroit Pistons": "活塞", "Golden State Warriors": "勇士",
    "Houston Rockets": "火箭", "Indiana Pacers": "溜馬",
    "LA Clippers": "快艇", "Los Angeles Clippers": "快艇",
    "Los Angeles Lakers": "湖人", "Memphis Grizzlies": "灰熊",
    "Miami Heat": "熱火", "Milwaukee Bucks": "公鹿",
    "Minnesota Timberwolves": "灰狼", "New Orleans Pelicans": "鵜鶘",
    "New York Knicks": "尼克", "Oklahoma City Thunder": "雷霆",
    "Orlando Magic": "魔術", "Philadelphia 76ers": "76人",
    "Phoenix Suns": "太陽", "Portland Trail Blazers": "拓荒者",
    "Sacramento Kings": "國王", "San Antonio Spurs": "馬刺",
    "Toronto Raptors": "暴龍", "Utah Jazz": "爵士",
    "Washington Wizards": "巫師",
}
# 跨資料源同隊異名（stats.nba.com 用 LA Clippers、schedule 曾用全名）
NBA_TEAM_RENAMES = {"Los Angeles Clippers": "LA Clippers"}


def zh_nba(name: str) -> str:
    return NBA_ZH.get(name, name)


# ---------------- 資料抓取 / 解析（解析為純函式，可離線測） ----------------
def _et_date(iso_utc: str | None) -> str:
    """UTC ISO → 美東賽程日（UTC−5 近似，與 MLB us_today 同慣例）。"""
    import datetime as _dt
    if not iso_utc:
        return ""
    s = iso_utc.replace("Z", "+00:00")
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError:
        return iso_utc[:10]
    return (dt.astimezone(_dt.timezone.utc) - _dt.timedelta(hours=5)).date().isoformat()


def parse_schedule_v2(payload: dict, finals_only: bool = True,
                      types: tuple = _GTYPE_OK) -> list[dict]:
    """解析 scheduleLeagueV2 成賽事列表（純函式）。

    回 [{date, home, away, home_goals, away_goals, game_pk, game_date_iso, status}]；
    finals_only=False 時未打完的比賽 home_goals/away_goals 為 None。
    """
    out = []
    for gd in (payload.get("leagueSchedule") or {}).get("gameDates", []):
        for g in gd.get("games", []):
            gid = str(g.get("gameId") or "")
            if types and gid[:3] not in types:
                continue
            ht, at = g.get("homeTeam") or {}, g.get("awayTeam") or {}
            home = f"{ht.get('teamCity', '')} {ht.get('teamName', '')}".strip()
            away = f"{at.get('teamCity', '')} {at.get('teamName', '')}".strip()
            home = NBA_TEAM_RENAMES.get(home, home)
            away = NBA_TEAM_RENAMES.get(away, away)
            if not home or not away:
                continue
            final = int(g.get("gameStatus") or 0) == 3
            if finals_only and not final:
                continue
            iso = g.get("gameDateTimeUTC") or ""
            out.append({
                "date": _et_date(iso),
                "home": home, "away": away,
                "home_goals": int(ht.get("score") or 0) if final else None,
                "away_goals": int(at.get("score") or 0) if final else None,
                "game_pk": int(gid) if gid.isdigit() else None,
                "game_date_iso": iso,
                "status": (g.get("gameStatusText") or "").strip(),
            })
    out.sort(key=lambda r: (r["date"], r["game_date_iso"]))
    return out


def fetch_schedule(timeout: float = 30.0) -> dict:
    """抓本季完整賽程 JSON（部署環境用；免金鑰）。"""
    import requests
    r = requests.get(SCHEDULE_URL, headers={"User-Agent": "Mozilla/5.0"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_leaguegamelog(payload: dict) -> list[dict]:
    """解析 stats.nba.com leaguegamelog（隊粒度，每場兩列）成單場列（純函式）。

    MATCHUP 含 " vs. " 的列是主隊、含 " @ " 的是客隊；以 GAME_ID 配對。
    回 [{date, home, away, home_goals, away_goals, game_pk}]。
    """
    rs = (payload.get("resultSets") or [{}])[0]
    heads = {h: i for i, h in enumerate(rs.get("headers") or [])}
    need = ("GAME_ID", "GAME_DATE", "TEAM_NAME", "MATCHUP", "PTS")
    if any(k not in heads for k in need):
        return []
    games: dict[str, dict] = {}
    for row in rs.get("rowSet") or []:
        gid = str(row[heads["GAME_ID"]])
        team = NBA_TEAM_RENAMES.get(row[heads["TEAM_NAME"]], row[heads["TEAM_NAME"]])
        matchup = row[heads["MATCHUP"]] or ""
        pts = row[heads["PTS"]]
        g = games.setdefault(gid, {"date": row[heads["GAME_DATE"]],
                                   "game_pk": int(gid) if gid.isdigit() else None})
        side = "home" if " vs. " in matchup else "away"
        g[side] = team
        g[f"{side}_goals"] = int(pts) if pts is not None else None
    out = [g for g in games.values()
           if g.get("home") and g.get("away")
           and g.get("home_goals") is not None and g.get("away_goals") is not None]
    out.sort(key=lambda r: (r["date"], r["game_pk"] or 0))
    return out


def fetch_gamelog(season: str, season_type: str = "Regular Season",
                  timeout: float = 30.0) -> list[dict]:
    """抓某季逐場（season 如 '2024-25'；部署環境用，沙箱擋）。"""
    import requests
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Accept": "application/json",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }
    params = {"Counter": 0, "DateFrom": "", "DateTo": "", "Direction": "DESC",
              "LeagueID": "00", "PlayerOrTeam": "T", "Season": season,
              "SeasonType": season_type, "Sorter": "DATE"}
    r = requests.get(GAMELOG_URL, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return parse_leaguegamelog(r.json())


_CSV_FIELDS = ["date", "home", "away", "home_goals", "away_goals", "game_pk"]


def write_games_csv(rows: list[dict], out_path: str | Path) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def load_with_history(data_path: str | Path,
                      hist_path: str | Path | None = "data/nba_hist.csv"):
    """載入本季賽果並（若存在）合併歷史，統一隊名、去重、依日排序。"""
    import pandas as pd
    frames = []
    if hist_path and Path(hist_path).exists():
        frames.append(pd.read_csv(hist_path))
    if Path(data_path).exists():
        frames.append(pd.read_csv(data_path))
    if not frames:
        raise FileNotFoundError(f"無資料：{data_path} / {hist_path}")
    df = pd.concat(frames, ignore_index=True)
    for c in ("home", "away"):
        df[c] = df[c].replace(NBA_TEAM_RENAMES)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_goals", "away_goals"])
    df = df.drop_duplicates(subset=["date", "home", "away"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------- 評分模型（加權嶺回歸 + 常態殘差） ----------------
@dataclass
class NBAModel:
    base: float                  # 聯盟平均單隊得分
    home_adv: float              # 主場優勢（分差）
    off: dict                    # 隊 → 進攻評分（相對聯盟，分）
    deff: dict                   # 隊 → 防守評分（正=好防守，壓低對手得分）
    sigma_margin: float          # 分差殘差標準差
    sigma_total: float           # 總分殘差標準差
    teams: list = field(default_factory=list)

    def expected_points(self, home: str, away: str) -> tuple[float, float]:
        mu_h = self.base + self.home_adv / 2 + self.off[home] - self.deff[away]
        mu_a = self.base - self.home_adv / 2 + self.off[away] - self.deff[home]
        return mu_h, mu_a

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "NBAModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_ratings(df, half_life_days: float = 300.0, reg: float = 8.0,
                reference_date=None) -> NBAModel:
    """加權嶺回歸擬合攻防評分。

    每場兩條方程：主隊得分 = base + hadv/2 + off_主 − def_客；客隊同理無主場。
    權重 = 0.5^(距 reference_date 天數 / half_life)；off/def 以 reg 收縮向 0
    （聯盟平均），球隊間可辨識。σ 由加權殘差估計。
    """
    import pandas as pd
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    ref = pd.Timestamp(reference_date) if reference_date is not None else df["date"].max()
    teams = sorted(set(df["home"]) | set(df["away"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    n_games = len(df)
    X = np.zeros((2 * n_games, 2 + 2 * n))
    y = np.zeros(2 * n_games)
    w = np.zeros(2 * n_games)
    age = (ref - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    wt = 0.5 ** (age / half_life_days) if half_life_days < 1e6 else np.ones_like(age)
    hi = df["home"].map(ti).to_numpy()
    ai = df["away"].map(ti).to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)
    r = np.arange(n_games)
    # 主隊得分列
    X[2 * r, 0] = 1.0; X[2 * r, 1] = 0.5
    X[2 * r, 2 + hi] = 1.0; X[2 * r, 2 + n + ai] = -1.0
    y[2 * r] = hg; w[2 * r] = wt
    # 客隊得分列
    X[2 * r + 1, 0] = 1.0; X[2 * r + 1, 1] = -0.5
    X[2 * r + 1, 2 + ai] = 1.0; X[2 * r + 1, 2 + n + hi] = -1.0
    y[2 * r + 1] = ag; w[2 * r + 1] = wt
    # 加權嶺回歸（只正則化 off/def，不動 base/hadv）
    sw = np.sqrt(w)
    A = X * sw[:, None]
    b = y * sw
    lam = np.zeros(2 + 2 * n)
    lam[2:] = reg
    beta = np.linalg.solve(A.T @ A + np.diag(lam), A.T @ b)
    base, hadv = float(beta[0]), float(beta[1])
    off = {t: float(beta[2 + i]) for t, i in ti.items()}
    deff = {t: float(beta[2 + n + i]) for t, i in ti.items()}
    # σ：加權殘差（分差與總分）
    mu_h = base + hadv / 2 + beta[2 + hi] - beta[2 + n + ai]
    mu_a = base - hadv / 2 + beta[2 + ai] - beta[2 + n + hi]
    wn = wt / wt.sum()
    res_m = (hg - ag) - (mu_h - mu_a)
    res_t = (hg + ag) - (mu_h + mu_a)
    sigma_m = float(np.sqrt((wn * res_m ** 2).sum()))
    sigma_t = float(np.sqrt((wn * res_t ** 2).sum()))
    return NBAModel(base=base, home_adv=hadv, off=off, deff=deff,
                    sigma_margin=sigma_m, sigma_total=sigma_t, teams=teams)


# ---------------- 市場分析（介面對齊 MLBMarkets，重用 MLB 訊號那套） ----------------
@dataclass
class NBAMarkets:
    home: str
    away: str
    p_home: float
    p_away: float
    exp_home: float
    exp_away: float
    total_line: float
    p_over: float
    p_under: float
    run_line: float        # 主隊讓分線（負=主讓）
    p_cover_home: float
    top_scores: list = field(default_factory=list)   # NBA 不列可能比分


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _half(x: float) -> float:
    """取最近的 .5 線（NBA 盤口慣例，無 push）。"""
    return math.floor(x) + 0.5


def analyze_game(model: NBAModel, home: str, away: str,
                 total_line: float | None = None,
                 run_line: float | None = None) -> NBAMarkets:
    """單場三盤口機率。未給線時用模型預期取最近 .5（顯示用）。"""
    mu_h, mu_a = model.expected_points(home, away)
    mu_m, mu_t = mu_h - mu_a, mu_h + mu_a
    if total_line is None:
        total_line = _half(mu_t)
    if run_line is None:
        run_line = -_half(mu_m) if mu_m >= 0 else _half(-mu_m)
    p_home = _phi(mu_m / model.sigma_margin)
    p_over = 1.0 - _phi((total_line - mu_t) / model.sigma_total)
    p_cover = 1.0 - _phi((-run_line - mu_m) / model.sigma_margin)
    return NBAMarkets(home=home, away=away, p_home=p_home, p_away=1.0 - p_home,
                      exp_home=mu_h, exp_away=mu_a,
                      total_line=float(total_line), p_over=p_over,
                      p_under=1.0 - p_over, run_line=float(run_line),
                      p_cover_home=p_cover)


def team_power(model: NBAModel) -> list[dict]:
    """戰力表：對聯盟平均對手的場均得/失分與淨值。"""
    avg_off = sum(model.off.values()) / len(model.off)
    avg_def = sum(model.deff.values()) / len(model.deff)
    rows = []
    for t in model.teams:
        rf = model.base + model.off[t] - avg_def
        ra = model.base + avg_off - model.deff[t]
        rows.append({"team": t, "rf": rf, "ra": ra, "diff": rf - ra})
    rows.sort(key=lambda r: r["diff"], reverse=True)
    return rows


def fetch_nba_odds(games: list, **kw) -> dict:
    """抓 NBA 盤口（the-odds-api sport=basketball_nba），回 {num: [MarketQuote...]}。"""
    from .live.providers import fetch_wc_odds
    return fetch_wc_odds(games, sport="basketball_nba", **kw)


# ---------------- 建站（重用 MLB 的訊號/帳本/頁面） ----------------
def build_site_page(model_path: str = "models/nba.pkl",
                    data_path: str = "data/nba.csv",
                    ledger_path: str = NBA_LEDGER,
                    date: str | None = None, with_odds: bool = True,
                    schedule_payload: dict | None = None) -> str:
    """建 NBA 分頁 HTML。休賽季/失敗都優雅降級（顯示說明頁）。"""
    from . import mlb, report, tracker
    date = date or mlb.us_today()
    if not Path(model_path).exists():
        return report.render_mlb_page(
            [], date=date, zh=zh_nba, sport="NBA",
            title="NBA 今日預測", note="尚未訓練 NBA 模型——部署環境跑 "
            "footy nba fetch + train 後，每日建站自動更新本頁。")
    model = NBAModel.load(model_path)
    note = ""
    sched: list[dict] = []
    try:
        payload = schedule_payload or fetch_schedule()
        sched = parse_schedule_v2(payload, finals_only=False)
    except Exception as e:  # noqa: BLE001
        note = f"抓不到 NBA 賽程（{e}）。"
    # 結算：賽程裡的已完賽比分直接用（零額外請求）
    try:
        results = {int(g["game_pk"]): (int(g["home_goals"]), int(g["away_goals"]))
                   for g in sched
                   if g.get("game_pk") and g.get("home_goals") is not None}
        if results and Path(ledger_path).exists():
            tracker.settle(ledger_path, results)
    except Exception:  # noqa: BLE001
        pass
    games = [g for g in sched if g["date"] == date
             and g["home"] in model.off and g["away"] in model.off]
    if not games and not note:
        import datetime as _dt
        m = _dt.date.fromisoformat(date).month
        note = ("NBA 休賽季（例行賽 10 月下旬開打，開季後本頁自動恢復每日預測）。"
                if 7 <= m <= 9 else "今日無 NBA 賽事。")
    odds_index = {}
    if with_odds and games:
        try:
            from .worldcup import Game as _Game
            gobjs = [_Game(i + 1, g["home"], g["away"]) for i, g in enumerate(games)]
            odds_index = fetch_nba_odds(gobjs)
        except Exception:  # noqa: BLE001
            odds_index = {}
    rows = []
    for i, g in enumerate(games):
        quotes = odds_index.get(i + 1)
        total_line = run_line = None
        if quotes:
            mu_h, mu_a = model.expected_points(g["home"], g["away"])
            ous = tracker._group_quotes(quotes, "OU")
            if ous:
                total_line = sorted(ous, key=lambda ln: abs(float(ln) - (mu_h + mu_a)))[0]
            mkt_rl = tracker.main_ah_line(quotes)
            if mkt_rl is not None:
                run_line = float(mkt_rl)
        m = analyze_game(model, g["home"], g["away"],
                         total_line=float(total_line) if total_line is not None else None,
                         run_line=run_line)
        sig = mlb.bet_signals(m, quotes)
        picks = mlb.picks_for_game(m, quotes)
        for p in picks:
            s = sig.get(p["market"])
            if s and s.get("edge") is not None:
                p["edge"] = s["edge"]
        try:
            mlb.log_picks(ledger_path, date, g, picks)
        except Exception:  # noqa: BLE001
            pass
        rows.append({"game": g, "m": m, "pf": 1.0, "wx": None, "wf": 1.0,
                     "hp_note": "", "ap_note": "",
                     "signals": sig, "best_edge": mlb.best_edge(sig),
                     "time": mlb.taipei_time(g.get("game_date_iso")),
                     "status": g.get("status", "")})
    power = None
    try:
        power = team_power(model)
    except Exception:  # noqa: BLE001
        power = None
    track = mlb.summary_text(ledger_path, label="NBA")
    n_buy = sum(1 for r in rows if r.get("best_edge") is not None)
    print(f"[nba-site] 賽程 {len(games)} 場｜盤口 {len(odds_index)} 場｜"
          f"買推薦 {n_buy} 場｜戰績卡 {'有' if track else '無'}", flush=True)
    return report.render_mlb_page(rows, date=date, power=power,
                                  track_text=track, note=note,
                                  mkt_conf=mlb.market_confidence(ledger_path),
                                  zh=zh_nba, sport="NBA", title="NBA 今日預測")
