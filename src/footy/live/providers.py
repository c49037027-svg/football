"""真實賠率資料來源（OddsFeed 實作）。

目前提供 The Odds API（the-odds-api.com）：免費方案即可取得多家博彩公司的
1X2(h2h) / 大小球(totals) / 讓盤(spreads) 賠率，並有 scores 端點可取即時比分。

設計原則：把「解析 JSON」(`parse_odds`, `parse_scores`) 與「網路請求」分離，
解析是純函式、可單元測試；要換別家 API 只要寫對應 parser + Feed 子類別。

走地分鐘數：The Odds API 的 scores 端點不含「比賽進行到第幾分鐘」，
這裡用 commence_time 到現在的經過時間估計（夾在 0~90+），並標註為估計值。
若要精準分鐘/紅牌/xG/傷停，建議改用 api-football（見 data/context.py 的說明）。

API key 由環境變數 ODDS_API_KEY 提供（不要寫進程式碼或入庫）。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

from .feed import MarketQuote, MatchState, OddsFeed

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


# ---------------- 純解析函式（可測試） ----------------
def _spread_to_home_line(outcome_name: str, point: float, home_team: str) -> float:
    """把 spreads 的 (隊名, point) 轉成『主隊視角』讓球線。

    home -0.5 -> line=-0.5；away +0.5（即 home -0.5）-> line=-0.5。
    """
    if outcome_name == home_team:
        return float(point)
    return -float(point)  # away 視角 point 取反即為 home 視角


def parse_odds(events: list[dict], bookmaker: str | None = None) -> dict[str, dict]:
    """解析 The Odds API /odds 回應。

    回傳 {event_id: {"home":..., "away":..., "commence": iso, "quotes": [MarketQuote...]}}。
    bookmaker：只取指定博彩商（如 "pinnacle"）；None 則取每市場第一家。
    """
    out: dict[str, dict] = {}
    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        books = ev.get("bookmakers", [])
        if bookmaker:
            books = [b for b in books if b.get("key") == bookmaker]
        quotes: list[MarketQuote] = []
        seen_markets: set[str] = set()
        for b in books:
            for m in b.get("markets", []):
                mkey = m.get("key")
                if mkey in seen_markets:
                    continue  # 已從較優先博彩商取得此市場
                if mkey == "h2h":
                    for o in m.get("outcomes", []):
                        name, price = o.get("name"), o.get("price")
                        if name == home:
                            quotes.append(MarketQuote("1X2", "home", float(price)))
                        elif name == away:
                            quotes.append(MarketQuote("1X2", "away", float(price)))
                        elif name and name.lower() == "draw":
                            quotes.append(MarketQuote("1X2", "draw", float(price)))
                    seen_markets.add(mkey)
                elif mkey == "totals":
                    for o in m.get("outcomes", []):
                        sel = "over" if o.get("name", "").lower() == "over" else "under"
                        quotes.append(MarketQuote("OU", sel, float(o["price"]),
                                                  line=float(o["point"])))
                    seen_markets.add(mkey)
                elif mkey == "spreads":
                    for o in m.get("outcomes", []):
                        name = o.get("name")
                        line = _spread_to_home_line(name, o["point"], home)
                        sel = "home" if name == home else "away"
                        quotes.append(MarketQuote("AH", sel, float(o["price"]), line=line))
                    seen_markets.add(mkey)
        out[ev.get("id")] = {
            "home": home, "away": away,
            "commence": ev.get("commence_time"), "quotes": quotes,
        }
    return out


def parse_scores(score_events: list[dict]) -> dict[str, dict]:
    """解析 The Odds API /scores 回應，回傳 {event_id: {home_goals, away_goals, completed}}。"""
    out: dict[str, dict] = {}
    for ev in score_events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        hg = ag = 0
        for s in ev.get("scores") or []:
            try:
                val = int(s.get("score"))
            except (TypeError, ValueError):
                continue
            if s.get("name") == home:
                hg = val
            elif s.get("name") == away:
                ag = val
        out[ev.get("id")] = {
            "home_goals": hg, "away_goals": ag,
            "completed": bool(ev.get("completed")),
        }
    return out


def estimate_minute(commence_iso: str | None, now: datetime | None = None) -> int:
    """由開賽時間估計目前比賽分鐘（夾在 0~95）。"""
    if not commence_iso:
        return 0
    now = now or datetime.now(timezone.utc)
    try:
        start = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except ValueError:
        return 0
    elapsed_min = (now - start).total_seconds() / 60.0
    # 中場休息約 15 分鐘，這裡簡化不扣除；夾在合理範圍。
    return int(max(0, min(95, elapsed_min)))


def find_ah_line(parsed: dict, home: str, away: str) -> "dict | None":
    """從 parse_odds 的結果裡，找指定對戰的亞盤讓球線（主隊視角）。

    用別名 + 模糊比對配對隊名。回傳 {"line", "home_odds", "away_odds"} 或 None。
    """
    import difflib

    from ..worldcup import TEAM_ALIASES

    def norm(n):
        return TEAM_ALIASES.get(n, n)

    th, ta = norm(home), norm(away)
    best_score, best = 1.4, None  # 需雙隊合計相似度 > 1.4 才算配對
    for info in parsed.values():
        eh, ea = norm(info.get("home") or ""), norm(info.get("away") or "")
        ah = [q for q in info.get("quotes", []) if q.market == "AH"]
        if not ah:
            continue
        score = (difflib.SequenceMatcher(None, eh, th).ratio()
                 + difflib.SequenceMatcher(None, ea, ta).ratio())
        if score <= best_score:
            continue
        home_q = next((q for q in ah if q.selection == "home"), None)
        away_q = next((q for q in ah if q.selection == "away"), None)
        if home_q is not None:
            best_score = score
            best = {"line": home_q.line, "home_odds": home_q.odds,
                    "away_odds": away_q.odds if away_q else None}
    return best


def fetch_ah_line(home: str, away: str, sport: str = "soccer_fifa_world_cup",
                  api_key: str | None = None, regions: str = "eu",
                  bookmaker: str | None = None, timeout: float = 15.0) -> "dict | None":
    """從 The Odds API 抓某場的亞盤讓球線（需 ODDS_API_KEY）。失敗回 None。"""
    import os

    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("缺少 ODDS_API_KEY")
    r = requests.get(f"{ODDS_API_BASE}/sports/{sport}/odds",
                     params={"regions": regions, "markets": "spreads",
                             "oddsFormat": "decimal", "apiKey": key},
                     timeout=timeout)
    r.raise_for_status()
    parsed = parse_odds(r.json(), bookmaker=bookmaker)
    return find_ah_line(parsed, home, away)


# ---------------- Feed 實作 ----------------
class TheOddsApiFeed(OddsFeed):
    """The Odds API 走地/初盤盤口來源。

    sport：聯賽鍵，如 'soccer_epl'、'soccer_spain_la_liga'（見 the-odds-api 文件）。
    in_play：True 只回傳已開賽的比賽（走地）；False 回傳未開賽（初盤）。
    """

    def __init__(self, sport: str = "soccer_epl", regions: str = "eu",
                 markets: str = "h2h,totals,spreads", bookmaker: str | None = None,
                 in_play: bool = True, api_key: str | None = None,
                 timeout: float = 15.0):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "缺少 The Odds API key。請設環境變數 ODDS_API_KEY，"
                "或到 https://the-odds-api.com 申請免費方案。")
        self.sport = sport
        self.regions = regions
        self.markets = markets
        self.bookmaker = bookmaker
        self.in_play = in_play
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, path: str, params: dict) -> list[dict]:
        params = {**params, "apiKey": self.api_key}
        r = self.session.get(f"{ODDS_API_BASE}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def poll(self) -> list[MatchState]:
        odds_raw = self._get(
            f"/sports/{self.sport}/odds",
            {"regions": self.regions, "markets": self.markets, "oddsFormat": "decimal"},
        )
        odds = parse_odds(odds_raw, bookmaker=self.bookmaker)

        # 取比分（走地才需要）
        scores: dict[str, dict] = {}
        if self.in_play:
            try:
                sc_raw = self._get(f"/sports/{self.sport}/scores", {"daysFrom": 1})
                scores = parse_scores(sc_raw)
            except Exception as e:  # noqa: BLE001 - 比分取不到仍可給初盤資訊
                print(f"[warn] 取比分失敗：{e}")

        states: list[MatchState] = []
        for eid, info in odds.items():
            sc = scores.get(eid, {})
            if sc.get("completed"):
                continue
            minute = estimate_minute(info["commence"])
            is_live = minute > 0
            if self.in_play and not is_live:
                continue
            if not self.in_play and is_live:
                continue
            states.append(MatchState(
                match_id=eid, home=info["home"], away=info["away"],
                minute=minute,
                home_goals=sc.get("home_goals", 0),
                away_goals=sc.get("away_goals", 0),
                quotes=info["quotes"],
            ))
        return states

    def close(self) -> None:
        self.session.close()
