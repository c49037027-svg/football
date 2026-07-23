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
QUOTA_PATH = "data/odds_quota.json"


def record_quota(resp, path: str = QUOTA_PATH) -> None:
    """把 the-odds-api 回應的額度 header 存進 JSON（介面顯示用）。best-effort。"""
    import json
    from pathlib import Path
    rem = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if rem is None and used is None:
        return
    try:
        rem_i = int(float(rem)) if rem is not None else None
        used_i = int(float(used)) if used is not None else None
        data = {"remaining": rem_i, "used": used_i,
                "total": (rem_i + used_i) if (rem_i is not None and used_i is not None) else None,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def read_quota(path: str = QUOTA_PATH) -> dict | None:
    """讀最近一次記錄的 the-odds-api 額度。無檔/壞檔 → None。"""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


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


# 隊名配對用：只去純法人後綴（保留 city/united/town 等辨識詞，靠前綴匹配處理
# Man↔Manchester；若去掉 city/united 會害曼城↔曼聯互相誤配）
_DROP_TOKENS = {"fc", "cf", "afc", "sc", "ac", "cd", "ss", "as", "calcio", "club"}
_MATCH_THRESHOLD = 1.35   # 雙隊合計相似度門檻（每隊 0~1，真配對通常近 2.0）


def _norm_tokens(name: str) -> tuple[str, set]:
    """隊名正規化：套別名→去重音→小寫→去標點→去冗詞。回 (正規字串, token 集)。"""
    import re
    import unicodedata

    from ..worldcup import TEAM_ALIASES
    n = TEAM_ALIASES.get(name, name)
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"[^a-z0-9 ]", " ", n.lower())
    toks = [t for t in n.split() if t]
    core = [t for t in toks if t not in _DROP_TOKENS] or toks   # 全被去光則保留原 token
    return " ".join(core), set(core)


def _name_sim(a: str, b: str) -> float:
    """兩隊名相似度 0~1：序列比對與 token 重疊（含前綴，如 man↔manchester）取大者。"""
    import difflib
    sa, ta = _norm_tokens(a)
    sb, tb = _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    matched = 0
    for x in ta:
        if x in tb or any(len(x) >= 3 and (y.startswith(x) or x.startswith(y))
                          for y in tb):
            matched += 1
    tok = matched / max(len(ta), len(tb))          # token 覆蓋率（含前綴匹配）
    seq = difflib.SequenceMatcher(None, sa, sb).ratio()
    return max(tok, seq)


def _best_event(parsed: dict, home: str, away: str, need=None):
    """在 parsed 事件中找與 (home, away) 最匹配者。need(info)→bool 額外條件。"""
    best_score, best = _MATCH_THRESHOLD, None
    for info in parsed.values():
        if need and not need(info):
            continue
        score = _name_sim(info.get("home") or "", home) + _name_sim(info.get("away") or "", away)
        if score > best_score:
            best_score, best = score, info
    return best


def find_ah_line(parsed: dict, home: str, away: str) -> "dict | None":
    """從 parse_odds 的結果裡，找指定對戰的亞盤讓球線（主隊視角）。"""
    info = _best_event(parsed, home, away,
                       need=lambda i: any(q.market == "AH" for q in i.get("quotes", [])))
    if not info:
        return None
    ah = [q for q in info["quotes"] if q.market == "AH"]
    home_q = next((q for q in ah if q.selection == "home"), None)
    away_q = next((q for q in ah if q.selection == "away"), None)
    if home_q is None:
        return None
    return {"line": home_q.line, "home_odds": home_q.odds,
            "away_odds": away_q.odds if away_q else None}


def find_quotes(parsed: dict, home: str, away: str) -> "list | None":
    """從 parse_odds 結果裡，找指定對戰的全部盤口報價（穩健隊名配對）。"""
    info = _best_event(parsed, home, away, need=lambda i: bool(i.get("quotes")))
    return info["quotes"] if info else None


def fetch_wc_odds(matches, sport: str = "soccer_fifa_world_cup",
                  api_key: str | None = None, regions: str = "eu",
                  markets: str = "h2h,totals,spreads", bookmaker: str | None = None,
                  timeout: float = 20.0) -> dict:
    """抓世界盃各場盤口，配對成 {match_num: [MarketQuote...]}（需 ODDS_API_KEY）。

    抓不到 key／網路被擋／無對應賽事時，個別略過；整體失敗則拋例外由呼叫端處理。
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("缺少 ODDS_API_KEY")
    r = requests.get(f"{ODDS_API_BASE}/sports/{sport}/odds",
                     params={"regions": regions, "markets": markets,
                             "oddsFormat": "decimal", "apiKey": key},
                     timeout=timeout)
    record_quota(r)          # 記錄剩餘額度（介面顯示用）
    r.raise_for_status()
    parsed = parse_odds(r.json(), bookmaker=bookmaker)
    index: dict = {}
    unmatched = []
    todo = [m for m in matches if not getattr(m, "played", False)]
    for m in todo:
        q = find_quotes(parsed, m.team1, m.team2)
        if q:
            index[m.num] = q
        else:
            unmatched.append(f"{m.team1} vs {m.team2}")
    # 診斷：the-odds-api 回了幾場、我方配對到幾場、哪些沒配對到（隊名對不上時看得見）
    if parsed:
        msg = f"[odds:{sport}] 來源 {len(parsed)} 場，配對 {len(index)}/{len(todo)}"
        if unmatched:
            msg += "；未配對：" + "、".join(unmatched[:6])
            if len(unmatched) > 6:
                msg += f" 等 {len(unmatched)} 場"
        print(msg, flush=True)
    return index


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
