"""走地機會 agent：掃即時比分 × 即時盤口 → 過所有風控閘門的真 +EV → 推播。

紀律（與整套系統一致，不可退讓）：「適合下注」＝**對即時盤口的真 +EV**，
不是模型信心高。沿用 tracker 的去 vig ＋ 融合 ＋ MAX_EDGE/MIN_PROB，走地門檻
更嚴（LIVE_MIN_EDGE，因走地盤 vig 高、比分源有延遲）；尊重 market_gates。

範圍：目前掃**錢線（MLB 1X2）**——唯一經半局邊界回測、且未關閘的市場；
足球 1X2 best-effort（世界盃/五大聯賽有即時盤時）。OU 關閘中、AH 待 line 對齊。
去重：同 賽事×盤口×方向 於 DEDUP_MINUTES 內不重複推（避免同一機會洗版）。

比分源延遲防呆：LIVE_MIN_EDGE 拉高（5%），因為 30~90 秒延遲造成的假 edge
通常小且不跨掃描持續；通知內文提醒「確認現場比分與此一致再下」。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

LIVE_MIN_EDGE = 0.05        # 走地 +EV 門檻（比賽前盤嚴：vig 高＋延遲風險）
DEDUP_MINUTES = 45
STATE_PATH = "data/live_alert_state.json"


# ---------------- 邊際計算（重用 tracker 去 vig＋融合） ----------------

def _best_edge(model_ps: dict, quotes, market: str, order: list,
               weight=None) -> dict | None:
    """某盤口的最佳 +EV 邊（過所有風控）。model_ps={sel: 機率}。無 → None。"""
    from .. import tracker
    sides = tracker._group_quotes(quotes, market).get("", {})
    if not all(s in sides for s in order):
        return None
    w = tracker.BLEND_WEIGHT if weight is None else weight
    bl = tracker._blended(model_ps, sides, order, w)
    best = None
    for sel in order:
        odds = sides.get(sel)
        if not odds or not (1.0 < odds <= tracker.MAX_ODDS):
            continue
        p = bl.get(sel, model_ps[sel])
        edge = p * odds - 1.0
        if LIVE_MIN_EDGE <= edge <= tracker.MAX_EDGE and p >= tracker.MIN_PROB:
            if best is None or edge > best["edge"]:
                best = {"side": sel, "odds": float(odds), "p": float(p),
                        "edge": float(edge)}
    return best


# ---------------- 各運動掃描 ----------------

def scan_mlb(snapshot=None, odds_index=None) -> list[dict]:
    """MLB 走地錢線 +EV。snapshot/odds_index 可注入（測試不打網路）。"""
    from .. import mlb, mlb_live
    if not mlb.market_allowed("1X2", "MLB"):     # 錢線被關閘 → 不掃
        return []
    snap = snapshot if snapshot is not None else mlb_live.live_snapshot()
    rows = snap.get("rows", [])
    if not rows:
        return []
    if odds_index is None:
        games = [mlb._Game(i + 1, r["game"]["home"], r["game"]["away"])
                 for i, r in enumerate(rows)]
        try:
            # 走地掃描只做錢線 → 只抓 h2h（省 the-odds-api 額度：1 credit 而非 3）
            odds_index = mlb.fetch_mlb_odds(games, markets="h2h")
        except Exception:  # noqa: BLE001
            return []
    out = []
    for i, r in enumerate(rows):
        quotes = odds_index.get(i + 1)
        if not quotes:
            continue
        ph = r["p_home"]
        best = _best_edge({"home": ph, "away": 1.0 - ph}, quotes, "1X2",
                          ["home", "away"])
        if not best:
            continue
        g, st = r["game"], r["state"]
        half = "上" if st.half == "top" else "下"
        pick = mlb.zh_mlb(g["home"]) if best["side"] == "home" else mlb.zh_mlb(g["away"])
        out.append({"sport": "⚾ MLB", "away": mlb.zh_mlb(g["away"]),
                    "home": mlb.zh_mlb(g["home"]),
                    "state": f"{st.inning}局{half} {st.away_score}-{st.home_score}",
                    "market": "錢線", "pick": pick, **best,
                    "key": f"MLB|{g['away']}@{g['home']}|1X2|{best['side']}"})
    return out


def scan_football(snapshot=None, odds_index=None, codes=None) -> list[dict]:
    """足球走地 1X2 +EV（best-effort：有即時盤才成）。"""
    from .. import foot_live, tracker
    from ..i18n import zh
    snap = snapshot if snapshot is not None else foot_live.live_snapshot()
    rows = snap.get("rows", [])
    if not rows:
        return []
    if odds_index is None:
        return []       # 足球即時盤來源未接（各聯賽 sport code 不同）→ v1 只在注入時掃
    out = []
    for i, r in enumerate(rows):
        quotes = odds_index.get(i + 1)
        if not quotes:
            continue
        p = r["p"]
        best = _best_edge({"home": p["p_home"], "draw": p["p_draw"],
                           "away": p["p_away"]}, quotes, "1X2",
                          ["home", "draw", "away"])
        if not best:
            continue
        g = r["game"]
        lbl = {"home": zh(g["home"]), "draw": "和", "away": zh(g["away"])}[best["side"]]
        out.append({"sport": "⚽ 足球", "away": zh(g["away"]), "home": zh(g["home"]),
                    "state": f"{g['minute']}' {g['home_goals']}-{g['away_goals']}",
                    "market": "1X2", "pick": lbl, **best,
                    "key": f"FOOT|{g['away']}@{g['home']}|1X2|{best['side']}"})
    return out


# ---------------- 去重 + 格式 + 主流程 ----------------

def _load_state(path) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _dedup(opps: list[dict], state: dict, now: datetime) -> list[dict]:
    """去掉 DEDUP_MINUTES 內已推過的同機會。"""
    fresh = []
    for o in opps:
        last = state.get(o["key"])
        if last:
            try:
                if now - datetime.fromisoformat(last) < timedelta(minutes=DEDUP_MINUTES):
                    continue
            except Exception:  # noqa: BLE001
                pass
        fresh.append(o)
    return fresh


def format_alert(o: dict) -> str:
    fair = 1.0 / max(o["p"], 0.005)
    return (f"🔔 走地 +EV\n{o['sport']}｜{o['away']} vs {o['home']}（{o['state']}）\n"
            f"{o['market']} {o['pick']} @{o['odds']:.2f}｜"
            f"模型 {o['p']:.0%}（公平 {fair:.2f}）｜edge +{o['edge']:.0%}\n"
            f"⚠️ 確認現場比分與此一致再下；走地 vig 高、比分源可能延遲。")


def run(dry_run: bool = False, state_path: str = STATE_PATH,
        now_iso: str | None = None, scans: list | None = None) -> dict:
    """跑一輪走地掃描：找 +EV → 去重 → 推播（或 dry_run 只回結果）。"""
    from . import notify
    now = (datetime.fromisoformat(now_iso) if now_iso
           else datetime.now(timezone.utc))
    opps = []
    for fn in (scans if scans is not None else [scan_mlb, scan_football]):
        try:
            opps += fn()
        except Exception:  # noqa: BLE001
            continue
    state = _load_state(state_path)
    fresh = _dedup(opps, state, now)
    channels = notify.configured()
    for o in fresh:
        if not dry_run and channels:
            notify.send(format_alert(o))
        if not dry_run:
            state[o["key"]] = now.isoformat()
    if fresh and not dry_run:
        Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(state_path).write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    return {"found": len(opps), "fresh": len(fresh), "channels": channels,
            "opps": fresh}
