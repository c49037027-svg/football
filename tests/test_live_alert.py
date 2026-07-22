"""走地機會 agent 測試：+EV 偵測、風控過濾、去重、格式、通知設定。"""
from footy.agents import live_alert, notify
from footy.mlb_live import LiveState
from footy.prematch import MarketQuote


def _mlb_snap(p_home):
    return {"rows": [{
        "game": {"home": "New York Yankees", "away": "Boston Red Sox"},
        "state": LiveState(6, "bottom", 1, "010", 3, 2),
        "p_home": p_home, "exp_total": 8.0, "sp_note": "",
        "fair": {"home_odds": 1.6, "away_odds": 2.7, "over_lines": [], "run_lines": []},
    }], "others": []}


def test_scan_mlb_finds_positive_ev():
    # 模型 62% 主勝，盤口 1.85/2.05（去 vig 後主約 53%）→ 融合 ~57%，edge ~6%（落在 5~10%）
    odds = {1: [MarketQuote("1X2", "home", 1.85), MarketQuote("1X2", "away", 2.05)]}
    opps = live_alert.scan_mlb(snapshot=_mlb_snap(0.62), odds_index=odds)
    assert len(opps) == 1
    o = opps[0]
    assert o["side"] == "home" and o["edge"] >= 0.05 and o["market"] == "錢線"
    assert "洋基" in o["pick"] and o["sport"] == "⚾ MLB"


def test_scan_mlb_no_edge_when_priced():
    # 盤口已把模型看法算進去 → 無 +EV
    odds = {1: [MarketQuote("1X2", "home", 1.5), MarketQuote("1X2", "away", 2.5)]}
    assert live_alert.scan_mlb(snapshot=_mlb_snap(0.62), odds_index=odds) == []


def test_scan_mlb_respects_gate(monkeypatch):
    from footy import mlb
    monkeypatch.setattr(mlb, "market_allowed", lambda m, s="MLB": False)
    odds = {1: [MarketQuote("1X2", "home", 2.0), MarketQuote("1X2", "away", 2.0)]}
    assert live_alert.scan_mlb(snapshot=_mlb_snap(0.62), odds_index=odds) == []


def test_scan_mlb_caps_fake_high_edge():
    # 灌爆的賠率 → edge 破 MAX_EDGE(10%) 應被擋（假高 edge 幾乎都是錯的）
    odds = {1: [MarketQuote("1X2", "home", 5.0), MarketQuote("1X2", "away", 1.2)]}
    opps = live_alert.scan_mlb(snapshot=_mlb_snap(0.62), odds_index=odds)
    assert all(o["edge"] <= 0.10 for o in opps)


def test_dedup_and_run(tmp_path):
    odds = {1: [MarketQuote("1X2", "home", 1.85), MarketQuote("1X2", "away", 2.05)]}
    sp = str(tmp_path / "state.json")
    scan = lambda: live_alert.scan_mlb(snapshot=_mlb_snap(0.62), odds_index=odds)
    # 第一次：找到、寫入去重狀態
    r1 = live_alert.run(state_path=sp, scans=[scan], now_iso="2026-07-20T18:00:00+00:00")
    assert r1["found"] == 1 and r1["fresh"] == 1
    # 20 分鐘後同機會 → 去重擋掉
    r2 = live_alert.run(state_path=sp, scans=[scan], now_iso="2026-07-20T18:20:00+00:00")
    assert r2["found"] == 1 and r2["fresh"] == 0
    # 46 分鐘後 → 可再推
    r3 = live_alert.run(state_path=sp, scans=[scan], now_iso="2026-07-20T18:46:00+00:00")
    assert r3["fresh"] == 1


def test_format_alert():
    o = {"sport": "⚾ MLB", "away": "紅襪", "home": "洋基", "state": "6局下 2-3",
         "market": "錢線", "pick": "洋基", "side": "home", "odds": 1.95,
         "p": 0.58, "edge": 0.12}
    t = live_alert.format_alert(o)
    assert "走地 +EV" in t and "洋基" in t and "@1.95" in t and "+12%" in t
    assert "確認現場比分" in t


def test_notify_configured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    assert notify.configured() == []
    monkeypatch.setenv("NOTIFY_WEBHOOK_URL", "https://example.com/hook")
    assert notify.configured() == ["webhook"]
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    assert set(notify.configured()) == {"telegram", "webhook"}


def test_live_ledger_logs_and_settles(tmp_path):
    """走地推薦記帳 → 結算 → 勝率摘要（不打網路，直接驗記帳＋tracker 結算）。"""
    from footy import mlb, tracker
    from footy.agents import live_alert
    lp = str(tmp_path / "live_bets.csv")
    # 一筆走地推薦（洋基主勝 @1.95）
    opps = [{"ledger": "MLB", "game_pk": 777, "raw_home": "New York Yankees",
             "raw_away": "Boston Red Sox", "side": "home", "odds": 1.95, "edge": 0.08}]
    assert live_alert._log_alerts(opps, lp, "2026-07-22") == 1
    df = tracker.load_ledger(lp)
    r = df[df["market"] == "1X2"].iloc[0]
    assert int(r["match_num"]) == 777 and r["selection"] == "home" and r["result"] == "pending"
    # 結算：洋基贏 5-3 → 該注 win
    tracker.settle(lp, {777: (5, 3)})
    s = mlb.summary_text(lp, label="🔴 走地推薦")
    assert s and "走地推薦" in s and "1 勝" in s
    # 不重複記帳
    assert live_alert._log_alerts(opps, lp, "2026-07-22") == 0
