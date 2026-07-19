"""推薦戰績追蹤測試（多盤口、過/沒過、勝率統整）。"""
from footy import tracker
from footy.models import dixon_coles as dc


class _M:
    def __init__(self, num, t1, t2, played=False, hg=0, ag=0):
        self.num, self.team1, self.team2 = num, t1, t2
        self.played, self.hg, self.ag, self.date = played, hg, ag, "2026-06-20"


def test_settle_one_markets():
    # 1X2
    assert tracker._settle_one("1X2", "home", "", 2, 0) == "win"
    assert tracker._settle_one("1X2", "home", "", 0, 1) == "loss"
    assert tracker._settle_one("1X2", "draw", "", 1, 1) == "win"
    # 大小 2.5
    assert tracker._settle_one("OU", "大", 2.5, 2, 1) == "win"
    assert tracker._settle_one("OU", "小", 2.5, 1, 0) == "win"
    assert tracker._settle_one("OU", "大", 2.5, 1, 0) == "loss"
    # 整數線走盤
    assert tracker._settle_one("OU", "大", 3.0, 2, 1) == "push"
    # BTTS
    assert tracker._settle_one("BTTS", "是", "", 1, 1) == "win"
    assert tracker._settle_one("BTTS", "否", "", 2, 0) == "win"
    # 亞盤：主 -1.5，主隊贏 2-0 → 過
    assert tracker._settle_one("AH", "主", -1.5, 2, 0) == "win"
    assert tracker._settle_one("AH", "主", -1.5, 1, 0) == "loss"
    # 主 -1.0 平半盤，主隊剛好贏 1 球 → 走盤
    assert tracker._settle_one("AH", "主", -1.0, 1, 0) == "push"


def test_log_settle_summary(tmp_path, synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    led = tmp_path / "bets.csv"
    matches = [_M(1, h, a), _M(2, a, h)]
    n = tracker.log_upcoming(matches, model, led)
    assert n >= 4  # 至少 1X2+亞盤 ×2 場
    # 不重複
    assert tracker.log_upcoming(matches, model, led) == 0
    settled = tracker.settle(led, {1: (3, 0), 2: (0, 2)})
    assert settled == n
    s = tracker.summary(led)
    assert s.settled == n
    assert 0.0 <= s.win_rate <= 1.0
    assert "勝率" in s.text()
    # 各盤口別都有統計
    assert "1X2" in s.by_market


def test_pl_half_outcomes():
    # 亞盤 quarter 線：主 -0.75，主隊贏 1 球 → 半贏（賠率 2.0 → +0.5u）
    w = tracker._settle_outcome("AH", "主", -0.75, 1, 0)
    assert abs(tracker._pl(w, 2.0) - 0.5) < 1e-9
    # 主 -0.25 主隊和局 → 半輸（-0.5u）
    w = tracker._settle_outcome("AH", "主", -0.25, 1, 1)
    assert abs(tracker._pl(w, 2.0) + 0.5) < 1e-9
    # 1X2 命中：賠率 2.5 → +1.5u；落空 → -1u
    assert abs(tracker._pl(tracker._settle_outcome("1X2", "home", "", 2, 0), 2.5) - 1.5) < 1e-9
    assert abs(tracker._pl(tracker._settle_outcome("1X2", "home", "", 0, 1), 2.5) + 1.0) < 1e-9


def test_logs_model_pick_with_odds_and_profit(tmp_path, synthetic_df):
    """記的是模型最看好的一邊（非 +EV 冷門），附真實賠率→可算準確度+收益。"""
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    top = max(o, key=o.get)            # 模型最看好的 1X2 結果
    odds = 3.0
    quotes = [MarketQuote("1X2", top, odds)]
    led = tmp_path / "m.csv"
    tracker.log_upcoming([_M(1, h, a)], model, led, odds_index={1: quotes})
    df = tracker.load_ledger(led)
    r = df[df["market"] == "1X2"].iloc[0]
    assert r["selection"] == top and r["source"] == "market" and float(r["odds"]) == odds
    # 讓模型那邊命中 → 收益 = odds-1
    score = {"home": (2, 0), "away": (0, 2), "draw": (1, 1)}[top]
    tracker.settle(led, {1: score})
    s = tracker.summary(led)
    assert s.market_bets == 1 and abs(s.pl_units - (odds - 1)) < 1e-6
    assert s.roi > 0 and "ROI" in s.text()


def test_ev_engine_blend(synthetic_df):
    """+EV 引擎（供 tune_weight 用）：純市場(weight=0)去 vig 後不該有假 +EV。"""
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    # 灌水主勝賠率：純模型(weight=1)會視為 +EV；純市場(weight=0)去 vig 後消失
    quotes = [MarketQuote("1X2", "home", round(1.0 / o["home"] * 1.08, 3)),   # 灌 8%（在 MAX_EDGE=10% 風控內）
              MarketQuote("1X2", "draw", round(1.0 / o["draw"] * 0.95, 3)),
              MarketQuote("1X2", "away", round(1.0 / o["away"] * 0.95, 3))]
    e1 = [r for r in tracker._market_edges(model, h, a, quotes, weight=1.0) if r["market"] == "1X2"]
    e0 = [r for r in tracker._market_edges(model, h, a, quotes, weight=0.0) if r["market"] == "1X2"]
    assert e1                       # 純模型找到灌水的 +EV
    if e0:                          # 純市場若仍下注，edge 必然更小
        assert e0[0]["edge"] < e1[0]["edge"] + 1e-9


def test_garbage_odds_not_recorded(tmp_path, synthetic_df):
    """模型推薦那一邊若賠率是髒資料(>MAX_ODDS)，不附賠率（不進收益）。"""
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    top = max(o, key=o.get)
    quotes = [MarketQuote("1X2", top, 100.0)]   # 模型推薦邊的賠率是髒資料
    recs = tracker._recommendations(model, h, a, quotes=quotes)
    r1x2 = next(r for r in recs if r[0] == "1X2")
    assert r1x2[1] == top and r1x2[3] is None   # 賠率被當無效，不附


def test_backfill_played(tmp_path, synthetic_df):
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    led = tmp_path / "b.csv"
    # 已踢比賽 → 回填並立即結算
    played = [_M(1, h, a, played=True, hg=2, ag=0),
              _M(2, a, h, played=True, hg=1, ag=1)]
    n = tracker.backfill_played(played, model, led)
    assert n >= 4  # 至少 1X2+亞盤 ×2 場
    df = tracker.load_ledger(led)
    assert (df["source"] == "model").all()
    assert (df["result"] != "pending").all()   # 全部已結算
    # 不重複
    assert tracker.backfill_played(played, model, led) == 0
    s = tracker.summary(led)
    assert s.settled == n and s.market_bets == 0   # 不進 ROI


def test_history_and_tune_weight(tmp_path, synthetic_df):
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    top = max(o, key=o.get)
    quotes = [MarketQuote("1X2", "home", round(1.0 / o["home"] * 1.05, 3)),
              MarketQuote("1X2", "draw", round(1.0 / o["draw"] * 1.05, 3)),
              MarketQuote("1X2", "away", round(1.0 / o["away"] * 1.05, 3))]
    led, snap = tmp_path / "b.csv", tmp_path / "odds_log.csv"
    ms = [_M(1, h, a)]
    tracker.log_upcoming(ms, model, led, odds_index={1: quotes})
    tracker.log_snapshots(ms, model, {1: quotes}, snap)
    # 快照含全部三柱（不過濾）
    assert len(tracker.load_snapshots(snap)) == 3
    # 讓模型最看好那邊命中 → 收益 > 0
    score = {"home": (2, 0), "away": (0, 2), "draw": (1, 1)}[top]
    tracker.settle(led, {1: score})
    tracker.settle_snapshots(snap, {1: score})
    # history：一筆已結算、累積損益>0
    hrows = tracker.history(led)
    assert len(hrows) == 1 and hrows[0]["cum_pl"] > 0
    assert hrows[0]["n"] == 1 and "cum_clv" in hrows[0]
    # tune_weight：回傳各權重列與最佳
    res = tracker.tune_weight(snap)
    assert res["n_matches"] == 1 and res["rows"]
    assert res["best_roi"]["n_bets"] >= 1
    # 空快照 → 安全回傳
    assert tracker.tune_weight(tmp_path / "none.csv")["rows"] == []


def test_tune_weight_incomplete_snapshot(tmp_path):
    """快照某盤口只有單邊（缺一個選項）時，tune_weight 不該丟例外。"""
    import pandas as pd
    snap = tmp_path / "odds_log.csv"
    # AH 只記了 home 一邊（缺 away）→ 舊版會 KeyError
    df = pd.DataFrame([dict(date="d", match_num=1, home="A", away="B", market="AH",
                            selection="home", line=-0.5, model_p=0.55, odds=1.95,
                            close_odds=1.95, hg=2, ag=0, played=1)])
    tracker.save_ledger(df, snap)
    res = tracker.tune_weight(snap)   # 不丟例外
    assert "rows" in res and res["n_matches"] == 1


def test_clv_recorded(tmp_path):
    from footy.live.feed import MarketQuote
    import pandas as pd
    led = tmp_path / "c.csv"
    df = pd.DataFrame([dict(date="d", match_num=1, home="A", away="B",
                            market="1X2", selection="home", line="", odds=2.10,
                            edge=0.05, source="market", close_odds=2.10,
                            result="pending", pl="")])
    tracker.save_ledger(df, led)
    # 收盤跌到 1.90（我們 2.10 拿得比收盤好）→ CLV 正
    tracker.refresh_close(led, {1: [MarketQuote("1X2", "home", 1.90)]})
    tracker.settle(led, {1: (1, 0)})
    s = tracker.summary(led)
    assert s.clv_n == 1 and s.clv > 0
    assert abs(s.clv - (2.10 / 1.90 - 1)) < 1e-9


def test_summary_counts(tmp_path):
    import pandas as pd
    led = tmp_path / "l.csv"
    df = pd.DataFrame([
        dict(date="d", match_num=1, home="A", away="B", market="1X2",
             selection="home", line="", result="win"),
        dict(date="d", match_num=1, home="A", away="B", market="OU",
             selection="大", line=2.5, result="loss"),
        dict(date="d", match_num=2, home="A", away="B", market="1X2",
             selection="away", line="", result="win"),
    ])
    tracker.save_ledger(df, led)
    s = tracker.summary(led)
    assert s.wins == 2 and s.losses == 1
    assert abs(s.win_rate - 2/3) < 1e-9
    assert s.by_market["1X2"] == (2, 0, 0)
