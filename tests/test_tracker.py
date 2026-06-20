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


def test_market_mode_ev_filter_and_roi(tmp_path, synthetic_df):
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    # 給主勝一個「高到必為 +EV」的賠率，客勝給很爛的賠率
    big = round(2.0 / max(o["home"], 1e-6), 3)  # p*odds = 2 → edge ~ +100%
    quotes = [MarketQuote("1X2", "home", big),
              MarketQuote("1X2", "away", 1.01)]
    led = tmp_path / "m.csv"
    # weight=1 純模型，讓 pl 與賠率精確對得上（融合行為另測）
    n = tracker.log_upcoming([_M(1, h, a)], model, led,
                             odds_index={1: quotes}, weight=1.0)
    assert n == 1  # 只記 +EV 的主勝，不記爛賠率客勝
    dfx = tracker.load_ledger(led)
    assert dfx.iloc[0]["selection"] == "home" and dfx.iloc[0]["source"] == "market"
    # 結算：主勝 2-0 命中 → pl = big-1
    tracker.settle(led, {1: (2, 0)})
    s = tracker.summary(led)
    assert s.market_bets == 1
    assert abs(s.pl_units - (big - 1)) < 1e-6
    assert s.roi > 0 and "ROI" in s.text()


def test_blending_shrinks_fake_edge(tmp_path, synthetic_df):
    """融合：純市場(weight=0)時，市場兩邊去 vig 後不該有正 edge → 不下注。"""
    from footy.live.feed import MarketQuote
    from footy.models import dixon_coles as dc
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    o = tracker.markets.outcome_1x2(model.score_matrix(h, a, neutral=True))
    # 給一組「公平」三柱賠率（模型機率取倒數，無 vig）
    quotes = [MarketQuote("1X2", s, round(1.0 / o[s], 3)) for s in ("home", "draw", "away")]
    led = tmp_path / "b.csv"
    # weight=1 純模型：edge≈0（賠率正好等於模型公平價）→ 不超過門檻
    n_model = tracker.log_upcoming([_M(1, h, a)], model, led,
                                   odds_index={1: quotes}, weight=1.0, min_edge=0.0)
    # weight=0 純市場：去 vig 後 p·odds=1 → edge≤0 → 必不下注
    led2 = tmp_path / "b2.csv"
    n_market = tracker.log_upcoming([_M(1, h, a)], model, led2,
                                    odds_index={1: quotes}, weight=0.0, min_edge=0.0)
    assert n_market == 0
    assert n_model <= n_market + 1  # 純市場下注數不多於純模型


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
