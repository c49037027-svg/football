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
