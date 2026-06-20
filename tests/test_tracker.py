"""推薦戰績追蹤測試。"""
from footy import tracker
from footy.models import dixon_coles as dc


class _M:
    def __init__(self, num, t1, t2, played=False, hg=0, ag=0):
        self.num, self.team1, self.team2 = num, t1, t2
        self.played, self.hg, self.ag, self.date = played, hg, ag, "2026-06-20"


def test_log_settle_summary(tmp_path, synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    led = tmp_path / "bets.csv"
    matches = [_M(1, h, a), _M(2, a, h)]
    n = tracker.log_upcoming(matches, model, led, min_prob=0.0)
    assert n == 2
    # 不重複記錄
    assert tracker.log_upcoming(matches, model, led, min_prob=0.0) == 0
    # 結算：第1場主隊大勝、第2場主隊(=a)輸
    settled = tracker.settle(led, {1: (3, 0), 2: (0, 2)})
    assert settled == 2
    s = tracker.summary(led)
    assert s.settled == 2 and s.pending == 0
    assert 0.0 <= s.hit_rate <= 1.0
    # 損益 = 命中場 *(odds-1) - 未命中場
    assert abs(s.staked - 2.0) < 1e-9
    assert "ROI" in s.text()


def test_min_prob_filter(tmp_path, synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    led = tmp_path / "b.csv"
    # 門檻設 0.99 → 幾乎不會有推薦達標
    n = tracker.log_upcoming([_M(1, h, a)], model, led, min_prob=0.99)
    assert n == 0
