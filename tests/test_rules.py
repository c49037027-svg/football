"""走地十六法則 規則引擎測試。"""
from footy.config import LiveConfig
from footy.live.rules import RuleContext, RuleEngine


def _ctx(**kw):
    base = dict(minute=30, home_goals=0, away_goals=0, home_red=0, away_red=0,
                market="OU", selection="over", line=2.5, exp_total_goals=2.7)
    base.update(kw)
    return RuleContext(**base)


def test_rule7_vetoes_early_bets():
    eng = RuleEngine(LiveConfig())
    v = eng.evaluate(_ctx(minute=5))
    assert not v.allow
    assert "法則7" in (v.vetoed_by or "")


def test_rule2_small_ball_floor_veto():
    eng = RuleEngine(LiveConfig())
    # 小球線 1.5 低於底線 2.0 -> 否決
    v = eng.evaluate(_ctx(market="OU", selection="under", line=1.5))
    assert not v.allow
    assert "法則2" in (v.vetoed_by or "")


def test_rule3_big_ball_cap_veto():
    eng = RuleEngine(LiveConfig())
    # 大球線 4.5 超過頂線 3.5（且非高進球環境）-> 否決
    v = eng.evaluate(_ctx(market="OU", selection="over", line=4.5, exp_total_goals=2.5))
    assert not v.allow
    assert "法則3" in (v.vetoed_by or "")


def test_rule3_relaxed_in_high_goal_env():
    eng = RuleEngine(LiveConfig())
    # 高進球環境（>=2.9）頂線放寬到 4.5，線 4.0 可過
    v = eng.evaluate(_ctx(market="OU", selection="over", line=4.0, exp_total_goals=3.2))
    assert v.allow


def test_rule5_handicap_requires_man_advantage():
    eng = RuleEngine(LiveConfig())
    # 無紅牌人數差，讓球被否決
    v = eng.evaluate(_ctx(market="AH", selection="home", line=-0.5))
    assert not v.allow
    assert "法則5" in (v.vetoed_by or "")
    # 客隊吃紅牌（away_red=1）後，讓球可考慮
    v2 = eng.evaluate(_ctx(market="AH", selection="home", line=-0.5, away_red=1))
    assert v2.allow


def test_rule9_late_game_reduces_stake():
    eng = RuleEngine(LiveConfig())
    v = eng.evaluate(_ctx(minute=85, market="OU", selection="over", line=2.5))
    assert v.allow
    # 終場前注碼折半 × 走地保守 0.8 = 0.4
    assert v.stake_factor < 0.5


def test_rule16_limits_overtrading():
    cfg = LiveConfig(max_live_bets_per_match=2)
    eng = RuleEngine(cfg)
    ctx = _ctx(minute=30)
    for _ in range(2):
        v = eng.evaluate(ctx, match_id="M1")
        assert v.allow
        eng.record_bet("M1")
    v3 = eng.evaluate(ctx, match_id="M1")
    assert not v3.allow
    assert "法則16" in (v3.vetoed_by or "")


def test_rules_disabled_passes_all():
    cfg = LiveConfig(rules_enabled=False)
    eng = RuleEngine(cfg)
    v = eng.evaluate(_ctx(minute=2))  # 即使很早也放行
    assert v.allow and v.stake_factor == 1.0
