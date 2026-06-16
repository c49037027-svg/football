from footy.config import RiskConfig, StakingConfig, ValueConfig
from footy.risk.manager import RiskManager
from footy.value import odds, staking
from footy.value.edge import evaluate


def test_remove_vig_proportional_sums_one():
    fair = odds.remove_vig_proportional([2.0, 3.5, 4.0])
    assert abs(sum(fair) - 1.0) < 1e-12


def test_overround_above_one():
    assert odds.overround([1.95, 1.95]) > 1.0  # 典型 vig


def test_kelly_positive_when_value():
    # 勝率 0.55、賠率 2.0 => 正凱利
    f = staking.kelly_fraction(0.55, 2.0)
    assert f > 0


def test_kelly_zero_when_no_edge():
    f = staking.kelly_fraction(0.45, 2.0)  # EV 為負
    assert f <= 0


def test_stake_respects_cap():
    cfg = StakingConfig(kelly_fraction=1.0, max_stake_fraction=0.02, min_stake=1.0)
    # 很大的 EV 也應被 2% 上限封頂
    amt = staking.stake(1000, ev=0.5, odds=3.0, cfg=cfg)
    assert amt <= 1000 * 0.02 + 1e-9


def test_evaluate_value_flag():
    cfg = ValueConfig(min_edge=0.03, min_ev=0.02, min_odds=1.3, max_odds=10)
    # 模型勝率 0.6，賠率 2.0（隱含 0.5），明顯 value
    vb = evaluate("1X2", "home", odds=2.0, prob_win=0.6, cfg=cfg, fair_prob=0.5)
    assert vb.is_value
    assert vb.ev > 0


def test_risk_daily_stop_loss():
    cfg = RiskConfig(bankroll=1000, daily_stop_loss=0.10)
    rm = RiskManager(cfg)
    import datetime
    day = datetime.date(2024, 1, 1)
    rm.roll_day(day)
    rm.daily_pnl = -120  # 已虧 12% > 10%
    d = rm.approve("M1", "home", 10, 2.0, day=day)
    assert not d.approved
    assert "停損" in d.reason


def test_risk_drawdown_halt():
    cfg = RiskConfig(bankroll=1000, max_drawdown=0.25)
    rm = RiskManager(cfg)
    rm.peak_bankroll = 1000
    rm.bankroll = 700  # 回撤 30% > 25%
    d = rm.approve("M1", "home", 10, 2.0)
    assert not d.approved
    assert rm.halted


def test_risk_match_exposure_cap_reduces():
    cfg = RiskConfig(bankroll=1000, max_match_exposure=0.05, max_daily_exposure=1.0)
    rm = RiskManager(cfg)
    import datetime
    day = datetime.date(2024, 1, 1)
    d1 = rm.approve("M1", "home", 40, 2.0, day=day)
    rm.place("M1", "home", d1.stake, 2.0)
    # 已用 40，單場上限 50，再要 30 應被縮到 10
    d2 = rm.approve("M1", "draw", 30, 3.0, day=day)
    assert d2.approved
    assert abs(d2.stake - 10) < 1e-6
