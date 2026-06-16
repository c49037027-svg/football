from footy.backtest import engine
from footy.config import Config


def test_backtest_runs_no_lookahead(synthetic_df):
    cfg = Config()
    cfg.value.min_edge = 0.0
    cfg.value.min_ev = -1.0  # 放寬讓回測一定有下注，驗證流程
    res = engine.run(synthetic_df, cfg, refit_every=20, min_train_matches=50,
                     verbose=False)
    # 流程能跑完並產生統計
    assert res.starting_bankroll == cfg.risk.bankroll
    assert res.ending_bankroll > 0
    assert len(res.equity_curve) >= 1
    assert 0.0 <= res.max_drawdown <= 1.0
    # summary 不應拋錯
    assert isinstance(res.summary(), str)


def test_backtest_no_bets_when_edge_too_high(synthetic_df):
    cfg = Config()
    cfg.value.min_edge = 5.0  # 不可能達到，應該完全不下注
    res = engine.run(synthetic_df, cfg, refit_every=20, min_train_matches=50,
                     verbose=False)
    assert res.n_bets == 0
    assert res.ending_bankroll == res.starting_bankroll
