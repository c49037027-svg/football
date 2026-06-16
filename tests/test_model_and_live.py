import numpy as np

from footy.config import Config
from footy.live.feed import SimulatedFeed
from footy.live.inplay import final_score_matrix
from footy.live.monitor import LiveMonitor
from footy.models import dixon_coles as dc
from footy.models import markets


def test_fit_recovers_home_advantage(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)  # 幾乎不衰減
    # 主場優勢應為正（合成資料設 0.25）
    assert model.home_adv > 0
    # attack 參數平均約為 0（可辨識性約束）
    assert abs(np.mean(list(model.attack.values()))) < 1e-6


def test_score_matrix_from_model(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    mat = model.score_matrix(h, a)
    assert abs(mat.sum() - 1.0) < 1e-9
    o = markets.outcome_1x2(mat)
    assert abs(sum(o.values()) - 1.0) < 1e-9


def test_inplay_matrix_shifts_with_score(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    # 第 80 分鐘 2-0：最終比分最低應是 2-0（不可能再變少）
    mat = final_score_matrix(model, h, a, minute=80, home_goals=2, away_goals=0)
    # P(最終主隊 < 2) 必須為 0
    assert mat[:2, :].sum() < 1e-12
    assert abs(mat.sum() - 1.0) < 1e-9
    # 80 分鐘 2-0 時，主隊不輸的機率應該很高
    o = markets.outcome_1x2(mat)
    assert o["home"] + o["draw"] > 0.8


def test_live_monitor_runs_with_simulated_feed(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    cfg = Config()
    cfg.value.min_edge = 0.0   # 放寬以確保模擬中至少評估到一些盤口
    cfg.value.min_ev = -1.0
    feed = SimulatedFeed(h, a, true_lambda=1.6, true_mu=1.0,
                         minutes_per_poll=10, seed=1)
    alerts = []
    monitor = LiveMonitor(model, cfg, alert_sink=alerts.append)
    monitor.run(feed, sleep_s=0.0)
    # 跑完整場不應拋例外；feed 結束後 minute 應達 90
    assert feed.minute >= 90
