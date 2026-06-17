"""正則化、防洩漏、超參數調校測試。"""
import numpy as np
import pandas as pd

from footy import tuning
from footy.config import Config
from footy.data import schema as S
from footy.models import dixon_coles as dc


def test_regularization_shrinks_params(synthetic_df):
    """較大的 reg 應把攻防參數整體往 0 收縮。"""
    m0 = dc.fit(synthetic_df, half_life_days=10_000, reg=0.0)
    m1 = dc.fit(synthetic_df, half_life_days=10_000, reg=5.0)
    spread0 = np.std(list(m0.attack.values())) + np.std(list(m0.defence.values()))
    spread1 = np.std(list(m1.attack.values())) + np.std(list(m1.defence.values()))
    assert spread1 < spread0  # 收縮後離散度變小


def test_reference_date_excludes_future(synthetic_df):
    """晚於 reference_date 的比賽不應被使用（防洩漏）。"""
    cutoff = synthetic_df[S.DATE].quantile(0.5)
    # 在 cutoff 後才首次出現的球隊，不應進入模型
    before = synthetic_df[synthetic_df[S.DATE] <= cutoff]
    after_only = (set(synthetic_df[S.HOME]) | set(synthetic_df[S.AWAY])) - \
                 (set(before[S.HOME]) | set(before[S.AWAY]))
    model = dc.fit(synthetic_df, half_life_days=10_000, reference_date=cutoff)
    # 用全量但給 cutoff，結果應等同只用 cutoff 前資料的球隊集合
    assert set(model.teams) == (set(before[S.HOME]) | set(before[S.AWAY]))
    for t in after_only:
        assert t not in model.attack


def test_future_match_not_weighted(synthetic_df):
    """給早一點的 reference_date，後面比賽被剔除，模型與『手動截斷』一致。"""
    cutoff = synthetic_df[S.DATE].quantile(0.6)
    m_auto = dc.fit(synthetic_df, half_life_days=10_000, reference_date=cutoff)
    m_manual = dc.fit(synthetic_df[synthetic_df[S.DATE] <= cutoff],
                      half_life_days=10_000, reference_date=cutoff)
    # 兩者攻擊參數應幾乎相同
    common = set(m_auto.teams) & set(m_manual.teams)
    diffs = [abs(m_auto.attack[t] - m_manual.attack[t]) for t in common]
    assert max(diffs) < 1e-6


def test_tune_runs_and_picks_best(synthetic_df):
    grid = {"half_life_days": [180.0, 10_000.0], "reg": [0.0, 2.0],
            "use_elo": [False], "xg_weight": [0.0]}
    res = tuning.tune(synthetic_df, base_cfg=Config(), grid=grid,
                      refit_every=40, min_train_matches=50, verbose=False)
    assert len(res.rows) == 4
    # best 應是 logloss 最小者
    assert res.best["logloss"] == min(r["logloss"] for r in res.rows)
    assert res.market_logloss > 0
    # apply_best 寫回 cfg
    cfg = Config()
    tuning.apply_best(cfg, res)
    assert cfg.model.half_life_days in (180.0, 10_000.0)
    assert isinstance(res.summary(), str)
