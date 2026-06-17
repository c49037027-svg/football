"""市場融合測試。"""
import numpy as np

from footy import blend
from footy.config import Config
from footy.evaluation import EvalResult


def test_blend_1x2_endpoints():
    model = {"home": 0.6, "draw": 0.25, "away": 0.15}
    odds = {"home": 2.0, "draw": 3.5, "away": 5.0}  # 含 vig
    # w=1 → 純模型
    b1 = blend.blend_1x2(model, odds, 1.0)
    assert abs(b1["home"] - 0.6) < 1e-9
    # w=0 → 純市場（去 vig 後）
    b0 = blend.blend_1x2(model, odds, 0.0)
    assert abs(sum(b0.values()) - 1.0) < 1e-9
    assert b0["home"] > b0["draw"] > b0["away"]  # 賠率越低機率越高
    # 融合結果恆為合法分布
    bm = blend.blend_1x2(model, odds, 0.5)
    assert abs(sum(bm.values()) - 1.0) < 1e-9


def test_blend_probs_normalized():
    out = blend.blend_probs([0.5, 0.3, 0.2], [0.4, 0.4, 0.2], 0.5)
    assert abs(sum(out) - 1.0) < 1e-9


def test_best_blend_prefers_better_source():
    """若市場完美、模型亂猜，最佳 w 應接近 0（偏向市場）。"""
    rng = np.random.default_rng(0)
    n = 800
    actuals = rng.integers(0, 3, size=n)
    res = EvalResult()
    for a in actuals:
        # 市場：對真實結果給高機率（近乎完美）
        mk = [0.1, 0.1, 0.1]; mk[a] = 0.8
        # 模型：均勻亂猜
        res.model_probs.append([1/3, 1/3, 1/3])
        res.market_probs.append(mk)
        res.actuals.append(int(a))
    bb = res.best_blend()
    assert bb["w"] <= 0.2  # 應大幅偏向市場
    assert bb["blended_logloss"] <= bb["market_logloss"] + 1e-6


def test_best_blend_improves_when_model_adds_info():
    """模型與市場各含部分資訊時，最佳融合應優於任一單獨來源。"""
    rng = np.random.default_rng(1)
    n = 1500
    res = EvalResult()
    for _ in range(n):
        a = int(rng.integers(0, 3))
        # 兩個來源各自把真值機率拉高一些、但加不同噪音
        m = rng.dirichlet([2, 2, 2]); m[a] += 0.25
        k = rng.dirichlet([2, 2, 2]); k[a] += 0.25
        m = m / m.sum(); k = k / k.sum()
        res.model_probs.append(list(m))
        res.market_probs.append(list(k))
        res.actuals.append(a)
    bb = res.best_blend()
    # 融合不應比兩者中較好的那個還差
    assert bb["blended_logloss"] <= min(bb["model_logloss"], bb["market_logloss"]) + 1e-6
    assert 0.0 <= bb["w"] <= 1.0
