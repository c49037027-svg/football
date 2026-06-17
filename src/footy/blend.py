"""市場融合（market blending）：把模型機率與市場賠率（去 vig）線性融合。

實證顯示基礎模型 log-loss 贏不過市場收盤盤，但**模型常帶有市場沒有的增量資訊**；
用 `最終 = w·模型 + (1-w)·市場` 融合，通常能得到比兩者單獨都低的 log-loss。
w 應由 `footy evaluate` 的 best_blend（樣本外）決定，而非隨手設。

用途：當某場有市場賠率時，把模型輸出「往市場拉」一個最佳比例，得到更準的機率，
再拿這個融合機率去算 value（仍要 +EV 才下注）。
"""
from __future__ import annotations

from .value.odds import remove_vig_proportional


def blend_1x2(model_probs: dict[str, float], odds: dict[str, float],
              w: float) -> dict[str, float]:
    """融合 1X2 機率。

    model_probs：{'home','draw','away'} 模型機率（和為 1）。
    odds：對應的小數賠率（含 vig），會先去 vig 成市場公平機率。
    w：模型權重（0~1）。w=1 純模型、w=0 純市場。
    """
    keys = ["home", "draw", "away"]
    fair = remove_vig_proportional([odds[k] for k in keys])
    market = dict(zip(keys, fair))
    blended = {k: w * model_probs[k] + (1.0 - w) * market[k] for k in keys}
    total = sum(blended.values())
    return {k: v / total for k, v in blended.items()}


def blend_probs(model_p: list[float], market_p: list[float], w: float) -> list[float]:
    """通用：融合兩組已正規化機率（皆和為 1）。"""
    blended = [w * m + (1.0 - w) * k for m, k in zip(model_p, market_p)]
    total = sum(blended)
    return [b / total for b in blended]
