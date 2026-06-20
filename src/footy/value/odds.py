"""賠率與機率互換、去除莊家抽水（vig / overround）。

歐洲（小數）賠率 odds 的「隱含機率」= 1 / odds。
但一個市場所有結果的隱含機率加總會 > 1，多出來的部分就是莊家利潤（overround）。
要得到「公平機率」必須去 vig。常見方法：
  - proportional（按比例縮放）：最簡單。
  - shin：考慮 insider trading 的方法，對 favourite-longshot bias 較穩健。
我們預設用 proportional，並提供 shin 供進階使用。
"""
from __future__ import annotations

import numpy as np


def odds_to_implied(odds: float) -> float:
    """小數賠率 -> 含 vig 的隱含機率。"""
    return 1.0 / odds


def overround(odds_list: list[float]) -> float:
    """市場 overround（總隱含機率），> 1 的部分為莊家利潤。"""
    return sum(1.0 / o for o in odds_list)


def remove_vig_proportional(odds_list: list[float]) -> list[float]:
    """按比例去 vig，回傳各結果的公平機率（加總為 1）。"""
    if any((o is None) or (o <= 1.0) for o in odds_list):
        raise ValueError(f"無效賠率（須 > 1）：{odds_list}")
    implied = np.array([1.0 / o for o in odds_list])
    return (implied / implied.sum()).tolist()


def remove_vig_shin(odds_list: list[float], max_iter: int = 100,
                    tol: float = 1e-10) -> list[float]:
    """Shin (1992) 去 vig 法，回傳公平機率（加總為 1）。

    對 favourite-longshot bias 比按比例法更合理。
    解出 insider 比例 z，使 sum(p_i) = 1，其中
        p_i = (sqrt(z^2 + 4(1-z) * pi_i^2 / B) - z) / (2(1-z))
    pi_i 為含 vig 隱含機率，B = sum(pi_i)。
    """
    pi = np.array([1.0 / o for o in odds_list])
    B = pi.sum()
    z = 0.0
    for _ in range(max_iter):
        p = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / B) - z) / (2 * (1 - z))
        s = p.sum()
        if abs(s - 1.0) < tol:
            break
        # 簡單修正：z 隨總和偏離調整
        z += (s - 1.0) * 0.5
        z = min(max(z, 0.0), 0.99)
    p = (np.sqrt(z**2 + 4 * (1 - z) * pi**2 / B) - z) / (2 * (1 - z))
    return (p / p.sum()).tolist()


def fair_odds(prob: float) -> float:
    """公平機率 -> 公平賠率（無 vig）。"""
    return float("inf") if prob <= 0 else 1.0 / prob


def blend_probs(model_p: list[float], market_fair_p: list[float],
                weight: float) -> list[float]:
    """市場融合：回傳 weight·模型 +(1-weight)·市場公平機率，重新正規化。

    weight=1 純模型、weight=0 純市場。實證上純模型常贏不過市場（見 FINDINGS），
    故融合可把「模型雜訊誤判成 edge」往市場拉回，只留真正的分歧。
    """
    m = np.array(model_p, dtype=float)
    k = np.array(market_fair_p, dtype=float)
    b = weight * m + (1.0 - weight) * k
    s = b.sum()
    return (b / s).tolist() if s > 0 else list(model_p)
