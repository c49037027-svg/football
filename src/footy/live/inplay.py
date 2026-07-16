"""走地（in-play）模型：依「當前比分 + 剩餘時間」即時重算各市場機率。

核心想法：
  - pre-match 的預期進球 (lambda, mu) 是「整場 90 分鐘」的量。
  - 比賽進行到第 t 分鐘時，剩餘時間佔比 r = (90 - t) / 90。
  - 在剩餘時間內，雙方「還會再進幾球」近似為 Poisson(lambda*r), Poisson(mu*r)。
  - 把「剩餘進球的比分矩陣」沿軸平移加上「目前已進球」，
    就得到整場最終比分的條件機率分布，再由此推導各市場（O/U、AH、1X2…）。

這是常用且穩健的近似。可選用 rate_scale 反映「比賽動態」（例如落後方壓上、
尾段更開放）。紅牌等情境可進一步調整 rate，這裡保留接口但預設不變。

注意：走地價值更難抓，因為市場反應快、流動性與延遲風險高，
所以決策時應加更高的安全邊際（見 LiveConfig.extra_edge）。
"""
from __future__ import annotations

import numpy as np

from ..models.dixon_coles import DixonColesModel, score_matrix_from_rates


def man_advantage_factors(home_red: int, away_red: int,
                          attack_boost: float = 0.30,
                          defense_penalty: float = 0.25) -> tuple[float, float]:
    """依紅牌造成的人數差，回傳 (主隊進球率乘數, 客隊進球率乘數)。

    每多一名球員（對方紅牌），己方進球率 ×(1+attack_boost*淨人數差)，
    己方少一人則 ×(1-defense_penalty*淨人數差)。淨人數差 = away_red - home_red
    （客隊被罰得多，主隊佔優）。這是穩健的一階近似，可由實證資料再校準。
    """
    net = away_red - home_red  # >0：主隊人數佔優
    home_factor = (1.0 + attack_boost * net) if net > 0 else (1.0 + defense_penalty * net)
    net_away = -net
    away_factor = (1.0 + attack_boost * net_away) if net_away > 0 else (1.0 + defense_penalty * net_away)
    return max(0.05, home_factor), max(0.05, away_factor)


def remaining_score_matrix(model: DixonColesModel, home: str, away: str,
                           minute: int, rate_scale: float = 1.0,
                           total_minutes: int = 90,
                           home_factor: float = 1.0,
                           away_factor: float = 1.0,
                           neutral: bool = False) -> np.ndarray:
    """剩餘時間內『還會再進 (h, a) 球』的機率矩陣。neutral=中立場（不套主場優勢）。"""
    lam, mu = model.expected_goals(home, away, neutral=neutral)
    r = max(0.0, (total_minutes - minute) / total_minutes) * rate_scale
    lam_r = max(1e-8, lam * r * home_factor)
    mu_r = max(1e-8, mu * r * away_factor)
    # 剩餘時間用同一 rho（低分相關性在剩餘子比賽上是近似）
    return score_matrix_from_rates(lam_r, mu_r, model.rho, model.max_goals)


def final_score_matrix(model: DixonColesModel, home: str, away: str,
                       minute: int, home_goals: int, away_goals: int,
                       rate_scale: float = 1.0, total_minutes: int = 90,
                       home_red: int = 0, away_red: int = 0,
                       red_attack_boost: float = 0.30,
                       red_defense_penalty: float = 0.25) -> np.ndarray:
    """整場最終比分的條件機率矩陣（給定目前比分、時間與紅牌人數差）。

    把剩餘進球矩陣平移 (home_goals, away_goals)，並擴大尺寸以容納。
    """
    home_factor, away_factor = man_advantage_factors(
        home_red, away_red, red_attack_boost, red_defense_penalty)
    rem = remaining_score_matrix(model, home, away, minute, rate_scale,
                                 total_minutes, home_factor, away_factor)
    n = rem.shape[0]
    size = n + max(home_goals, away_goals)
    final = np.zeros((size, size))
    # rem[i, j] 代表剩餘再進 i 主 j 客 -> 最終 (home_goals+i, away_goals+j)
    final[home_goals:home_goals + n, away_goals:away_goals + n] = rem
    return final
