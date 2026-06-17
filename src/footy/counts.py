"""角球（corners）與黃牌（cards）的 Poisson 計數模型。

⚠️ 誠實說明：國際賽公開資料**沒有**角球/黃牌統計，所以這裡用
「可調先驗 + 依雙方實力/強弱傾斜」的 Poisson 模型，**不是**從資料擬合的。
俱樂部資料（football-data.co.uk）有角球/黃牌欄位，可用 `fit_count_rates` 校準
聯賽級別的基準，但跨到國際賽仍屬近似。請把這兩個面板當作「參考」而非精算。

模型：
  - 總數 ~ Poisson(λ_total)；主客分配依實力差（強隊通常拿較多角球）。
  - 黃牌總數 ~ Poisson(λ_cards)，可依賽事重要性（淘汰賽更兇）加成。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass
class CountPriors:
    corners_total: float = 10.5     # 一場總角球先驗（國際賽略低於頂級聯賽）
    cards_total: float = 4.0        # 一場總黃牌先驗
    corner_tilt: float = 2.5        # 實力差→角球分配傾斜強度
    knockout_card_boost: float = 1.15  # 淘汰賽黃牌加成


@dataclass
class CountEstimate:
    home: float
    away: float
    total: float
    line: float                 # 建議盤口線（最接近的 .5）
    over_prob: float            # P(實際 > line)
    recommend: str              # "買大" / "買小"
    confidence: float           # = max(over_prob, 1-over_prob)
    edge_vs_line: float         # 點估計 − line（顯示用，正=偏大）


def _split_by_strength(total: float, strength_diff: float, tilt: float) -> tuple[float, float]:
    """把總數依實力差分成主/客。strength_diff>0 代表主隊較強，分到較多。"""
    share_home = 1.0 / (1.0 + np.exp(-tilt * strength_diff))
    return total * share_home, total * (1.0 - share_home)


def _line_and_reco(total: float, lam_total: float) -> tuple[float, float, str, float, float]:
    """由期望總數與 Poisson 率，給最接近的 .5 線、P(over)、建議與信心。"""
    line = np.floor(total) + 0.5  # 最接近且在點估計附近的半線
    # P(total > line) = P(count >= ceil(line)) = 1 - cdf(floor(line))
    over_prob = float(1.0 - poisson.cdf(np.floor(line), lam_total))
    recommend = "買大" if over_prob >= 0.5 else "買小"
    confidence = max(over_prob, 1.0 - over_prob)
    edge = total - line
    return line, over_prob, recommend, confidence, edge


def estimate_corners(home_strength: float, away_strength: float,
                     priors: CountPriors | None = None) -> CountEstimate:
    """home/away_strength：球隊整體強度（如 attack-defence 或 Elo 標準化值）。"""
    p = priors or CountPriors()
    diff = home_strength - away_strength
    # 強隊壓著打，總角球略增
    lam_total = p.corners_total * (1.0 + 0.05 * abs(diff))
    h, a = _split_by_strength(lam_total, diff, p.corner_tilt)
    line, over, reco, conf, edge = _line_and_reco(lam_total, lam_total)
    return CountEstimate(home=round(h, 1), away=round(a, 1), total=round(lam_total, 1),
                         line=line, over_prob=over, recommend=reco,
                         confidence=conf, edge_vs_line=round(edge, 1))


def estimate_cards(home_strength: float, away_strength: float,
                   knockout: bool = False,
                   priors: CountPriors | None = None) -> CountEstimate:
    p = priors or CountPriors()
    lam_total = p.cards_total * (p.knockout_card_boost if knockout else 1.0)
    # 勢均力敵的比賽通常更兇（黃牌略增）
    closeness = 1.0 + 0.10 * np.exp(-abs(home_strength - away_strength))
    lam_total *= closeness
    # 黃牌主客大致對半，弱隊常犯規略多
    diff = away_strength - home_strength
    h, a = _split_by_strength(lam_total, diff, 0.8)
    line, over, reco, conf, edge = _line_and_reco(lam_total, lam_total)
    return CountEstimate(home=round(h, 1), away=round(a, 1), total=round(lam_total, 1),
                         line=line, over_prob=over, recommend=reco,
                         confidence=conf, edge_vs_line=round(edge, 1))


def fit_count_rates(df, home_col: str, away_col: str) -> CountPriors:
    """選用：從含角球/黃牌欄位的資料估「平均總數」先驗（聯賽級別校準）。"""
    import pandas as pd
    h = pd.to_numeric(df[home_col], errors="coerce")
    a = pd.to_numeric(df[away_col], errors="coerce")
    total = (h + a).dropna()
    pr = CountPriors()
    if len(total):
        pr.corners_total = float(total.mean())
    return pr
