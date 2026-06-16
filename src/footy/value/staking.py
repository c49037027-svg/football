"""凱利下注（Kelly criterion）與分數凱利。

全凱利下注比例（單一結果、小數賠率 odds、勝率 p）：
    f* = (b * p - q) / b
其中 b = odds - 1，q = 1 - p。
全凱利長期成長最快，但波動極大、且對機率估計誤差敏感，
所以實務上用「分數凱利」（如 0.25 倍）並加硬上限。

對含退款（半勝/半輸/push）的亞盤，凱利沒有簡潔閉式解，
這裡用「以 EV 為基礎的近似」：先把含退款情形折算成等效 (p, b)，
再套標準凱利。折算方式：保留真實 EV 與賠率，反推等效勝率。
"""
from __future__ import annotations

from ..config import StakingConfig


def kelly_fraction(prob: float, odds: float) -> float:
    """標準凱利比例（可能為負，代表不應下注）。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    return (b * prob - q) / b


def kelly_from_ev(ev: float, odds: float) -> float:
    """由 EV 與賠率反推等效勝率後計算凱利（給含退款盤口用）。"""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    # ev = p*b - (1-p)  =>  p = (ev + 1) / (b + 1) = (ev + 1) / odds
    p = (ev + 1.0) / odds
    p = min(max(p, 0.0), 1.0)
    return kelly_fraction(p, odds)


def stake(bankroll: float, ev: float, odds: float, cfg: StakingConfig) -> float:
    """回傳建議下注金額（已套用分數凱利與上下限）。

    回傳 0 代表不下注（凱利為負，即無正期望）。
    """
    f_full = kelly_from_ev(ev, odds)
    if f_full <= 0:
        return 0.0
    f = f_full * cfg.kelly_fraction
    f = min(f, cfg.max_stake_fraction)  # 硬上限
    amount = bankroll * f
    if amount < cfg.min_stake:
        return 0.0
    return round(amount, 2)
