"""價值偵測：比較模型機率與盤口，找出 +EV 的下注。

對單一可下注選項，給定：
  - 模型估計的真實機率（或對亞盤/含 push 盤口的逐結果機率拆解）
  - 莊家提供的小數賠率
計算：
  - EV（每單位本金的期望淨報酬）
  - edge（模型機率 / 公平機率 - 1，相對於去 vig 後的市場機率）
並依設定的門檻判斷是否為 value bet。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ValueConfig
from ..models import markets


@dataclass
class ValueBet:
    market: str          # 例如 "1X2:home" / "OU2.5:over" / "AH-0.5:home"
    selection: str
    odds: float
    model_prob: float    # 模型給的「等效勝率」（對含退款盤口為 EV 反推的等效值，僅供顯示）
    fair_prob: float     # 去 vig 後的市場公平機率（若可得）
    ev: float            # 每單位本金期望淨報酬
    edge: float          # 相對市場公平機率的優勢
    is_value: bool


def _equivalent_winprob_from_ev(ev: float, odds: float) -> float:
    """由 EV 反推「等效勝率」p，使 p*(odds-1) - (1-p) = ev。僅供顯示直覺。"""
    if odds <= 1:
        return 0.0
    return (ev + 1.0) / odds


def evaluate(market: str, selection: str, odds: float,
             prob_win: float, cfg: ValueConfig,
             prob_push: float = 0.0, prob_half_win: float = 0.0,
             prob_half_loss: float = 0.0, fair_prob: float | None = None,
             extra_edge: float = 0.0) -> ValueBet:
    """評估單一選項是否為 value bet。

    prob_* 為模型對該選項的逐結果機率（亞盤/含 push 用得到）。
    fair_prob：若已由整個市場去 vig 得到該選項的公平機率，可傳入做 edge 比較；
              否則用 1/odds（含 vig）保守估計，edge 會偏低（更嚴格）。
    extra_edge：走地時可加的額外安全邊際。
    """
    ev = markets.ev_per_unit(prob_win, odds, prob_push, prob_half_win, prob_half_loss)
    eq_p = _equivalent_winprob_from_ev(ev, odds)

    if fair_prob is None:
        fair_prob = 1.0 / odds  # 含 vig，保守
    edge = (eq_p / fair_prob - 1.0) if fair_prob > 0 else 0.0

    min_edge = cfg.min_edge + extra_edge
    is_value = (
        ev >= cfg.min_ev
        and edge >= min_edge
        and cfg.min_odds <= odds <= cfg.max_odds
    )
    return ValueBet(
        market=market, selection=selection, odds=odds,
        model_prob=eq_p, fair_prob=fair_prob, ev=ev, edge=edge, is_value=is_value,
    )
