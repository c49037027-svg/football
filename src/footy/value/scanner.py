"""盤口掃描器：給定『比分機率矩陣 + 一組盤口報價』，逐一評估價值。

初盤（pre-match）與走地（in-play）共用這支：
  - 初盤：用 model.score_matrix(home, away)
  - 走地：用 inplay.final_score_matrix(...)
兩者都產出一個 (size x size) 的最終比分機率矩陣，掃描邏輯完全一致。
"""
from __future__ import annotations

import numpy as np

from ..config import ValueConfig
from ..models import markets
from .edge import ValueBet, evaluate
from .odds import remove_vig_proportional


def _quote_key(q) -> tuple:
    return (q.market, q.selection, q.line)


def scan_market(mat: np.ndarray, quotes: list, cfg: ValueConfig,
                extra_edge: float = 0.0) -> list[ValueBet]:
    """評估一組盤口報價，回傳所有評估結果（含非 value 的，方便記錄）。

    quotes：list[MarketQuote]（feed.MarketQuote 或等價物件，需有 market/selection/odds/line）。
    """
    results: list[ValueBet] = []

    # 為了算 edge，盡量對「同一市場的對立面」一起去 vig 得到公平機率。
    by_group: dict[tuple, list] = {}
    for q in quotes:
        key = (q.market, q.line)
        by_group.setdefault(key, []).append(q)

    for (market, line), group in by_group.items():
        fair_probs = None
        if len(group) >= 2:
            try:
                fair_probs = remove_vig_proportional([g.odds for g in group])
            except Exception:  # noqa: BLE001
                fair_probs = None

        for i, q in enumerate(group):
            fair_prob = fair_probs[i] if fair_probs is not None else None
            probs = _model_probs(mat, q)
            if probs is None:
                continue
            vb = evaluate(
                market=f"{market}{'' if line is None else line}",
                selection=q.selection, odds=q.odds,
                prob_win=probs["win"], prob_push=probs.get("push", 0.0),
                prob_half_win=probs.get("half_win", 0.0),
                prob_half_loss=probs.get("half_loss", 0.0),
                cfg=cfg, fair_prob=fair_prob, extra_edge=extra_edge,
            )
            results.append(vb)
    return results


def value_bets(mat: np.ndarray, quotes: list, cfg: ValueConfig,
               extra_edge: float = 0.0) -> list[ValueBet]:
    """只回傳判定為 value 的下注，依 EV 由高到低排序。"""
    res = [r for r in scan_market(mat, quotes, cfg, extra_edge) if r.is_value]
    return sorted(res, key=lambda v: v.ev, reverse=True)


def _model_probs(mat: np.ndarray, q) -> dict | None:
    """把某個盤口選項對應到模型的逐結果機率拆解。"""
    m = q.market.upper()
    if m == "1X2":
        o = markets.outcome_1x2(mat)
        key = {"home": "home", "draw": "draw", "away": "away"}.get(q.selection)
        if key is None:
            return None
        return {"win": o[key]}
    if m in ("OU", "O/U", "TOTAL"):
        if q.line is None:
            return None
        ou = markets.over_under(mat, q.line)
        if q.selection == "over":
            return {"win": ou["over_win"], "push": ou["over_push"]}
        if q.selection == "under":
            return {"win": ou["under_win"], "push": ou["under_push"]}
        return None
    if m in ("AH", "HANDICAP", "ASIAN"):
        if q.line is None:
            return None
        # line 一律以 home 視角；下注 away 時 side 切換、handicap 取反由 markets 處理。
        side = q.selection  # "home" / "away"
        hcap = q.line
        ah = markets.asian_handicap(mat, hcap, side)
        return {"win": ah.p_win, "push": ah.p_push,
                "half_win": ah.p_half_win, "half_loss": ah.p_half_loss}
    return None
