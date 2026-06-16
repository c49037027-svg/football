"""初盤（pre-match）價值掃描。

讀入一份「即將開賽」的盤口 CSV，對每場用模型掃描各市場價值並輸出建議。
CSV 欄位（彈性，缺的市場略過）：
  home, away,
  odds_home, odds_draw, odds_away,           # 1X2
  ou_line, odds_over, odds_under,            # 大小球（單一條線）
  ah_line, odds_ah_home, odds_ah_away        # 亞盤（home 視角的 line）
"""
from __future__ import annotations

import pandas as pd

from .config import Config
from .live.feed import MarketQuote
from .models.dixon_coles import DixonColesModel
from .risk.manager import RiskManager
from .value import scanner, staking


def _row_to_quotes(row) -> list[MarketQuote]:
    q: list[MarketQuote] = []

    def has(col):
        return col in row and pd.notna(row[col])

    if has("odds_home") and has("odds_draw") and has("odds_away"):
        q.append(MarketQuote("1X2", "home", float(row["odds_home"])))
        q.append(MarketQuote("1X2", "draw", float(row["odds_draw"])))
        q.append(MarketQuote("1X2", "away", float(row["odds_away"])))
    if has("ou_line") and has("odds_over") and has("odds_under"):
        line = float(row["ou_line"])
        q.append(MarketQuote("OU", "over", float(row["odds_over"]), line=line))
        q.append(MarketQuote("OU", "under", float(row["odds_under"]), line=line))
    if has("ah_line") and has("odds_ah_home") and has("odds_ah_away"):
        line = float(row["ah_line"])
        q.append(MarketQuote("AH", "home", float(row["odds_ah_home"]), line=line))
        q.append(MarketQuote("AH", "away", float(row["odds_ah_away"]), line=line))
    return q


def scan(model: DixonColesModel, fixtures: pd.DataFrame, cfg: Config,
         risk: RiskManager | None = None,
         adjustments: dict | None = None) -> list[dict]:
    """回傳所有 value 下注建議（dict 清單，方便印表/匯出）。

    adjustments：選用的 {(home, away): ContextAdjustment}，
    用來把傷停/輪休等情境疊加到模型的預期進球上（見 context 模組）。
    """
    risk = risk or RiskManager(cfg.risk)
    adjustments = adjustments or {}
    out: list[dict] = []
    for _, row in fixtures.iterrows():
        home, away = str(row["home"]), str(row["away"])
        if home not in model.attack or away not in model.attack:
            print(f"[skip] 模型未包含：{home} 或 {away}")
            continue
        lam, mu = model.expected_goals(home, away)
        adj = adjustments.get((home, away))
        if adj is not None:
            lam, mu = adj.apply(lam, mu)
        mat = model.score_matrix(home, away, lam=lam, mu=mu)
        quotes = _row_to_quotes(row)
        for vb in scanner.value_bets(mat, quotes, cfg.value):
            req = staking.stake(risk.bankroll, vb.ev, vb.odds, cfg.staking)
            decision = risk.approve(f"{home}-{away}", f"{vb.market}:{vb.selection}",
                                    req, vb.odds)
            out.append({
                "match": f"{home} v {away}",
                "market": vb.market,
                "selection": vb.selection,
                "odds": round(vb.odds, 2),
                "model_prob": round(vb.model_prob, 4),
                "fair_prob": round(vb.fair_prob, 4),
                "edge": round(vb.edge, 4),
                "ev": round(vb.ev, 4),
                "suggested_stake": decision.stake if decision.approved else 0.0,
                "risk_note": decision.reason if decision.approved else f"否決：{decision.reason}",
            })
    return out
