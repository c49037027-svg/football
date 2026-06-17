"""產生「預測站」風格的每場比賽預測內容。

像 Forebet / PredictZ / soccermaddy 那類網站，每場比賽會呈現：
  - 1X2 勝/平/負機率（%）
  - 預測比分（最可能比分）與「正確比分」機率排行
  - 大小球 Over/Under（1.5 / 2.5 / 3.5）機率
  - 雙方都進球 BTTS 機率
  - 雙方預期進球
  - 近期戰績（最近 N 場 W/D/L）與對戰紀錄 H2H
  - 一句推薦 Tip

本模組由 Dixon–Coles 的比分機率矩陣推導以上全部，並可選擇性地把賽前情境
（傷停/輪休，見 context）疊加到預期進球上。輸出為結構化的 MatchPrediction，
再交給 report 模組渲染成 console / Markdown / HTML。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .context import ContextAdjustment
from .data import schema as S
from .models import markets
from .models.dixon_coles import DixonColesModel


@dataclass
class MatchPrediction:
    home: str
    away: str
    exp_home_goals: float
    exp_away_goals: float
    p_home: float
    p_draw: float
    p_away: float
    predicted_score: tuple[int, int]
    correct_scores: list[tuple[tuple[int, int], float]]
    over_under: dict[float, dict[str, float]]   # line -> {over, under, push}
    btts_yes: float
    btts_no: float
    home_form: str = ""        # 例如 "WWDLW"（最近→最舊）
    away_form: str = ""
    h2h: str = ""              # 簡短對戰摘要
    tip: str = ""              # 推薦下注方向（純預測，不含 odds）
    confidence: str = ""       # 信心度標籤

    @property
    def fav(self) -> str:
        if self.p_home >= self.p_draw and self.p_home >= self.p_away:
            return "home"
        if self.p_away >= self.p_draw and self.p_away >= self.p_home:
            return "away"
        return "draw"


def _form_string(df: pd.DataFrame, team: str, before_date, n: int = 5) -> str:
    """最近 n 場（該日期前）的 W/D/L，最近在前。"""
    past = df[(df[S.DATE] < before_date)
              & ((df[S.HOME] == team) | (df[S.AWAY] == team))]
    past = past.sort_values(S.DATE).tail(n)
    out = []
    for _, r in past.iterrows():
        hg, ag = r[S.HOME_GOALS], r[S.AWAY_GOALS]
        if r[S.HOME] == team:
            out.append("W" if hg > ag else "L" if hg < ag else "D")
        else:
            out.append("W" if ag > hg else "L" if ag < hg else "D")
    return "".join(reversed(out))  # 最近在前


def _h2h_string(df: pd.DataFrame, home: str, away: str, before_date, n: int = 5) -> str:
    """兩隊近 n 次交手摘要：'主隊勝-平-客隊勝'。"""
    mask = (df[S.DATE] < before_date) & (
        ((df[S.HOME] == home) & (df[S.AWAY] == away))
        | ((df[S.HOME] == away) & (df[S.AWAY] == home)))
    past = df[mask].sort_values(S.DATE).tail(n)
    hw = dw = aw = 0
    for _, r in past.iterrows():
        hg, ag = r[S.HOME_GOALS], r[S.AWAY_GOALS]
        winner = home if hg > ag else away if hg < ag else None
        if winner == home:
            hw += 1
        elif winner == away:
            aw += 1
        else:
            dw += 1
    if hw + dw + aw == 0:
        return "無近期交手紀錄"
    return f"近{hw+dw+aw}次：{home} {hw}勝 / {dw}平 / {away} {aw}勝"


def _make_tip(pred: MatchPrediction) -> tuple[str, str]:
    """由機率產生一句推薦與信心度（純預測導向，非投注建議）。"""
    probs = {"主勝": pred.p_home, "和局": pred.p_draw, "客勝": pred.p_away}
    pick, p = max(probs.items(), key=lambda x: x[1])
    conf = "高" if p >= 0.55 else "中" if p >= 0.42 else "低"
    ou25 = pred.over_under.get(2.5, {})
    ou_side = "大2.5" if ou25.get("over", 0) >= 0.5 else "小2.5"
    btts = "雙方進球" if pred.btts_yes >= 0.5 else "非雙方進球"
    tip = f"{pick}（{p:.0%}）、{ou_side}、{btts}"
    return tip, conf


def predict_match(model: DixonColesModel, home: str, away: str,
                  history: pd.DataFrame | None = None,
                  adjustment: ContextAdjustment | None = None,
                  ou_lines: tuple[float, ...] = (1.5, 2.5, 3.5),
                  before_date=None) -> MatchPrediction:
    lam, mu = model.expected_goals(home, away)
    if adjustment is not None:
        lam, mu = adjustment.apply(lam, mu)
    mat = model.score_matrix(home, away, lam=lam, mu=mu)

    o = markets.outcome_1x2(mat)
    eh, ea = markets.expected_goals_from_matrix(mat)
    ou = {}
    for line in ou_lines:
        d = markets.over_under(mat, line)
        ou[line] = {"over": d["over_win"], "under": d["under_win"],
                    "push": d["over_push"]}
    bt = markets.btts(mat)

    pred = MatchPrediction(
        home=home, away=away,
        exp_home_goals=round(eh, 2), exp_away_goals=round(ea, 2),
        p_home=o["home"], p_draw=o["draw"], p_away=o["away"],
        predicted_score=markets.most_likely_score(mat),
        correct_scores=markets.correct_score(mat, top_n=5),
        over_under=ou, btts_yes=bt["yes"], btts_no=bt["no"],
    )

    if history is not None:
        bd = before_date if before_date is not None else history[S.DATE].max() + pd.Timedelta(days=1)
        pred.home_form = _form_string(history, home, bd)
        pred.away_form = _form_string(history, away, bd)
        pred.h2h = _h2h_string(history, home, away, bd)

    pred.tip, pred.confidence = _make_tip(pred)
    return pred


def predict_fixtures(model: DixonColesModel, fixtures: pd.DataFrame,
                     history: pd.DataFrame | None = None,
                     adjustments: dict | None = None) -> list[MatchPrediction]:
    """對一批賽程產生預測。fixtures 需含 home, away 欄位。"""
    adjustments = adjustments or {}
    out: list[MatchPrediction] = []
    for _, row in fixtures.iterrows():
        home, away = str(row["home"]), str(row["away"])
        if home not in model.attack or away not in model.attack:
            print(f"[skip] 模型未包含：{home} 或 {away}")
            continue
        out.append(predict_match(
            model, home, away, history=history,
            adjustment=adjustments.get((home, away))))
    return out
