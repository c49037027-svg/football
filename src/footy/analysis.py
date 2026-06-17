"""世界盃單場深度分析：整合 Poisson（解析）與蒙地卡羅，產出全部面板。

涵蓋（對應 UI）：
  - AI 預測比分、總進球、xG 區間
  - 大小球 1.5 / 2.5 / 3.5（買大/買小）
  - 兩隊都進球 BTTS（含各自進球機率）
  - 亞盤讓球（模型估線、xG 差、建議）
  - 角球預測（合計、估線、建議、信心）
  - 黃牌預測（合計、估線、建議、信心）
  - 上半場走向（1X2 + 上半場大小 0.5 / 1.5）
  - 影響因子：FIFA/Elo 評分、近5場、歷史交手、球員狀態、戰術對比

方法：
  - 單場各市場的「機率」用解析比分矩陣（精確）。
  - 上半場、聯合分布、信心等用蒙地卡羅抽樣（n_sims），因為涉及把進球分配到上下半場
    與多面板的一致抽樣。角球/黃牌用 Poisson 計數模型（見 counts，屬先驗近似）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import counts
from .context import ContextAdjustment
from .data import schema as S
from .models import markets
from .models.dixon_coles import DixonColesModel
from .predict import _form_string, _h2h_string


@dataclass
class MatchAnalysis:
    home: str
    away: str
    neutral: bool
    # 進球
    exp_home_goals: float
    exp_away_goals: float
    predicted_score: tuple[int, int]
    predicted_score_prob: float
    total_goals: float
    xg_low: float
    xg_high: float
    # 1X2
    p_home: float
    p_draw: float
    p_away: float
    # 大小球：line -> {over, under}
    over_under: dict[float, dict[str, float]]
    # BTTS
    btts_yes: float
    p_home_scores: float
    p_away_scores: float
    # 亞盤
    ah_fav: str
    ah_line: float
    ah_supremacy: float
    ah_cover_prob: float
    ah_reco: str
    # 角球 / 黃牌
    corners: counts.CountEstimate
    cards: counts.CountEstimate
    # 上半場
    fh_home: float
    fh_draw: float
    fh_away: float
    fh_over: dict[float, float]   # line -> P(over)
    # 影響因子
    elo_home: float = 0.0
    elo_away: float = 0.0
    home_form: str = ""
    away_form: str = ""
    h2h: str = ""
    home_style: str = ""
    away_style: str = ""
    player_note_home: str = "資料不足（中性）"
    player_note_away: str = "資料不足（中性）"
    n_sims: int = 0


def _ah_model_line(lam: float, mu: float, mat: np.ndarray) -> tuple[str, float, float, float, str]:
    """由讓步（supremacy=lam-mu）給亞盤建議。

    回傳 (受讓方視角的熱門隊, 讓球線(熱門 -X), supremacy, 熱門covers機率, 建議文字)。
    """
    supremacy = lam - mu
    fav_is_home = supremacy >= 0
    sup_abs = abs(supremacy)
    line_q = round(sup_abs * 4) / 4.0  # 最接近的 0.25
    # 用比熱門 supremacy 略低的線當建議（保守），至少 0.25
    bet_line = max(0.25, line_q)
    side = "home" if fav_is_home else "away"
    # home 視角讓球：熱門讓球為負
    handicap_home_view = -bet_line if fav_is_home else bet_line
    ah = markets.asian_handicap(mat, handicap_home_view, side)
    cover = ah.p_win + ah.p_half_win + 0.5 * ah.p_push
    fav = "home" if fav_is_home else "away"
    reco = f"買{'主' if fav_is_home else '客'}隊 讓{bet_line}"
    return fav, bet_line, supremacy, float(cover), reco


def _style(attack: float, defence: float) -> str:
    """由攻防參數給一句風格描述（相對全聯盟平均，已中心化）。"""
    if attack > 0.15 and defence > 0.15:
        return "攻守兼備"
    if attack > 0.15:
        return "攻擊型"
    if defence > 0.15:
        return "防守反擊"
    if attack < -0.15 and defence < -0.15:
        return "整體偏弱"
    return "均衡"


def analyze(model: DixonColesModel, home: str, away: str,
            history: pd.DataFrame | None = None, neutral: bool = True,
            knockout: bool = False, n_sims: int = 50000,
            first_half_fraction: float = 0.45,
            adjustment: ContextAdjustment | None = None,
            priors: counts.CountPriors | None = None,
            seed: int | None = 42) -> MatchAnalysis:
    lam, mu = model.expected_goals(home, away, neutral=neutral)
    if adjustment is not None:
        lam, mu = adjustment.apply(lam, mu)
    mat = model.score_matrix(home, away, lam=lam, mu=mu)

    # ---- 解析面板 ----
    o = markets.outcome_1x2(mat)
    eh, ea = markets.expected_goals_from_matrix(mat)
    ou = {}
    for line in (1.5, 2.5, 3.5):
        d = markets.over_under(mat, line)
        ou[line] = {"over": d["over_win"], "under": d["under_win"]}
    bt = markets.btts(mat)
    p_home_scores = float(1.0 - mat[0, :].sum())
    p_away_scores = float(1.0 - mat[:, 0].sum())

    fav, bet_line, supremacy, cover, ah_reco = _ah_model_line(lam, mu, mat)

    # ---- 角球 / 黃牌（Poisson 先驗）----
    sh = model.attack.get(home, 0.0) - model.defence.get(home, 0.0)
    sa = model.attack.get(away, 0.0) - model.defence.get(away, 0.0)
    corners = counts.estimate_corners(sh, sa, priors)
    cards = counts.estimate_cards(sh, sa, knockout=knockout, priors=priors)

    # ---- 蒙地卡羅：上半場與一致抽樣 ----
    rng = np.random.default_rng(seed)
    flat = mat.ravel() / mat.sum()
    ncol = mat.shape[1]
    draws = rng.choice(flat.size, size=n_sims, p=flat)
    g_home = draws // ncol
    g_away = draws % ncol
    fh_home_goals = rng.binomial(g_home, first_half_fraction)
    fh_away_goals = rng.binomial(g_away, first_half_fraction)
    fh_diff = fh_home_goals - fh_away_goals
    fh_home_p = float((fh_diff > 0).mean())
    fh_away_p = float((fh_diff < 0).mean())
    fh_draw_p = float((fh_diff == 0).mean())
    fh_total = fh_home_goals + fh_away_goals
    fh_over = {0.5: float((fh_total > 0.5).mean()),
               1.5: float((fh_total > 1.5).mean())}

    ml = markets.most_likely_score(mat)
    ml_prob = float(mat[ml[0], ml[1]])
    total = eh + ea

    a = MatchAnalysis(
        home=home, away=away, neutral=neutral,
        exp_home_goals=round(eh, 2), exp_away_goals=round(ea, 2),
        predicted_score=ml, predicted_score_prob=ml_prob, total_goals=round(total, 1),
        xg_low=round(min(eh, ea), 1), xg_high=round(max(eh, ea), 1),
        p_home=o["home"], p_draw=o["draw"], p_away=o["away"],
        over_under=ou, btts_yes=bt["yes"],
        p_home_scores=p_home_scores, p_away_scores=p_away_scores,
        ah_fav=fav, ah_line=bet_line, ah_supremacy=round(supremacy, 2),
        ah_cover_prob=cover, ah_reco=ah_reco,
        corners=corners, cards=cards,
        fh_home=fh_home_p, fh_draw=fh_draw_p, fh_away=fh_away_p, fh_over=fh_over,
        elo_home=round(model.team_elo.get(home, 0.0), 0) if model.team_elo else 0.0,
        elo_away=round(model.team_elo.get(away, 0.0), 0) if model.team_elo else 0.0,
        home_style=_style(model.attack.get(home, 0), model.defence.get(home, 0)),
        away_style=_style(model.attack.get(away, 0), model.defence.get(away, 0)),
        n_sims=n_sims,
    )
    if history is not None:
        bd = history[S.DATE].max() + pd.Timedelta(days=1)
        a.home_form = _form_string(history, home, bd)
        a.away_form = _form_string(history, away, bd)
        a.h2h = _h2h_string(history, home, away, bd)
    return a
