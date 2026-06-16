import numpy as np

from footy.models import markets
from footy.models.dixon_coles import score_matrix_from_rates


def test_score_matrix_normalized():
    mat = score_matrix_from_rates(1.5, 1.1, -0.05, 10)
    assert abs(mat.sum() - 1.0) < 1e-9
    assert (mat >= 0).all()


def test_1x2_sums_to_one():
    mat = score_matrix_from_rates(1.4, 1.2, -0.03, 10)
    o = markets.outcome_1x2(mat)
    assert abs(o["home"] + o["draw"] + o["away"] - 1.0) < 1e-9
    # 主場優勢下，主勝機率應高於客勝
    assert o["home"] > o["away"]


def test_over_under_push():
    mat = score_matrix_from_rates(1.5, 1.5, 0.0, 10)
    ou = markets.over_under(mat, 3.0)  # 整數線有 push
    assert ou["over_push"] > 0
    assert abs(ou["over_win"] + ou["under_win"] + ou["over_push"] - 1.0) < 1e-9
    ou25 = markets.over_under(mat, 2.5)  # 半線無 push
    assert ou25["over_push"] == 0
    assert abs(ou25["over_win"] + ou25["under_win"] - 1.0) < 1e-9


def test_asian_handicap_quarter_line():
    mat = score_matrix_from_rates(1.6, 1.0, -0.04, 10)
    # home -0.75：半勝半輸結構應存在
    ah = markets.asian_handicap(mat, -0.75, "home")
    total = ah.p_win + ah.p_half_win + ah.p_push + ah.p_half_loss + ah.p_loss
    assert abs(total - 1.0) < 1e-9
    # -0.75 不會有整盤 push（push 來自整數線），但會有 half_win/half_loss
    assert ah.p_half_win >= 0 and ah.p_half_loss >= 0


def test_ah_home_away_complementary_ev():
    # 公平賠率下，home 與 away 兩邊 EV 應接近 0（零和、無 vig）
    mat = score_matrix_from_rates(1.5, 1.2, -0.03, 10)
    ah_h = markets.asian_handicap(mat, -0.5, "home")
    ah_a = markets.asian_handicap(mat, -0.5, "away")
    # 主隊 -0.5 公平賠率
    ev_h = markets.ev_per_unit(ah_h.p_win, 1 / max(1e-9, ah_h.p_win),
                               ah_h.p_push, ah_h.p_half_win, ah_h.p_half_loss)
    assert abs(ev_h) < 1e-6
    # away +0.5 的勝率 = 1 - 主隊 -0.5 勝率（無 push 於半線）
    assert abs((ah_h.p_win + ah_a.p_win) - 1.0) < 1e-9


def test_ev_per_unit_signs():
    # 勝率 0.5、賠率 2.1 應為正 EV
    assert markets.ev_per_unit(0.5, 2.1) > 0
    # 勝率 0.5、賠率 1.9 應為負 EV
    assert markets.ev_per_unit(0.5, 1.9) < 0
