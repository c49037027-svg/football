"""陣型與先發陣容調整測試。"""
from footy import context
from footy.models import dixon_coles as dc


def test_formation_factor_attacking_vs_defensive():
    assert context.formation_factor("4-3-3") > 1.0      # 進攻
    assert context.formation_factor("5-4-1") < 1.0      # 防守
    assert context.formation_factor("4-4-2") == 1.0     # 均衡
    assert context.formation_factor(None) == 1.0        # 未知/未填


def test_formation_adjustment_opponent_bus():
    # 對方擺大巴（5-4-1）會再壓低我方進攻率
    adj = context.formation_adjustment("4-3-3", "5-4-1")
    assert adj.home_attack_mult < context.formation_factor("4-3-3")  # 被大巴折扣
    assert adj.away_attack_mult == context.formation_factor("5-4-1")


def test_combine_adjustments_multiplies():
    a1 = context.ContextAdjustment(0.9, 1.1)
    a2 = context.ContextAdjustment(0.8, 1.0)
    c = context.combine_adjustments(a1, a2, None)
    assert abs(c.home_attack_mult - 0.72) < 1e-9
    assert abs(c.away_attack_mult - 1.1) < 1e-9


def test_parse_lineups():
    payload = {"response": [
        {"team": {"name": "Brazil"}, "formation": "4-2-3-1",
         "startXI": [{"player": {"id": 1, "name": "A"}},
                     {"player": {"id": 2, "name": "B"}}]},
        {"team": {"name": "France"}, "formation": "4-3-3",
         "startXI": [{"player": {"id": 9, "name": "X"}}]},
    ]}
    lu = context.parse_lineups(payload)
    assert lu["Brazil"]["formation"] == "4-2-3-1"
    assert lu["Brazil"]["start_ids"] == [1, 2]
    assert lu["France"]["starters"] == ["X"]


def test_lineup_strength_missing_key_players():
    baseline = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 缺 3 名先發（用替補替換）→ 乘數 < 1
    actual = [1, 2, 3, 4, 5, 6, 7, 8, 21, 22, 23]
    f = context.lineup_strength_adjustment(actual, baseline)
    assert f < 1.0
    # 完整原班 → 1.0
    assert context.lineup_strength_adjustment(baseline, baseline) == 1.0


def test_lineup_strength_weighted_by_rating():
    baseline = list(range(1, 12))  # 11 名先發
    rating = {i: 5.0 for i in baseline}
    rating[1] = 12.0  # 1 號是關鍵球員（評分遠高）
    # 缺關鍵球員 1 號 vs 缺一般球員 2 號（各用替補 99 替換）
    miss_key = context.lineup_strength_adjustment(
        [x for x in baseline if x != 1] + [99], baseline, rating)
    miss_reg = context.lineup_strength_adjustment(
        [x for x in baseline if x != 2] + [99], baseline, rating)
    assert miss_key < miss_reg  # 缺關鍵球員影響更大


def test_formation_changes_expected_goals(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    lam0, mu0 = model.expected_goals(h, a)
    adj = context.formation_adjustment("4-3-3", "4-4-2")
    lam1, _ = adj.apply(lam0, mu0)
    assert lam1 > lam0  # 主隊用進攻陣型，預期進球升
