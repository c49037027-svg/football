"""整屆世界盃模擬與網站渲染測試（用 examples/wc2026.json）。"""
from pathlib import Path

from footy import report, worldcup
from footy.models import dixon_coles as dc

SCHED = Path(__file__).resolve().parents[1] / "examples" / "wc2026.json"


def test_parse_wc_json():
    groups, matches, num_map = worldcup.parse_wc_json(SCHED)
    assert len(groups) == 12
    assert sum(len(v) for v in groups.values()) == 48
    # 場次編號唯一且涵蓋所有比賽
    nums = [m.num for m in matches]
    assert len(set(nums)) == len(matches)
    # 別名生效：USA → United States
    all_teams = {t for ts in groups.values() for t in ts}
    assert "United States" in all_teams and "USA" not in all_teams


def test_alias_only_on_group_names():
    _, matches, _ = worldcup.parse_wc_json(SCHED)
    ko = [m for m in matches if m.round == "Round of 32"]
    # 淘汰賽槽位碼不應被別名破壞
    assert any(m.team1.startswith(("1", "2", "3", "W")) for m in ko)


def test_bipartite_assign_perfect_matching():
    # 8 個第三名組別，8 個槽位（各允許一組合）
    thirds = list("ABCDEFGH")
    slots = [["A", "B", "C"], ["B", "C", "D"], ["C", "D", "E"], ["D", "E", "F"],
             ["E", "F", "G"], ["F", "G", "H"], ["A", "G", "H"], ["A", "B", "H"]]
    assign = worldcup._bipartite_assign(thirds, slots)
    # 每槽都配到、且不重複
    assert len(assign) == 8
    assert len(set(assign.values())) == 8


def test_simulate_worldcup_probabilities():
    # 用一個能涵蓋所有 48 隊的玩具模型：每隊強度=0，確保不缺隊
    groups, matches, _ = worldcup.parse_wc_json(SCHED)
    teams = sorted({t for ts in groups.values() for t in ts})
    model = _toy_model(teams)
    res = worldcup.simulate_worldcup(model, SCHED, n_sims=400, seed=1)
    # 冠軍機率總和 ≈ 1
    assert abs(sum(res.champion.values()) - 1.0) < 1e-9
    # 每隊晉級機率 >= 各更深輪次機率（單調性）
    for t in teams:
        assert res.qualify[t] + 1e-9 >= res.r16[t] >= res.quarter[t] - 1e-9
        assert res.semi[t] + 1e-9 >= res.final[t] >= res.champion[t] - 1e-9
    # 大約每組有 ~2.67 隊晉級（前2+8/12），全部加總應為 32
    assert abs(sum(res.qualify.values()) - 32) < 0.5


def test_worldcup_render():
    groups, matches, _ = worldcup.parse_wc_json(SCHED)
    teams = sorted({t for ts in groups.values() for t in ts})
    model = _toy_model(teams)
    res = worldcup.simulate_worldcup(model, SCHED, n_sims=200, seed=2)
    html = report.render_worldcup_html(res, model, matches, "測試世界盃")
    assert "<!doctype html>" in html
    for lab in ("奪冠機率", "晉級展望", "小組賽程", "A 組"):
        assert lab in html
    # 隊名應顯示中文（且不出現英文原名）
    assert "阿根廷" in html and "Argentina" not in html


def _toy_model(teams):
    """造一個涵蓋所有隊的最小 DixonColesModel。"""
    return dc.DixonColesModel(
        teams=list(teams),
        attack={t: 0.0 for t in teams},
        defence={t: 0.0 for t in teams},
        home_adv=0.2, rho=0.0, max_goals=8,
    )
