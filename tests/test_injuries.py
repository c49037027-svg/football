"""api-football 傷停整合測試（離線，模擬 JSON）。"""
from footy import context, report, worldcup
from footy.models import dixon_coles as dc


def test_parse_league_injuries_shape():
    payload = {
        "paging": {"current": 1, "total": 1},
        "response": [
            {"team": {"name": "France"}, "player": {"id": 1}},
            {"team": {"name": "France"}, "player": {"id": 2}},
            {"team": {"name": "Brazil"}, "player": {"id": 3}},
        ],
    }
    counts = context.parse_api_football_injuries(payload)
    assert counts["France"] == 2 and counts["Brazil"] == 1


def test_map_injury_counts_alias_and_fuzzy():
    known = ["United States", "South Korea", "Brazil", "Ivory Coast"]
    raw = {"USA": 3, "Korea Republic": 1, "Brasil": 2}  # 別名 + 拼寫
    mapped = context.map_injury_counts(raw, known)
    assert mapped.get("United States") == 3
    assert mapped.get("South Korea") == 1
    assert mapped.get("Brazil") == 2  # 模糊比對 Brasil→Brazil


def test_injury_adjustment_reduces_expected_goals(synthetic_df):
    """缺陣會降低該隊預期進球。"""
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    lam0, mu0 = model.expected_goals(h, a)
    adj = context.injuries_to_adjustment(home_injuries=5, away_injuries=0)
    lam1, mu1 = adj.apply(lam0, mu0)
    assert lam1 < lam0 and abs(mu1 - mu0) < 1e-9


def test_site_shows_injury_note(tmp_path):
    from pathlib import Path
    sched = Path(__file__).resolve().parents[1] / "examples" / "wc2026.json"
    groups, matches, _ = worldcup.parse_wc_json(sched)
    teams = sorted({t for ts in groups.values() for t in ts})
    model = dc.DixonColesModel(teams=list(teams), attack={t: 0.0 for t in teams},
                               defence={t: 0.0 for t in teams}, home_adv=0.2,
                               rho=0.0, max_goals=8)
    res = worldcup.simulate_worldcup(model, sched, n_sims=200, seed=1)
    # 給其中一隊一些缺陣
    some_team = teams[0]
    counts = {some_team: 3}
    outdir, n = report.write_worldcup_site(res, model, matches, tmp_path,
                                           n_sims=300, injury_counts=counts)
    # 找含該隊的分析頁，應顯示「缺陣 3 人」
    found = False
    for m in matches:
        if m.team1 == some_team and m.num in range(1, 999):
            p = Path(outdir) / f"match_{m.num}.html"
            if p.exists() and "缺陣 3 人" in p.read_text(encoding="utf-8"):
                found = True
                break
    assert found
