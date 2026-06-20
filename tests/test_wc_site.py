"""世界盃多頁網站產生測試。"""
from pathlib import Path

from footy import report, worldcup
from footy.models import dixon_coles as dc

SCHED = Path(__file__).resolve().parents[1] / "examples" / "wc2026.json"


def _toy_model(teams):
    return dc.DixonColesModel(
        teams=list(teams), attack={t: 0.0 for t in teams},
        defence={t: 0.0 for t in teams}, home_adv=0.2, rho=0.0, max_goals=8)


def test_write_worldcup_site(tmp_path):
    groups, matches, _ = worldcup.parse_wc_json(SCHED)
    teams = sorted({t for ts in groups.values() for t in ts})
    model = _toy_model(teams)
    result = worldcup.simulate_worldcup(model, SCHED, n_sims=200, seed=1)
    outdir, n = report.write_worldcup_site(result, model, matches, tmp_path,
                                           n_sims=500)
    # 首頁存在
    index = Path(outdir) / "index.html"
    assert index.exists()
    # 小組賽（72 場）皆有分析頁
    assert n == 72
    pages = list(Path(outdir).glob("match_*.html"))
    assert len(pages) == 72
    # 首頁含可點連結與淘汰賽區塊
    html = index.read_text(encoding="utf-8")
    assert "match_" in html and "晉級&對陣" in html  # 對陣圖移到 knockout 頁
    ko = (Path(outdir) / "knockout.html").read_text(encoding="utf-8")
    assert "淘汰賽對陣圖" in ko and "bracket" in ko and "晉級展望" in ko
    # 分析頁含返回首頁與行動裝置 viewport
    sample = pages[0].read_text(encoding="utf-8")
    assert "返回首頁" in sample and "width=device-width" in sample
