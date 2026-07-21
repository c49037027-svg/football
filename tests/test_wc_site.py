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
    # 績效頁存在（無帳本時顯示引導訊息）且首頁導覽列有連結
    perf = (Path(outdir) / "performance.html").read_text(encoding="utf-8")
    assert "尚無收益紀錄" in perf and "performance.html" in html
    # MLB 分頁存在（沒給內容時是引導頁）且導覽列有連結
    mlb_page = (Path(outdir) / "mlb.html").read_text(encoding="utf-8")
    assert "MLB" in mlb_page and "mlb.html" in html
    # 分析頁含返回首頁與行動裝置 viewport
    sample = pages[0].read_text(encoding="utf-8")
    assert "返回首頁" in sample and "width=device-width" in sample


def test_write_site_wc_over(tmp_path):
    """賽會結束：index→hub、世界盃→wc.html、加 leagues.html、導覽換季外版。"""
    from footy import report
    groups, matches, _ = worldcup.parse_wc_json(SCHED)
    teams = sorted({t for m in matches for t in (m.team1, m.team2) if t})
    model = _toy_model(teams)
    result = worldcup.simulate_worldcup(model, SCHED, n_sims=200, seed=1)
    try:
        outdir, _ = report.write_worldcup_site(result, model, matches, tmp_path,
                                               n_sims=500, wc_over=True)
        idx = (Path(outdir) / "index.html").read_text(encoding="utf-8")
        assert "體育預測" in idx and "MLB 今日預測" in idx      # hub
        assert "奪冠機率" not in idx                            # 世界盃內容不在首頁
        wc = (Path(outdir) / "wc.html").read_text(encoding="utf-8")
        assert "奪冠機率" in wc                                 # 世界盃收存檔
        lg = (Path(outdir) / "leagues.html").read_text(encoding="utf-8")
        assert "球季尚未開打" in lg
        # 導覽換版：世界盃存檔/足球聯賽在、晉級&對陣/自訂分析不在
        assert "世界盃存檔" in idx and "足球聯賽" in idx
        assert "晉級&對陣" not in idx and "自訂分析" not in idx
        # 分析頁返回連結指向 wc.html（存檔）
        sample = next(Path(outdir).glob("match_*.html")).read_text(encoding="utf-8")
        assert "wc.html" in sample
    finally:
        report._WC_OVER = False            # 復原模組旗標，免污染其他測試
