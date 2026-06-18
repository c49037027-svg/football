"""互動網頁與盤口讓球線解析測試。"""
from footy import webapp
from footy.live import providers
from footy.live.feed import MarketQuote
from footy.models import dixon_coles as dc


def test_find_ah_line_basic():
    parsed = {"e1": {"home": "Spain", "away": "Haiti", "quotes": [
        MarketQuote("AH", "home", 1.5, line=-1.5),
        MarketQuote("AH", "away", 2.5, line=-1.5),
        MarketQuote("1X2", "home", 1.2)]}}
    r = providers.find_ah_line(parsed, "Spain", "Haiti")
    assert r["line"] == -1.5 and r["home_odds"] == 1.5 and r["away_odds"] == 2.5


def test_find_ah_line_alias():
    parsed = {"e": {"home": "United States", "away": "Mexico", "quotes": [
        MarketQuote("AH", "home", 1.9, line=-0.5),
        MarketQuote("AH", "away", 1.9, line=-0.5)]}}
    r = providers.find_ah_line(parsed, "USA", "Mexico")  # 別名
    assert r is not None and r["line"] == -0.5


def test_find_ah_line_no_match():
    parsed = {"e": {"home": "Brazil", "away": "Argentina", "quotes": [
        MarketQuote("AH", "home", 1.9, line=-0.25)]}}
    assert providers.find_ah_line(parsed, "Japan", "Qatar") is None


def test_render_form_has_controls():
    teams = ["Brazil", "France", "Spain"]
    h = webapp.render_form(teams)
    assert "主隊" in h and "客隊" in h and "陣型" in h
    assert "盤口讓球線" in h and "缺陣主力" in h
    assert "巴西" in h  # 隊名顯示中文


def test_handler_analyze(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    webapp._Handler.model = model
    webapp._Handler.history = synthetic_df
    h, a = model.teams[0], model.teams[1]
    # 直接呼叫分析邏輯（繞過 HTTP）
    handler = webapp._Handler.__new__(webapp._Handler)
    html = handler._analyze({"home": h, "away": a, "hf": "4-3-3", "af": "5-4-1",
                             "hm": "0", "am": "2", "ah": "-1.5", "neutral": "1"})
    assert "<!doctype html>" in html
    assert "-1.5" in html and "陣型" in html and "缺主力 2 人" in html
