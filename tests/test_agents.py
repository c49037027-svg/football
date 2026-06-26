"""AI agents 測試：用假的 LLM 客戶端（不打網路）驗證提示組裝與解析。"""
from footy.agents import llm, roles


class _A:
    """精簡的 MatchAnalysis 替身。"""
    home, away = "Brazil", "Croatia"
    odds_home, odds_draw, odds_away = 0.55, 0.25, 0.20
    ah_line, ah_supremacy, ah_from_market = -0.75, 0.62, True
    over_under = {2.5: {"over": 0.58, "under": 0.42}}
    btts_yes = 0.51
    top_scores = [((2, 1), 0.11), ((1, 0), 0.10)]
    elo_home, elo_away = 2010, 1820
    home_form, away_form = "WWDLW", "LWDWL"
    h2h = "近 3 戰 2 勝 1 和"
    home_style, away_style = "高壓", "防反"
    home_formation, away_formation = "4-3-3", "4-4-2"
    player_note_home, player_note_away = "主力盡出", "缺主力 1 人"
    data_support = "充足"


def test_parse_json_fenced():
    assert llm.parse_json('前綴 ```json\n{"a":1}\n``` 後綴') == {"a": 1}
    assert llm.parse_json("亂講沒有json") is None


def test_match_facts_contains_key_numbers():
    f = roles.match_facts(_A())
    assert "Brazil" in f and "Croatia" in f
    assert "55%" in f and "-0.75" in f and "真實盤口" in f


def test_preview_builds_prompt_and_calls_llm(monkeypatch):
    seen = {}

    def fake_complete(prompt, system=None, **kw):
        seen["prompt"], seen["system"] = prompt, system
        return "巴西明顯被看好。"
    monkeypatch.setattr(llm, "available", lambda: True)
    out = roles.preview(_A(), complete=fake_complete)
    assert out == "巴西明顯被看好。"
    assert "Brazil" in seen["prompt"] and "賽前分析" in seen["prompt"]


def test_preview_skips_without_key(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    assert roles.preview(_A()) is None


def test_extract_news_parses_json(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    payload = {"home": {"missing": 2, "formation": "4-3-3", "note": "x"},
               "away": {"missing": 0, "formation": "", "note": ""}}
    out = roles.extract_news("Brazil", "Croatia", "Neymar out, Silva injured",
                             complete_json=lambda *a, **k: payload)
    assert out["home"]["missing"] == 2


def test_extract_news_empty_text_returns_none(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    assert roles.extract_news("A", "B", "   ") is None


def test_debate_runs_panel_and_judge(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    res = roles.debate(
        _A(),
        complete=lambda *a, **k: "傾向主勝。",
        complete_json=lambda *a, **k: {"lean": "主勝", "confidence": "中", "summary": "巴西較強"},
        lenses=[("數據派", "看數字"), ("狀態派", "看近況")])
    assert len(res["analysts"]) == 2
    assert res["verdict"]["lean"] == "主勝"


def test_risk_and_postmortem(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    picks = [{"home": "A", "away": "B", "market": "1X2", "selection": "home",
              "odds": 1.8, "line": ""}]
    assert roles.risk_review(picks, complete=lambda *a, **k: "提醒…") == "提醒…"
    rows = [{"home": "A", "away": "B", "market": "1X2", "selection": "home",
             "odds": 1.8, "result": "win", "pl": 0.8}]
    assert roles.postmortem(rows, complete=lambda *a, **k: "檢討…") == "檢討…"
    # 無資料 → None
    assert roles.risk_review([]) is None
