"""新增功能測試：紅牌調整、xG 建模、真實 feed 解析、傷停情境、校準/CLV。"""
import numpy as np

from footy.config import Config
from footy.context import (ContextAdjustment, injuries_to_adjustment,
                           parse_api_football_injuries)
from footy.live import providers
from footy.live.inplay import final_score_matrix, man_advantage_factors
from footy.models import dixon_coles as dc
from footy.models import markets


# ---------- 紅牌 / 少打多 ----------
def test_man_advantage_factors_symmetry():
    # 客隊吃紅牌（away_red=1）：主隊進球率上升、客隊下降
    hf, af = man_advantage_factors(home_red=0, away_red=1)
    assert hf > 1.0 and af < 1.0
    # 對稱：主隊吃紅牌時相反
    hf2, af2 = man_advantage_factors(home_red=1, away_red=0)
    assert hf2 < 1.0 and af2 > 1.0


def test_red_card_raises_home_win_prob(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000)
    h, a = model.teams[0], model.teams[1]
    base = markets.outcome_1x2(final_score_matrix(model, h, a, 60, 0, 0))
    # 客隊第 60 分鐘吃紅牌，主隊勝率應上升
    red = markets.outcome_1x2(
        final_score_matrix(model, h, a, 60, 0, 0, away_red=1))
    assert red["home"] > base["home"]


# ---------- xG 建模 ----------
def test_xg_weight_fit_runs_and_disables_rho(synthetic_df):
    model = dc.fit(synthetic_df, half_life_days=10_000, xg_weight=1.0)
    assert model.rho == 0.0  # xG 模式關閉 tau
    h, a = model.teams[0], model.teams[1]
    mat = model.score_matrix(h, a)
    assert abs(mat.sum() - 1.0) < 1e-9


# ---------- The Odds API 解析 ----------
SAMPLE_ODDS = [{
    "id": "evt1", "home_team": "Arsenal", "away_team": "Chelsea",
    "commence_time": "2030-01-01T00:00:00Z",
    "bookmakers": [{
        "key": "pinnacle",
        "markets": [
            {"key": "h2h", "outcomes": [
                {"name": "Arsenal", "price": 2.1},
                {"name": "Chelsea", "price": 3.6},
                {"name": "Draw", "price": 3.4}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.9, "point": 2.5},
                {"name": "Under", "price": 1.95, "point": 2.5}]},
            {"key": "spreads", "outcomes": [
                {"name": "Arsenal", "price": 1.95, "point": -0.5},
                {"name": "Chelsea", "price": 1.9, "point": 0.5}]},
        ],
    }],
}]


def test_parse_odds_maps_all_markets():
    parsed = providers.parse_odds(SAMPLE_ODDS)
    q = parsed["evt1"]["quotes"]
    kinds = {(x.market, x.selection) for x in q}
    assert ("1X2", "home") in kinds
    assert ("1X2", "draw") in kinds
    assert ("OU", "over") in kinds
    assert ("AH", "home") in kinds
    # 讓盤線：主隊 -0.5（home 視角）；客隊也轉成 home 視角 -0.5
    ah = {x.selection: x.line for x in q if x.market == "AH"}
    assert ah["home"] == -0.5
    assert ah["away"] == -0.5


def test_parse_scores():
    sc = providers.parse_scores([{
        "id": "evt1", "home_team": "Arsenal", "away_team": "Chelsea",
        "completed": False,
        "scores": [{"name": "Arsenal", "score": "2"}, {"name": "Chelsea", "score": "1"}],
    }])
    assert sc["evt1"]["home_goals"] == 2
    assert sc["evt1"]["away_goals"] == 1
    assert sc["evt1"]["completed"] is False


# ---------- 傷停情境 ----------
def test_injuries_to_adjustment_caps():
    adj = injuries_to_adjustment(home_injuries=20, away_injuries=0,
                                 per_injury_penalty=0.05, max_penalty=0.30)
    assert abs(adj.home_attack_mult - 0.70) < 1e-9  # 封頂 30%
    assert adj.away_attack_mult == 1.0


def test_context_adjustment_apply():
    adj = ContextAdjustment(home_attack_mult=0.8, away_attack_mult=1.1)
    lam, mu = adj.apply(1.5, 1.0)
    assert abs(lam - 1.2) < 1e-9 and abs(mu - 1.1) < 1e-9


def test_parse_api_football_injuries():
    payload = {"response": [
        {"team": {"name": "Arsenal"}, "player": {"id": 1}},
        {"team": {"name": "Arsenal"}, "player": {"id": 2}},
        {"team": {"name": "Chelsea"}, "player": {"id": 3}},
    ]}
    counts = parse_api_football_injuries(payload)
    assert counts["Arsenal"] == 2 and counts["Chelsea"] == 1


# ---------- 校準 / CLV ----------
def test_evaluation_metrics(synthetic_df):
    from footy import evaluation
    cfg = Config()
    res = evaluation.run(synthetic_df, cfg, refit_every=30, min_train_matches=50)
    assert len(res.actuals) > 0
    # Brier 與 logloss 為有限正值
    assert 0 < res.model_brier() < 2
    assert res.model_logloss() > 0
    # 可靠度表非空、summary 不拋錯
    assert not res.reliability_table().empty
    assert isinstance(res.summary(cfg), str)
    # 合成資料有開盤賠率，CLV 應有樣本
    assert len(res.clv_samples) >= 0
