"""定期審查 agents 測試：法醫 edge 診斷、自動關閘、閘門生效、報告組裝。"""
import pandas as pd

from footy.agents import review as rv


def _ledger(path, market, n, win_rate, edge=0.05, odds=1.9):
    """合成帳本：n 筆已結算、指定勝率、全部正 edge。"""
    rows = []
    for i in range(n):
        win = i < int(n * win_rate)
        rows.append(dict(date="2026-07-01", match_num=i, home="A", away="B",
                         market=market, selection="over", line=8.5, odds=odds,
                         edge=edge, source="market", close_odds=odds,
                         result="win" if win else "loss",
                         pl=(odds - 1) if win else -1.0))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_forensics_detects_inverted_edge_and_gates(tmp_path):
    lp = tmp_path / "x_bets.csv"
    _ledger(lp, "OU", 60, 0.33)          # 正 edge 60 注只贏 33% → 顯著爛
    fx = rv.ledger_forensics({"X": str(lp)})
    assert fx["stats"][0]["pos_edge_n"] == 60
    acts = [a for a in fx["gate_actions"] if a["action"] == "disable"]
    assert acts and acts[0]["key"] == "X:OU"
    # 自動關閘寫檔
    gp = tmp_path / "gates.json"
    applied = rv.apply_gate_actions(fx["gate_actions"], gp, today="2026-07-20")
    assert applied and rv.load_gates(gp)["X:OU"]["disabled"]


def test_forensics_healthy_market_no_gate(tmp_path):
    lp = tmp_path / "y_bets.csv"
    _ledger(lp, "AH", 80, 0.62)          # 正 edge 62% 勝率 → 健康
    fx = rv.ledger_forensics({"Y": str(lp)})
    assert not [a for a in fx["gate_actions"] if a["action"] == "disable"]
    # 樣本 ≥100 且顯著優於兩平 → 建議重開
    _ledger(lp, "AH", 120, 0.62)
    fx2 = rv.ledger_forensics({"Y": str(lp)})
    assert [a for a in fx2["gate_actions"] if a["action"] == "suggest_enable"]


def test_default_gates_block_ou():
    from footy import mlb
    assert not mlb.market_allowed("OU", "MLB")
    assert not mlb.market_allowed("OU", "NBA")
    assert mlb.market_allowed("1X2", "MLB")
    assert mlb.market_allowed("AH", "NBA")


def test_run_review_and_report(tmp_path):
    lp = tmp_path / "m_bets.csv"
    _ledger(lp, "OU", 60, 0.30)
    gp = tmp_path / "gates.json"
    res = rv.run_review(ledger_paths={"測": str(lp)},
                        odds_log_path=str(tmp_path / "none.csv"),
                        gates_path=gp, today="2026-07-20")
    assert res["gates_applied"]
    rep = rv.compose_report(res, "2026-07-20")
    assert "帳本法醫" in rep and "自動關閘" in rep and "測" in rep
    assert "校準哨兵" in rep and "權重調參" in rep
