"""xgabora 彙整資料的 xG 代理（射正→home_xg/away_xg）測試。"""
import pandas as pd

from footy.data import loader
from footy.data import schema as S


def _raw():
    return pd.DataFrame({
        "Division": ["E0", "E0", "SP1"],
        "MatchDate": ["2023-08-12", "2023-08-19", "2023-08-13"],
        "HomeTeam": ["Arsenal", "Man City", "Barcelona"],
        "AwayTeam": ["Forest", "Newcastle", "Getafe"],
        "FTHome": [2, 1, 0], "FTAway": [1, 0, 0],
        "HomeTarget": [6, 5, 4], "AwayTarget": [3, 2, 5],
        "OddHome": [1.3, 1.8, 1.4], "OddDraw": [5.0, 3.6, 4.5],
        "OddAway": [9.0, 4.2, 8.0],
    })


def test_xg_proxy_from_shots_on_target():
    df = loader.normalize_consolidated(_raw(), division="E0")
    assert len(df) == 2
    assert S.HOME_XG in df.columns and S.AWAY_XG in df.columns
    # xG 代理 = 0.33 × 射正
    assert abs(df[S.HOME_XG].iloc[0] - 0.33 * 6) < 1e-9
    assert abs(df[S.AWAY_XG].iloc[0] - 0.33 * 3) < 1e-9
    # 過濾聯賽正確
    assert set(df[S.HOME]) == {"Arsenal", "Man City"}


def test_xg_proxy_absent_when_no_shots():
    raw = _raw().drop(columns=["HomeTarget", "AwayTarget"])
    df = loader.normalize_consolidated(raw, division="E0")
    # 無射正欄位 → 不建立 xG 欄位（dc.fit 會自動退回實際進球）
    assert S.HOME_XG not in df.columns


def test_dc_fit_accepts_xg_weight():
    # 有 xG 欄位時 xg_weight>0 可正常擬合、且與純進球結果不同
    from footy.models import dixon_coles as dc
    df = loader.normalize_consolidated(
        pd.concat([_raw()] * 30, ignore_index=True), division="E0")
    m0 = dc.fit(df, xg_weight=0.0, max_goals=8)
    m1 = dc.fit(df, xg_weight=1.0, max_goals=8)
    a0 = m0.expected_goals("Arsenal", "Man City")
    a1 = m1.expected_goals("Arsenal", "Man City")
    assert a0 != a1   # xG 目標改變了擬合
