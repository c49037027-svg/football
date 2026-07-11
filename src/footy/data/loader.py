"""下載與載入 football-data.co.uk 歷史資料。

URL 格式：https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
season 例如 2324（代表 2023/24 球季），league 例如 E0（英超）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import schema as S

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# 常用聯賽代碼
LEAGUES = {
    "E0": "English Premier League",
    "E1": "English Championship",
    "SP1": "Spanish La Liga",
    "I1": "Italian Serie A",
    "D1": "German Bundesliga",
    "F1": "French Ligue 1",
    "N1": "Dutch Eredivisie",
    "P1": "Portuguese Primeira Liga",
}


def _season_code(start_year: int) -> str:
    """2023 -> '2324'（2023/24 球季）。"""
    yy = start_year % 100
    return f"{yy:02d}{(yy + 1) % 100:02d}"


def download_season(league: str, start_year: int, timeout: float = 30.0) -> pd.DataFrame:
    """下載單一球季的原始 CSV 為 DataFrame。"""
    url = f"{BASE_URL}/{_season_code(start_year)}/{league}.csv"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    from io import StringIO

    # 該站 CSV 偶有壞行/編碼問題，用寬鬆解析。
    df = pd.read_csv(StringIO(resp.content.decode("latin-1")), on_bad_lines="skip")
    df["__season"] = start_year
    return df


def fetch(league: str, seasons: list[int], out_path: str | Path | None = None,
          timeout: float = 30.0) -> pd.DataFrame:
    """下載多個球季並合併、標準化，可選擇存檔。"""
    frames = []
    for yr in seasons:
        try:
            frames.append(download_season(league, yr, timeout=timeout))
        except Exception as e:  # noqa: BLE001 - 某季缺檔不應中斷全部
            print(f"[warn] 下載 {league} {yr} 失敗：{e}")
    if not frames:
        raise RuntimeError("沒有成功下載任何球季資料。")
    raw = pd.concat(frames, ignore_index=True)
    df = normalize(raw)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ok] 已存 {len(df)} 場比賽到 {out_path}")
    return df


def _first_present(df: pd.DataFrame, candidates: list[str]) -> "pd.Series | None":
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """把原始 football-data.co.uk 欄位轉成內部標準格式。"""
    cols = {}
    for src, dst in S.RAW_RESULT_MAP.items():
        if src not in raw.columns:
            raise KeyError(f"原始資料缺少必要欄位 {src}")
        cols[dst] = raw[src]
    df = pd.DataFrame(cols)

    # 賠率（可能缺，缺則為 NaN，建模不需要、回測/掃描才需要）
    for dst, cands in (
        (S.ODDS_HOME, S.ODDS_HOME_CANDIDATES),
        (S.ODDS_DRAW, S.ODDS_DRAW_CANDIDATES),
        (S.ODDS_AWAY, S.ODDS_AWAY_CANDIDATES),
    ):
        s = _first_present(raw, cands)
        df[dst] = s.values if s is not None else float("nan")

    # 選用開盤賠率（用於 CLV）：若原始資料含早盤欄位則帶入
    for dst, cands in (
        (S.ODDS_HOME_OPEN, S.ODDS_HOME_OPEN_CANDIDATES),
        (S.ODDS_DRAW_OPEN, S.ODDS_DRAW_OPEN_CANDIDATES),
        (S.ODDS_AWAY_OPEN, S.ODDS_AWAY_OPEN_CANDIDATES),
    ):
        s = _first_present(raw, cands)
        if s is not None:
            df[dst] = pd.to_numeric(s.values, errors="coerce")

    # 選用 xG 欄位：若原始資料含常見命名則帶入
    for dst, cands in ((S.HOME_XG, ["HxG", "home_xg", "xG_home"]),
                       (S.AWAY_XG, ["AxG", "away_xg", "xG_away"])):
        s = _first_present(raw, cands)
        if s is not None:
            df[dst] = pd.to_numeric(s.values, errors="coerce")

    # 解析日期（該站有 dd/mm/yy 與 dd/mm/yyyy 兩種）
    df[S.DATE] = pd.to_datetime(df[S.DATE], dayfirst=True, errors="coerce")

    # 清理：必要欄位不可缺
    df = df.dropna(subset=[S.DATE, S.HOME, S.AWAY, S.HOME_GOALS, S.AWAY_GOALS])
    df[S.HOME_GOALS] = df[S.HOME_GOALS].astype(int)
    df[S.AWAY_GOALS] = df[S.AWAY_GOALS].astype(int)
    df = df.sort_values(S.DATE).reset_index(drop=True)
    return df


# ---- GitHub 鏡像（含賠率的彙整資料集），用於網路 allowlist 擋住官方站時 ----
GITHUB_CONSOLIDATED_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv")

# 彙整檔（xgabora）欄位 -> 內部欄位。Division 沿用 football-data 代碼（E0/SP1…）。
CONSOLIDATED_MAP = {
    "MatchDate": S.DATE,
    "HomeTeam": S.HOME,
    "AwayTeam": S.AWAY,
    "FTHome": S.HOME_GOALS,
    "FTAway": S.AWAY_GOALS,
    "OddHome": S.ODDS_HOME,
    "OddDraw": S.ODDS_DRAW,
    "OddAway": S.ODDS_AWAY,
    "HomeElo": S.HOME_ELO,
    "AwayElo": S.AWAY_ELO,
}


def normalize_consolidated(raw: pd.DataFrame, division: str | None = None) -> pd.DataFrame:
    """正規化 xgabora 彙整資料集（含 1X2 收盤賠率與紅牌）。"""
    if division is not None:
        raw = raw[raw["Division"] == division]
    cols = {dst: raw[src] for src, dst in CONSOLIDATED_MAP.items() if src in raw.columns}
    df = pd.DataFrame(cols)
    for c in (S.ODDS_HOME, S.ODDS_DRAW, S.ODDS_AWAY, S.HOME_ELO, S.AWAY_ELO):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 小數賠率必須 > 1；0 或 <=1 視為缺值（彙整檔偶有 0 佔位）。
    for c in (S.ODDS_HOME, S.ODDS_DRAW, S.ODDS_AWAY):
        if c in df.columns:
            df.loc[df[c] <= 1.0, c] = float("nan")
    # 紅牌（選用，走地/分析可用）
    if "HomeRed" in raw.columns:
        df["home_red"] = pd.to_numeric(raw["HomeRed"].values, errors="coerce")
        df["away_red"] = pd.to_numeric(raw["AwayRed"].values, errors="coerce")
    # 射正（shots on target）→ 粗略 xG 代理，填入 home_xg/away_xg（DC 的 xg_weight 會用）。
    # 每次射正約 K_SOT 進球轉換（近年五大聯賽 ~0.33）。這是「去運氣的攻防訊號」替代品，
    # 比純進球雜訊低；真實 xG（understat）可日後以相同欄位覆蓋。缺射正資料時留 NaN，
    # dc.fit 會自動退回實際進球，故舊球季不受影響。
    K_SOT = 0.33
    if "HomeTarget" in raw.columns and "AwayTarget" in raw.columns:
        hsot = pd.to_numeric(raw["HomeTarget"].values, errors="coerce")
        asot = pd.to_numeric(raw["AwayTarget"].values, errors="coerce")
        df[S.HOME_XG] = K_SOT * hsot
        df[S.AWAY_XG] = K_SOT * asot
    df[S.DATE] = pd.to_datetime(df[S.DATE], errors="coerce")
    df = df.dropna(subset=[S.DATE, S.HOME, S.AWAY, S.HOME_GOALS, S.AWAY_GOALS])
    df[S.HOME_GOALS] = df[S.HOME_GOALS].astype(int)
    df[S.AWAY_GOALS] = df[S.AWAY_GOALS].astype(int)
    return df.sort_values(S.DATE).reset_index(drop=True)


def fetch_github(division: str, out_path: str | Path | None = None,
                 timeout: float = 120.0) -> pd.DataFrame:
    """從 GitHub 鏡像下載彙整資料集並過濾指定聯賽（官方站被擋時的備援）。"""
    from io import StringIO

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(GITHUB_CONSOLIDATED_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text), low_memory=False)
    df = normalize_consolidated(raw, division=division)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ok] 已存 {len(df)} 場 {division} 比賽到 {out_path}")
    return df


def _sanitize_odds(df: pd.DataFrame) -> pd.DataFrame:
    """小數賠率必須 > 1；把 0/<=1/非數值轉成 NaN，避免後續除以零。"""
    for c in (S.ODDS_HOME, S.ODDS_DRAW, S.ODDS_AWAY,
              S.ODDS_HOME_OPEN, S.ODDS_DRAW_OPEN, S.ODDS_AWAY_OPEN):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= 1.0, c] = float("nan")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """載入已標準化（或原始）的 CSV。自動偵測格式。"""
    df = pd.read_csv(path, low_memory=False)
    # 已是內部格式？
    if set(S.REQUIRED_INTERNAL).issubset(df.columns):
        df[S.DATE] = pd.to_datetime(df[S.DATE], errors="coerce")
        df = _sanitize_odds(df)
        # neutral 由 CSV 讀回常是字串 "True"/"False"，需轉回布林（bool("False")=True 是陷阱）
        if S.NEUTRAL in df.columns:
            df[S.NEUTRAL] = (df[S.NEUTRAL].map({"True": True, "False": False,
                                                True: True, False: False, 1: True, 0: False})
                             .fillna(False).astype(bool))
        return df.dropna(subset=[S.DATE]).sort_values(S.DATE).reset_index(drop=True)
    # 彙整檔格式？
    if {"Division", "MatchDate", "FTHome", "OddHome"}.issubset(df.columns):
        return normalize_consolidated(df)
    # 否則當作原始 football-data.co.uk 格式
    return normalize(df)
