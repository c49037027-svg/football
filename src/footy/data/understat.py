"""understat.com 真實 xG 抓取（每場 h/a xG）。

understat 聯賽頁把整季賽果嵌在 `datesData = JSON.parse('...')`（十六進位轉義字串）。
解析為純函式、可離線測試；抓取需連 understat（沙箱擋，跑在 Actions/Render）。
產出內部標準欄位含真實 home_xg/away_xg，供 Dixon–Coles 的 xg_weight 使用。

五大聯賽代碼對應 understat slug：E0=EPL、SP1=La_liga、I1=Serie_A、D1=Bundesliga、F1=Ligue_1。
球季以起始年表示（2023 = 2023/24）。
"""
from __future__ import annotations

import codecs
import json
import re
from pathlib import Path

from . import schema as S

LEAGUE_SLUG = {
    "E0": "EPL", "SP1": "La_liga", "I1": "Serie_A",
    "D1": "Bundesliga", "F1": "Ligue_1",
}


def _decode(s: str) -> str:
    """把 understat 的 \\xHH 轉義字串還原成 UTF-8（正確處理重音隊名）。"""
    try:
        return codecs.escape_decode(s.encode())[0].decode("utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return s.encode("utf-8").decode("unicode_escape")


def parse_understat_page(html: str) -> list[dict]:
    """解析 understat 聯賽頁 HTML → 賽果列（純函式）。只回已完賽（isResult）。

    回傳 [{date, home, away, home_goals, away_goals, home_xg, away_xg}]。
    """
    m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", html, re.S)
    if not m:
        return []
    try:
        data = json.loads(_decode(m.group(1)))
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for g in data:
        if not g.get("isResult"):
            continue
        goals = g.get("goals") or {}
        xg = g.get("xG") or {}
        try:
            rows.append({
                S.DATE: (g.get("datetime") or "")[:10],
                S.HOME: (g.get("h") or {}).get("title", ""),
                S.AWAY: (g.get("a") or {}).get("title", ""),
                S.HOME_GOALS: int(goals.get("h") or 0),
                S.AWAY_GOALS: int(goals.get("a") or 0),
                S.HOME_XG: float(xg.get("h") or 0.0),
                S.AWAY_XG: float(xg.get("a") or 0.0),
            })
        except (TypeError, ValueError):
            continue
    return rows


def fetch_understat(league_code: str, season: int, timeout: float = 30.0) -> list[dict]:
    """抓某聯賽某季的真實 xG 賽果（部署環境用；沙箱擋 understat）。"""
    import requests
    slug = LEAGUE_SLUG[league_code]
    r = requests.get(f"https://understat.com/league/{slug}/{season}",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return parse_understat_page(r.text)


def build_league_csv(league_code: str, seasons: list[int],
                     out_path: str | Path | None = None):
    """抓多季 understat 真實 xG，合併成內部標準 CSV（含 home_xg/away_xg）。"""
    import pandas as pd
    rows = []
    for yr in seasons:
        try:
            rows.extend(fetch_understat(league_code, yr))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] understat {league_code} {yr} 失敗：{e}")
    if not rows:
        raise RuntimeError("understat 無資料。")
    df = pd.DataFrame(rows)
    df[S.DATE] = pd.to_datetime(df[S.DATE], errors="coerce")
    df = df.dropna(subset=[S.DATE, S.HOME, S.AWAY]).sort_values(S.DATE).reset_index(drop=True)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[ok] 已存 {len(df)} 場 {league_code} 真實 xG 到 {out_path}")
    return df
