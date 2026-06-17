"""賽前情境調整：傷停 / 停賽 / 輪休對球隊攻防強度的修正。

模型的攻防參數來自歷史「整隊表現」，無法反映「本場主力傷缺」這類即時資訊。
這個模組把外部情境轉成對 (lambda, mu) 的乘數修正，疊加在模型之上。

兩種來源：
  1) 手動 CSV：你自己評估（最可控，建議用於少數重點場次）。
  2) api-football 傷停 API：自動抓缺陣名單（需 API_FOOTBALL_KEY）。

重要：傷停對戰力的量化沒有公認常數，這裡的預設只是「透明、可調的一階近似」。
請務必用你自己的資料校準（例如比較有/無主力時的實際 xG 差異）後再信任它。
解析函式為純函式、可測試；網路抓取另外封裝。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import requests


@dataclass
class ContextAdjustment:
    """對某場比賽雙方進球率的乘數修正（1.0 = 不變）。"""

    home_attack_mult: float = 1.0
    away_attack_mult: float = 1.0

    def apply(self, lam: float, mu: float) -> tuple[float, float]:
        return lam * self.home_attack_mult, mu * self.away_attack_mult


# ---------------- 手動 CSV ----------------
def load_adjustments_csv(path: str) -> dict[tuple[str, str], ContextAdjustment]:
    """讀手動調整 CSV。欄位：home, away, home_attack_mult, away_attack_mult。

    缺的乘數欄位視為 1.0。
    """
    out: dict[tuple[str, str], ContextAdjustment] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["home"].strip(), row["away"].strip())
            out[key] = ContextAdjustment(
                home_attack_mult=float(row.get("home_attack_mult") or 1.0),
                away_attack_mult=float(row.get("away_attack_mult") or 1.0),
            )
    return out


# ---------------- 傷停 -> 戰力修正（可調啟發式） ----------------
def injuries_to_adjustment(home_injuries: int, away_injuries: int,
                           per_injury_penalty: float = 0.05,
                           max_penalty: float = 0.30) -> ContextAdjustment:
    """把雙方缺陣人數轉成攻擊乘數修正。

    每名缺陣 -per_injury_penalty 的攻擊率（封頂 max_penalty）。
    這是粗略近似：未區分球員重要性。若 api-football 提供傷停球員，
    可進一步用該球員的出賽分鐘/評分加權（見 parse_api_football_injuries 註解）。
    """
    h_pen = min(max_penalty, per_injury_penalty * max(0, home_injuries))
    a_pen = min(max_penalty, per_injury_penalty * max(0, away_injuries))
    return ContextAdjustment(home_attack_mult=1.0 - h_pen,
                             away_attack_mult=1.0 - a_pen)


# ---------------- api-football 抓取 ----------------
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"


def parse_api_football_injuries(payload: dict) -> dict[str, int]:
    """解析 api-football /injuries 回應，回傳 {team_name: 缺陣人數}。

    回應結構：{"response": [{"player": {...}, "team": {"name": ...}, ...}, ...]}
    進階：可改回傳每隊的球員清單，再用球員重要性加權（此處先計數）。
    """
    counts: dict[str, int] = {}
    for item in payload.get("response", []):
        team = (item.get("team") or {}).get("name")
        if team:
            counts[team] = counts.get(team, 0) + 1
    return counts


def fetch_injuries(fixture_id: int, api_key: str | None = None,
                   timeout: float = 15.0) -> dict[str, int]:
    """抓某場比賽的傷停名單計數（需 API_FOOTBALL_KEY）。"""
    api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise RuntimeError("缺少 api-football key。請設環境變數 API_FOOTBALL_KEY。")
    r = requests.get(f"{API_FOOTBALL_BASE}/injuries",
                     params={"fixture": fixture_id},
                     headers={"x-apisports-key": api_key}, timeout=timeout)
    r.raise_for_status()
    return parse_api_football_injuries(r.json())


def fetch_league_injuries(league: int, season: int, api_key: str | None = None,
                          timeout: float = 20.0, max_pages: int = 5) -> dict[str, int]:
    """一次抓整個賽事的傷停（依 league+season），回傳 {隊名: 缺陣人數}。

    比逐隊抓省很多 API 額度（世界盃 league=1）。會自動翻頁。
    """
    api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        raise RuntimeError("缺少 api-football key。請設環境變數 API_FOOTBALL_KEY。")
    headers = {"x-apisports-key": api_key}
    counts: dict[str, int] = {}
    page = 1
    while page <= max_pages:
        r = requests.get(f"{API_FOOTBALL_BASE}/injuries",
                         params={"league": league, "season": season, "page": page},
                         headers=headers, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        for team, c in parse_api_football_injuries(payload).items():
            counts[team] = counts.get(team, 0) + c
        paging = payload.get("paging") or {}
        if page >= int(paging.get("total", 1)):
            break
        page += 1
    return counts


def map_injury_counts(counts: dict[str, int], known_teams: list[str]) -> dict[str, int]:
    """把 api-football 隊名對到我們模型的隊名（別名 + 模糊比對），合併計數。"""
    import difflib
    from .worldcup import TEAM_ALIASES
    known = set(known_teams)
    out: dict[str, int] = {}
    for name, c in counts.items():
        target = TEAM_ALIASES.get(name, name)
        if target not in known:
            match = difflib.get_close_matches(target, known_teams, n=1, cutoff=0.8)
            target = match[0] if match else None
        if target:
            out[target] = out.get(target, 0) + c
    return out


def build_injury_adjustments(counts: dict[str, int], matches,
                             per_injury_penalty: float = 0.04,
                             max_penalty: float = 0.25) -> dict:
    """由各隊缺陣人數，為每場（home, away）建 ContextAdjustment。"""
    adj: dict = {}
    for m in matches:
        h, a = getattr(m, "team1", None), getattr(m, "team2", None)
        if not h or not a:
            continue
        ch, ca = counts.get(h, 0), counts.get(a, 0)
        if ch == 0 and ca == 0:
            continue
        adj[(h, a)] = injuries_to_adjustment(ch, ca, per_injury_penalty, max_penalty)
    return adj

