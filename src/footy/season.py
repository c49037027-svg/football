"""整季蒙地卡羅模擬（像 FiveThirtyEight 的賽季預測）。

用 Dixon–Coles 模型對「整季所有比賽」反覆抽樣，統計每隊的：
  - 奪冠機率、前四（歐冠席）機率、前六機率
  - 降級（末三）機率
  - 預期積分、預期名次、完整名次分布

做法：
  1. 取得賽程（remaining fixtures）。未提供則用球隊清單生成雙循環（主客各一場）。
  2. 可選擇性帶入「目前積分榜」（賽季已踢部分），只模擬剩餘賽程後累加。
  3. 每場比賽：由模型算出比分機率矩陣（含 DC 低分修正），直接從該分布抽樣比分
     （不是只抽 Poisson 邊際，這樣低比分/平局的相關性才正確）。
  4. 累積積分（勝3平1）、淨勝球、進球，依 積分→淨勝球→進球 排名（英超規則；
     未含對戰成績這個更細的 tiebreak，註明之）。
  5. N 次模擬後彙整各種機率。

效能：每場賽事的比分分布只算一次，再用 numpy 對 N 次模擬向量化抽樣。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import schema as S
from .models.dixon_coles import DixonColesModel


@dataclass
class TeamStanding:
    team: str
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class TeamSeasonResult:
    team: str
    title_pct: float
    top4_pct: float
    top6_pct: float
    relegation_pct: float
    exp_points: float
    exp_position: float
    position_dist: np.ndarray  # 長度=隊數，position_dist[i]=排到第 i+1 名的機率


@dataclass
class SeasonSimResult:
    teams: list[str]
    results: dict[str, TeamSeasonResult]
    n_sims: int
    relegation_spots: int
    top4_spots: int = 4
    top6_spots: int = 6

    def table(self) -> pd.DataFrame:
        """依奪冠機率（再依預期積分）排序的彙整表。"""
        rows = []
        for t in self.teams:
            r = self.results[t]
            rows.append({
                "team": t,
                "冠軍%": round(r.title_pct * 100, 1),
                "前四%": round(r.top4_pct * 100, 1),
                "前六%": round(r.top6_pct * 100, 1),
                "降級%": round(r.relegation_pct * 100, 1),
                "預期積分": round(r.exp_points, 1),
                "預期名次": round(r.exp_position, 1),
            })
        df = pd.DataFrame(rows)
        return df.sort_values(["冠軍%", "預期積分"], ascending=False).reset_index(drop=True)


def round_robin_fixtures(teams: list[str]) -> list[tuple[str, str]]:
    """雙循環賽程：每對球隊互為主客各一場。"""
    fixtures = []
    for h in teams:
        for a in teams:
            if h != a:
                fixtures.append((h, a))
    return fixtures


def standings_from_matches(df: pd.DataFrame) -> dict[str, TeamStanding]:
    """由已踢比賽（內部格式）累計目前積分榜。"""
    table: dict[str, TeamStanding] = {}

    def _get(t: str) -> TeamStanding:
        if t not in table:
            table[t] = TeamStanding(team=t)
        return table[t]

    for _, r in df.iterrows():
        h, a = str(r[S.HOME]), str(r[S.AWAY])
        hg, ag = int(r[S.HOME_GOALS]), int(r[S.AWAY_GOALS])
        th, ta = _get(h), _get(a)
        th.played += 1
        ta.played += 1
        th.gf += hg
        th.ga += ag
        ta.gf += ag
        ta.ga += hg
        if hg > ag:
            th.points += 3
        elif hg < ag:
            ta.points += 3
        else:
            th.points += 1
            ta.points += 1
    return table


def simulate_season(model: DixonColesModel, teams: list[str],
                    fixtures: list[tuple[str, str]] | None = None,
                    start_standings: dict[str, TeamStanding] | None = None,
                    n_sims: int = 10000, relegation_spots: int = 3,
                    top4_spots: int = 4, top6_spots: int = 6,
                    seed: int | None = 42) -> SeasonSimResult:
    """跑整季蒙地卡羅模擬。"""
    teams = list(teams)
    n = len(teams)
    tidx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)

    if fixtures is None:
        fixtures = round_robin_fixtures(teams)
    # 只保留雙方都在模型與 teams 內的賽事
    fixtures = [(h, a) for (h, a) in fixtures
                if h in tidx and a in tidx and h in model.attack and a in model.attack]

    # 起始積分/淨勝球/進球
    base_pts = np.zeros(n)
    base_gd = np.zeros(n)
    base_gf = np.zeros(n)
    if start_standings:
        for t, st in start_standings.items():
            if t in tidx:
                base_pts[tidx[t]] = st.points
                base_gd[tidx[t]] = st.gd
                base_gf[tidx[t]] = st.gf

    pts = np.tile(base_pts, (n_sims, 1))
    gd = np.tile(base_gd, (n_sims, 1))
    gf = np.tile(base_gf, (n_sims, 1))

    for h, a in fixtures:
        mat = model.score_matrix(h, a)
        flat = mat.ravel()
        flat = flat / flat.sum()
        ncol = mat.shape[1]
        draws = rng.choice(flat.size, size=n_sims, p=flat)
        hg = draws // ncol
        ag = draws % ncol
        ih, ia = tidx[h], tidx[a]

        home_win = hg > ag
        away_win = ag > hg
        draw = ~(home_win | away_win)
        pts[home_win, ih] += 3
        pts[away_win, ia] += 3
        pts[draw, ih] += 1
        pts[draw, ia] += 1
        gd[:, ih] += hg - ag
        gd[:, ia] += ag - hg
        gf[:, ih] += hg
        gf[:, ia] += ag

    # 排名：依 積分→淨勝球→進球（皆越大越前）。用複合鍵一次排序。
    # 為了向量化，把三者組成單一遞減排序鍵。
    # 名次：對每個模擬列，argsort 後得到排名。
    # 複合分數需避免溢位：用足夠大的權重。
    score = pts * 1e6 + (gd + 1000) * 1e2 + gf  # gd 可能為負，平移確保非負影響排序一致
    # 每列由大到小排序，得到名次（1=最佳）
    order = np.argsort(-score, axis=1, kind="stable")  # order[s, k] = 第 k+1 名的隊 index
    # 反查每隊名次
    positions = np.empty((n_sims, n), dtype=int)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(n)[None, :]  # 0-based 名次
    positions += 1  # 1-based

    results: dict[str, TeamSeasonResult] = {}
    for t in teams:
        i = tidx[t]
        pos_i = positions[:, i]
        dist = np.bincount(pos_i, minlength=n + 1)[1:n + 1] / n_sims
        results[t] = TeamSeasonResult(
            team=t,
            title_pct=float((pos_i == 1).mean()),
            top4_pct=float((pos_i <= top4_spots).mean()),
            top6_pct=float((pos_i <= top6_spots).mean()),
            relegation_pct=float((pos_i > n - relegation_spots).mean()),
            exp_points=float(pts[:, i].mean()),
            exp_position=float(pos_i.mean()),
            position_dist=dist,
        )

    return SeasonSimResult(
        teams=teams, results=results, n_sims=n_sims,
        relegation_spots=relegation_spots, top4_spots=top4_spots, top6_spots=top6_spots,
    )
