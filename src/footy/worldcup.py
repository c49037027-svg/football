"""整屆世界盃（2026，48 隊新賽制）蒙地卡羅模擬與賽程預測。

讀 openfootball 的官方賽程 JSON（含分組、賽程、已踢比分），用國際賽 Dixon–Coles
模型模擬整屆：
  - 小組賽：已踢的用真實比分，未踢的抽樣 → 小組排名、出線（前二 + 8 個最佳第三）機率
  - 淘汰賽：依官方對陣（R32 用 1A/2B/最佳第三；之後用 W{場次} 勝者）建樹模擬到冠軍
  - 產出每隊：晉級各輪（R32/R16/8強/4強/決賽/冠軍）機率、各小組出線機率、預期積分

2026 賽制：12 組各 4 隊，每組前 2 + 8 個成績最好的第三名 → 32 強淘汰賽。
⚠️ 「最佳第三名」分配到 R32 各槽位有官方對照表，這裡用「符合槽位允許組別」的
二分匹配近似（晉級機率正確，特定對陣槽位為近似）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models.dixon_coles import DixonColesModel


@dataclass
class WCMatch:
    round: str
    group: str | None
    date: str
    team1: str       # 小組賽為隊名；淘汰賽為槽位碼（1A / 3A/B/C/D/F / W74）
    team2: str
    played: bool = False
    hg: int = 0
    ag: int = 0
    num: int = 0     # 官方場次編號（用於 W{num} 引用）


# 賽程 JSON 隊名 → 模型/國際賽資料隊名（martj42 命名）。
TEAM_ALIASES = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
}


def _alias(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


WC2026_SCHEDULE_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "world-cup.json/master/2026/worldcup.json")


def fetch_schedule(out_path: str | Path = "data/wc2026.json",
                   timeout: float = 30.0) -> Path:
    """下載 openfootball 2026 世界盃賽程 JSON（含最新已踢比分）。"""
    import requests
    r = requests.get(WC2026_SCHEDULE_URL, headers={"User-Agent": "Mozilla/5.0"},
                     timeout=timeout)
    r.raise_for_status()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(r.text, encoding="utf-8")
    return out_path


GROUP_ROUNDS = {f"Matchday {i}" for i in range(1, 18)}
KO_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final",
            "Match for third place", "Final"]


def parse_wc_json(path: str | Path) -> tuple[dict[str, list[str]], list[WCMatch], dict[int, WCMatch]]:
    """解析賽程 JSON。回傳 (groups, matches, num→match)。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = d["matches"]

    groups: dict[str, set] = {}
    matches: list[WCMatch] = []
    for x in raw:
        rnd = x.get("round", "")
        grp = x.get("group")
        sc = x.get("score") or {}
        ft = sc.get("ft")
        is_group = bool(grp) and rnd in GROUP_ROUNDS
        # 小組賽 team1/team2 是隊名，套用別名；淘汰賽是槽位碼，保持原樣。
        t1 = _alias(str(x.get("team1", ""))) if is_group else str(x.get("team1", ""))
        t2 = _alias(str(x.get("team2", ""))) if is_group else str(x.get("team2", ""))
        m = WCMatch(
            round=rnd, group=grp, date=x.get("date", ""),
            team1=t1, team2=t2,
            played=bool(ft), hg=ft[0] if ft else 0, ag=ft[1] if ft else 0,
        )
        matches.append(m)
        if grp and rnd in GROUP_ROUNDS:
            groups.setdefault(grp, set()).update([m.team1, m.team2])

    # 指派官方場次編號：小組賽（依日期）1..72，再各淘汰輪依日期接續編號。
    num = 1
    group_matches = sorted([m for m in matches if m.round in GROUP_ROUNDS],
                           key=lambda m: (m.date, m.team1))
    for m in group_matches:
        m.num = num
        num += 1
    for rnd in KO_ORDER:
        rms = sorted([m for m in matches if m.round == rnd], key=lambda m: (m.date,))
        for m in rms:
            m.num = num
            num += 1

    num_map = {m.num: m for m in matches}
    groups_sorted = {g: sorted(ts) for g, ts in sorted(groups.items())}
    return groups_sorted, matches, num_map


# ---------------- 兩隊勝負機率（含點球） ----------------
def _build_adv_matrix(model: DixonColesModel, teams: list[str]) -> np.ndarray:
    """P_adv[i,j] = i 在淘汰賽中淘汰 j 的機率（90 分鐘平手則點球 50/50）。"""
    n = len(teams)
    P = np.full((n, n), 0.5)
    from .models import markets
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if teams[i] not in model.attack or teams[j] not in model.attack:
                continue
            mat = model.score_matrix(teams[i], teams[j], neutral=True)
            o = markets.outcome_1x2(mat)
            P[i, j] = o["home"] + 0.5 * o["draw"]
    return P


def _bipartite_assign(thirds_groups: list[str], slots: list[list[str]]) -> dict[int, str]:
    """把已晉級的第三名（其組別）指派到 R32 第三名槽位（各槽允許特定組別）。

    回傳 slot_index -> group。用增廣路徑做二分匹配（官方保證有完美匹配）。
    """
    match_slot: dict[int, str] = {}
    used_group: set[str] = set()

    def try_assign(si: int, visited: set[int]) -> bool:
        for g in slots[si]:
            if g in thirds_groups and g not in used_group:
                used_group.add(g)
                match_slot[si] = g
                return True
        return False

    # 簡單貪婪（槽位允許組別少的先配）+ 退讓重試
    order = sorted(range(len(slots)), key=lambda s: len(slots[s]))
    for si in order:
        if not try_assign(si, set()):
            # 退讓：找個已配但可換的
            for sj in list(match_slot.keys()):
                gj = match_slot[sj]
                # sj 換成別的允許組別，騰出 gj 給 si
                for g2 in slots[sj]:
                    if g2 in thirds_groups and g2 not in used_group and g2 != gj:
                        used_group.discard(gj)
                        used_group.add(g2)
                        match_slot[sj] = g2
                        if gj in slots[si]:
                            used_group.add(gj)
                            match_slot[si] = gj
                        break
                if si in match_slot:
                    break
    return match_slot


_SLOT_RE = re.compile(r"^([123])([A-L])$")
_THIRD_RE = re.compile(r"^3([A-L/]+)$")
_W_RE = re.compile(r"^[WL](\d+)$")


@dataclass
class WCResult:
    teams: list[str]
    n_sims: int
    # 每隊機率
    champion: dict[str, float] = field(default_factory=dict)
    final: dict[str, float] = field(default_factory=dict)
    semi: dict[str, float] = field(default_factory=dict)
    quarter: dict[str, float] = field(default_factory=dict)
    r16: dict[str, float] = field(default_factory=dict)
    r32: dict[str, float] = field(default_factory=dict)
    # 小組
    group_first: dict[str, float] = field(default_factory=dict)
    group_top2: dict[str, float] = field(default_factory=dict)
    qualify: dict[str, float] = field(default_factory=dict)   # 含最佳第三晉級 R32
    exp_points: dict[str, float] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)


def simulate_worldcup(model: DixonColesModel, path: str | Path,
                      n_sims: int = 10000, seed: int | None = 42) -> WCResult:
    groups, matches, num_map = parse_wc_json(path)
    teams = sorted({t for ts in groups.values() for t in ts})
    tidx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(seed)

    known = [t for t in teams if t in model.attack]
    adv = _build_adv_matrix(model, teams)  # 未知隊以 0.5 處理

    # 預算每場小組賽的比分分布（未踢者）
    group_fix = {g: [m for m in matches if m.group == g and m.round in GROUP_ROUNDS]
                 for g in groups}

    # R32 槽位定義（slot_index 0..15）
    r32 = sorted([m for m in matches if m.round == "Round of 32"], key=lambda m: m.num)

    # 累計器
    cnt = {k: np.zeros(len(teams)) for k in
           ["champ", "final", "semi", "quarter", "r16", "r32",
            "g1", "g2", "qual"]}
    pts_sum = np.zeros(len(teams))

    # 預先抽好每場未踢小組賽的所有模擬比分
    def sample_scores(home, away, k):
        if home in model.attack and away in model.attack:
            mat = model.score_matrix(home, away, neutral=True)
            flat = mat.ravel() / mat.sum()
            ncol = mat.shape[1]
            d = rng.choice(flat.size, size=k, p=flat)
            return d // ncol, d % ncol
        z = np.zeros(k, dtype=int)
        return z, z

    pre = {}
    for g, fix in group_fix.items():
        for m in fix:
            if not m.played:
                pre[m.num] = sample_scores(m.team1, m.team2, n_sims)

    # 逐模擬（小組用向量化彙整，淘汰賽逐場用 adv 查表）
    # 先建立每組每場 (t1,t2,hg[n],ag[n])
    for s in range(n_sims):
        # 小組積分
        gpos: dict[str, list[str]] = {}
        third_info = []  # (group, team, pts, gd, gf)
        for g, fix in group_fix.items():
            tlist = groups[g]
            li = {t: k for k, t in enumerate(tlist)}
            pts = np.zeros(len(tlist)); gd = np.zeros(len(tlist)); gf = np.zeros(len(tlist))
            for m in fix:
                if m.played:
                    hg, ag = m.hg, m.ag
                else:
                    hg = int(pre[m.num][0][s]); ag = int(pre[m.num][1][s])
                ih, ia = li[m.team1], li[m.team2]
                gf[ih] += hg; gf[ia] += ag; gd[ih] += hg - ag; gd[ia] += ag - hg
                if hg > ag: pts[ih] += 3
                elif hg < ag: pts[ia] += 3
                else: pts[ih] += 1; pts[ia] += 1
            score = pts * 1e6 + (gd + 100) * 1e3 + gf
            order = np.argsort(-score, kind="stable")
            ranked = [tlist[k] for k in order]
            gpos[g] = ranked
            for t in tlist:
                pts_sum[tidx[t]] += pts[li[t]]
            cnt["g1"][tidx[ranked[0]]] += 1
            cnt["g2"][tidx[ranked[0]]] += 1
            cnt["g2"][tidx[ranked[1]]] += 1
            k3 = li[ranked[2]]
            third_info.append((g, ranked[2], pts[k3], gd[k3], gf[k3]))

        # 最佳第三名：取前 8。以「組別字母」為鍵（對齊 R32 槽位碼如 3A/B/...）。
        third_info.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
        qual_thirds = third_info[:8]
        third_by_group = {g.split()[-1]: t for (g, t, *_) in qual_thirds}
        qual_third_groups = list(third_by_group.keys())

        # 指派第三名到 R32 槽位
        slot_defs = []
        for m in r32:
            for slot in (m.team1, m.team2):
                tm = _THIRD_RE.match(slot)
                if tm:
                    slot_defs.append((m, slot, list(tm.group(1).replace("/", ""))))
        assign = _bipartite_assign(qual_third_groups, [sd[2] for sd in slot_defs])
        slot_to_group = {}
        for si, (m, slot, _allowed) in enumerate(slot_defs):
            if si in assign:
                slot_to_group[(m.num, slot)] = assign[si]

        # 解析槽位 -> 隊
        def resolve(slot: str, mnum: int):
            sm = _SLOT_RE.match(slot)
            if sm:
                rank = int(sm.group(1)); grp = "Group " + sm.group(2)
                if grp in gpos:
                    return gpos[grp][rank - 1]
            tm = _THIRD_RE.match(slot)
            if tm:
                grp = slot_to_group.get((mnum, slot))
                if grp:
                    return third_by_group.get(grp)
            return None

        # 標記晉級 R32
        winners_cache: dict[int, str] = {}

        def play(mnum: int):
            if mnum in winners_cache:
                return winners_cache[mnum]
            m = num_map[mnum]
            t1 = ref(m.team1, m)
            t2 = ref(m.team2, m)
            if t1 is None and t2 is None:
                w = None
            elif t1 is None:
                w = t2
            elif t2 is None:
                w = t1
            else:
                p = adv[tidx[t1], tidx[t2]]
                w = t1 if rng.random() < p else t2
            winners_cache[mnum] = w
            return w

        def ref(token: str, m: WCMatch):
            wm = _W_RE.match(token)
            if wm:
                return play(int(wm.group(1)))
            return resolve(token, m.num)

        # R32 參賽者
        for m in r32:
            for slot in (m.team1, m.team2):
                t = resolve(slot, m.num)
                if t:
                    cnt["r32"][tidx[t]] += 1
                    cnt["qual"][tidx[t]] += 1

        # 跑各輪，記錄晉級
        def round_nums(rnd):
            return sorted(mm.num for mm in matches if mm.round == rnd)

        for m in [num_map[n] for n in round_nums("Round of 16")]:
            for tok in (m.team1, m.team2):
                t = ref(tok, m)
                if t: cnt["r16"][tidx[t]] += 1
        for m in [num_map[n] for n in round_nums("Quarter-final")]:
            for tok in (m.team1, m.team2):
                t = ref(tok, m)
                if t: cnt["quarter"][tidx[t]] += 1
        for m in [num_map[n] for n in round_nums("Semi-final")]:
            for tok in (m.team1, m.team2):
                t = ref(tok, m)
                if t: cnt["semi"][tidx[t]] += 1
        final_nums = round_nums("Final")
        for m in [num_map[n] for n in final_nums]:
            for tok in (m.team1, m.team2):
                t = ref(tok, m)
                if t: cnt["final"][tidx[t]] += 1
            champ = play(m.num)
            if champ: cnt["champ"][tidx[champ]] += 1

    def to_pct(arr):
        return {teams[i]: float(arr[i] / n_sims) for i in range(len(teams))}

    return WCResult(
        teams=teams, n_sims=n_sims,
        champion=to_pct(cnt["champ"]), final=to_pct(cnt["final"]),
        semi=to_pct(cnt["semi"]), quarter=to_pct(cnt["quarter"]),
        r16=to_pct(cnt["r16"]), r32=to_pct(cnt["r32"]),
        group_first=to_pct(cnt["g1"]), group_top2=to_pct(cnt["g2"]),
        qualify=to_pct(cnt["qual"]),
        exp_points={teams[i]: float(pts_sum[i] / n_sims) for i in range(len(teams))},
        groups=groups,
    )
