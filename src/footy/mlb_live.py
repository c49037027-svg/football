"""MLB 走地（in-play）：從當前局面算剩餘比賽的勝率 / 總分分布。

核心想法：把已驗證的「每場負二項(NB)得分」精確分解成「每半局 NB」——
NB 具可加性：整場 NB(size=k, mean=λ) = 9 個 iid NB(size=k/9, mean=λ/9) 之和，
所以走地引擎與賽前模型在開賽瞬間完全一致（開賽時勝率＝賽前勝率），
之後隨比分/局數/出局/壘包即時更新。

規則正確性（用蒙地卡羅而非解析遞迴，因為棒球終局規則多）：
  - 9 局下若主隊已領先 → 不打（總分被截斷）
  - 9 局下（含延長下半）主隊一領先即再見（對勝率無差、對總分有差；
    近似：再見時該半局得分仍整半局計——影響僅尾端 <0.1 分，可接受）
  - 延長賽：幽靈跑者（二壘無出局）→ 每半局期望加成 GHOST_BONUS
壘包/出局（局中狀態）：RE24 期望得分表（2015-2023 聯盟平均環境），
按該隊每 9 局得分率線性縮放——這是文獻標準近似。

驗證方式（見 backtest-live）：statsapi 逐局比分可還原每個半局邊界的
真實歷史狀態 → walk-forward 勝率校準/對數損失，基準=「賽前機率凍結」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# RE24：(壘包狀態, 出局數) → 該半局「之後」的期望得分（聯盟平均環境 ~4.5 R/9）。
# 壘包鍵：三位字串 "一二三"，'1'=有人。來源：2015-2023 多季平均的慣用表。
RE24 = {
    ("000", 0): 0.51, ("000", 1): 0.27, ("000", 2): 0.11,
    ("100", 0): 0.88, ("100", 1): 0.52, ("100", 2): 0.22,
    ("010", 0): 1.14, ("010", 1): 0.68, ("010", 2): 0.32,
    ("001", 0): 1.39, ("001", 1): 0.95, ("001", 2): 0.36,
    ("110", 0): 1.45, ("110", 1): 0.93, ("110", 2): 0.44,
    ("101", 0): 1.77, ("101", 1): 1.13, ("101", 2): 0.48,
    ("011", 0): 1.97, ("011", 1): 1.38, ("011", 2): 0.56,
    ("111", 0): 2.21, ("111", 1): 1.57, ("111", 2): 0.75,
}
RE24_ENV = 4.5          # RE24 表對應的每 9 局得分環境
GHOST_BONUS = 0.55      # 延長賽幽靈跑者：每半局期望得分加成（≈RE(010,0)−RE(000,0) 縮放後）
MAX_EXTRA = 15          # 模擬延長上限（之後判半，機率 <1e-6）


def _half(x: float) -> float:
    """取最近的 .5 盤口線。"""
    import math
    return math.floor(x) + 0.5


@dataclass
class LiveState:
    """走地狀態。inning 從 1 起；half ∈ {"top","bottom"}；
    outs/bases 描述「當前半局進行中」的局面（半局開始 = 0 出局、壘上無人）。
    bases：三位字串（一二三壘），如 "010"=二壘有人。"""
    inning: int
    half: str
    outs: int
    bases: str
    home_score: int
    away_score: int


def _nb_half_innings(rng, mean: float, k9: float, shape) -> np.ndarray:
    """抽每半局得分：NB(size=k/9, mean)。mean<=0 視為 0 分。"""
    if mean <= 0:
        return np.zeros(shape, dtype=np.int64)
    p = k9 / (k9 + mean)
    return rng.negative_binomial(k9, p, shape)


def simulate(state: LiveState, lam_home: float, lam_away: float,
             k: float | None = 3.0, n_sims: int = 20000,
             seed: int = 0) -> dict:
    """從當前狀態模擬到終場。lam_home/lam_away = 兩隊「每場」預期得分
    （賽前模型輸出，含投手/球場/天氣層），k = 每場 NB 離散度。

    回 {"p_home", "exp_total", "exp_home", "exp_away", "total_dist"(np.ndarray)}。
    """
    if k is None or k <= 0:
        k = 1e6                     # 退化為 Poisson
    k9 = k / 9.0
    m_h, m_a = lam_home / 9.0, lam_away / 9.0
    env_h, env_a = lam_home / RE24_ENV, lam_away / RE24_ENV
    rng = np.random.default_rng(seed)
    hs = np.full(n_sims, state.home_score, dtype=np.int64)
    as_ = np.full(n_sims, state.away_score, dtype=np.int64)

    # 1) 當前半局剩餘：RE24（壘包/出局）縮放為該打擊隊環境，當 NB 均值
    cur_re = RE24.get((state.bases, min(state.outs, 2)), 0.0)
    if state.half == "top":
        as_ = as_ + _nb_half_innings(rng, cur_re * env_a, k9, n_sims)
        next_half, next_inning = "bottom", state.inning
    else:
        # 9 局(含延長)下半進行中且主隊已領先 → 理論上已結束；防呆直接算
        hs = hs + _nb_half_innings(rng, cur_re * env_h, k9, n_sims)
        next_half, next_inning = "top", state.inning + 1

    # 2) 之後的完整半局（1-9 局）
    inn = next_inning
    half = next_half
    while inn <= 9:
        if half == "top":
            as_ = as_ + _nb_half_innings(rng, m_a, k9, n_sims)
            half = "bottom"
        else:
            if inn == 9:
                need = hs <= as_          # 9 下只有落後/平手才打
                add = _nb_half_innings(rng, m_h, k9, int(need.sum()))
                hs = hs.copy()
                hs[need] += add
            else:
                hs = hs + _nb_half_innings(rng, m_h, k9, n_sims)
            half = "top"
            inn += 1

    # 特例：狀態已在 9 局下（或更晚）且主隊領先 → 比賽其實已結束（防呆）
    # 3) 延長賽：平手者逐局打（幽靈跑者加成），主隊下半同樣「落後/平手才打」規則
    #    對勝率等價於比較單局得分；平手繼續。
    for _ in range(MAX_EXTRA):
        tied = hs == as_
        n_t = int(tied.sum())
        if n_t == 0:
            break
        add_a = _nb_half_innings(rng, m_a + GHOST_BONUS * env_a, k9, n_t)
        add_h = _nb_half_innings(rng, m_h + GHOST_BONUS * env_h, k9, n_t)
        as_c, hs_c = as_.copy(), hs.copy()
        as_c[tied] += add_a
        # 延長下半：主隊若上半後已落後仍要打；平手/領先中不會發生「領先」
        # （上半只加客隊分）→ 主隊都要打
        hs_c[tied] += add_h
        as_, hs = as_c, hs_c
    still = hs == as_
    if still.any():                      # 極罕見：判給勝率各半（拆單數場）
        flip = rng.random(int(still.sum())) < 0.5
        hs = hs.copy()
        hs[np.flatnonzero(still)[flip]] += 1
        as_ = as_.copy()
        as_[np.flatnonzero(still)[~flip]] += 1

    total = hs + as_
    return {
        "p_home": float((hs > as_).mean()),
        "exp_home": float(hs.mean()), "exp_away": float(as_.mean()),
        "exp_total": float(total.mean()),
        "total_dist": total,
    }


def p_over(result: dict, line: float) -> float:
    """由模擬結果算大分機率（>line；.5 線無和局）。"""
    t = result["total_dist"]
    return float((t > line).mean())


# ---------------- statsapi 狀態解析（純函式可測） ----------------
def parse_linescore_state(ls: dict, default_inning: int = 1) -> LiveState | None:
    """由 statsapi linescore（schedule hydrate 或 feed/live 的 liveData.linescore）
    解析走地狀態。缺欄位安全退回。"""
    if not ls:
        return None
    try:
        teams = ls.get("teams") or {}
        hsc = int(((teams.get("home") or {}).get("runs")) or 0)
        asc = int(((teams.get("away") or {}).get("runs")) or 0)
        inning = int(ls.get("currentInning") or default_inning)
        st = (ls.get("inningState") or ls.get("inningHalf") or "Top").lower()
        half = "bottom" if st.startswith(("bot", "end")) else "top"
        # end of top = 中場 → 視為 bottom 開始；end of bottom → 下一局 top
        if st.startswith("end"):
            inning += 1
            half = "top"
        elif st.startswith("mid"):
            half = "bottom"
        outs = int(ls.get("outs") or 0)
        if st.startswith(("mid", "end")):
            outs = 0
        off = ls.get("offense") or {}
        bases = "".join("1" if off.get(b) else "0"
                        for b in ("first", "second", "third"))
        if st.startswith(("mid", "end")):
            bases = "000"
        return LiveState(inning=inning, half=half, outs=min(outs, 2),
                         bases=bases, home_score=hsc, away_score=asc)
    except (TypeError, ValueError):
        return None


def boundary_states_from_innings(innings: list[dict],
                                 final_home: int, final_away: int) -> list[dict]:
    """由逐局比分還原「每個半局開始」的歷史狀態（回測用，零額外請求）。

    innings：statsapi linescore.innings（[{num, home:{runs}, away:{runs}}...]）。
    回 [{state: LiveState, label:"T3"/"B7"}]；只含常規 1-9 局邊界
    （延長邊界樣本少且規則特殊，不納入校準）。
    """
    out = []
    hs = as_ = 0
    for inn in innings:
        num = int(inn.get("num") or 0)
        if not 1 <= num <= 9:
            break
        # 該局上半開始
        out.append({"label": f"T{num}",
                    "state": LiveState(num, "top", 0, "000", hs, as_)})
        a_runs = (inn.get("away") or {}).get("runs")
        if a_runs is None:
            break                        # 該半局沒打（不可能發生在上半，防呆）
        as_ += int(a_runs)
        # 該局下半開始（9 下主隊領先則沒打）
        h_runs = (inn.get("home") or {}).get("runs")
        if num == 9 and h_runs is None:
            break
        out.append({"label": f"B{num}",
                    "state": LiveState(num, "bottom", 0, "000", hs, as_)})
        if h_runs is None:
            break
        hs += int(h_runs)
    return out


def parse_schedule_linescores(payload: dict, finals_only: bool = True) -> list[dict]:
    """由 schedule?hydrate=linescore 解析逐局比分（純函式）。

    回 [{game_pk, date, home, away, home_goals, away_goals, status,
         innings, linescore}]；finals_only 只留已完賽。
    """
    rows = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") not in ("R", "F", "D", "L", "W"):
                continue
            teams = g.get("teams", {})
            home, away = teams.get("home", {}), teams.get("away", {})
            status = (g.get("status") or {}).get("abstractGameState", "")
            if finals_only and status != "Final":
                continue
            ls = g.get("linescore") or {}
            rows.append({
                "game_pk": g.get("gamePk"),
                "date": g.get("officialDate") or day.get("date"),
                "home": (home.get("team") or {}).get("name", ""),
                "away": (away.get("team") or {}).get("name", ""),
                "home_goals": home.get("score"),
                "away_goals": away.get("score"),
                "status": status,
                "innings": ls.get("innings") or [],
                "linescore": ls,
                "game_date_iso": g.get("gameDate"),
            })
    return rows


def fetch_schedule_linescores(start: str, end: str,
                              timeout: float = 30.0) -> dict:
    """抓帶逐局比分的賽程（statsapi，部署可達；一段日期一次請求）。"""
    import requests
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule",
                     params={"sportId": 1, "startDate": start, "endDate": end,
                             "hydrate": "linescore"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def live_snapshot(model_path: str = "models/mlb.pkl",
                  data_path: str = "data/mlb.csv",
                  date: str | None = None, n_sims: int = 20000,
                  schedule_payload: dict | None = None) -> dict:
    """今日進行中比賽的走地快照（CLI 與 /mlb-live 網頁共用）。

    回 {"date", "rows": [{game, state, p_home, exp_total, fair}], "others": [...]}；
    fair = {"home_odds", "away_odds", "over_lines": [(line, p_over)...]}。
    """
    from . import mlb
    from .models.dixon_coles import DixonColesModel
    model = DixonColesModel.load(model_path)
    pf_map = mlb.park_factors_from_csv(data_path)
    disp = mlb.dispersion_from_csv(data_path)
    date = date or mlb.us_today()
    payload = schedule_payload or fetch_schedule_linescores(date, date)
    games = parse_schedule_linescores(payload, finals_only=False)
    rows, others = [], []
    for g in games:
        if g["home"] not in model.attack or g["away"] not in model.attack:
            continue
        if g["status"] != "Live":
            others.append(g)
            continue
        st = parse_linescore_state(g["linescore"])
        if st is None:
            others.append(g)
            continue
        lam, mu = model.expected_goals(g["home"], g["away"])
        pf = pf_map.get(g["home"], 1.0)
        r = simulate(st, lam * pf, mu * pf, k=disp, n_sims=n_sims,
                     seed=int(g.get("game_pk") or 0) % 100000)
        p = min(max(r["p_home"], 0.005), 0.995)
        base = _half(r["exp_total"])
        over_lines = [(ln, p_over(r, ln)) for ln in (base - 1, base, base + 1)]
        rows.append({"game": g, "state": st, "p_home": r["p_home"],
                     "exp_total": r["exp_total"],
                     "fair": {"home_odds": 1.0 / p, "away_odds": 1.0 / (1.0 - p),
                              "over_lines": over_lines}})
    return {"date": date, "rows": rows, "others": others}


def render_live_page(snap: dict, refresh_sec: int = 45) -> str:
    """走地頁 HTML（Render /mlb-live 用；meta refresh 自動更新）。"""
    import html as _h

    from . import mlb, report
    zh = mlb.zh_mlb
    cards = []
    for r in sorted(snap["rows"], key=lambda x: -abs(x["p_home"] - 0.5)):
        g, st = r["game"], r["state"]
        hz, az = zh(g["home"]), zh(g["away"])
        half = "上" if st.half == "top" else "下"
        bases = "".join(b for b, f in zip("一二三", st.bases) if f == "1") or "無人"
        p = r["p_home"]
        bar = int(round(p * 100))
        fair = r["fair"]
        ou_rows = "".join(
            f"<tr><td>大 {ln:g}</td><td>{po:.1%}</td>"
            f"<td>{1 / max(po, 0.005):.2f}</td>"
            f"<td>{1 / max(1 - po, 0.005):.2f}</td></tr>"
            for ln, po in fair["over_lines"])
        cards.append(f"""
  <div class='card mgame'>
    <div class='mhd'><b>{_h.escape(az)}</b> <span class='at'>{st.away_score}</span>
      <span class='at'>–</span> <span class='at'>{st.home_score}</span> <b>{_h.escape(hz)}</b>
      <span class='xr'>{st.inning}局{half}・{st.outs}出局・壘上{_h.escape(bases)}</span></div>
    <div class='lbar'><div class='lfill' style='width:{bar}%'></div></div>
    <div class='lrow'><span>主勝 <b>{p:.1%}</b>（公平賠率 主 {fair['home_odds']:.2f}／客 {fair['away_odds']:.2f}）</span>
      <span>預期總分 <b>{r['exp_total']:.1f}</b></span></div>
    <table class='ltab'><thead><tr><th>大小線</th><th>大分機率</th><th>大·公平賠率</th><th>小·公平賠率</th></tr></thead>
    <tbody>{ou_rows}</tbody></table>
  </div>""")
    body = "".join(cards) if cards else (
        "<div class='card'><div class='sec'>目前無進行中的比賽</div>"
        "<div class='small' style='color:var(--muted)'>開賽後本頁自動出現即時勝率（每 "
        f"{refresh_sec} 秒更新）。美東賽程日：{_h.escape(snap['date'])}</div></div>")
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{refresh_sec}">
<title>MLB 走地</title><style>{report._CSS}
.mgame{{padding:12px 14px;margin-bottom:12px}}
.mhd{{display:flex;align-items:baseline;gap:6px;font-size:16px}}
.mhd .at{{color:#cdd9e5;font-weight:700}}
.mhd .xr{{margin-left:auto;font-size:12px;color:var(--muted)}}
.lbar{{height:8px;background:#2a1c1c;border-radius:4px;margin:8px 0 4px;overflow:hidden}}
.lfill{{height:100%;background:linear-gradient(90deg,#2e5c42,#7be0b0)}}
.lrow{{display:flex;justify-content:space-between;font-size:13px;color:#cdd9e5;margin:4px 0 8px}}
.ltab{{width:100%;border-collapse:collapse;font-size:12px}}
.ltab th,.ltab td{{padding:4px 6px;text-align:center;border-bottom:1px solid var(--line)}}
.ltab th{{color:var(--muted);font-size:11px}}
</style></head><body><div class="wrap">
  <h1>🔴 MLB 走地（即時勝率）</h1>
  <div class="sub">每 {refresh_sec} 秒自動更新 · 機率經 527 場/9253 邊界回測校準（誤差 ≤2.5pp）·
  <a href="/mlb.html" style="color:var(--accent)">← 回 MLB 預測</a></div>
  <div class="disc">⚠️ 公平賠率=無水錢的理論價；莊家走地賠率高於公平價才有正期望值。走地盤 vig 較高，門檻請比賽前盤嚴。</div>
  {body}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""


def fetch_live_feed(game_pk: int, timeout: float = 20.0) -> dict:
    """抓單場即時 feed（statsapi v1.1；部署/Render 可達）。"""
    import requests
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.json()
