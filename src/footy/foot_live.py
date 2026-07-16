"""足球走地：即時比分（ESPN）→ 1X2/大小 的即時機率與公平賠率。

引擎重用 live/inplay（剩餘時間 Poisson-DC 重算），本模組加上三層
**經 990 場英超半場狀態 walk-forward 驗證**的修正（docs/FINDINGS.md；
重現：scripts/backtest_foot_live.py。驗證樣本不含升級新軍且採納含
test-set reuse，警語見 FINDINGS「足球走地」節）：
  1. 下半場升溫：進球率隨比賽進行上升（半場後 ×1.11，由 22 季訓練期擬合）
  2. 比分狀態效應：落後方壓上 ×1.10、領先方收縮 ×0.95
  3. 溫度縮放 τ=0.95：修正殘餘過度自信
半場邊界驗證：三向 log-loss 0.825 vs 賽前凍結 1.012（-0.19，大勝）。

即時比分源：ESPN scoreboard（雲端可達；cdn/官方源常擋雲端 IP）。
世界盃=fifa.world；五大聯賽開季後加 eng.1/esp.1/ita.1/ger.1/fra.1。
"""
from __future__ import annotations

import numpy as np

ESPN_SOCCER = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
LEAGUE_CODES = ["fifa.world"]        # 世界盃結束後改五大聯賽 codes
BIG5_CODES = ["eng.1", "esp.1", "ita.1", "ger.1", "fra.1"]
NEUTRAL_CODES = {"fifa.world"}       # 中立場賽事：不套主場優勢（與賽前 worldcup.py 一致）

# 驗證出的走地修正參數（FINDINGS：足球走地）
SECOND_HALF_SCALE = 1.11     # 半場後的進球升溫係數
TRAIL_BOOST = 1.10           # 落後方進球率乘數（2018-22 擬合；該期強度屬 in-sample，但僅 2 全域參數且 2022-25 測試期增益成立）
LEAD_DAMP = 0.95             # 領先方進球率乘數
TAU = 0.95                   # 機率溫度縮放（<1 = 軟化）；只驗證/套用於英超 1X2。
                             # 大小盤與中立場國際賽（域外）不套——見 live_probs。

# ESPN → 模型隊名（國家隊；不在表中且不在模型 → 跳過該場）
ESPN_RENAMES = {
    "USA": "United States", "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic", "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea", "Türkiye": "Turkey",
}


def _rate_scale(minute: int) -> float:
    """進球升溫：開賽=1.0（與賽前一致）、線性升至半場 1.11 後持平（僅 45' 驗證過）。"""
    return 1.0 + (SECOND_HALF_SCALE - 1.0) * min(max(minute, 0), 45) / 45.0


def live_probs(model, home: str, away: str, minute: int,
               home_goals: int, away_goals: int,
               home_red: int = 0, away_red: int = 0,
               neutral: bool = False) -> dict:
    """走地 1X2 + 總進球分布。回 {p_home, p_draw, p_away, exp_total, total_dist}。"""
    from .live import inplay
    hf = af = 1.0
    if home_goals > away_goals:
        af, hf = TRAIL_BOOST, LEAD_DAMP
    elif away_goals > home_goals:
        hf, af = TRAIL_BOOST, LEAD_DAMP
    if home_red or away_red:
        rf_h, rf_a = inplay.man_advantage_factors(home_red, away_red)
        hf, af = hf * rf_h, af * rf_a
    rem = inplay.remaining_score_matrix(model, home, away, minute,
                                        _rate_scale(minute),
                                        home_factor=hf, away_factor=af,
                                        neutral=neutral)
    n = rem.shape[0]
    size = n + max(home_goals, away_goals)
    mat = np.zeros((size, size))
    mat[home_goals:home_goals + n, away_goals:away_goals + n] = rem
    ph = float(np.tril(mat, -1).sum())
    pd_ = float(np.trace(mat))
    pa = float(np.triu(mat, 1).sum())
    p = np.array([ph, pd_, pa])
    # τ 只在英超（擬合域）套用；中立場國際賽（世界盃）屬域外、τ 又是三層中
    # 證據最弱的一層（增益在噪音內、無訓練期依據），域外不套。
    tau = 1.0 if neutral else TAU
    p = np.maximum(p / p.sum(), 1e-12) ** tau
    p /= p.sum()
    tot = np.zeros(2 * size - 1)
    for i in range(size):
        for j in range(size):
            tot[i + j] += mat[i, j]
    tot /= tot.sum()
    exp_total = float(sum(k * v for k, v in enumerate(tot)))
    return {"p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2]),
            "exp_total": exp_total, "total_dist": tot}


def p_over(result: dict, line: float) -> float:
    t = result["total_dist"]
    return float(sum(v for k, v in enumerate(t) if k > line))


def parse_espn_soccer(payload: dict) -> list[dict]:
    """解析 ESPN soccer scoreboard 成走地列（純函式）。

    回 [{home, away, home_goals, away_goals, minute, phase, league}]；
    只留進行中（state=="in"）。phase："in"/"ht"。
    """
    out = []
    for ev in payload.get("events") or []:
        st = (ev.get("status") or {})
        tp = st.get("type") or {}
        if tp.get("state") != "in":
            continue
        comp = (ev.get("competitions") or [{}])[0]
        home = away = None
        hs = as_ = 0
        home_id = away_id = ""
        for c in comp.get("competitors") or []:
            team = c.get("team") or {}
            name = (team.get("displayName") or "").strip()
            name = ESPN_RENAMES.get(name, name)
            tid = str(team.get("id") or "")
            try:
                sc = int(c.get("score") or 0)
            except (TypeError, ValueError):
                sc = 0
            if c.get("homeAway") == "home":
                home, hs, home_id = name, sc, tid
            else:
                away, as_, away_id = name, sc, tid
        if not home or not away:
            continue
        # 紅牌：details 事件流（redCard 旗標 + team.id）。缺 details 時安全為 0。
        hred = ared = 0
        for d in comp.get("details") or []:
            if d.get("redCard"):
                tid = str((d.get("team") or {}).get("id") or "")
                if tid and tid == home_id:
                    hred += 1
                elif tid and tid == away_id:
                    ared += 1
        ht = "HALFTIME" in (tp.get("name") or "")
        clock = str(st.get("displayClock") or "")
        # 只取分鐘：先去補時（+），再去秒（:）。"45:30" 若不切會被讀成 4530。
        digits = "".join(ch for ch in clock.split("+")[0].split(":")[0]
                         if ch.isdigit())
        minute = 45 if ht else min(int(digits) if digits else 0, 90)
        out.append({"home": home, "away": away, "home_goals": hs,
                    "away_goals": as_, "minute": minute,
                    "phase": "ht" if ht else "in",
                    "home_red": hred, "away_red": ared,
                    "league": (ev.get("season") or {}).get("slug") or ""})
    return out


def fetch_live_scores(codes: list[str] | None = None,
                      timeout: float = 8.0,
                      errors: list[str] | None = None) -> list[dict]:
    """抓各聯賽進行中比賽（每聯賽一請求，免金鑰）。

    errors：呼叫端傳 list 進來可收到抓取失敗的聯賽代碼——「來源故障」
    和「真的沒比賽」必須能區分，否則故障會被顯示成無比賽。
    timeout 取 8 秒：/live 是同步序列抓取，逾時直接墊高頁面延遲。
    """
    import requests
    rows = []
    for code in codes or LEAGUE_CODES:
        try:
            r = requests.get(ESPN_SOCCER.format(code=code),
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            r.raise_for_status()
            parsed = parse_espn_soccer(r.json())
            for row in parsed:
                row["neutral"] = code in NEUTRAL_CODES
            rows.extend(parsed)
        except Exception:  # noqa: BLE001
            if errors is not None:
                errors.append(code)
            continue
    return rows


def live_snapshot(model_path: str = "models/intl.pkl",
                  codes: list[str] | None = None,
                  matches: list[dict] | None = None) -> dict:
    """足球走地快照（網頁/CLI 共用）。matches 供測試注入。"""
    from .models.dixon_coles import DixonColesModel
    model = DixonColesModel.load(model_path)
    src_errors: list[str] = []
    games = (matches if matches is not None
             else fetch_live_scores(codes, errors=src_errors))
    rows, skipped = [], []
    for g in games:
        if g["home"] not in model.attack or g["away"] not in model.attack:
            skipped.append(f"{g['home']} vs {g['away']}")
            continue
        r = live_probs(model, g["home"], g["away"], g["minute"],
                       g["home_goals"], g["away_goals"],
                       home_red=g.get("home_red", 0),
                       away_red=g.get("away_red", 0),
                       neutral=g.get("neutral", False))
        base = int(g["home_goals"] + g["away_goals"]) + 0.5
        lines = [base, base + 1, base + 2]
        fair = {
            "home_odds": 1.0 / max(r["p_home"], 0.005),
            "draw_odds": 1.0 / max(r["p_draw"], 0.005),
            "away_odds": 1.0 / max(r["p_away"], 0.005),
            "over_lines": [(ln, p_over(r, ln)) for ln in lines],
        }
        rows.append({"game": g, "p": r, "fair": fair})
    return {"rows": rows, "skipped": skipped, "source_errors": src_errors}


def render_live_section(snap: dict) -> str:
    """足球走地區塊 HTML（嵌入 /live 綜合頁）。"""
    import html as _h

    from .i18n import zh
    cards = []
    for r in snap["rows"]:
        g, p, fair = r["game"], r["p"], r["fair"]
        hz, az = zh(g["home"]), zh(g["away"])
        phase = "中場" if g["phase"] == "ht" else f"{g['minute']}'"
        if g["phase"] != "ht" and g["minute"] >= 90:
            phase += "＋ ⚠️補時/加時未建模，本場價格不可用"
        reds = ""
        if g.get("home_red") or g.get("away_red"):
            reds = (f" 🟥{g.get('home_red', 0)}-{g.get('away_red', 0)}"
                    if g.get("home_red") and g.get("away_red")
                    else (f" 🟥主{g.get('home_red')}" if g.get("home_red")
                          else f" 🟥客{g.get('away_red')}"))
        # 已知校準偏差（FINDINGS：足球走地）：主隊下半場落後、模型主勝 10-20% 時
        # 歷史實際逆轉率 ~21%——低估 6-7pp，公平價會偏貴
        warn = ""
        if (g["phase"] != "ht" and 45 <= g["minute"] < 90
                and g["home_goals"] < g["away_goals"]
                and 0.10 <= p["p_home"] <= 0.20):
            warn = ("<div class='small'>⚠️ 已知校準偏差：此類狀態（主隊下半場落後）"
                    "的主勝實際機率歷史上偏高約 6pp，主勝公平價僅供參考</div>")
        ou_rows = "".join(
            f"<tr><td>大 {ln:g}</td><td>{po:.1%}</td>"
            f"<td>{1 / max(po, 0.005):.2f}</td><td>{1 / max(1 - po, 0.005):.2f}</td></tr>"
            for ln, po in fair["over_lines"])
        cards.append(f"""
  <div class='card mgame'>
    <div class='mhd'><b>{_h.escape(hz)}</b> <span class='at'>{g['home_goals']}</span>
      <span class='at'>–</span> <span class='at'>{g['away_goals']}</span> <b>{_h.escape(az)}</b>
      <span class='xr'>{_h.escape(phase + reds)}</span></div>{warn}
    <div class='lrow'><span>主勝 <b>{p['p_home']:.1%}</b>（{fair['home_odds']:.2f}）</span>
      <span>和 <b>{p['p_draw']:.1%}</b>（{fair['draw_odds']:.2f}）</span>
      <span>客勝 <b>{p['p_away']:.1%}</b>（{fair['away_odds']:.2f}）</span>
      <span>預期總球 <b>{p['exp_total']:.2f}</b></span></div>
    <table class='ltab'><thead><tr><th>大小線</th><th>大球機率</th><th>大·公平賠率</th><th>小·公平賠率</th></tr></thead>
    <tbody>{ou_rows}</tbody></table>
  </div>""")
    if not cards:
        if snap.get("source_errors"):
            codes = ", ".join(snap["source_errors"])
            return (f"<div class='card'><div class='sec'>⚽ 足球：比分源暫時故障"
                    f"（{_h.escape(codes)}）</div>"
                    "<div class='small' style='color:var(--muted)'>ESPN 抓取失敗，"
                    "非「無比賽」——稍後自動重試。</div></div>")
        return ("<div class='card'><div class='sec'>⚽ 足球：目前無進行中的比賽</div>"
                "<div class='small' style='color:var(--muted)'>開賽後自動出現。"
                "世界盃期間追蹤 WC；之後換五大聯賽。</div></div>")
    note = ""
    if snap["skipped"]:
        note = ("<div class='small' style='color:var(--muted)'>未涵蓋（隊名不在模型）："
                + _h.escape("、".join(snap["skipped"][:5])) + "</div>")
    return "<h2 style='margin:14px 0 8px'>⚽ 足球走地</h2>" + "".join(cards) + note
