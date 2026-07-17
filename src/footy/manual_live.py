"""手動走地：使用者看直播自行輸入當前比分/狀態 → 立即回機率與公平賠率。

存在理由：免費比分源（ESPN/statsapi）本身延遲 30~90 秒，走地頁的價格
永遠慢半拍；但使用者看的直播只慢幾秒。讓使用者手動輸入當前狀態，
就繞過了整條比分源延遲鏈——模型計算本身是毫秒級的。

路由（webapp）：GET /manual（表單＋結果同頁，改參數即重算）。
結果渲染重用 foot_live/mlb_live 的走地卡片，樣式與 /live 一致。
"""
from __future__ import annotations

import html as _h


def _num(q: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        return min(max(int(q.get(key, default) or default), lo), hi)
    except (TypeError, ValueError):
        return default


def _odds(q: dict, key: str) -> float | None:
    """解析莊家賠率欄位：>1 的小數賠率才有效，其餘視為未填。"""
    try:
        v = float(q.get(key, "") or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 1.0 else None


def _edge_card(q: dict, entries: list[tuple[str, float, str]]) -> str:
    """盤口對照卡：entries=[(盤名, 模型機率, 賠率欄位名)]，只列有填賠率的盤。

    EV = p×賠率 − 1（每 1 元注的期望淨利）。正 EV 常來自「模型錯」而非
    「莊家錯」——尤其模型未涵蓋的資訊（先發、傷停、輪休），門檻要嚴。
    """
    rows = []
    for name, p, key in entries:
        o = _odds(q, key)
        if o is None:
            continue
        ev = p * o - 1.0
        cls = "pos" if ev > 0 else "neg"
        rows.append(f"<tr><td>{_h.escape(name)}</td><td>{p:.1%}</td>"
                    f"<td>{1 / max(p, 0.005):.2f}</td><td>{o:.2f}</td>"
                    f"<td class='{cls}'>{ev:+.1%}</td></tr>")
    if not rows:
        return ""
    return ("<div class='card'><div class='sec'>盤口對照（EV = 機率×賠率 −1）</div>"
            "<table class='ltab'><thead><tr><th>盤</th><th>模型機率</th>"
            "<th>公平賠率</th><th>莊家賠率</th><th>EV</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            "<div class='small' style='color:var(--muted)'>正 EV ≠ 必賺：先檢查"
            "模型是否漏了莊家知道的事（先發/傷停/輪休/紅牌）。走地盤 vig 高，"
            "EV 門檻建議 ≥5%。</div></div>")


# ---------------- 足球 ----------------


def foot_manual_result(model, q: dict) -> str:
    """由表單參數算足球走地卡片。q 缺隊伍時回空字串。"""
    from . import foot_live
    home, away = q.get("home", ""), q.get("away", "")
    if not home or not away:
        return ""
    if home == away:
        return "<div class='card'><div class='small'>主客隊不能相同</div></div>"
    if home not in model.attack or away not in model.attack:
        return "<div class='card'><div class='small'>球隊不在模型中</div></div>"
    minute = _num(q, "minute", 0, 0, 90)
    hg, ag = _num(q, "hg", 0, 0, 15), _num(q, "ag", 0, 0, 15)
    hr, ar = _num(q, "hr", 0, 0, 3), _num(q, "ar", 0, 0, 3)
    neutral = q.get("neutral") == "1"
    r = foot_live.live_probs(model, home, away, minute, hg, ag,
                             home_red=hr, away_red=ar, neutral=neutral)
    base = hg + ag + 0.5
    fair = {
        "home_odds": 1.0 / max(r["p_home"], 0.005),
        "draw_odds": 1.0 / max(r["p_draw"], 0.005),
        "away_odds": 1.0 / max(r["p_away"], 0.005),
        "over_lines": [(ln, foot_live.p_over(r, ln))
                       for ln in (base, base + 1, base + 2)],
    }
    g = {"home": home, "away": away, "home_goals": hg, "away_goals": ag,
         "minute": minute, "phase": "in", "home_red": hr, "away_red": ar}
    snap = {"rows": [{"game": g, "p": r, "fair": fair}], "skipped": []}
    # 盤口對照：大小盤用使用者自己的線（僅 .5 線；整數/四分之一線的
    # push/半退未支援，四捨五入到最近 .5）
    entries = [("主勝", r["p_home"], "f_oh"), ("和局", r["p_draw"], "f_od"),
               ("客勝", r["p_away"], "f_oa")]
    try:
        tl = float(q.get("f_otl", "") or 0)
    except (TypeError, ValueError):
        tl = 0.0
    if tl > 0:
        tl = round(tl * 2) / 2
        if tl == int(tl):
            tl += 0.5
        po = foot_live.p_over(r, tl)
        entries += [(f"大 {tl:g}", po, "f_oto"), (f"小 {tl:g}", 1 - po, "f_otu")]
    return foot_live.render_live_section(snap) + _edge_card(q, entries)


# ---------------- MLB ----------------

_MLB_CTX: dict = {}


def _mlb_ctx(model_path: str = "models/mlb.pkl", data_path: str = "data/mlb.csv"):
    """MLB 模型/球場係數/離散度（模組層快取，載一次）。"""
    key = (model_path, data_path)
    if key not in _MLB_CTX:
        from . import mlb
        from .models.dixon_coles import DixonColesModel
        _MLB_CTX[key] = (DixonColesModel.load(model_path),
                         mlb.park_factors_from_csv(data_path),
                         mlb.dispersion_from_csv(data_path))
    return _MLB_CTX[key]


def mlb_manual_result(q: dict, model_path: str = "models/mlb.pkl",
                      data_path: str = "data/mlb.csv") -> str:
    """由表單參數算 MLB 走地卡片。q 缺隊伍時回空字串。"""
    from . import mlb_live
    home, away = q.get("mhome", ""), q.get("maway", "")
    if not home or not away:
        return ""
    if home == away:
        return "<div class='card'><div class='small'>主客隊不能相同</div></div>"
    model, pf_map, disp = _mlb_ctx(model_path, data_path)
    if home not in model.attack or away not in model.attack:
        return "<div class='card'><div class='small'>球隊不在模型中</div></div>"
    st = mlb_live.LiveState(
        inning=_num(q, "inning", 1, 1, 15),
        half="bottom" if q.get("half") == "bottom" else "top",
        outs=_num(q, "outs", 0, 0, 2),
        bases="".join("1" if q.get(b) == "1" else "0" for b in ("b1", "b2", "b3")),
        home_score=_num(q, "hs", 0, 0, 30), away_score=_num(q, "as", 0, 0, 30))
    lam, mu = model.expected_goals(home, away)
    pf = pf_map.get(home, 1.0)
    import time
    r = mlb_live.simulate(st, lam * pf, mu * pf, k=disp,
                          seed=time.time_ns() % (2 ** 32))
    p = min(max(r["p_home"], 0.005), 0.995)
    base = mlb_live._half(r["exp_total"])
    fair = {"home_odds": 1.0 / p, "away_odds": 1.0 / (1.0 - p),
            "over_lines": [(ln, mlb_live.p_over(r, ln))
                           for ln in (base - 1, base, base + 1)]}
    snap = {"date": "-", "rows": [{
        "game": {"home": home, "away": away}, "state": st,
        "p_home": r["p_home"], "exp_total": r["exp_total"], "fair": fair}]}
    entries = [("主勝", p, "m_oh"), ("客勝", 1.0 - p, "m_oa")]
    try:
        tl = float(q.get("m_otl", "") or 0)
    except (TypeError, ValueError):
        tl = 0.0
    if tl > 0:
        tl = round(tl * 2) / 2
        if tl == int(tl):
            tl += 0.5
        po = mlb_live.p_over(r, tl)
        entries += [(f"大 {tl:g}", po, "m_oto"), (f"小 {tl:g}", 1 - po, "m_otu")]
    return mlb_live.render_live_section(snap) + _edge_card(q, entries)


# ---------------- NBA ----------------

_NBA_CTX: dict = {}
NBA_REG_MINUTES = 48.0


def _nba_ctx(model_path: str = "models/nba.pkl"):
    if model_path not in _NBA_CTX:
        from .nba import NBAModel
        _NBA_CTX[model_path] = NBAModel.load(model_path)
    return _NBA_CTX[model_path]


def nba_manual_result(q: dict, model_path: str = "models/nba.pkl") -> str:
    """NBA 手動走地：時間衰減常態模型（剩餘時間縮放均值/變異）。

    ⚠️ 未回測的模型延伸：賽前常態模型已驗證（5 季 64-70%），走地版
    只是把剩餘分數視為 frac 比例的縮小版比賽（mu×frac、sigma×√frac），
    未對真實走地資料驗證，結果卡片帶警語。
    """
    import math
    home, away = q.get("nhome", ""), q.get("naway", "")
    if not home or not away:
        return ""
    if home == away:
        return "<div class='card'><div class='small'>主客隊不能相同</div></div>"
    model = _nba_ctx(model_path)
    if home not in model.off or away not in model.off:
        return "<div class='card'><div class='small'>球隊不在模型中</div></div>"
    quarter = _num(q, "quarter", 1, 1, 5)          # 5 = OT
    qmin = _num(q, "qmin", 12, 0, 12)              # 該節剩餘分鐘
    hs, as_ = _num(q, "nhs", 0, 0, 200), _num(q, "nas", 0, 0, 200)
    if quarter == 5:
        remaining = min(qmin, 5)
    else:
        remaining = (4 - quarter) * 12.0 + qmin
    frac = max(remaining / NBA_REG_MINUTES, 1.0 / 480.0)   # 避免 0 變異退化
    mu_h, mu_a = model.expected_points(home, away)
    diff = hs - as_
    mu_m_rem = (mu_h - mu_a) * frac
    sd_m = model.sigma_margin * math.sqrt(frac)
    p_home = 0.5 * (1.0 + math.erf((diff + mu_m_rem) / sd_m / math.sqrt(2)))
    p_home = min(max(p_home, 0.005), 0.995)
    exp_total = hs + as_ + (mu_h + mu_a) * frac
    sd_t = model.sigma_total * math.sqrt(frac)

    def p_over(line: float) -> float:
        return 1.0 - 0.5 * (1.0 + math.erf((line - exp_total) / sd_t / math.sqrt(2)))

    from .nba import zh_nba
    base = round(exp_total * 2) / 2
    if base == int(base):
        base += 0.5
    ou_rows = "".join(
        f"<tr><td>大 {ln:g}</td><td>{p_over(ln):.1%}</td>"
        f"<td>{1 / max(p_over(ln), 0.005):.2f}</td>"
        f"<td>{1 / max(1 - p_over(ln), 0.005):.2f}</td></tr>"
        for ln in (base - 5, base, base + 5))
    ph_bar = int(round(p_home * 100))
    qlab = "OT" if quarter == 5 else f"第{quarter}節"
    card = f"""
  <div class='card mgame'>
    <div class='mhd'><b>{_h.escape(zh_nba(away))}</b> <span class='at'>{as_}</span>
      <span class='at'>–</span> <span class='at'>{hs}</span> <b>{_h.escape(zh_nba(home))}</b>
      <span class='xr'>{qlab}・剩 {qmin} 分</span></div>
    <div class='lbar'><div class='lfill' style='width:{ph_bar}%'></div></div>
    <div class='lrow'><span>主勝 <b>{p_home:.1%}</b>（公平賠率 主 {1 / p_home:.2f}／客 {1 / (1 - p_home):.2f}）</span>
      <span>預期總分 <b>{exp_total:.1f}</b></span></div>
    <table class='ltab'><thead><tr><th>大小線</th><th>大分機率</th><th>大·公平賠率</th><th>小·公平賠率</th></tr></thead>
    <tbody>{ou_rows}</tbody></table>
    <div class='small'>⚠️ NBA 走地為賽前模型的時間衰減延伸，<b>未經走地資料回測</b>（末節分差保護、
    犯規戰術等未建模），越接近終場越不可靠；賽前版已驗證（5 季勝負準確率 64-70%）。</div>
  </div>"""
    entries = [("主勝", p_home, "n_oh"), ("客勝", 1.0 - p_home, "n_oa")]
    try:
        tl = float(q.get("n_otl", "") or 0)
    except (TypeError, ValueError):
        tl = 0.0
    if tl > 0:
        po = p_over(tl)
        entries += [(f"大 {tl:g}", po, "n_oto"), (f"小 {tl:g}", 1 - po, "n_otu")]
    try:
        sl = float(q.get("n_osl", "") or 0)     # 主隊讓分線（讓 5.5 填 -5.5）
    except (TypeError, ValueError):
        sl = 0.0
    if sl != 0:
        p_cover = 1.0 - 0.5 * (1.0 + math.erf(
            (-sl - diff - mu_m_rem) / sd_m / math.sqrt(2)))
        entries += [(f"主 {sl:+g} 過盤", p_cover, "n_osho"),
                    (f"客 {-sl:+g} 過盤", 1 - p_cover, "n_osao")]
    return card + _edge_card(q, entries)


# ---------------- 頁面 ----------------

def _opts(names: list[str], zh, selected: str) -> str:
    return "".join(
        f'<option value="{_h.escape(n)}"{" selected" if n == selected else ""}>'
        f'{_h.escape(zh(n))}</option>' for n in names)


def render_manual_page(foot_model, q: dict | None = None,
                       mlb_model_path: str = "models/mlb.pkl",
                       mlb_data_path: str = "data/mlb.csv",
                       nba_model_path: str = "models/nba.pkl") -> str:
    """手動走地整頁：結果（若有輸入）＋ ⚽/⚾/🏀 三個表單（保留上次輸入）。"""
    from . import mlb_live, report
    from .i18n import zh
    q = q or {}
    sport = q.get("sport", "")
    result = ""
    if sport == "foot":
        result = foot_manual_result(foot_model, q)
    elif sport == "mlb":
        try:
            result = mlb_manual_result(q, mlb_model_path, mlb_data_path)
        except Exception as e:  # noqa: BLE001（模型檔缺失等）
            result = (f"<div class='card'><div class='small'>MLB 模型未載入："
                      f"{_h.escape(str(e))}</div></div>")
    elif sport == "nba":
        try:
            result = nba_manual_result(q, nba_model_path)
        except Exception as e:  # noqa: BLE001
            result = (f"<div class='card'><div class='small'>NBA 模型未載入："
                      f"{_h.escape(str(e))}</div></div>")
    if result:
        result = ("<h2 style='margin:14px 0 8px'>分析結果</h2>" + result
                  + "<div class='small' style='color:var(--muted)'>"
                  "改比分/時間後再按一次即可重算。</div>")

    fteams = sorted(foot_model.attack)
    def v(key, default=""):
        return _h.escape(str(q.get(key, default)))
    neutral_chk = " checked" if q.get("neutral", "1") == "1" else ""
    foot_form = f"""
  <div class='card'><div class='sec'>⚽ 足球（看直播輸入當前狀態）</div>
  <form method='get' action='/manual'><input type='hidden' name='sport' value='foot'>
    <div class='frow'>主隊 <select name='home'>{_opts(fteams, zh, q.get('home', ''))}</select>
      客隊 <select name='away'>{_opts(fteams, zh, q.get('away', ''))}</select></div>
    <div class='frow'>比分 主 <input name='hg' type='number' min='0' max='15' value='{v("hg", "0")}'>
      – 客 <input name='ag' type='number' min='0' max='15' value='{v("ag", "0")}'>
      第 <input name='minute' type='number' min='0' max='90' value='{v("minute", "0")}'> 分鐘</div>
    <div class='frow'>紅牌 主 <input name='hr' type='number' min='0' max='3' value='{v("hr", "0")}'>
      客 <input name='ar' type='number' min='0' max='3' value='{v("ar", "0")}'>
      <label><input type='checkbox' name='neutral' value='1'{neutral_chk}> 中立場（世界盃）</label></div>
    <div class='frow odds'>莊家賠率（選填）：主勝 <input name='f_oh' value='{v("f_oh")}'>
      和 <input name='f_od' value='{v("f_od")}'> 客勝 <input name='f_oa' value='{v("f_oa")}'>
      ｜大小線 <input name='f_otl' value='{v("f_otl")}'>
      大 <input name='f_oto' value='{v("f_oto")}'> 小 <input name='f_otu' value='{v("f_otu")}'>
      <button type='submit'>計算</button></div>
  </form></div>"""

    mlb_form_body = "<div class='small'>MLB 模型未載入（需 models/mlb.pkl）。</div>"
    try:
        mmodel, _, _ = _mlb_ctx(mlb_model_path, mlb_data_path)
        from . import mlb as _mlb
        mteams = sorted(mmodel.attack)
        half_bot = " selected" if q.get("half") == "bottom" else ""
        chk = {b: " checked" if q.get(b) == "1" else "" for b in ("b1", "b2", "b3")}
        outs_opts = "".join(f"<option value='{o}'{' selected' if v('outs', '0') == str(o) else ''}>{o}</option>"
                            for o in (0, 1, 2))
        mlb_form_body = f"""
  <form method='get' action='/manual'><input type='hidden' name='sport' value='mlb'>
    <div class='frow'>主隊 <select name='mhome'>{_opts(mteams, _mlb.zh_mlb, q.get('mhome', ''))}</select>
      客隊 <select name='maway'>{_opts(mteams, _mlb.zh_mlb, q.get('maway', ''))}</select></div>
    <div class='frow'>比分 客 <input name='as' type='number' min='0' max='30' value='{v("as", "0")}'>
      – 主 <input name='hs' type='number' min='0' max='30' value='{v("hs", "0")}'>
      第 <input name='inning' type='number' min='1' max='15' value='{v("inning", "1")}'> 局
      <select name='half'><option value='top'>上</option><option value='bottom'{half_bot}>下</option></select></div>
    <div class='frow'><select name='outs'>{outs_opts}</select> 出局
      <label><input type='checkbox' name='b1' value='1'{chk["b1"]}>一壘</label>
      <label><input type='checkbox' name='b2' value='1'{chk["b2"]}>二壘</label>
      <label><input type='checkbox' name='b3' value='1'{chk["b3"]}>三壘</label></div>
    <div class='frow odds'>莊家賠率（選填）：主勝 <input name='m_oh' value='{v("m_oh")}'>
      客勝 <input name='m_oa' value='{v("m_oa")}'>
      ｜大小線 <input name='m_otl' value='{v("m_otl")}'>
      大 <input name='m_oto' value='{v("m_oto")}'> 小 <input name='m_otu' value='{v("m_otu")}'>
      <button type='submit'>計算</button></div>
  </form>"""
    except Exception:  # noqa: BLE001
        pass
    mlb_form = f"<div class='card'><div class='sec'>⚾ MLB（看轉播輸入當前狀態）</div>{mlb_form_body}</div>"

    nba_form_body = "<div class='small'>NBA 模型未載入（10 月開季後 deploy 產生 models/nba.pkl）。</div>"
    try:
        nmodel = _nba_ctx(nba_model_path)
        from .nba import zh_nba
        nteams = sorted(nmodel.off)
        qsel = "".join(
            f"<option value='{i}'{' selected' if v('quarter', '1') == str(i) else ''}>"
            f"{'OT' if i == 5 else f'第{i}節'}</option>" for i in range(1, 6))
        nba_form_body = f"""
  <form method='get' action='/manual'><input type='hidden' name='sport' value='nba'>
    <div class='frow'>主隊 <select name='nhome'>{_opts(nteams, zh_nba, q.get('nhome', ''))}</select>
      客隊 <select name='naway'>{_opts(nteams, zh_nba, q.get('naway', ''))}</select></div>
    <div class='frow'>比分 客 <input name='nas' type='number' min='0' max='200' value='{v("nas", "0")}'>
      – 主 <input name='nhs' type='number' min='0' max='200' value='{v("nhs", "0")}'>
      <select name='quarter'>{qsel}</select>
      剩 <input name='qmin' type='number' min='0' max='12' value='{v("qmin", "12")}'> 分</div>
    <div class='frow odds'>莊家賠率（選填）：主勝 <input name='n_oh' value='{v("n_oh")}'>
      客勝 <input name='n_oa' value='{v("n_oa")}'>
      ｜大小線 <input name='n_otl' value='{v("n_otl")}'>
      大 <input name='n_oto' value='{v("n_oto")}'> 小 <input name='n_otu' value='{v("n_otu")}'>
      ｜主讓分線 <input name='n_osl' value='{v("n_osl")}' placeholder='-5.5'>
      主過盤 <input name='n_osho' value='{v("n_osho")}'> 客過盤 <input name='n_osao' value='{v("n_osao")}'>
      <button type='submit'>計算</button></div>
  </form>"""
    except Exception:  # noqa: BLE001
        pass
    nba_form = f"<div class='card'><div class='sec'>🏀 NBA（看轉播輸入當前狀態）</div>{nba_form_body}</div>"

    form_css = """
.frow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0;font-size:14px}
.frow input[type=number]{width:64px;padding:6px;background:#161b22;color:var(--fg);
  border:1px solid #30363d;border-radius:6px}
.frow select{padding:6px;background:#161b22;color:var(--fg);border:1px solid #30363d;border-radius:6px}
.frow button{padding:8px 18px;background:#238636;color:#fff;border:0;border-radius:6px;cursor:pointer}
.frow label{display:flex;align-items:center;gap:4px}
.frow.odds input{width:56px;padding:6px;background:#161b22;color:var(--fg);
  border:1px solid #30363d;border-radius:6px}
.frow.odds{font-size:13px;color:var(--muted)}
.ltab td.pos{color:#7be0b0;font-weight:700}
.ltab td.neg{color:#e0837b}
"""
    return f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>手動走地</title><style>{report._CSS}{mlb_live.LIVE_CSS}{form_css}
</style></head><body><div class="wrap">
  <h1>✍️ 手動走地（即時分析）</h1>
  <div class="sub">你看的直播只慢幾秒，比任何免費比分源（延遲 30~90 秒）都快——
  自己輸入當前狀態，模型立即重算，繞過整條比分延遲鏈。
  <a href="/live" style="color:var(--accent)">← 自動走地</a> ·
  <a href="/index.html" style="color:var(--accent)">回首頁</a></div>
  <div class="disc">⚠️ 公平賠率=無水錢理論價。輸入的狀態要是「當下」的：進球後別忘了同步更新比分與分鐘。</div>
  {result}
  {foot_form}
  {mlb_form}
  {nba_form}
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""
