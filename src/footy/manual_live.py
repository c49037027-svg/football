"""手動走地：使用者看直播自行輸入當前比分/狀態 → 立即回機率與公平賠率。

存在理由：免費比分源（ESPN/statsapi）本身延遲 30~90 秒，走地頁的價格
永遠慢半拍；但使用者看的直播只慢幾秒。讓使用者手動輸入當前狀態，
就繞過了整條比分源延遲鏈——模型計算本身是毫秒級的。

路由（webapp）：GET /manual（表單＋結果同頁，改參數即重算）。
結果渲染重用 foot_live/mlb_live 的走地卡片，樣式與 /live 一致。
"""
from __future__ import annotations

import html as _h

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

    def num(key, default, lo, hi):
        try:
            return min(max(int(q.get(key, default) or default), lo), hi)
        except (TypeError, ValueError):
            return default

    minute = num("minute", 0, 0, 90)
    hg, ag = num("hg", 0, 0, 15), num("ag", 0, 0, 15)
    hr, ar = num("hr", 0, 0, 3), num("ar", 0, 0, 3)
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
    return foot_live.render_live_section(snap)


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

    def num(key, default, lo, hi):
        try:
            return min(max(int(q.get(key, default) or default), lo), hi)
        except (TypeError, ValueError):
            return default

    st = mlb_live.LiveState(
        inning=num("inning", 1, 1, 15),
        half="bottom" if q.get("half") == "bottom" else "top",
        outs=num("outs", 0, 0, 2),
        bases="".join("1" if q.get(b) == "1" else "0" for b in ("b1", "b2", "b3")),
        home_score=num("hs", 0, 0, 30), away_score=num("as", 0, 0, 30))
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
    return mlb_live.render_live_section(snap)


# ---------------- 頁面 ----------------

def _opts(names: list[str], zh, selected: str) -> str:
    return "".join(
        f'<option value="{_h.escape(n)}"{" selected" if n == selected else ""}>'
        f'{_h.escape(zh(n))}</option>' for n in names)


def render_manual_page(foot_model, q: dict | None = None,
                       mlb_model_path: str = "models/mlb.pkl",
                       mlb_data_path: str = "data/mlb.csv") -> str:
    """手動走地整頁：結果（若有輸入）＋ ⚽/⚾ 兩個表單（保留上次輸入）。"""
    from . import mlb_live, report
    from .i18n import zh
    q = q or {}
    sport = q.get("sport", "")
    foot_html = mlb_html = ""
    err = ""
    if sport == "foot":
        foot_html = foot_manual_result(foot_model, q)
    elif sport == "mlb":
        try:
            mlb_html = mlb_manual_result(q, mlb_model_path, mlb_data_path)
        except Exception as e:  # noqa: BLE001（模型檔缺失等）
            err = (f"<div class='card'><div class='small'>MLB 模型未載入："
                   f"{_h.escape(str(e))}</div></div>")
    result = foot_html + mlb_html + err
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
      <label><input type='checkbox' name='neutral' value='1'{neutral_chk}> 中立場（世界盃）</label>
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
      <label><input type='checkbox' name='b3' value='1'{chk["b3"]}>三壘</label>
      <button type='submit'>計算</button></div>
  </form>"""
    except Exception:  # noqa: BLE001
        pass
    mlb_form = f"<div class='card'><div class='sec'>⚾ MLB（看轉播輸入當前狀態）</div>{mlb_form_body}</div>"

    form_css = """
.frow{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0;font-size:14px}
.frow input[type=number]{width:64px;padding:6px;background:#161b22;color:var(--fg);
  border:1px solid #30363d;border-radius:6px}
.frow select{padding:6px;background:#161b22;color:var(--fg);border:1px solid #30363d;border-radius:6px}
.frow button{padding:8px 18px;background:#238636;color:#fff;border:0;border-radius:6px;cursor:pointer}
.frow label{display:flex;align-items:center;gap:4px}
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
  <div class="foot">Generated by footy · 研究與教育用途</div>
</div></body></html>"""
