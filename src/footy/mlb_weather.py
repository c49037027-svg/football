"""MLB 天氣預報接入（openweathermap）——把回測確認的天氣訊號接上生產大小盤。

流程：球場座標 → openweathermap 5 天/3 小時預報（幾天前就有，解決凌晨建站抓不到
當晚天氣的時機問題）→ 取最接近開賽時刻的時段 → 溫度 + 風速 + 風向。
風向（氣象度數，風「從」何方吹來）配合各場「本壘→中外野」方位角，判斷吹出（助攻，
+）/ 吹進（壓制，−）/ 橫風（0）。輸出 {temp, wind_speed, wind_sign} 給
mlb_sim.weather_total_factor（併入 park_factor、只動大小盤）。

需要環境變數 OPENWEATHER_KEY；未設 → game_weather 回 None → 不調整（安全降級）。

⚠️ 資料註記：球場座標為公開值（溫度只需城市級精度，容忍小誤差）；「中外野方位角」
為近似值，僅用於**粗略**判斷風吹出/進（cos 寬容忍帶 + 保守係數），巨蛋場一律中性
（室內無天氣）。方位角可日後校正，不影響溫度訊號。
"""
from __future__ import annotations

import math

# team → (lat, lon, 本壘→中外野方位角[度,正北0順時針], 是否巨蛋/可開合頂)
# 方位角為近似值；巨蛋（dome=True）視為室內、天氣中性。
# 註：需 OPENWEATHER_KEY 已「啟用」（新金鑰約 10 分~2 小時生效，之前一律回 401）。
MLB_PARKS: dict[str, tuple] = {
    "Arizona Diamondbacks": (33.4453, -112.0667, 0, True),
    "Athletics": (38.5800, -121.5180, 60, False),          # Sutter Health Park（沙加緬度）
    "Oakland Athletics": (38.5800, -121.5180, 60, False),
    "Atlanta Braves": (33.8907, -84.4677, 30, False),
    "Baltimore Orioles": (39.2839, -76.6217, 30, False),
    "Boston Red Sox": (42.3467, -71.0972, 45, False),
    "Chicago Cubs": (41.9484, -87.6553, 30, False),        # Wrigley（風向對總分影響最大）
    "Chicago White Sox": (41.8299, -87.6338, 50, False),
    "Cincinnati Reds": (39.0975, -84.5069, 40, False),
    "Cleveland Guardians": (41.4962, -81.6852, 0, False),
    "Colorado Rockies": (39.7559, -104.9942, 0, False),    # Coors（高海拔，球場因子已含部分）
    "Detroit Tigers": (42.3390, -83.0485, 30, False),
    "Houston Astros": (29.7573, -95.3555, 20, True),
    "Kansas City Royals": (39.0517, -94.4803, 0, False),
    "Los Angeles Angels": (33.8003, -117.8827, 30, False),
    "Los Angeles Dodgers": (34.0739, -118.2400, 25, False),
    "Miami Marlins": (25.7781, -80.2197, 40, True),
    "Milwaukee Brewers": (43.0280, -87.9712, 0, True),
    "Minnesota Twins": (44.9817, -93.2776, 100, False),
    "New York Mets": (40.7571, -73.8458, 20, False),
    "New York Yankees": (40.8296, -73.9262, 20, False),
    "Philadelphia Phillies": (39.9061, -75.1665, 10, False),
    "Pittsburgh Pirates": (40.4469, -80.0057, 60, False),
    "San Diego Padres": (32.7073, -117.1566, 0, False),
    "San Francisco Giants": (37.7786, -122.3893, 30, False),
    "Seattle Mariners": (47.5914, -122.3325, 0, True),
    "St. Louis Cardinals": (38.6226, -90.1928, 0, False),
    "Tampa Bay Rays": (27.7683, -82.6534, 0, True),
    "Texas Rangers": (32.7473, -97.0842, 0, True),
    "Toronto Blue Jays": (43.6414, -79.3894, 0, True),
    "Washington Nationals": (38.8730, -77.0074, 30, False),
}


def wind_sign_from_deg(wind_from_deg: float | None, cf_deg: float,
                       band_cos: float = 0.5) -> float:
    """風向（氣象度數，風從何方來）+ 中外野方位角 → 吹出/進係數（cos 縮放）。

    風實際吹「向」= wind_from_deg + 180。與本壘→中外野方位角夾角小 → 吹向外野（出，+）；
    夾角大（風吹向本壘）→ 進（−）。|cos| < band 視為橫風（0）。回 [-1,1] 浮點。
    """
    if wind_from_deg is None:
        return 0.0
    blow_to = (wind_from_deg + 180.0) % 360.0
    diff = abs(((blow_to - cf_deg + 180.0) % 360.0) - 180.0)   # 0..180 度
    c = math.cos(math.radians(diff))
    return c if abs(c) >= band_cos else 0.0


def _nearest_slot(slots: list, target_unix: float):
    best = None
    for s in slots:
        dt = s.get("dt")
        if dt is None:
            continue
        if best is None or abs(dt - target_unix) < abs(best["dt"] - target_unix):
            best = s
    return best


def parse_openweather(forecast_json: dict, target_unix: float,
                      cf_deg: float) -> dict | None:
    """解析 openweathermap /forecast 回應（units=imperial）→ 天氣 dict。純函式、可測。

    取最接近 target_unix 的 3 小時時段；回 {temp(°F), wind_speed(mph), wind_sign}。
    """
    s = _nearest_slot(forecast_json.get("list") or [], target_unix)
    if not s:
        return None
    main = s.get("main") or {}
    wind = s.get("wind") or {}
    temp = main.get("temp")
    speed = wind.get("speed") or 0.0
    sign = wind_sign_from_deg(wind.get("deg"), cf_deg)
    if temp is None and not speed:
        return None
    return {"temp": float(temp) if temp is not None else None,
            "wind_speed": float(speed), "wind_sign": sign}


def _to_unix(iso: str | None) -> float | None:
    if not iso:
        return None
    import datetime as _dt
    try:
        dt = _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        return dt.timestamp()
    except Exception:  # noqa: BLE001
        return None


def probe(team: str, when_iso: str | None, key: str | None = None,
          timeout: float = 20.0) -> str:
    """診斷用：回傳單場天氣抓取的結果原因（不吞錯），供 weather-probe CLI。

    可能值：no-key / no-park:<隊> / dome / no-time / neterr:<型別> /
    http:<狀態碼>（如 401=金鑰未啟用或錯誤）/ no-slot / ok temp=.. wind=.. f=..
    """
    import os
    key = key or os.environ.get("OPENWEATHER_KEY")
    if not key:
        return "no-key"
    park = MLB_PARKS.get(team)
    if not park:
        return f"no-park:{team}"
    lat, lon, cf_deg, dome = park
    if dome:
        return "dome"
    target = _to_unix(when_iso)
    if target is None:
        return f"no-time:{when_iso}"
    import requests
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                         params={"lat": lat, "lon": lon, "appid": key,
                                 "units": "imperial"}, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return f"neterr:{type(e).__name__}"
    if r.status_code != 200:
        return f"http:{r.status_code}"
    w = parse_openweather(r.json(), target, cf_deg)
    if w is None:
        return "no-slot"
    from . import mlb_sim
    return (f"ok temp={w.get('temp')} wind={w.get('wind_speed')} "
            f"sign={(w.get('wind_sign') or 0):.2f} f={mlb_sim.weather_total_factor(w):.3f}")


def game_weather(team: str, when_iso: str | None, key: str | None = None,
                 timeout: float = 20.0) -> dict | None:
    """抓某場（主隊球場）開賽時刻的預報天氣。無金鑰/巨蛋/查無球場/失敗 → None（不調整）。"""
    import os
    key = key or os.environ.get("OPENWEATHER_KEY")
    park = MLB_PARKS.get(team)
    if not key or not park:
        return None
    lat, lon, cf_deg, dome = park
    if dome:                       # 室內／可開合頂關閉時無天氣影響 → 中性
        return None
    target = _to_unix(when_iso)
    if target is None:
        return None
    import requests
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                         params={"lat": lat, "lon": lon, "appid": key,
                                 "units": "imperial"}, timeout=timeout)
        r.raise_for_status()
        return parse_openweather(r.json(), target, cf_deg)
    except Exception:  # noqa: BLE001
        return None
