"""openweathermap 天氣接入測試（純函式；無金鑰時安全 no-op）。"""
from footy import mlb_weather as wx


def test_wind_sign_direction():
    # 中外野方位角 0（正北）。風「從南吹來」(180) → 吹向北(0)=中外野 → 出(+)
    assert wx.wind_sign_from_deg(180, 0) > 0.5
    # 風從北吹來(0) → 吹向南 → 進本壘方向 → 進(−)
    assert wx.wind_sign_from_deg(0, 0) < -0.5
    # 橫風（從東 90 → 吹向西）與南北中外野線垂直 → 0
    assert wx.wind_sign_from_deg(90, 0) == 0.0
    # 無風向 → 0
    assert wx.wind_sign_from_deg(None, 0) == 0.0


def test_parse_openweather_nearest_slot():
    forecast = {"list": [
        {"dt": 1000, "main": {"temp": 60.0}, "wind": {"speed": 5.0, "deg": 0}},
        {"dt": 2000, "main": {"temp": 88.0}, "wind": {"speed": 15.0, "deg": 180}},
        {"dt": 3000, "main": {"temp": 70.0}, "wind": {"speed": 8.0, "deg": 90}},
    ]}
    # 目標接近 2000 → 取熱、風大、吹出（deg180、cf0）
    w = wx.parse_openweather(forecast, 2100, cf_deg=0)
    assert w["temp"] == 88.0 and w["wind_speed"] == 15.0 and w["wind_sign"] > 0.5
    # 併入總分係數 > 1（熱 + 吹出）
    from footy import mlb_sim
    assert mlb_sim.weather_total_factor(w) > 1.0
    # 空清單 → None
    assert wx.parse_openweather({"list": []}, 2000, 0) is None


def test_game_weather_safe_fallback():
    # 無金鑰 → None（不調整）
    assert wx.game_weather("Chicago Cubs", "2026-07-06T23:05:00Z", key=None) is None
    # 巨蛋（Rogers Centre）即使有金鑰也中性（室內）
    assert wx.game_weather("Toronto Blue Jays", "2026-07-06T23:05:00Z", key="x") is None
    # 查無球場 → None
    assert wx.game_weather("Unknown Team", "2026-07-06T23:05:00Z", key="x") is None
    # 30 隊都在表內（含 Athletics 別名）
    assert "Chicago Cubs" in wx.MLB_PARKS and "Athletics" in wx.MLB_PARKS
