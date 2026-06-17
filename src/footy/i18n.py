"""隊名中文對照（顯示用）。模型與資料一律用英文當鍵，只在渲染時轉中文。"""
from __future__ import annotations

# 英文國家隊名（martj42 命名）→ 繁體中文
TEAM_ZH = {
    "Algeria": "阿爾及利亞", "Argentina": "阿根廷", "Australia": "澳洲",
    "Austria": "奧地利", "Belgium": "比利時", "Bosnia and Herzegovina": "波士尼亞",
    "Brazil": "巴西", "Canada": "加拿大", "Cape Verde": "維德角",
    "Colombia": "哥倫比亞", "Croatia": "克羅埃西亞", "Curaçao": "古拉索",
    "Czech Republic": "捷克", "DR Congo": "剛果民主共和國", "Ecuador": "厄瓜多",
    "Egypt": "埃及", "England": "英格蘭", "France": "法國", "Germany": "德國",
    "Ghana": "迦納", "Haiti": "海地", "Iran": "伊朗", "Iraq": "伊拉克",
    "Ivory Coast": "象牙海岸", "Japan": "日本", "Jordan": "約旦", "Mexico": "墨西哥",
    "Morocco": "摩洛哥", "Netherlands": "荷蘭", "New Zealand": "紐西蘭",
    "Norway": "挪威", "Panama": "巴拿馬", "Paraguay": "巴拉圭", "Portugal": "葡萄牙",
    "Qatar": "卡達", "Saudi Arabia": "沙烏地阿拉伯", "Scotland": "蘇格蘭",
    "Senegal": "塞內加爾", "South Africa": "南非", "South Korea": "南韓",
    "Spain": "西班牙", "Sweden": "瑞典", "Switzerland": "瑞士", "Tunisia": "突尼西亞",
    "Turkey": "土耳其", "United States": "美國", "Uruguay": "烏拉圭",
    "Uzbekistan": "烏茲別克",
    # 其他常見國家隊（H2H/熱身賽可能出現）
    "Italy": "義大利", "Wales": "威爾斯", "Denmark": "丹麥", "Poland": "波蘭",
    "Serbia": "塞爾維亞", "Nigeria": "奈及利亞", "Cameroon": "喀麥隆",
    "Chile": "智利", "Peru": "秘魯", "Russia": "俄羅斯", "Ukraine": "烏克蘭",
    "Greece": "希臘", "Hungary": "匈牙利", "Romania": "羅馬尼亞",
    "China PR": "中國", "China": "中國", "North Korea": "北韓", "Thailand": "泰國",
    "Costa Rica": "哥斯大黎加", "Honduras": "宏都拉斯", "Jamaica": "牙買加",
    "Mali": "馬利", "Burkina Faso": "布吉納法索", "Cameroon ": "喀麥隆",
}

# 小組名 Group A → A 組
def group_zh(name: str) -> str:
    if name and name.startswith("Group "):
        return f"{name.split()[-1]} 組"
    return name


def zh(name: str) -> str:
    """英文隊名 → 中文；查不到就原樣回傳（不致缺字）。"""
    return TEAM_ZH.get(name, name)
