"""由比分機率矩陣推導各市場的真實機率與（亞盤的）期望報酬。

支援：
  - 1X2（主勝/平/客勝）
  - Over/Under 總進球（支援 .5 與整數/半整數線，含 push）
  - 亞洲讓盤 Asian Handicap（含 .0 / .25 / .5 / .75 線的半勝半退邏輯）

亞盤與含 push 的大小球無法只用「勝率」表達，因為有退款（stake back）情形，
所以這裡對這類盤口直接回傳「每 1 單位本金、賠率為 odds 時的期望淨報酬」，
方便 value 模組統一用 EV 判斷。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------- 1X2 ----------------
def outcome_1x2(mat: np.ndarray) -> dict[str, float]:
    """回傳 {'home', 'draw', 'away'} 三者機率。"""
    n = mat.shape[0]
    # mat[h, a]：列=主隊進球、行=客隊進球。主勝為 h > a（下三角）。
    home = mat[np.tril_indices(n, k=-1)].sum()  # h > a
    away = mat[np.triu_indices(n, k=1)].sum()   # h < a
    draw = np.trace(mat)
    total = home + draw + away
    return {"home": float(home / total), "draw": float(draw / total),
            "away": float(away / total)}


# ---------------- Over / Under ----------------
def total_goals_probs(mat: np.ndarray) -> np.ndarray:
    """回傳 P(總進球 = k)，k = 0..2*max_goals。"""
    n = mat.shape[0]
    totals = np.zeros(2 * (n - 1) + 1)
    for h in range(n):
        for a in range(n):
            totals[h + a] += mat[h, a]
    return totals


# ---------------- BTTS / 正確比分 / 預期進球（給預測內容用） ----------------
def btts(mat: np.ndarray) -> dict[str, float]:
    """雙方都進球（both teams to score）的機率。"""
    n = mat.shape[0]
    yes = mat[1:, 1:].sum()  # 主客皆 >=1
    total = mat.sum()
    return {"yes": float(yes / total), "no": float(1.0 - yes / total)}


def correct_score(mat: np.ndarray, top_n: int = 5) -> list[tuple[tuple[int, int], float]]:
    """回傳機率最高的前 top_n 個比分 [((h, a), prob), ...]。"""
    n = mat.shape[0]
    flat = [((h, a), float(mat[h, a])) for h in range(n) for a in range(n)]
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat[:top_n]


def most_likely_score(mat: np.ndarray) -> tuple[int, int]:
    """單一最可能比分。"""
    idx = int(np.argmax(mat))
    return divmod(idx, mat.shape[1])


def expected_goals_from_matrix(mat: np.ndarray) -> tuple[float, float]:
    """由比分矩陣算雙方期望進球（含截斷後的實際分布期望）。"""
    n = mat.shape[0]
    goals = np.arange(n)
    eh = float((mat.sum(axis=1) * goals).sum())
    ea = float((mat.sum(axis=0) * goals).sum())
    return eh, ea



def over_under(mat: np.ndarray, line: float) -> dict[str, float]:
    """回傳大小球機率/退款。

    line 例如 2.5（無 push）、3.0（總分=3 時 push 退款）。
    回傳 dict：over_win, over_push, under_win, under_push（push 為平局退本金）。
    """
    totals = total_goals_probs(mat)
    ks = np.arange(len(totals))
    over_win = totals[ks > line].sum()
    under_win = totals[ks < line].sum()
    push = totals[ks == line].sum() if float(line).is_integer() else 0.0
    return {
        "over_win": float(over_win),
        "over_push": float(push),
        "under_win": float(under_win),
        "under_push": float(push),
    }


# ---------------- Asian Handicap ----------------
@dataclass
class AHResult:
    """亞盤單邊在給定賠率下的逐結果機率拆解。"""

    p_win: float    # 全贏（淨賺 odds-1）
    p_half_win: float  # 半贏（淨賺 (odds-1)/2，另一半退本金）
    p_push: float   # 走盤退本金（淨 0）
    p_half_loss: float  # 半輸（輸一半本金）
    p_loss: float   # 全輸（輸全部本金）


def _ah_single_line(diff_probs: dict[int, float], handicap: float, side: str) -> dict[str, float]:
    """對「整數或半整數」單線（如 -1, -0.5, 0, +0.5）計算結果分布。

    diff_probs：P(主隊淨勝球 = d)，d 為整數（可負）。
    handicap：站在 home 視角的讓球（負數=讓，正數=受讓）。
    side：'home' 或 'away'。
    """
    res = {"win": 0.0, "push": 0.0, "loss": 0.0}
    for d, p in diff_probs.items():
        # 站在投注方視角的「調整後淨勝球」
        margin = (d + handicap) if side == "home" else (-d - handicap)
        if margin > 0:
            res["win"] += p
        elif margin == 0:
            res["push"] += p
        else:
            res["loss"] += p
    return res


def goal_diff_probs(mat: np.ndarray) -> dict[int, float]:
    """P(主隊淨勝球 = d)。"""
    n = mat.shape[0]
    out: dict[int, float] = {}
    for h in range(n):
        for a in range(n):
            d = h - a
            out[d] = out.get(d, 0.0) + mat[h, a]
    return out


def asian_handicap(mat: np.ndarray, handicap: float, side: str) -> AHResult:
    """支援 .0 / .25 / .5 / .75 的亞盤。

    .25/.75 線視為「兩條相鄰半線各押一半本金」（split ball）。
    handicap 一律以 home 視角表示，例如 home -0.75；下注 away 時 side='away'。
    """
    diffs = goal_diff_probs(mat)

    # 把 handicap 拆成最近的 0.25 倍數，判斷是否為 quarter 線
    q = round(handicap * 4) / 4.0
    frac = abs(q * 4) % 4  # 0,1,2,3 -> .0,.25,.5,.75

    if frac in (0, 2):  # 純整數或純半線
        r = _ah_single_line(diffs, q, side)
        return AHResult(p_win=r["win"], p_half_win=0.0, p_push=r["push"],
                        p_half_loss=0.0, p_loss=r["loss"])

    lo = (np.floor(q * 2) / 2.0)
    hi = (np.ceil(q * 2) / 2.0)
    # quarter 線拆成相鄰兩條半線（lo, hi）各押一半本金，逐結果組合：
    # 兩線都贏=全贏；一贏一退=半贏；一輸一退=半輸；兩線都輸=全輸。
    win = 0.0
    half_win = 0.0
    push = 0.0
    half_loss = 0.0
    loss = 0.0
    # 列舉兩線結果組合機率：因兩線是同一場比賽的同一隨機事件（淨勝球），
    # 用逐 d 計算最精確。
    for d, p in diffs.items():
        m_lo = (d + lo) if side == "home" else (-d - lo)
        m_hi = (d + hi) if side == "home" else (-d - hi)
        # 各線結果：+1 贏、0 退、-1 輸
        s_lo = 1 if m_lo > 0 else (0 if m_lo == 0 else -1)
        s_hi = 1 if m_hi > 0 else (0 if m_hi == 0 else -1)
        combo = s_lo + s_hi  # 範圍 -2..+2
        if combo == 2:
            win += p
        elif combo == 1:
            half_win += p
        elif combo == 0:
            push += p
        elif combo == -1:
            half_loss += p
        else:
            loss += p
    return AHResult(p_win=win, p_half_win=half_win, p_push=push,
                    p_half_loss=half_loss, p_loss=loss)


# ---------------- 期望報酬（統一給 value 模組用） ----------------
def ev_per_unit(prob_win: float, odds: float, prob_push: float = 0.0,
                prob_half_win: float = 0.0, prob_half_loss: float = 0.0) -> float:
    """每 1 單位本金的期望淨報酬（已扣本金，即 EV，可正可負）。

    - 全贏：+ (odds - 1)
    - 半贏：+ (odds - 1) / 2（另一半退款，淨 0）
    - 走盤 push：0
    - 半輸：- 0.5
    - 全輸：- 1
    其餘（隱含的全輸）由 1 - 其它機率推得。
    """
    prob_full_loss = max(0.0, 1.0 - prob_win - prob_push - prob_half_win - prob_half_loss)
    ev = (prob_win * (odds - 1.0)
          + prob_half_win * (odds - 1.0) / 2.0
          + prob_push * 0.0
          + prob_half_loss * (-0.5)
          + prob_full_loss * (-1.0))
    return float(ev)
