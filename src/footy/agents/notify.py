"""推播通知：Telegram 與 Discord/Slack webhook（單一 POST，無額外套件）。

設定（環境變數，設哪個就送哪個、可都設）：
  - Telegram：TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Webhook：NOTIFY_WEBHOOK_URL（Discord/Slack incoming webhook 皆可，
    送 {"content":…, "text":…}，Discord 讀 content、Slack 讀 text）
都沒設 → send 回 []、configured 回 []（呼叫端據此改為只印不推）。
"""
from __future__ import annotations

import os


def configured() -> list[str]:
    """已設定的通知管道清單。"""
    ch = []
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        ch.append("telegram")
    if os.environ.get("NOTIFY_WEBHOOK_URL"):
        ch.append("webhook")
    return ch


def send(text: str) -> list[str]:
    """送到所有已設定管道；回實際送出的管道清單（個別失敗容忍、不拋例外）。"""
    import requests
    sent = []
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if tok and chat:
        try:
            r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                              json={"chat_id": chat, "text": text,
                                    "disable_web_page_preview": True}, timeout=15)
            r.raise_for_status()
            sent.append("telegram")
        except Exception:  # noqa: BLE001
            pass
    hook = os.environ.get("NOTIFY_WEBHOOK_URL")
    if hook:
        try:
            r = requests.post(hook, json={"content": text, "text": text}, timeout=15)
            r.raise_for_status()
            sent.append("webhook")
        except Exception:  # noqa: BLE001
            pass
    return sent
