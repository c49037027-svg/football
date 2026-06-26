"""供應商無關的 LLM 客戶端（OpenAI 相容 chat/completions）。

預設接 Gemini 的 OpenAI 相容端點；改 base_url/model 即可換 OpenAI/OpenRouter/
DeepSeek/本地 Ollama 等。只用 requests，無額外相依。
"""
from __future__ import annotations

import json
import os
import re

import requests

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-2.5-flash"


def config() -> dict:
    return {
        "base": os.environ.get("LLM_BASE_URL", DEFAULT_BASE).rstrip("/"),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "key": os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY"),
    }


def available() -> bool:
    """是否已設金鑰（沒設則所有 agent 略過）。"""
    return bool(config()["key"])


def complete(prompt: str, system: str | None = None, temperature: float = 0.4,
             max_tokens: int = 900, timeout: float = 45.0) -> str:
    """送一則 chat 請求，回純文字。未設金鑰則丟 RuntimeError。"""
    c = config()
    if not c["key"]:
        raise RuntimeError("缺少 LLM_API_KEY / GEMINI_API_KEY")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        f"{c['base']}/chat/completions",
        headers={"Authorization": f"Bearer {c['key']}",
                 "Content-Type": "application/json"},
        json={"model": c["model"], "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def complete_json(prompt: str, system: str | None = None, **kw):
    """要求模型輸出 JSON 並寬鬆解析（容忍 ```json 圍欄與前後雜訊）。失敗回 None。"""
    txt = complete(prompt, system=system, **kw)
    return parse_json(txt)


def parse_json(txt: str):
    """從文字裡抽出第一個 JSON 物件/陣列。"""
    if not txt:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if fenced:
        txt = fenced.group(1)
    m = re.search(r"[\{\[].*[\}\]]", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
