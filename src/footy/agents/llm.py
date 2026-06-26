"""Claude（Anthropic）LLM 客戶端，供 AI agents 使用。

用官方 anthropic SDK（非 OpenAI 相容層）。設定：
  ANTHROPIC_API_KEY  ── 金鑰（SDK 預設讀此環境變數）
  LLM_MODEL          ── 模型，預設 claude-opus-4-8（要省錢可設 claude-haiku-4-5）
未設金鑰時 available() 回 False，所有 agent 安全略過、不影響主流程。
"""
from __future__ import annotations

import json
import os
import re

# 預設用最強的 Opus 4.8；成本是使用者的決定，要省錢就設 LLM_MODEL。
DEFAULT_MODEL = "claude-opus-4-8"


def _model() -> str:
    return os.environ.get("LLM_MODEL") or DEFAULT_MODEL


def _key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")


def available() -> bool:
    """是否已設金鑰（沒設則所有 agent 略過）。"""
    return bool(_key())


def config() -> dict:
    return {"model": _model(), "key": _key()}


def _client(timeout: float):
    import anthropic
    key = _key()
    if not key:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key, timeout=timeout)


def complete(prompt: str, system: str | None = None, temperature: float | None = None,
             max_tokens: int = 900, timeout: float = 60.0) -> str:
    """送一則 Messages 請求，回純文字。未設金鑰則丟 RuntimeError。

    temperature 參數保留相容性但不轉送——Opus 4.8 已移除取樣參數（傳了會 400）。
    """
    client = _client(timeout)
    kwargs = {"model": _model(), "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    return _extract_text(msg)


def complete_json(prompt: str, system: str | None = None, **kw):
    """要求模型輸出 JSON 並寬鬆解析（容忍 ```json 圍欄與前後雜訊）。失敗回 None。"""
    return parse_json(complete(prompt, system=system, **kw))


def _extract_text(msg) -> str:
    """從 Message.content（區塊清單）取出文字。"""
    parts = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


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
