"""Minimal OpenAI-compatible chat client (vLLM, llama.cpp server, TGI ...).

Uses only the standard library so that it adds no dependency. In real mode the
system talks to a *self-hosted* endpoint; patient data never leaves the host.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request

from .config import Settings, get_settings


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.served_model = self.s.llm_model  # updated from each response (the server may serve another model)

    def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> str:
        body = json.dumps({"model": self.s.llm_model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(
            self.s.llm_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.s.llm_api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.s.llm_timeout) as r:  # noqa: S310 - self-hosted URL
                data = json.loads(r.read())
        except Exception as e:  # pragma: no cover - network
            raise LLMError(f"LLM endpoint unavailable: {e}") from e
        self.served_model = data.get("model") or self.served_model
        return data["choices"][0]["message"]["content"]


def image_part(png_bytes: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png_bytes).decode()}}


def extract_json(text: str):
    """Pull the first JSON object/array out of a model response."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise LLMError("no JSON in model output")
    return json.loads(text[start:])
