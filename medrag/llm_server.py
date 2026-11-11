"""A minimal OpenAI-compatible chat server for running real mode WITHOUT a GPU server.

In production the vision-language model is served by vLLM on a GPU (see docker-compose.yml). This module serves the
same ``/v1/chat/completions`` API from a local ``transformers`` model, so real mode can be developed and evaluated on
any machine (slowly on CPU). It accepts text and base64 ``image_url`` parts exactly as the system sends them.

    medrag llm-server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8001
"""

from __future__ import annotations

import base64
import io
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_lock = threading.Lock()


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.0


def _to_hf_messages(messages: list[dict]):
    from PIL import Image

    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = []
            for p in content or []:
                if p.get("type") == "text":
                    parts.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
                    if not url.startswith("data:"):
                        raise HTTPException(400, "only base64 data URLs are accepted (no remote fetching)")
                    data = base64.b64decode(url.split(",", 1)[1])
                    parts.append({"type": "image", "image": Image.open(io.BytesIO(data)).convert("RGB")})
        out.append({"role": m.get("role", "user"), "content": parts})
    return out


def create_llm_app(model_id: str, max_pixels: int = 256 * 28 * 28, threads: int | None = None) -> FastAPI:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if threads:
        torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
    dtype = torch.bfloat16 if not torch.cuda.is_available() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).eval()
    if torch.cuda.is_available():  # pragma: no cover
        model = model.to("cuda")
    app = FastAPI(title="MedRAG local LLM server", version="1.0")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        msgs = _to_hf_messages(req.messages)
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(model.device)
        gen = {"max_new_tokens": min(req.max_tokens, 2048), "do_sample": req.temperature > 0}
        if req.temperature > 0:
            gen["temperature"] = req.temperature
        t = time.time()
        with _lock, torch.no_grad():
            out = model.generate(**inputs, **gen)
        new = out[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(new, skip_special_tokens=True)[0]
        n_in, n_out = int(inputs["input_ids"].shape[1]), int(new.shape[1])
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
                          "seconds": round(time.time() - t, 2)}}

    return app
