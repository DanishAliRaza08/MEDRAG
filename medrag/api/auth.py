"""Stateless signed bearer tokens (HMAC-SHA256) and role-based access control."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class TokenError(ValueError):
    pass


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(username: str, role: str, secret: str, ttl: int) -> str:
    payload = _b64(json.dumps({"sub": username, "role": role, "exp": int(time.time()) + ttl}).encode())
    sig = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str, secret: str) -> dict:
    try:
        payload, sig = token.split(".")
    except ValueError as e:
        raise TokenError("malformed token") from e
    expect = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expect):
        raise TokenError("bad signature")
    data = json.loads(_unb64(payload))
    if data["exp"] < time.time():
        raise TokenError("token expired")
    return data


# permission -> roles allowed
PERMISSIONS = {
    "read": {"clinician", "admin", "viewer"},
    "ingest": {"clinician", "admin"},
    "analyze": {"clinician"},
    "sign_off": {"clinician"},
    "kb_write": {"admin"},
    "audit": {"admin"},
}
