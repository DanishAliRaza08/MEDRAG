"""Irreversible patient pseudonyms and coarse demographic bands."""

from __future__ import annotations

import hashlib
import hmac


def pseudonymize(patient_id: str, salt: str) -> str:
    """Keyed HMAC-SHA256: stable for linkage, irreversible without the secret salt."""
    if not patient_id or not patient_id.strip():
        raise ValueError("patient_id is required")
    digest = hmac.new(salt.encode(), patient_id.strip().upper().encode(), hashlib.sha256).hexdigest()
    return "pt_" + digest[:32]


def age_band(age: int | None) -> str:
    if age is None:
        return "unknown"
    if age < 0 or age > 130:
        raise ValueError("implausible age")
    if age < 18:
        return "0-17"
    if age < 40:
        return "18-39"
    if age < 65:
        return "40-64"
    return "65+"  # ages > 89 are themselves identifiers under HIPAA; banding removes them
