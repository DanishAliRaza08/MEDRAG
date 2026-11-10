"""A minimal content-addressed object store on the local filesystem.

(The same ``put/get`` interface can be backed by MinIO/S3 in a hospital deployment.)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_KEY_RE = re.compile(r"^[a-z0-9_-]+/[a-f0-9]{64}\.[a-z]{2,4}$")


class ObjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, prefix: str, ext: str) -> str:
        key = f"{prefix}/{hashlib.sha256(data).hexdigest()}.{ext}"
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def _path(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            raise ValueError("invalid object key")
        return self.root / key
