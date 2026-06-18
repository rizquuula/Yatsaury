"""Content-hash disk cache for pipeline stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DiskCache:
    """Content-hash based disk cache for pipeline stages.

    Cache key = SHA-256 of (uri + config_fingerprint).
    Cache value = list of Sample dicts for that URI + config.
    Storage: one JSON file per cache entry under cache_dir/.
    """

    def __init__(self, cache_dir: Path = Path(".cache")) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, uri: str, config_fingerprint: str) -> str:
        raw = f"{uri}|{config_fingerprint}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, uri: str, config_fingerprint: str) -> list[dict] | None:
        """Return cached samples or None if not cached."""
        path = self._dir / f"{self._key(uri, config_fingerprint)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set(self, uri: str, config_fingerprint: str, samples: list[dict]) -> None:
        """Write samples to cache."""
        path = self._dir / f"{self._key(uri, config_fingerprint)}.json"
        path.write_text(json.dumps(samples))

    def invalidate(self, uri: str, config_fingerprint: str) -> bool:
        """Delete cache entry. Returns True if it existed."""
        path = self._dir / f"{self._key(uri, config_fingerprint)}.json"
        if path.exists():
            path.unlink()
            return True
        return False
