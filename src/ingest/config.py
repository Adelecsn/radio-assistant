"""Configuration describing an authorized dataset source."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceConfig:
    """Provenance fields copied into every manifest record."""

    name: str
    version: str
    license: str
    access_url: str
    redistribution_allowed: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "SourceConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"name", "version", "license", "access_url", "redistribution_allowed"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Source configuration is missing: {sorted(missing)}")

        for field in required - {"redistribution_allowed"}:
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise ValueError(f"Source configuration field '{field}' must be non-empty")
        if not isinstance(payload["redistribution_allowed"], bool):
            raise ValueError("redistribution_allowed must be a boolean")

        return cls(**{field: payload[field] for field in required})
