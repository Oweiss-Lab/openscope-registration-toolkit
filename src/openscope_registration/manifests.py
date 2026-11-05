"""Manifest helpers for calcium imaging skill workflows."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    *,
    artifact_type: str,
    created_by: str,
    parameters: dict[str, Any],
    output_files: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "artifact_type": artifact_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "source_script": created_by,
        "parameters": parameters,
        "output_files": output_files,
        "output_sha256": {k: sha256(Path(v)) for k, v in output_files.items() if Path(v).exists()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "status": "completed",
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest
