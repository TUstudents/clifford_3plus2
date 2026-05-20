"""Generic result persistence helpers for simulation sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def save_npz_json(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Save arrays to ``.npz`` and metadata to a same-stem JSON sidecar."""

    npz_path = Path(path)
    if npz_path.suffix != ".npz":
        raise ValueError("simulation result path must end with .npz")
    json_path = npz_path.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **{key: np.asarray(value) for key, value in arrays.items()})
    json_path.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def load_json_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata written by :func:`save_npz_json`."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
