from __future__ import annotations
import hashlib, json, pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def load_or_run(cache_dir: Path, stage: str, params: dict, fn: Callable[[], T]) -> T:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stage}_{_hash({**params, 'stage': stage})}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)  # type: ignore[return-value]
    result = fn()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result
