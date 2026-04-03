"""
utils.py — Shared utilities: JSONL I/O, timing, logging.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of dicts, skipping blank/bad lines."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                logging.getLogger(__name__).warning(
                    "Skipped malformed line %d in %s: %s", lineno, path, exc
                )
    return records


def save_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context-manager timer.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed)   # seconds as float
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - (self._start or 0.0)

    def split(self) -> float:
        """Elapsed time since the timer started (without stopping it)."""
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a simple stdout handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Path resolution (handles data/images/ → images/ mapping in this project)
# ---------------------------------------------------------------------------

def resolve_image_path(raw: str, jsonl_dir: Path | None = None) -> str:
    """
    Try multiple candidate locations for an image file and return the first
    that exists.  Falls back to the original string if nothing is found so
    that errors surface naturally when the file is actually opened.

    Strategy (in order):
      1. Exact path as provided.
      2. Relative to the directory containing the JSONL file.
      3. Strip a leading ``data/`` segment (matches this project's layout where
         JSONL stores ``data/images/X.png`` but images live in ``images/``).
      4. Bare filename looked up in ``images/``.
    """
    candidates: list[Path] = []

    p = Path(raw)
    candidates.append(p)

    if jsonl_dir is not None:
        candidates.append(jsonl_dir / p)

    # Strip leading "data/" — common mismatch in this dataset
    parts = p.parts
    if parts and parts[0] == "data":
        stripped = Path(*parts[1:])
        candidates.append(stripped)
        if jsonl_dir is not None:
            candidates.append(jsonl_dir / stripped)

    # Bare filename in images/
    candidates.append(Path("images") / p.name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Nothing found — return original and let the caller raise FileNotFoundError
    return raw
