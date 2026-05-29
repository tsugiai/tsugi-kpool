"""Instrumentation for tokens-per-second, buffer-variance, and sideband
heartbeat health. Writes JSONL lines to `diagnostics_dir/sdk.jsonl`.

This is the data file that benchmark plots are generated from.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DiagnosticsWriter:
    """Append-only JSONL writer. One line per event."""

    def __init__(self, diagnostics_dir: str | None) -> None:
        self._enabled = diagnostics_dir is not None
        if not self._enabled:
            return
        path = Path(diagnostics_dir) if diagnostics_dir else None
        assert path is not None
        path.mkdir(parents=True, exist_ok=True)
        self._fh = open(path / f"sdk_pid{os.getpid()}.jsonl", "a")

    def emit(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        record = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._enabled:
            self._fh.close()
