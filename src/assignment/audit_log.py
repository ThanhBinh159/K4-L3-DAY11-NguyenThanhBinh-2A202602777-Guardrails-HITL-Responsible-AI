"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, tuple[dict, float]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the input and start timestamp under request_id or user_id."""
        key = request_id or user_id
        entry = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }
        self.logs.append(entry)
        self._open[key] = (entry, time.perf_counter())

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the response, decision, and elapsed time for this request."""
        key = request_id or user_id
        opened = self._open.pop(key, None)
        if opened is None:
            entry = {
                "request_id": request_id,
                "user_id": user_id,
                "started_at": None,
            }
            self.logs.append(entry)
            latency = 0.0
        else:
            entry, started = opened
            latency = max(0.0, time.perf_counter() - started)

        entry.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "completed_at": utc_now_iso(),
                "latency_seconds": round(latency, 6),
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target = Path(filepath) if filepath else Path(default_audit_log_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
