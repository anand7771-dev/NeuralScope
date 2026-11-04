"""
app/instrumentation/profiler.py
────────────────────────────────
Lightweight wall-clock and memory profiler for the inference pipeline.

``InferenceProfiler`` is used inside the GPT-2 adapter to record per-stage
timings and memory snapshots. All times are in milliseconds; memory figures
are in MiB.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import psutil

__all__ = ["InferenceProfiler"]

logger = logging.getLogger(__name__)


class InferenceProfiler:
    """Records wall-clock timings and memory snapshots for inference stages.

    Usage::

        profiler = InferenceProfiler()
        t0 = profiler.start_stage("tokenization")
        # ... do work ...
        dur = profiler.end_stage("tokenization", t0)

        t1 = profiler.start_stage("layer_0")
        # ... run layer ...
        profiler.end_stage("layer_0", t1)

        summary = profiler.get_summary()
    """

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = {}   # stage -> [duration_ms, ...]
        self._snapshots: List[Dict[str, Any]] = []
        self._global_start: float = time.perf_counter()

    # ── Timing ────────────────────────────────────────────────────────────────

    def start_stage(self, name: str) -> float:
        """Mark the beginning of a named stage and return the start time.

        Args:
            name: Arbitrary label for the stage (e.g. ``"layer_3"``).

        Returns:
            High-resolution start time (seconds since epoch, via
            ``time.perf_counter()``).
        """
        return time.perf_counter()

    def end_stage(self, name: str, start_time: float) -> float:
        """Record the end of a stage and return elapsed milliseconds.

        Args:
            name:       Same label used in :meth:`start_stage`.
            start_time: Return value from :meth:`start_stage`.

        Returns:
            Duration in milliseconds.
        """
        duration_ms = (time.perf_counter() - start_time) * 1_000.0
        self._timings.setdefault(name, []).append(duration_ms)
        return duration_ms

    # ── Memory ────────────────────────────────────────────────────────────────

    def record_memory_snapshot(self) -> Dict[str, Any]:
        """Capture current RAM (and GPU if available) memory usage.

        Returns:
            A dict with keys:
            - ``ram_used_mb``    – Process RSS in MiB.
            - ``ram_total_mb``   – System total RAM in MiB.
            - ``ram_percent``    – Process RSS as % of system RAM.
            - ``gpu_allocated_mb`` – GPU memory allocated by PyTorch (or 0).
            - ``gpu_reserved_mb``  – GPU memory reserved by PyTorch (or 0).
            - ``timestamp_ms``   – Elapsed ms since profiler creation.
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        vm = psutil.virtual_memory()

        snapshot: Dict[str, Any] = {
            "ram_used_mb": mem_info.rss / (1024**2),
            "ram_total_mb": vm.total / (1024**2),
            "ram_percent": process.memory_percent(),
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "timestamp_ms": (time.perf_counter() - self._global_start) * 1_000.0,
        }

        # Optionally add CUDA stats
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                snapshot["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
                snapshot["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        except Exception:  # noqa: BLE001
            pass

        self._snapshots.append(snapshot)
        return snapshot

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a complete profiling summary.

        Returns:
            A dict with:
            - ``total_duration_ms``   – Wall-clock time since profiler creation.
            - ``stages``              – Per-stage timing breakdown.
            - ``memory_snapshots``    – All recorded memory snapshots.
        """
        total_ms = (time.perf_counter() - self._global_start) * 1_000.0

        stages: Dict[str, Any] = {}
        for stage, durations in self._timings.items():
            stages[stage] = {
                "calls": len(durations),
                "total_ms": sum(durations),
                "mean_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
            }

        return {
            "total_duration_ms": total_ms,
            "stages": stages,
            "memory_snapshots": self._snapshots,
        }
