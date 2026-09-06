"""CPU/RAM sampling for evaluation runs."""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any

import psutil


class ResourceMonitor:
    """Collect CPU and memory samples while a model inference runs."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self._samples: list[dict[str, float]] = []
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples = []
        self._event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def _collect_loop(self) -> None:
        while not self._event.is_set():
            self._samples.append(
                {
                    "cpu_percent": self.process.cpu_percent(interval=None),
                    "rss_bytes": self.process.memory_info().rss,
                    "timestamp": time.time(),
                }
            )
            time.sleep(self.interval)

    def stop(self) -> dict[str, float | None]:
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if not self._samples:
            self._samples = [{
                "cpu_percent": self.process.cpu_percent(interval=None),
                "rss_bytes": self.process.memory_info().rss,
                "timestamp": time.time(),
            }]

        cpu_values = [float(item["cpu_percent"]) for item in self._samples]
        memory_values = [float(item["rss_bytes"]) for item in self._samples]

        return {
            "avg_cpu_percent": statistics.mean(cpu_values) if cpu_values else None,
            "avg_memory_bytes": statistics.mean(memory_values) if memory_values else None,
            "peak_memory_bytes": max(memory_values) if memory_values else None,
            "peak_memory_mb": (max(memory_values) / (1024 * 1024)) if memory_values else None,
        }
