"""Resource usage monitoring for FinQuery evaluation runs."""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Callable

import psutil


class ResourceMonitor:
    """Sample CPU and memory usage during model execution."""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self._stop_event = threading.Event()
        self._samples: list[dict[str, float]] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._samples = []
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._samples.append(
                {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(interval=None),
                    "rss_bytes": self.process.memory_info().rss,
                }
            )
            time.sleep(self.interval)

    def stop(self) -> dict[str, float | None]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if not self._samples:
            cpu = self.process.cpu_percent(interval=None)
            rss = self.process.memory_info().rss
            self._samples = [{"timestamp": time.time(), "cpu_percent": cpu, "rss_bytes": rss}]

        avg_cpu = statistics.mean(sample["cpu_percent"] for sample in self._samples)
        peak_memory = max(sample["rss_bytes"] for sample in self._samples)
        avg_memory = statistics.mean(sample["rss_bytes"] for sample in self._samples)

        return {
            "avg_cpu_percent": float(avg_cpu),
            "peak_memory_bytes": float(peak_memory),
            "avg_memory_bytes": float(avg_memory),
        }

    def run(self, func: Callable[[], Any]) -> tuple[Any, dict[str, float | None]]:
        self.start()
        try:
            result = func()
            return result, self.stop()
        finally:
            self.stop()
