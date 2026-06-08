"""Background system sampler: peak/avg memory during a run (no sudo needed).

macOS psutil exposes `wired` in virtual_memory(); we track total used + wired,
plus the engine process RSS (matched by pid or substring). Thermal throttling is
measured downstream as decode-rate decay over a soak, not via temp sensors.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
import psutil


@dataclass
class MemStats:
    samples: int = 0
    used_peak_gb: float = 0.0
    wired_peak_gb: float = 0.0
    proc_rss_peak_gb: float = 0.0


class Sampler:
    def __init__(self, proc_match: str | None = None, interval: float = 0.5):
        self.proc_match = proc_match
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self.stats = MemStats()

    def _find_rss(self) -> float:
        if not self.proc_match:
            return 0.0
        total = 0
        for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
            try:
                hay = " ".join([p.info["name"] or ""] + (p.info["cmdline"] or []))
                if self.proc_match in hay:
                    total += p.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / 1e9

    def _loop(self):
        while not self._stop.is_set():
            vm = psutil.virtual_memory()
            self.stats.samples += 1
            self.stats.used_peak_gb = max(self.stats.used_peak_gb, vm.used / 1e9)
            wired = getattr(vm, "wired", 0) / 1e9
            self.stats.wired_peak_gb = max(self.stats.wired_peak_gb, wired)
            self.stats.proc_rss_peak_gb = max(self.stats.proc_rss_peak_gb, self._find_rss())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
