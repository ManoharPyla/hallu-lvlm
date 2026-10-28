"""Efficiency measurement: latency, peak memory, parameter count."""
from __future__ import annotations

import logging
import time
import tracemalloc
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)


class LatencyTimer:
    """Accumulates wall-clock latencies; use ``with timer: ...`` repeatedly."""

    def __init__(self):
        self.times: list[float] = []
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append(time.perf_counter() - self._t0)
        return False

    def summary(self) -> dict:
        if not self.times:
            return dict(n=0, mean_s=float("nan"), p50_s=float("nan"), p95_s=float("nan"),
                        total_s=0.0)
        a = np.asarray(self.times)
        return dict(n=len(a), mean_s=float(a.mean()), p50_s=float(np.median(a)),
                    p95_s=float(np.quantile(a, 0.95)), total_s=float(a.sum()))


@contextmanager
def peak_memory(device: str = "auto"):
    """Yields a dict filled on exit with ``peak_mb`` and ``backend`` ('cuda'|'tracemalloc').

    Note: tracemalloc only tracks Python-allocator memory; torch CPU tensor storage may be
    under-counted. Report the backend alongside the number.
    """
    out: dict = {}
    torch = None
    try:
        import torch  # noqa: F811
    except ImportError:
        pass
    use_cuda = torch is not None and torch.cuda.is_available() and device in ("auto", "cuda")
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        try:
            yield out
        finally:
            torch.cuda.synchronize()
            out.update(peak_mb=torch.cuda.max_memory_allocated() / 2**20, backend="cuda")
    else:
        started = not tracemalloc.is_tracing()
        if started:
            tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            yield out
        finally:
            _, peak = tracemalloc.get_traced_memory()
            if started:
                tracemalloc.stop()
            out.update(peak_mb=peak / 2**20, backend="tracemalloc")


def count_params(model, trainable_only: bool = False) -> int:
    """Parameter count for torch modules; sklearn linear models via coef_/intercept_."""
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters()
                       if p.requires_grad or not trainable_only))
    target = getattr(model, "model", model)  # e.g. wrapper around sklearn estimator
    n = 0
    for attr in ("coef_", "intercept_"):
        if hasattr(target, attr):
            n += int(np.asarray(getattr(target, attr)).size)
    return n
