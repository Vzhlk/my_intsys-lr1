"""Метрики запросов. Всё считается из записей results.jsonl."""
import statistics

import numpy as np


def ttft_tpot(t0, t1, tn, n_tokens):
    """TTFT = t1 - t0; TPOT = (tN - t1) / (N - 1) (None при N <= 1)."""
    ttft = t1 - t0
    tpot = (tn - t1) / (n_tokens - 1) if n_tokens > 1 else None
    return ttft, tpot


def median_range(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)}


def percentile(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def linear_fit(xs, ys):
    """Прямая y = a + b x по методу наименьших квадратов: (a, b)."""
    b, a = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return float(a), float(b)


def kv_bytes_per_token(n_layers=36, n_kv_heads=8, head_dim=128, dtype_bytes=2):
    """2 (K и V) x слои x головы KV x размерность головы x байт на число."""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
