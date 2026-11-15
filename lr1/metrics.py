"""Сводные показатели экспериментов — только из записей results.jsonl (знаменатель — все запланированные записи)."""
import statistics

from .stats import linear_fit, median_range, percentile


def _measured(results, kind="measure"):
    return [r for r in results if r.get("kind") == kind]


def _ok(rs):
    return [r for r in rs if r.get("status") == "ok"]


def prefill_speed(rec):
    """Токенов запроса в секунду: prompt_tokens / TTFT (в TTFT входит и первый шаг decode)."""
    if rec.get("status") != "ok" or not rec.get("ttft_s"):
        return None
    return rec["prompt_tokens"] / rec["ttft_s"]


def metrics_exp1(results, cfg):
    out = {"by_length": {}}
    meas = _measured(results)
    for length in cfg["lengths"]:
        rs = [r for r in meas if r["length"] == length]
        ok = _ok(rs)
        out["by_length"][str(length)] = {
            "planned": cfg["repeats"], "recorded": len(rs), "ok": len(ok), "error": len(rs) - len(ok),
            "prompt_tokens": median_range([r["prompt_tokens"] for r in ok]),
            "ttft_s": median_range([r["ttft_s"] for r in ok]),
            "prefill_tok_s": median_range([prefill_speed(r) for r in ok]),
        }
    cache = sorted(_measured(results, "cache_rep"), key=lambda r: r["rep"])
    if cache and all(r["status"] == "ok" for r in cache):
        t = [r["ttft_s"] for r in cache]
        out["prefix_cache"] = {"length": cfg["prefix_cache_length"], "ttft_s": t,
                               "speedup_2nd": t[0] / t[1], "speedup_3rd": t[0] / t[2] if len(t) > 2 else None}
    return out


def metrics_exp2(results, cfg):
    out = {"by_output": {}}
    meas = _measured(results)
    all_x, all_y = [], []
    for n in cfg["output_lengths"]:
        rs = [r for r in meas if r["output_len"] == n]
        ok = _ok(rs)
        exact = sum(1 for r in ok if r["output_tokens"] == n)
        out["by_output"][str(n)] = {
            "planned": cfg["repeats"], "recorded": len(rs), "ok": len(ok), "error": len(rs) - len(ok),
            "output_tokens_exact": exact,
            "ttft_s": median_range([r["ttft_s"] for r in ok]),
            "tpot_s": median_range([r["tpot_s"] for r in ok]),
            "e2e_s": median_range([r["e2e_s"] for r in ok]),
        }
        all_x += [r["output_tokens"] for r in ok]
        all_y += [r["e2e_s"] for r in ok]
    if len(set(all_x)) > 1:
        a, b = linear_fit(all_x, all_y)
        out["fit"] = {"intercept_s": a, "slope_s_per_token": b, "n_points": len(all_x)}
    return out


def metrics_exp3(results, cfg):
    lim = cfg["limits"]
    out = {"limits": lim, "levels": {}}
    meas = _measured(results)
    for c in cfg["concurrency_levels"]:
        rs = [r for r in meas if r["level"] == c]
        ok = _ok(rs)
        planned = max(cfg["min_requests"], 4 * c)
        entry = {"planned": planned, "recorded": len(rs), "ok": len(ok), "error": len(rs) - len(ok)}
        if ok:
            wall = max(r["t_end"] for r in rs) - min(r["t_start"] for r in rs)
            n_out = sum(r["output_tokens"] for r in ok)
            good = [r for r in ok if r["ttft_s"] <= lim["ttft_p90_s"] and (r["tpot_s"] or 0) <= lim["tpot_median_s"]]
            ttfts = [r["ttft_s"] for r in ok]
            tpots = [r["tpot_s"] for r in ok]
            entry.update({
                "wall_s": wall, "throughput_tok_s": n_out / wall, "rps": len(ok) / wall,
                "ttft_median_s": statistics.median(ttfts), "ttft_p90_s": percentile(ttfts, 90),
                "tpot_median_s": statistics.median(tpots), "goodput_rps": len(good) / wall,
                "within_limits": bool(percentile(ttfts, 90) <= lim["ttft_p90_s"]
                                      and statistics.median(tpots) <= lim["tpot_median_s"]),
            })
        srv = [r for r in _measured(results, "server_metrics") if r["level"] == c]
        if srv:
            entry["server"] = srv[0]["summary"]
        out["levels"][str(c)] = entry
    return out


COMPUTE = {"exp1": metrics_exp1, "exp2": metrics_exp2, "exp3": metrics_exp3}
