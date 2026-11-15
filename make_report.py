"""Таблицы, графики и числа отчёта ЛР1 — только из каталогов запусков runs/ (и MLflow).

    python make_report.py

Берёт последний завершённый запуск каждой серии (exp1_ttft_vs_prompt, exp2_latency_vs_output, exp3_load),
пересчитывает метрики из results.jsonl и пишет report/numbers.json, report/*.csv и рисунки report/fig_*.png.
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from lr1.metrics import COMPUTE, prefill_speed  # noqa: E402
from lr1.stats import kv_bytes_per_token, linear_fit  # noqa: E402
from lr1.tracking import read_results, setup_mlflow  # noqa: E402

OUT = ROOT / "report"
SERIES = {"exp1": "exp1_ttft_vs_prompt", "exp2": "exp2_latency_vs_output", "exp3": "exp3_load"}
BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"


def latest_run(series):
    best = None
    for d in sorted(Path("runs").glob("*")):
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("series") == series and meta.get("status") == "completed":
            best = (d, meta)
    if best is None:
        sys.exit(f"Нет завершённого запуска серии {series}")
    return best


def f(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",") if x is not None else "—"


def load(exp):
    d, meta = latest_run(SERIES[exp])
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["experiment"]
    results = read_results(d)
    m = COMPUTE[exp](results, cfg)
    stored = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    for k in m:  # сверка: metrics.json совпадает с пересчётом из results.jsonl
        assert json.dumps(m[k], sort_keys=True) == json.dumps(stored[k], sort_keys=True), (exp, k)
    return d, meta, cfg, results, m


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def exp1(num):
    d, meta, cfg, results, m = load("exp1")
    num["exp1_run_id"] = meta["run_id"]
    rows, lens, ttft_med, ttft_lo, ttft_hi, pf = [], [], [], [], [], []
    xs, ys = [], []
    for L in cfg["lengths"]:
        e = m["by_length"][str(L)]
        t, s = e["ttft_s"], e["prefill_tok_s"]
        rows.append([L, int(e["prompt_tokens"]["median"]), f"{e['ok']}/{e['planned']}", f(t["median"]),
                     f"{f(t['min'])}–{f(t['max'])}", f(s["median"], 1), f"{f(s['min'], 1)}–{f(s['max'], 1)}"])
        lens.append(L); ttft_med.append(t["median"]); ttft_lo.append(t["min"]); ttft_hi.append(t["max"])
        pf.append(s["median"])
        num[f"exp1_ttft_{L}"] = f(t["median"]); num[f"exp1_prefill_{L}"] = f(s["median"], 1)
    meas = [r for r in results if r.get("kind") == "measure" and r["status"] == "ok"]
    xs = [r["prompt_tokens"] for r in meas]; ys = [r["ttft_s"] for r in meas]
    a, b = linear_fit(xs, ys)
    num["exp1_fit_intercept_s"] = f(a); num["exp1_fit_slope_ms_per_token"] = f(b * 1000, 1)
    write_csv("exp1_ttft.csv", ["length", "prompt_tokens", "ok/planned", "ttft_median_s", "ttft_range_s",
                                "prefill_tok_s_median", "prefill_range"], rows)
    num["exp1_table"] = rows
    pc = m.get("prefix_cache")
    if pc:
        num["exp1_cache_ttft"] = [f(x) for x in pc["ttft_s"]]
        num["exp1_cache_speedup_2"] = f(pc["speedup_2nd"], 1)
        num["exp1_cache_speedup_3"] = f(pc["speedup_3rd"], 1)
        num["exp1_cache_length"] = pc["length"]
    # рисунок 1
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    yerr = [[m_ - lo for m_, lo in zip(ttft_med, ttft_lo)], [hi - m_ for m_, hi in zip(ttft_med, ttft_hi)]]
    ax[0].errorbar(lens, ttft_med, yerr=yerr, fmt="o-", color=BLUE, capsize=4, label="медиана и диапазон (5 запросов)")
    ax[0].plot(lens, [a + b * x for x in lens], "--", color=ORANGE, label=f"прямая: {a:.1f} с + {b * 1000:.0f} мс/токен")
    ax[0].set_xlabel("длина запроса, токенов"); ax[0].set_ylabel("TTFT, с"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title("TTFT от длины запроса")
    ax[1].plot(lens, pf, "o-", color=GREEN)
    ax[1].set_xlabel("длина запроса, токенов"); ax[1].set_ylabel("prefill, токенов/с"); ax[1].grid(alpha=.3)
    ax[1].set_ylim(0, max(pf) * 1.25); ax[1].set_title("Скорость prefill = токены запроса / TTFT")
    fig.tight_layout(); fig.savefig(OUT / "fig_ttft_vs_prompt.png", dpi=150); plt.close(fig)


def exp2(num):
    d, meta, cfg, results, m = load("exp2")
    num["exp2_run_id"] = meta["run_id"]
    rows, xs, ys = [], [], []
    for n in cfg["output_lengths"]:
        e = m["by_output"][str(n)]
        rows.append([n, f"{e['ok']}/{e['planned']}", f"{e['output_tokens_exact']}/{e['ok']}", f(e["ttft_s"]["median"]),
                     f(e["tpot_s"]["median"], 3), f"{f(e['tpot_s']['min'], 3)}–{f(e['tpot_s']['max'], 3)}",
                     f(e["e2e_s"]["median"], 1), f"{f(e['e2e_s']['min'], 1)}–{f(e['e2e_s']['max'], 1)}"])
        num[f"exp2_tpot_{n}"] = f(e["tpot_s"]["median"], 3); num[f"exp2_e2e_{n}"] = f(e["e2e_s"]["median"], 1)
    meas = [r for r in results if r.get("kind") == "measure" and r["status"] == "ok"]
    xs = [r["output_tokens"] for r in meas]; ys = [r["e2e_s"] for r in meas]
    fit = m["fit"]
    num["exp2_fit_intercept_s"] = f(fit["intercept_s"], 1)
    num["exp2_fit_slope_s_per_token"] = f(fit["slope_s_per_token"], 3)
    num["exp2_ttft_median_all"] = f(statistics.median([r["ttft_s"] for r in meas]), 1)
    num["exp2_tpot_median_all"] = f(statistics.median([r["tpot_s"] for r in meas]), 3)
    weights_gb = Path("models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf").stat().st_size / 1e9
    bw = 46.9  # ГБ/с: DDR4-2933 в два канала, паспортная пропускная способность памяти i5-10300H
    num["bound_tpot_s"] = f(weights_gb / bw, 3); num["weights_gb"] = f(weights_gb, 2); num["mem_bw_gbs"] = f(bw, 1)
    num["exp2_tpot_vs_bound"] = f(statistics.median([r["tpot_s"] for r in meas]) / (weights_gb / bw), 1)
    write_csv("exp2_latency.csv", ["output_tokens", "ok/planned", "exact_len", "ttft_median_s", "tpot_median_s",
                                   "tpot_range_s", "e2e_median_s", "e2e_range_s"], rows)
    num["exp2_table"] = rows
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(xs, ys, color=BLUE, s=18, label="запросы (5 на длину)")
    xx = [0, max(xs)]
    ax.plot(xx, [fit["intercept_s"] + fit["slope_s_per_token"] * x for x in xx], "--", color=ORANGE,
            label=f"прямая: {fit['intercept_s']:.1f} с + {fit['slope_s_per_token'] * 1000:.0f} мс/токен")
    ax.set_xlabel("длина ответа, токенов"); ax.set_ylabel("полное время ответа, с"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Полное время ответа от длины ответа (запрос 1024 токена)")
    fig.tight_layout(); fig.savefig(OUT / "fig_latency_vs_output.png", dpi=150); plt.close(fig)


def exp3(num):
    d, meta, cfg, results, m = load("exp3")
    num["exp3_run_id"] = meta["run_id"]
    lim = cfg["limits"]
    rows, cs, thr, ttft90, tpot, good = [], [], [], [], [], []
    for c in cfg["concurrency_levels"]:
        e = m["levels"][str(c)]
        srv = e.get("server", {})
        mx = srv.get("max_during", {})
        rows.append([c, f"{e['ok']}/{e['planned']}", f(e["throughput_tok_s"], 2), f(e["rps"], 3), f(e["ttft_median_s"], 1),
                     f(e["ttft_p90_s"], 1), f(e["tpot_median_s"], 3), f(e["goodput_rps"], 3),
                     "да" if e["within_limits"] else "нет",
                     f"{mx.get('requests_processing')}/{mx.get('requests_deferred')}"])
        cs.append(c); thr.append(e["throughput_tok_s"]); ttft90.append(e["ttft_p90_s"]); tpot.append(e["tpot_median_s"])
        good.append(e["goodput_rps"])
        num[f"exp3_thr_{c}"] = f(e["throughput_tok_s"], 2); num[f"exp3_ttft90_{c}"] = f(e["ttft_p90_s"], 1)
        num[f"exp3_tpot_{c}"] = f(e["tpot_median_s"], 3); num[f"exp3_ok_{c}"] = e["within_limits"]
    write_csv("exp3_load.csv", ["clients", "ok/planned", "throughput_tok_s", "rps", "ttft_median_s", "ttft_p90_s",
                                "tpot_median_s", "goodput_rps", "within_limits", "max_processing/deferred"], rows)
    num["exp3_table"] = rows
    within = [c for c in cs if m["levels"][str(c)]["within_limits"]]
    num["exp3_max_within_limits"] = max(within) if within else 0
    num["exp3_limits"] = lim
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    ax[0].plot(cs, thr, "o-", color=BLUE); ax[0].set_ylabel("выходных токенов/с"); ax[0].set_title("Пропускная способность")
    ax[1].plot(cs, ttft90, "o-", color=ORANGE); ax[1].axhline(lim["ttft_p90_s"], color=RED, ls="--", label="предел")
    ax[1].set_ylabel("TTFT p90, с"); ax[1].set_title("TTFT (90-й процентиль)"); ax[1].legend(fontsize=8)
    ax[2].plot(cs, tpot, "o-", color=GREEN); ax[2].axhline(lim["tpot_median_s"], color=RED, ls="--", label="предел")
    ax[2].set_ylabel("медиана TPOT, с"); ax[2].set_title("TPOT (медиана)"); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("одновременных клиентов"); a.set_xticks(cs); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_load.png", dpi=150); plt.close(fig)


def memory(num):
    kv = kv_bytes_per_token()
    num["kv_bytes_per_token"] = kv
    num["kv_32k_gib"] = f(32768 * kv / 2**30, 1)
    num["kv_8x4000_gib"] = f(8 * 4000 * kv / 2**30, 2)
    num["kv_ctx8192_mib"] = f(8192 * kv / 2**20, 0)


def main():
    OUT.mkdir(exist_ok=True)
    num = {}
    memory(num); exp1(num); exp2(num); exp3(num)
    mlflow = setup_mlflow("lr1")
    df = mlflow.search_runs(experiment_names=["lr1"])
    df.to_csv(OUT / "mlflow_runs.csv", index=False)
    num["mlflow_n_runs"] = len(df)
    (OUT / "numbers.json").write_text(json.dumps(num, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in num.items() if not k.endswith("_table")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
