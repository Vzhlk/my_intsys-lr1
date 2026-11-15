"""Запуск одного эксперимента ЛР1: каталог запуска, паспорт сервера, журнал MLflow.

    python scripts/run_exp.py --exp 1 --server-log logs/server_np1.log --command "<команда запуска сервера>"
    python scripts/run_exp.py --exp 2 ...
    python scripts/run_exp.py --exp 3 ...    (сервер с 4 слотами)

Итоговый запуск требует чистого git-дерева; --debug пишет в runs_debug/ и эксперимент lr1_debug.
"""
import argparse
import asyncio
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from lr1.client import make_aclient, make_client, run_load, timed_chat  # noqa: E402
from lr1.metrics import COMPUTE  # noqa: E402
from lr1.prompts import ServerTokenizer, build_messages, load_book_tokens, sha256_text  # noqa: E402
from lr1.server import scrape_metrics, server_passport  # noqa: E402
from lr1.tracking import (RunDir, git_commit, git_dirty, library_versions, now_iso,  # noqa: E402
                          require_clean_tree, setup_mlflow)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Ctx:
    def __init__(self, ep, cfg, rd, tok, book, client, model):
        self.ep, self.cfg, self.rd, self.tok, self.book = ep, cfg, rd, tok, book
        self.client, self.model = client, model
        self.counter = 0

    def messages(self, target, label, offset=None):
        """Запрос нужной длины; сдвиг по книге зависит от номера запроса (тексты разные)."""
        self.counter += 1
        start = (offset if offset is not None else self.counter * 7919) % max(1, len(self.book) - target - 10)
        msgs, n = build_messages(self.tok, self.book, label, target, start)
        return msgs, n, start


def record(ctx, kind, rec, **fields):
    row = {"kind": kind, **fields, **rec}
    ctx.rd.append_result(row)
    return row


def warmup(ctx, target, label):
    """Прогревочный запрос (в статистику не входит): та же длина, ответ в один токен."""
    msgs, n, start = ctx.messages(target, label)
    rec = timed_chat(ctx.client, ctx.model, msgs, max_tokens=1, cache_prompt=False)
    record(ctx, "warmup", rec, planned_prompt_tokens=target, label=label, prompt_sha256=sha256_text(msgs[0]["content"]))
    log(f"прогрев {label}: ttft={rec['ttft_s']}")


def exp1(ctx):
    cfg = ctx.cfg
    for length in cfg["lengths"]:
        warmup(ctx, length, f"warm-e1-{length}")
        for i in range(cfg["repeats"]):
            label = f"e1-L{length}-{i}-{uuid.uuid4().hex[:6]}"
            msgs, n_planned_actual, start = ctx.messages(length, label)
            # кеш префикса на этой серии выключен: и параметром запроса, и уникальной меткой в начале
            rec = timed_chat(ctx.client, ctx.model, msgs, cfg["max_tokens"], cache_prompt=False)
            record(ctx, "measure", rec, length=length, i=i, label=label, planned_prompt_tokens=length,
                   tokenized_prompt_tokens=n_planned_actual, book_offset=start,
                   prompt_sha256=sha256_text(msgs[0]["content"]))
            log(f"L={length} #{i}: prompt={rec['prompt_tokens']} ttft={rec['ttft_s']} status={rec['status']}")
    # кеширование префикса: один и тот же запрос несколько раз подряд, кеш включён
    L = cfg["prefix_cache_length"]
    label = f"e1-cache-{uuid.uuid4().hex[:6]}"
    msgs, n, start = ctx.messages(L, label)
    for rep in range(1, cfg["prefix_cache_repeats"] + 1):
        rec = timed_chat(ctx.client, ctx.model, msgs, cfg["max_tokens"], cache_prompt=True)
        record(ctx, "cache_rep", rec, rep=rep, length=L, label=label, planned_prompt_tokens=L,
               book_offset=start, prompt_sha256=sha256_text(msgs[0]["content"]))
        log(f"cache rep {rep}: ttft={rec['ttft_s']}")


def exp2(ctx):
    cfg = ctx.cfg
    for n_out in cfg["output_lengths"]:
        warmup(ctx, cfg["prompt_tokens"], f"warm-e2-{n_out}")
        for i in range(cfg["repeats"]):
            label = f"e2-N{n_out}-{i}-{uuid.uuid4().hex[:6]}"
            msgs, n, start = ctx.messages(cfg["prompt_tokens"], label)
            rec = timed_chat(ctx.client, ctx.model, msgs, n_out, ignore_eos=True, cache_prompt=False)
            record(ctx, "measure", rec, output_len=n_out, i=i, label=label,
                   planned_prompt_tokens=cfg["prompt_tokens"], tokenized_prompt_tokens=n,
                   book_offset=start, prompt_sha256=sha256_text(msgs[0]["content"]))
            log(f"N={n_out} #{i}: out={rec['output_tokens']} e2e={rec['e2e_s']} tpot={rec['tpot_s']} {rec['status']}")


def sample_metrics(base_url, stop, samples):
    while not stop.is_set():
        try:
            samples.append({"t": time.time(), **scrape_metrics(base_url)})
        except Exception:
            pass
        stop.wait(1.0)


def exp3(ctx):
    cfg = ctx.cfg
    warmup(ctx, cfg["prompt_tokens"], "warm-e3")
    aclient = make_aclient(ctx.ep["base_url"])
    for c in cfg["concurrency_levels"]:
        n_req = max(cfg["min_requests"], 4 * c)
        jobs = []
        for j in range(n_req):
            label = f"e3-c{c}-{j}-{uuid.uuid4().hex[:6]}"
            msgs, n, start = ctx.messages(cfg["prompt_tokens"], label)
            jobs.append({"id": f"c{c}-{j}", "messages": msgs, "label": label, "start": start,
                         "tokenized": n, "sha": sha256_text(msgs[0]["content"])})
        before = scrape_metrics(ctx.ep["base_url"])
        stop, samples = threading.Event(), []
        th = threading.Thread(target=sample_metrics, args=(ctx.ep["base_url"], stop, samples), daemon=True)
        th.start()
        log(f"уровень c={c}: {n_req} запросов")
        results = asyncio.run(run_load(aclient, ctx.model, jobs, c, cfg["max_tokens"]))
        stop.set(); th.join()
        after = scrape_metrics(ctx.ep["base_url"])
        by_id = {j["id"]: j for j in jobs}
        for rec in sorted(results, key=lambda r: r["t_start"]):
            j = by_id[rec["job_id"]]
            record(ctx, "measure", rec, level=c, label=j["label"], planned_prompt_tokens=cfg["prompt_tokens"],
                   tokenized_prompt_tokens=j["tokenized"], book_offset=j["start"], prompt_sha256=j["sha"])
        keys = ("requests_processing", "requests_deferred", "kv_cache_usage_ratio")
        summary = {"before": {k: before.get(k) for k in keys}, "after": {k: after.get(k) for k in keys},
                   "max_during": {k: max([s.get(k, 0) for s in samples] or [None]) for k in keys},
                   "n_samples": len(samples), "available_keys": sorted(after)}
        record(ctx, "server_metrics", {}, level=c, summary=summary)
        okn = sum(1 for r in results if r["status"] == "ok")
        log(f"c={c}: готово, успешных {okn}/{n_req}; сервер: {summary['max_during']}")


EXPS = {"1": exp1, "2": exp2, "3": exp3}


def main():
    p = argparse.ArgumentParser(description="ЛР1: один эксперимент")
    p.add_argument("--exp", required=True, choices=list(EXPS))
    p.add_argument("--server-log", required=True)
    p.add_argument("--command", required=True, help="команда запуска сервера (со всеми флагами)")
    p.add_argument("--series-suffix", default="", help="суффикс имени серии (например, для повтора)")
    p.add_argument("--config", help="файл конфигурации эксперимента (по умолчанию configs/exp<N>.yaml)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    ep = yaml.safe_load(open("endpoint.yaml", encoding="utf-8"))
    cfg = yaml.safe_load(open(args.config or f"configs/exp{args.exp}.yaml", encoding="utf-8"))
    cfg["series"] += args.series_suffix
    commit, dirty = git_commit(), git_dirty()
    if not args.debug:
        require_clean_tree()

    passport = server_passport(ep["base_url"], args.server_log, ep["server"]["model_path"], args.command)
    if passport["total_slots"] != cfg["server_slots"]:
        sys.exit(f"Сервер запущен с {passport['total_slots']} слотами, а эксперимент требует {cfg['server_slots']}")

    runs_root = "runs_debug" if args.debug else "runs"
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg['series']}-{uuid.uuid4().hex[:6]}"
    rd = RunDir(runs_root, run_id)
    full_cfg = {"endpoint": ep, "experiment": cfg}
    rd.write_config(full_cfg)
    rd.update_metadata(
        run_id=run_id, experiment=cfg["experiment"], series=cfg["series"], status="running", debug=args.debug,
        started_at=now_iso(), finished_at=None, commit=commit, dirty_files=dirty.splitlines() if dirty else [],
        versions=library_versions(), command=" ".join(sys.argv), server=passport,
        endpoint=ep["base_url"], model=ep["model"], temperature=0,
        data_manifest_sha256=sha256_text(open("data/manifest.json", encoding="utf-8").read()))

    mlflow = setup_mlflow("lr1_debug" if args.debug else cfg["experiment"])
    with mlflow.start_run(run_name=run_id) as mlrun:
        mlflow.set_tags({"experiment": cfg["experiment"], "series": cfg["series"], "commit": commit, "run_id": run_id,
                         "model": passport["model_file"], "server": passport["build_info"]})
        mlflow.log_params({"exp": args.exp, "context": passport["log"].get("n_ctx_slot"),
                           "slots": passport["total_slots"], "threads": passport["log"].get("n_threads")})
        rd.update_metadata(mlflow_run_id=mlrun.info.run_id)
        try:
            tok = ServerTokenizer(ep["base_url"])
            book, _ = load_book_tokens(tok)
            ctx = Ctx(ep, cfg, rd, tok, book, make_client(ep["base_url"]), ep["model"])
            t0 = time.perf_counter()
            EXPS[args.exp](ctx)
            from lr1.tracking import read_results
            metrics = COMPUTE[f"exp{args.exp}"](read_results(rd.path), cfg)
            metrics["duration_s"] = time.perf_counter() - t0
            rd.write_metrics(metrics)
            rd.update_metadata(status="completed", finished_at=now_iso())
            mlflow.set_tag("status", "completed")
        except BaseException as exc:
            rd.update_metadata(status="failed", finished_at=now_iso(), error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc())
            mlflow.set_tag("status", "failed")
            mlflow.log_artifacts(str(rd.path))
            raise
        mlflow.log_artifacts(str(rd.path))
    print(f"Запуск завершён: {rd.path}")


if __name__ == "__main__":
    main()
