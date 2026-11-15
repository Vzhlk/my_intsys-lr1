"""Проверки с работающим сервером (п. 3.4 методички); результат — report/server_checks.json.

    python scripts/server_checks.py --server-log logs/server_np1.log
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from lr1.client import make_client, timed_chat  # noqa: E402
from lr1.prompts import ServerTokenizer, build_messages, load_book_tokens  # noqa: E402
from lr1.server import KV_KIB_PER_TOKEN, parse_server_log  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server-log", required=True)
    args = p.parse_args()
    ep = yaml.safe_load(open("endpoint.yaml", encoding="utf-8"))
    client, model = make_client(ep["base_url"]), ep["model"]
    out = {}

    # 1. max_tokens = 64 и ignore_eos возвращают ровно 64 токена
    msgs = [{"role": "user", "content": "Write a long story about a dragon."}]
    rec = timed_chat(client, model, msgs, max_tokens=64, ignore_eos=True)
    out["exact_64_tokens"] = {"completion_tokens": rec["output_tokens"], "ok": rec["output_tokens"] == 64}

    # 2. Один и тот же запрос при температуре 0 дважды подряд: совпали ли тексты
    msgs = [{"role": "user", "content": "List three primary colors."}]
    a = timed_chat(client, model, msgs, max_tokens=48, cache_prompt=False)
    b = timed_chat(client, model, msgs, max_tokens=48, cache_prompt=False)
    out["temperature_0_repeat"] = {"identical": a["text"] == b["text"], "sha256": [a["text_sha256"], b["text_sha256"]],
                                   "tokens": [a["output_tokens"], b["output_tokens"]]}

    # 3. Запрос заданной длины: число токенов по токенизатору сервера и по usage
    tok = ServerTokenizer(ep["base_url"])
    book, _ = load_book_tokens(tok)
    msgs, n = build_messages(tok, book, "check", 1024, 12345)
    rec = timed_chat(client, model, msgs, max_tokens=1, cache_prompt=False)
    out["prompt_length_1024"] = {"tokenized": n, "usage_prompt_tokens": rec["prompt_tokens"],
                                 "ok": rec["prompt_tokens"] == 1024 == n}

    # 4. Ёмкость KV-кеша из журнала запуска сравнивается с расчётом (144 КиБ на токен)
    info = parse_server_log(args.server_log)
    n_ctx_total = info["n_ctx_slot"] * (1 if info.get("kv_unified") == "true" else info["n_slots"])
    calc_mib = n_ctx_total * KV_KIB_PER_TOKEN / 1024
    log_mib = info["memory_mib"]["context_kv"]
    out["kv_cache"] = {"n_ctx_total_tokens": n_ctx_total, "calc_mib": calc_mib, "log_context_mib": log_mib,
                       "ok": abs(calc_mib - log_mib) <= 1.0}
    Path("report").mkdir(exist_ok=True)
    name = f"report/server_checks_np{info['n_slots']}.json"
    Path(name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
