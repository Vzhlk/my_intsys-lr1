"""Дополнительная проверка (вне итоговых серий): одинаков ли текст при температуре 0, если запрос идёт один и если параллельно с другими.

    python scripts/extra_temp0_load.py        (сервер с 4 слотами на :8080)

Результат — report/extra_temp0_check.json (в отчёт ЛР1 не входит; нужен для ответа на вопрос 6 к защите).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from lr1.client import atimed_chat, make_aclient, make_client, timed_chat  # noqa: E402

URL = "http://localhost:8080/v1"
TARGET = [{"role": "user", "content": "Explain in four sentences why the sky is blue."}]
OTHERS = [[{"role": "user", "content": q}] for q in (
    "Name five rivers in Europe and one fact about each.",
    "Write a short poem about a lighthouse.",
    "List the steps to boil an egg.")]


async def concurrent(n_tokens):
    c = make_aclient(URL)
    try:
        jobs = [atimed_chat(c, "x", TARGET, n_tokens, cache_prompt=False)]
        jobs += [atimed_chat(c, "x", m, n_tokens, cache_prompt=False) for m in OTHERS]
        jobs += [atimed_chat(c, "x", TARGET, n_tokens, cache_prompt=False)]
        return await asyncio.gather(*jobs)
    finally:
        await c.close()


def main():
    n = 96
    c = make_client(URL)
    alone = [timed_chat(c, "x", TARGET, n, cache_prompt=False) for _ in range(2)]
    conc = asyncio.run(concurrent(n))
    target_conc = [conc[0], conc[-1]]
    out = {
        "tokens": n,
        "alone_1_vs_alone_2_identical": alone[0]["text"] == alone[1]["text"],
        "alone_vs_concurrent_identical": [alone[0]["text"] == r["text"] for r in target_conc],
        "concurrent_copies_identical": target_conc[0]["text"] == target_conc[1]["text"],
        "sha256_alone": [r["text_sha256"] for r in alone],
        "sha256_concurrent": [r["text_sha256"] for r in target_conc],
        "output_tokens": [r["output_tokens"] for r in alone + target_conc],
    }
    first_diff = None
    for r in target_conc:
        a, b = alone[0]["text"], r["text"]
        if a != b:
            first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            out.setdefault("first_difference_char_index", []).append(first_diff)
    Path("report/extra_temp0_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
