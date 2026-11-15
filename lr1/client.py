"""Клиент-измеритель: потоковый запрос с отметками времени (асинхронный вариант — для нагрузки)."""
import asyncio
import hashlib
import time

from openai import AsyncOpenAI, OpenAI

from .stats import ttft_tpot


def make_client(base_url, timeout=3600):
    return OpenAI(base_url=base_url, api_key="none", timeout=timeout, max_retries=0)


def make_aclient(base_url, timeout=3600):
    return AsyncOpenAI(base_url=base_url, api_key="none", timeout=timeout, max_retries=0)


def _record(t0, t1, t_last, usage, text, finish, params):
    n = usage.completion_tokens if usage else None
    ttft, tpot = (None, None)
    if t1 is not None and n:
        ttft, tpot = ttft_tpot(t0, t1, t_last, n)
    return {
        "status": "ok" if (usage and t1 is not None) else "error",
        "ttft_s": ttft, "tpot_s": tpot,
        "e2e_s": (t_last - t0) if t_last is not None else None,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": n,
        "finish_reason": finish,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "params": params,
    }


def _extra(ignore_eos, cache_prompt):
    extra = {}
    if ignore_eos:
        extra["ignore_eos"] = True
    if cache_prompt is not None:
        extra["cache_prompt"] = cache_prompt
    return extra


def timed_chat(client, model, messages, max_tokens, ignore_eos=False, cache_prompt=None, **params):
    """Один потоковый запрос. Температура всегда передаётся явно (по умолчанию 0)."""
    params = {"temperature": 0, **params}
    extra = _extra(ignore_eos, cache_prompt)
    t0 = time.perf_counter()
    t1 = t_last = None
    usage, parts, finish = None, [], None
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=extra, **params)
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
                piece = chunk.choices[0].delta.content
                if piece:
                    now = time.perf_counter()
                    t1 = t1 or now
                    t_last = now
                    parts.append(piece)
    except Exception as exc:  # ошибка запроса — неуспешная запись, а не пропуск
        rec = _record(t0, t1, t_last, usage, "".join(parts), finish, {**params, **extra})
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return rec
    return _record(t0, t1, t_last, usage, "".join(parts), finish, {**params, **extra})


async def atimed_chat(client, model, messages, max_tokens, ignore_eos=False, cache_prompt=None, **params):
    params = {"temperature": 0, **params}
    extra = _extra(ignore_eos, cache_prompt)
    t0 = time.perf_counter()
    t1 = t_last = None
    usage, parts, finish = None, [], None
    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=extra, **params)
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                if chunk.choices[0].finish_reason:
                    finish = chunk.choices[0].finish_reason
                piece = chunk.choices[0].delta.content
                if piece:
                    now = time.perf_counter()
                    t1 = t1 or now
                    t_last = now
                    parts.append(piece)
    except Exception as exc:
        rec = _record(t0, t1, t_last, usage, "".join(parts), finish, {**params, **extra})
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rec["t_start"], rec["t_end"] = t0, time.perf_counter()
        return rec
    rec = _record(t0, t1, t_last, usage, "".join(parts), finish, {**params, **extra})
    rec["t_start"], rec["t_end"] = t0, (t_last if t_last is not None else time.perf_counter())
    return rec


async def run_load(aclient, model, jobs, concurrency, max_tokens, cache_prompt=None):
    """c клиентов берут запросы из общей очереди: каждый шлёт следующий, получив ответ на предыдущий."""
    queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    results = []

    async def worker():
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            rec = await atimed_chat(aclient, model, job["messages"], max_tokens,
                                    ignore_eos=True, cache_prompt=cache_prompt)
            rec["job_id"] = job["id"]
            results.append(rec)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results
