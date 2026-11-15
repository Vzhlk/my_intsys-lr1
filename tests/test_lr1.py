"""Проверки реализации ЛР1 без сервера модели (их результаты в таблицы отчёта не входят)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lr1.metrics import metrics_exp3, prefill_speed  # noqa: E402
from lr1.server import KV_KIB_PER_TOKEN, parse_server_log  # noqa: E402
from lr1.stats import kv_bytes_per_token, linear_fit, median_range, percentile, ttft_tpot  # noqa: E402


def test_ttft_tpot_known_example():
    """t0 = 0, t1 = 0,4 с, tN = 2,4 с, N = 101 -> TTFT 400 мс, TPOT 20 мс (в знаменателе N - 1)."""
    ttft, tpot = ttft_tpot(0.0, 0.4, 2.4, 101)
    assert ttft == pytest.approx(0.4)
    assert tpot == pytest.approx(0.020)


def test_tpot_single_token_is_none():
    assert ttft_tpot(0.0, 0.5, 0.5, 1)[1] is None


def test_kv_bytes_per_token_qwen3_4b():
    assert kv_bytes_per_token() == 147456            # 2 * 36 * 8 * 128 * 2
    assert kv_bytes_per_token() == KV_KIB_PER_TOKEN * 1024


def test_kv_32k_context_is_4_5_gib():
    assert 32768 * kv_bytes_per_token() / 2**30 == pytest.approx(4.5)


def test_median_range_and_percentile():
    r = median_range([1.0, 3.0, 2.0, None])
    assert (r["median"], r["min"], r["max"], r["n"]) == (2.0, 1.0, 3.0, 3)
    assert percentile(list(range(1, 11)), 90) == pytest.approx(9.1)


def test_linear_fit_recovers_line():
    a, b = linear_fit([16, 64, 256], [0.5 + 0.02 * x for x in (16, 64, 256)])
    assert (a, b) == (pytest.approx(0.5), pytest.approx(0.02))


def test_prefill_speed_from_prompt_tokens_and_ttft():
    assert prefill_speed({"status": "ok", "prompt_tokens": 1000, "ttft_s": 20.0}) == pytest.approx(50.0)
    assert prefill_speed({"status": "error", "prompt_tokens": 1000, "ttft_s": 20.0}) is None


def test_failed_requests_stay_in_denominator():
    """Неуспешные запросы не исчезают из знаменателя: planned/recorded/error считаются явно."""
    cfg = {"limits": {"ttft_p90_s": 40.0, "tpot_median_s": 0.3}, "concurrency_levels": [2], "min_requests": 4}
    ok = {"kind": "measure", "level": 2, "status": "ok", "ttft_s": 1.0, "tpot_s": 0.1, "output_tokens": 10,
          "t_start": 0.0, "t_end": 5.0}
    bad = {"kind": "measure", "level": 2, "status": "error", "ttft_s": None, "tpot_s": None, "output_tokens": None,
           "t_start": 1.0, "t_end": 2.0}
    m = metrics_exp3([ok, ok, ok, bad], cfg)["levels"]["2"]
    assert (m["planned"], m["recorded"], m["ok"], m["error"]) == (8, 4, 3, 1)
    assert m["throughput_tok_s"] == pytest.approx(30 / 5.0)
    assert m["goodput_rps"] == pytest.approx(3 / 5.0)


def test_parse_server_log(tmp_path):
    log = tmp_path / "s.log"
    log.write_text(
        "srv    load_model: initializing, n_slots = 4, n_ctx_slot = 2048, kv_unified = 'false'\n"
        "common_memory_breakdown_print: |   - Host               |                 1934 =   692 +    1152 +      90   |\n"
        "common_param: system_info: n_threads = 4 (n_threads_batch = 4) / 8 | CPU : AVX2 = 1 |\n", encoding="utf-8")
    info = parse_server_log(log)
    assert info["n_slots"] == 4 and info["n_ctx_slot"] == 2048
    assert info["memory_mib"] == {"host_total": 1934, "model": 692, "context_kv": 1152, "compute": 90}
    assert info["n_threads"] == 4


def test_dirty_tree_refuses_final_run(tmp_path):
    """Если в репозитории есть незакоммиченный файл, итоговый запуск отказывается стартовать."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from lr1.tracking import require_clean_tree; require_clean_tree()" % ROOT
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "x.py").write_text("print(1)\n")
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode != 0
    assert "незакоммиченные" in (res.stderr + res.stdout)
