"""Паспорт сервера report/server_passport.md — из metadata.json итоговых запусков (без ручного переписывания)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def latest(series):
    best = None
    for d in sorted(Path("runs").glob("*")):
        m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        if m.get("series") == series and m.get("status") == "completed":
            best = m
    return best


def main():
    m1 = latest("exp1_ttft_vs_prompt")
    m3 = latest("exp3_load_v2")
    s1, s3 = m1["server"], m3["server"]
    hw = s1["hardware"]
    kv = s1["log"]["memory_mib"]["context_kv"]
    lines = [
        "# Паспорт сервера ЛР1",
        "",
        "Собран `scripts/make_passport.py` из `runs/*/metadata.json` (запуски экспериментов 1 и 3).",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Сервер и версия | llama.cpp `llama-server`, сборка {s1['build_info']} (релиз b12406, Windows, CPU x64) |",
        f"| Модель | Qwen3-4B-Instruct-2507, файл `{s1['model_file']}` |",
        f"| Источник модели | {s1['model_source']} |",
        f"| Формат весов | {s1['weights_format']} ({s1['model_size_bytes'] / 1e9:.3f} ГБ) |",
        f"| SHA-256 файла модели | `{s1['model_sha256']}` (совпал с хешем на Hugging Face) |",
        "| Ревизия модели | репозиторий bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF, ревизия `ae44f08e1392f39c0e474af10c3ff8355c8b6688` (последнее изменение 2025-08-06; запрошена 2026-11-15, файл скачан 2026-11-15) |",
        f"| Предел контекста | {s1['log']['n_ctx_slot']} токенов на весь сервер при 1 слоте; при 4 слотах {s3['log']['n_ctx_slot']} токенов на слот (`kv_unified = {s3['log']['kv_unified']}`) |",
        f"| Ёмкость KV-кеша из журнала запуска | {kv} МиБ (`common_memory_breakdown_print`, столбец context) = 8192 токена × 144 КиБ |",
        f"| Память (МиБ): всего / модель / KV / расчётные буферы | {s1['log']['memory_mib']['host_total']} / {s1['log']['memory_mib']['model']} / {kv} / {s1['log']['memory_mib']['compute']} |",
        f"| Устройство | CPU {hw['cpu']}, {hw['cores']} ядра, {hw['logical_threads']} потоков, ОЗУ {hw['ram_gib']} ГиБ, без GPU |",
        f"| Потоки сервера | {s1['log']['n_threads']} (`-t 4`), расширения: AVX2, FMA, F16C, LLAMAFILE, OPENMP, REPACK |",
        f"| ОС | {hw['os']} |",
        "| Адрес | http://localhost:8080/v1 (`endpoint.yaml`) |",
        "| Генерация | температура 0 всегда передаётся явно; `ignore_eos` в экспериментах 2 и 3; кеш префикса: в экспериментах 1–2 выключен параметром запроса `cache_prompt: false` плюс уникальная метка в начале запроса, в эксперименте 3 включён (по умолчанию) |",
        "",
        "Команда запуска (эксперименты 1 и 2, 1 слот):",
        "",
        "```",
        s1["command"],
        "```",
        "",
        "Команда запуска (эксперимент 3, 4 слота):",
        "",
        "```",
        s3["command"],
        "```",
        "",
        f"Журналы запуска (`-lv 4`): `logs/server_np1.log`, `logs/server_np4.log` (первая серия эксперимента 3: `logs/server_np4_first_attempt.log`) — в архиве артефактов `lr1_artifacts.zip`.",
        "",
        "Отклонение от методички: профиль CPU рекомендует контекст 16384; на ноутбуке с 8 ГиБ ОЗУ (при работающей ОС и приложениях свободно около 3–4 ГиБ) "
        "сервер с таким контекстом (KV-кеш 2,25 ГиБ) не помещается без обращения к файлу подкачки, поэтому контекст — 8192 (KV-кеш 1152 МиБ). "
        "Для запросов эксперимента 3 (1024 + 128 токенов) при 4 слотах хватает 2048 токенов на слот.",
        "",
    ]
    Path("report/server_passport.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
