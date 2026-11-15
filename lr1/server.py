"""Паспорт сервера: что запущено, с какими флагами, сколько памяти ушло на KV-кеш."""
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import httpx

KV_KIB_PER_TOKEN = 144  # Qwen3-4B-Instruct-2507: 2 * 36 слоёв * 8 голов KV * 128 * 2 байта = 147 456 байт


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def cached_model_sha256(model_path):
    """SHA-256 файла модели (считается один раз и кладётся рядом в *.sha256)."""
    model_path = Path(model_path)
    cache = model_path.with_suffix(model_path.suffix + ".sha256")
    if cache.exists():
        return cache.read_text().split()[0]
    digest = sha256_file(model_path)
    cache.write_text(f"{digest}  {model_path.name}\n")
    return digest


def parse_server_log(log_path):
    """Из журнала запуска: слоты, контекст слота, разбивка памяти (MiB), число потоков."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    info = {}
    m = re.search(r"n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'", text)
    if m:
        info.update(n_slots=int(m.group(1)), n_ctx_slot=int(m.group(2)), kv_unified=m.group(3))
    m = re.search(r"\|\s+- Host\s+\|\s+(\d+) =\s+(\d+) \+\s+(\d+) \+\s+(\d+)", text)
    if m:
        info["memory_mib"] = {"host_total": int(m.group(1)), "model": int(m.group(2)),
                              "context_kv": int(m.group(3)), "compute": int(m.group(4))}
    m = re.search(r"n_threads = (\d+) \(n_threads_batch = (\d+)\)", text)
    if m:
        info["n_threads"] = int(m.group(1))
    m = re.search(r"system_info:.*", text)
    if m:
        info["system_info"] = m.group(0).split("system_info:")[1].strip()
    return info


def hardware():
    """Процессор, память и ОС (через PowerShell/CIM; на другой ОС — platform)."""
    info = {"os": platform.platform(), "machine": platform.machine()}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$c=Get-CimInstance Win32_Processor; $m=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
             "ConvertTo-Json @{cpu=$c.Name; cores=$c.NumberOfCores; threads=$c.NumberOfLogicalProcessors; ram_bytes=$m}"],
            capture_output=True, text=True, timeout=60)
        d = json.loads(out.stdout)
        info.update(cpu=d["cpu"], cores=d["cores"], logical_threads=d["threads"],
                    ram_gib=round(d["ram_bytes"] / 2**30, 2))
    except Exception:
        info["cpu"] = platform.processor()
    return info


def server_passport(base_url, log_path, model_path, command):
    base = base_url.rstrip("/").removesuffix("/v1")
    http = httpx.Client(timeout=30)
    props = http.get(f"{base}/props").json()
    model_path = Path(model_path)
    passport = {
        "server": "llama.cpp llama-server",
        "build_info": props.get("build_info"),
        "total_slots": props.get("total_slots"),
        "command": command,
        "model_file": model_path.name,
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": cached_model_sha256(model_path),
        "model_source": "huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF (файл Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf)",
        "weights_format": "GGUF Q4_K_M",
        "log": parse_server_log(log_path),
        "hardware": hardware(),
        "default_generation_settings_n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
    }
    return passport


def scrape_metrics(base_url):
    """Метрики Prometheus сервера llama.cpp (/metrics): очередь и загрузка KV-кеша."""
    base = base_url.rstrip("/").removesuffix("/v1")
    r = httpx.get(f"{base}/metrics", timeout=10)
    out = {}
    for line in r.text.splitlines():
        if line.startswith("llamacpp:") and " " in line:
            k, v = line.rsplit(" ", 1)
            try:
                out[k.replace("llamacpp:", "")] = float(v)
            except ValueError:
                pass
    return out
