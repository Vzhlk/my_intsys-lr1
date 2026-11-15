#!/usr/bin/env bash
# Итоговая серия ЛР1 (профиль CPU): эксперименты 1 и 2 на сервере с 1 слотом, эксперимент 3 — с 4 слотами.
# Сервер с 1 слотом уже запущен (команда CMD1); скрипт перезапускает его с 4 слотами для эксперимента 3.
# Запускать из корня репозитория, чистое git-дерево обязательно (итоговые запуски).
set -euo pipefail
PY=.venv/Scripts/python.exe
CMD1='tools/llama.cpp/llama-server.exe -m models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 8192 -np 1 -t 4 --port 8080 --metrics -lv 4'
CMD4='tools/llama.cpp/llama-server.exe -m models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 8192 -np 4 -t 4 --port 8080 --metrics -lv 4'

wait_ready() { for _ in $(seq 1 100); do curl -s localhost:8080/health | grep -q '"ok"' && return 0; sleep 3; done; return 1; }

wait_ready
$PY scripts/run_exp.py --exp 1 --server-log logs/server_np1.log --command "$CMD1"
$PY scripts/run_exp.py --exp 2 --server-log logs/server_np1.log --command "$CMD1"

taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
sleep 3
(nohup $CMD4 > logs/server_np4.log 2>&1 &)
wait_ready
$PY scripts/server_checks.py --server-log logs/server_np4.log
git add report/server_checks_np4.json
git commit -q -m "ЛР1: проверки сервера с 4 слотами

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
$PY scripts/run_exp.py --exp 3 --server-log logs/server_np4.log --command "$CMD4"
taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
echo "ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ"
