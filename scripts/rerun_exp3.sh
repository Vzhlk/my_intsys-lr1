#!/usr/bin/env bash
# Повтор эксперимента 3 после исправления ошибки клиента (серия exp3_load_v2): те же конфигурация и сервер с 4 слотами.
set -euo pipefail
PY=.venv/Scripts/python.exe
CMD4='tools/llama.cpp/llama-server.exe -m models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 8192 -np 4 -t 4 --port 8080 --metrics -lv 4'
mv logs/server_np4.log logs/server_np4_first_attempt.log
(nohup $CMD4 > logs/server_np4.log 2>&1 &)
for _ in $(seq 1 100); do curl -s localhost:8080/health | grep -q '"ok"' && break; sleep 3; done
$PY scripts/run_exp.py --exp 3 --series-suffix _v2 --server-log logs/server_np4.log --command "$CMD4"
taskkill //F //IM llama-server.exe > /dev/null 2>&1 || true
echo "EXP3 V2 DONE"
