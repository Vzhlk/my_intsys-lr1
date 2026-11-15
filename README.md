# ЛР1. Инференс и развёртывание языковой модели: от одного запроса к серверу под нагрузкой

Курс «Интеллектуальные системы», задание: `LR1_llm_inference_serving.pdf` (репозиторий курса `dm780009/intsys`).

**Вопрос работы:** из чего складывается время ответа языковой модели и как оно меняется, когда пользователей становится много.
Прогноз — [hypothesis.md](hypothesis.md). Отчёт — `report/REPORT.md` (и `Лаба_1_отчет.docx` рядом с репозиторием).

## Профиль

CPU: `llama.cpp` (`llama-server`, сборка b12406), модель Qwen3-4B-Instruct-2507, GGUF Q4_K_M, контекст 8192 (методичка предлагает 16384;
на ноутбуке с 8 ГБ ОЗУ он не помещается), 1 слот в экспериментах 1–2 и 4 слота в эксперименте 3, 4 потока. Паспорт сервера, команды запуска
и журналы — `report/server_passport.md`, `report/server_np*.log`; адрес и параметры генерации — [endpoint.yaml](endpoint.yaml).

## Структура

| Файл | Что делает |
|------|------------|
| `prepare_data.py` | из `data/pg1342.txt` (Project Gutenberg № 1342) вырезает текст, пишет `data/manifest.json` с SHA-256 |
| `lr1/client.py` | клиент-измеритель: потоковый запрос, `t0`, `t1`, `tN`, usage, TTFT, TPOT; асинхронная нагрузка |
| `lr1/prompts.py` | запросы заданной длины (токенизатор и шаблон диалога берутся у сервера) с уникальной меткой в начале |
| `lr1/metrics.py`, `lr1/stats.py` | сводки из `results.jsonl`: медианы, диапазоны, p90, пропускная способность, goodput |
| `lr1/server.py` | паспорт сервера, разбор журнала запуска, метрики `/metrics` |
| `lr1/tracking.py` | каркас ЛР0: git, каталог запуска, MLflow |
| `scripts/run_exp.py` | один эксперимент (`--exp 1|2|3`) → каталог `runs/<run_id>/`, запись MLflow |
| `scripts/server_checks.py` | проверки с работающим сервером (64 токена, температура 0, длина запроса, KV-кеш) |
| `make_report.py` | таблицы, графики и числа отчёта из `runs/` |
| `configs/exp*.yaml` | параметры экспериментов; пределы интерактивности — в `exp3.yaml` |
| `tests/` | проверки без модели |

Каталог запуска `runs/<run_id>/`: `config.yaml`, `metadata.json` (коммит, версии, паспорт сервера, статус), `results.jsonl` (запись на каждый запрос),
`metrics.json` (пересчитано из `results.jsonl`).

## Установка и запуск (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py
python -m pytest -q
```

Скачать (источник и хеши — в `report/server_passport.md` и `data/manifest.json`):
- `llama-b12406-bin-win-cpu-x64.zip` из релиза b12406 llama.cpp (github.com/ggml-org/llama.cpp) → `tools/llama.cpp/`;
- `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF) → `models/`;
- `https://www.gutenberg.org/cache/epub/1342/pg1342.txt` → `data/pg1342.txt`.

Эксперименты 1 и 2 (1 слот):

```powershell
.\tools\llama.cpp\llama-server.exe -m models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 8192 -np 1 -t 4 --port 8080 --metrics -lv 4 *> logs\server_np1.log
python scripts\server_checks.py --server-log logs\server_np1.log
python scripts\run_exp.py --exp 1 --server-log logs\server_np1.log --command "<команда запуска выше>"
python scripts\run_exp.py --exp 2 --server-log logs\server_np1.log --command "<команда запуска выше>"
```

Эксперимент 3 (4 слота, тот же сервер с `-np 4`, журнал `logs\server_np4.log`):

```powershell
python scripts\run_exp.py --exp 3 --server-log logs\server_np4.log --command "<команда запуска с -np 4>"
```

Итоговые запуски работают только из закоммиченного кода (иначе отказ); `--debug` — отладка (`runs_debug/`).
Отчёт: `python make_report.py`.

Артефакты итоговых запусков (`runs/`, `mlflow.db`, `mlruns/`, журналы сервера) не входят в git (`.gitignore`) и сдаются архивом `lr1_artifacts.zip`.
