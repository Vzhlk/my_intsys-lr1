"""Git, метаданные, каталог запуска и журнал MLflow (каркас ЛР0, адаптированный под ЛР1)."""
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

TRACKING_URI = "sqlite:///mlflow.db"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_commit():
    return _git("rev-parse", "HEAD")


def git_dirty():
    """Изменённые и новые неигнорируемые файлы (пусто — дерево чистое)."""
    return _git("status", "--porcelain")


def require_clean_tree():
    """Итоговый запуск только из закоммиченного кода."""
    dirty = git_dirty()
    if dirty:
        raise SystemExit("Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty
                         + "\n(для отладки добавьте --debug)")
    return git_commit()


def library_versions():
    libs = ["openai", "httpx", "numpy", "pandas", "matplotlib", "mlflow", "pyyaml"]
    return {"python": platform.python_version(), **{lib: version(lib) for lib in libs}}


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class RunDir:
    """runs/<run_id>/: config.yaml, metadata.json, results.jsonl, metrics.json."""

    def __init__(self, root, run_id):
        self.path = Path(root) / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {}

    def write_config(self, cfg):
        import yaml
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    def update_metadata(self, **fields):
        self.metadata.update(fields)
        (self.path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_result(self, rec):
        """Результат дописывается сразу: при сбое записанное остаётся на диске."""
        with open(self.path / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def write_metrics(self, metrics):
        (self.path / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def read_results(run_path):
    with open(Path(run_path) / "results.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def setup_mlflow(experiment):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow
