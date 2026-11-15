"""Данные ЛР1: текст «Pride and Prejudice» (Project Gutenberg № 1342) для запросов заданной длины.

    python prepare_data.py

Из data/pg1342.txt (скачан с https://www.gutenberg.org/cache/epub/1342/pg1342.txt) вырезается
основной текст между маркерами START и END, переводы строк приводятся к \\n. Хеши исходного
файла и результата пишутся в data/manifest.json (проверка воспроизводимости).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "pg1342.txt"
BODY = ROOT / "data" / "pride_and_prejudice.txt"
MANIFEST = ROOT / "data" / "manifest.json"
URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    raw_bytes = RAW.read_bytes()
    text = raw_bytes.decode("utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\n", text)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not start or not end:
        sys.exit("Не найдены маркеры START/END Project Gutenberg")
    body = text[start.end():end.start()].strip() + "\n"
    BODY.write_text(body, encoding="utf-8", newline="\n")
    body_bytes = BODY.read_bytes()
    manifest = {
        "datasets": {
            "pride_and_prejudice": {
                "source": "Project Gutenberg eBook № 1342, Jane Austen, Pride and Prejudice (общественное достояние)",
                "url": URL,
                "raw_file": "data/pg1342.txt",
                "raw_sha256": sha256_bytes(raw_bytes),
                "raw_size_bytes": len(raw_bytes),
                "selection_rule": "текст между строками «*** START OF …» и «*** END OF …», "
                                  "\\r\\n -> \\n, обрезка пробелов по краям",
                "body_file": "data/pride_and_prejudice.txt",
                "body_sha256": sha256_bytes(body_bytes),
                "body_size_bytes": len(body_bytes),
                "body_chars": len(body),
                "records_total": 1,
                "records_used": 1,
                "excluded": 0,
                "excluded_reason": "нет",
                "use": "нарезка запросов нужной длины в токенах сервера (lr1/prompts.py)",
            }
        }
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["datasets"]["pride_and_prejudice"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
