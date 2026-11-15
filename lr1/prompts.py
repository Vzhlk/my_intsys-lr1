"""Запросы заданной длины в токенах (токенизатор и шаблон диалога — сервера)."""
import hashlib
from pathlib import Path

import httpx

INSTRUCTION = "Summarize the following text in one sentence."
BOOK = Path("data/pride_and_prejudice.txt")


class ServerTokenizer:
    """Токенизатор и шаблон диалога берутся у самого сервера llama.cpp."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/").removesuffix("/v1")
        self.http = httpx.Client(timeout=120)

    def tokenize(self, text):
        r = self.http.post(f"{self.base}/tokenize", json={"content": text, "add_special": False})
        r.raise_for_status()
        return r.json()["tokens"]

    def detokenize(self, tokens):
        r = self.http.post(f"{self.base}/detokenize", json={"tokens": tokens})
        r.raise_for_status()
        return r.json()["content"]

    def apply_template(self, messages):
        r = self.http.post(f"{self.base}/apply-template", json={"messages": messages})
        r.raise_for_status()
        return r.json()["prompt"]

    def prompt_tokens(self, messages):
        """Число токенов запроса после применения шаблона диалога."""
        return len(self.tokenize(self.apply_template(messages)))


def load_book_tokens(tok):
    text = BOOK.read_text(encoding="utf-8")
    return tok.tokenize(text), text


def build_messages(tok, book_tokens, label, target, start, max_iter=6):
    """Сообщение, дающее target токенов запроса (после шаблона диалога).

    Уникальная метка стоит в самом начале: кеш префикса не сможет переиспользовать
    предыдущие запросы. Если точное попадание не получилось (повторная токенизация
    текста), возвращается ближайший вариант: фактическое число токенов записывается.
    """
    header = f"[request {label}] {INSTRUCTION}\n\n"
    base = tok.prompt_tokens([{"role": "user", "content": header}])
    k = max(1, target - base)
    best = None
    for _ in range(max_iter):
        seg = book_tokens[start:start + k]
        if len(seg) < k:  # книга закончилась — продолжаем с начала
            seg = (book_tokens[start:] + book_tokens)[:k]
        msgs = [{"role": "user", "content": header + tok.detokenize(seg)}]
        n = tok.prompt_tokens(msgs)
        best = (msgs, n)
        if n == target:
            break
        k = max(1, k + target - n)
    return best


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
