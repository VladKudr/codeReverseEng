"""Маскирование секретов в артефактах.

Цитаты `evidence` копируют код дословно — вместе с тем, что в нём захардкожено:
строки подключения, токены, пароли, приватные ключи. Артефакты уходят в отчёты
и чужие руки, поэтому секреты маскируются ДО того, как артефакт признаётся
принятым: значение заменяется на ***MASKED***, факт маскирования логируется.

Детектор регулярный, без эвристик энтропии: в закрытом контуре важнее
предсказуемость, чем полнота. Классы: строки подключения с паролем, известные
формы токенов, пары ключ=значение с секретным именем, приватные ключи PEM.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MASK = "***MASKED***"

# Каждый паттерн маскирует group('secret'), остальное сохраняется как контекст.
PATTERNS: list[tuple[str, re.Pattern]] = [
    # postgres://user:pass@host, amqp://..., mongodb+srv://..., jdbc:...
    ("строка подключения", re.compile(
        r"(?P<scheme>[a-z][a-z0-9+.-]{1,30}):\/\/(?P<user>[^\s:@\/'\"]+):(?P<secret>[^\s@\/'\"]+)@",
        re.IGNORECASE)),
    # приватные ключи PEM (маскируется тело целиком)
    ("приватный ключ", re.compile(
        r"(?P<secret>-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----)")),
    # известные формы токенов
    ("токен AWS", re.compile(r"(?P<secret>\bAKIA[0-9A-Z]{16}\b)")),
    ("токен GitHub", re.compile(r"(?P<secret>\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b)")),
    ("токен Slack", re.compile(r"(?P<secret>\bxox[baprs]-[A-Za-z0-9-]{10,}\b)")),
    ("токен JWT", re.compile(r"(?P<secret>\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b)")),
    ("ключ API", re.compile(r"(?P<secret>\bsk-[A-Za-z0-9_-]{20,}\b)")),
    # пары ключ=значение / ключ: значение с секретным именем
    ("пароль/секрет в присваивании", re.compile(
        r"(?P<key>\b(?:password|passwd|pwd|secret|secret_key|api_key|apikey|"
        r"access_key|access_token|auth_token|token|private_key|client_secret)\b"
        r"\s*[=:]\s*)(?P<q>[\"']?)(?P<secret>[^\s\"',;]{4,})(?P=q)",
        re.IGNORECASE)),
]

# значения-заглушки, которые маскировать не нужно
_PLACEHOLDER_RE = re.compile(
    r"^(\*+|x+|<[^>]+>|\{[^}]+\}|\$\{?[A-Z_]+\}?|none|null|nil|true|false|"
    r"changeme|example|dummy|test|секрет|пример)$",
    re.IGNORECASE)


def mask_text(text: str) -> tuple[str, list[dict]]:
    """Возвращает (текст с масками, список фактов маскирования)."""
    events: list[dict] = []

    for kind, pattern in PATTERNS:
        def repl(m: re.Match, kind: str = kind) -> str:
            secret = m.group("secret")
            if _PLACEHOLDER_RE.match(secret) or secret == MASK:
                return m.group(0)
            if kind == "пароль/секрет в присваивании" and (
                "(" in secret or secret.startswith(("os.", "self.", "config.", "settings."))
            ):
                # значение — выражение кода, а не литерал секрета
                return m.group(0)
            line_no = text.count("\n", 0, m.start()) + 1
            events.append({
                "kind": kind,
                "line": line_no,
                "preview": secret[:4] + "…" if len(secret) > 8 else "…",
            })
            whole = m.group(0)
            return whole.replace(secret, MASK)

        text = pattern.sub(repl, text)
    return text, events


def mask_file(path: Path) -> list[dict]:
    """Маскирует файл на месте; возвращает факты маскирования."""
    original = path.read_text(encoding="utf-8", errors="replace")
    masked, events = mask_text(original)
    if events:
        path.write_text(masked, encoding="utf-8")
    return events


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Маскирование секретов в артефактах")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    total = 0
    for path in args.files:
        events = mask_file(path)
        for e in events:
            print(f"{path}:{e['line']}: замаскировано ({e['kind']}, начиналось с {e['preview']})")
        total += len(events)
    print(f"замаскировано значений: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
