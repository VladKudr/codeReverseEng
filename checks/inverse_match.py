"""Инверсная верификация, механическая половина (этап 92).

Идея (spec-gen/OpenLore): не «спека подтверждается кодом», а «по спеке
предсказывается код, предсказание сверяется с фактом». LLM-половина — промпт
92: модель видит ТОЛЬКО statement'ы правил (без evidence и source) и
предсказывает, какие проверки, сообщения, пороги и где должны встретиться.
Этот модуль механически ищет предсказанное в исходниках.

Поиск нормализованный, не точное совпадение: регистр опускается, пробелы
сжимаются, плейсхолдеры чисел (`<N>`, `<NUM>`, `<число>`) в тексте предсказания
сверяются с любым числовым литералом.

Выход — доля подтверждённых предсказаний по каждому правилу и сводно.
Низкая доля по правилу: из формулировки не восстанавливается поведение —
кандидат на возврат в этап 4. Метрика дополняет этап 90, а не заменяет его;
пороги приёмки в v3 не калибруются.

Отчёт детерминирован: одинаковые предсказания и исходники дают одинаковые числа.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from ._common import squash_ws

# плейсхолдеры чисел в тексте предсказания
_NUM_PLACEHOLDER = re.compile(r"<\s*(?:n|num|число)\s*>", re.IGNORECASE)
_NUM_RE = r"[0-9][0-9_]*(?:\.[0-9]+)?"

# что считаем исходниками при глобальном поиске
SOURCE_SUFFIXES = {".py", ".pyi", ".js", ".ts", ".tsx", ".go", ".java", ".kt",
                   ".rb", ".rs", ".c", ".h", ".cpp", ".cs", ".php", ".sql",
                   ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json",
                   ".md", ".rst", ".txt"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist",
             "build", ".tox"}


def normalize(text: str) -> str:
    return squash_ws(text).lower()


def needle_regex(prediction_text: str) -> re.Pattern:
    """Текст предсказания -> regex по нормализованному исходнику."""
    norm = normalize(prediction_text)
    parts = _NUM_PLACEHOLDER.split(norm)
    return re.compile(_NUM_RE.join(re.escape(p) for p in parts))


def _iter_source_files(repo_root: Path):
    for p in sorted(repo_root.rglob("*")):
        if not p.is_file() or p.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(repo_root).parts):
            continue
        yield p


class SourceIndex:
    """Нормализованные тексты исходников (файл читается один раз)."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._cache: dict[str, str] = {}

    def text(self, rel: str) -> str | None:
        if rel not in self._cache:
            p = self.repo_root / rel
            self._cache[rel] = (
                normalize(p.read_text(encoding="utf-8", errors="replace"))
                if p.is_file() else "")
        return self._cache[rel] or None

    def all_files(self) -> list[str]:
        return [str(p.relative_to(self.repo_root))
                for p in _iter_source_files(self.repo_root)]


def match_expect(index: SourceIndex, expect: dict) -> dict:
    """Одно предсказание -> {confirmed, found_in}."""
    pattern = needle_regex(str(expect.get("text", "")))
    hint = expect.get("module_hint")
    if hint and hint != "UNCERTAIN":
        text = index.text(str(hint))
        if text and pattern.search(text):
            return {"confirmed": True, "found_in": str(hint)}
    for rel in index.all_files():
        text = index.text(rel)
        if text and pattern.search(text):
            return {"confirmed": True, "found_in": rel}
    return {"confirmed": False, "found_in": None}


def analyze_file(index: SourceIndex, predictions_path: Path) -> list[dict]:
    """Файл предсказаний -> строки по правилам."""
    data = yaml.safe_load(predictions_path.read_text(encoding="utf-8")) or {}
    rows = []
    for pred in data.get("predictions") or []:
        if not isinstance(pred, dict):
            continue
        expects = pred.get("expect") or []
        matched = [match_expect(index, e) for e in expects if isinstance(e, dict)]
        confirmed = sum(1 for m in matched if m["confirmed"])
        rows.append({
            "rule_id": str(pred.get("rule_id", "?")),
            "capability": str(data.get("capability", predictions_path.stem)),
            "expects_total": len(matched),
            "expects_confirmed": confirmed,
            "share_pct": round(100 * confirmed / len(matched), 1) if matched else 0.0,
            "unconfirmed": [str(e.get("text"))
                            for e, m in zip(expects, matched) if not m["confirmed"]],
        })
    return rows


def analyze(repo_root: Path, predictions: Path) -> dict:
    """Каталог (или файл) предсказаний -> сводный отчёт."""
    index = SourceIndex(repo_root)
    files = (sorted(predictions.glob("*.predictions.yaml"))
             if predictions.is_dir() else [predictions])
    rules = [row for f in files for row in analyze_file(index, f)]
    total = sum(r["expects_total"] for r in rules)
    confirmed = sum(r["expects_confirmed"] for r in rules)
    return {
        "rules": sorted(rules, key=lambda r: r["rule_id"]),
        "total": total,
        "confirmed": confirmed,
        "share_pct": round(100 * confirmed / total, 1) if total else 0.0,
    }


def render_md(report: dict) -> str:
    out = ["# Инверсная верификация (этап 92): предсказания против кода", ""]
    out.append("| Правило | Capability | Предсказаний | Подтверждено | % |")
    out.append("|---|---|---|---|---|")
    for r in report["rules"]:
        out.append(f"| {r['rule_id']} | `{r['capability']}` | {r['expects_total']} "
                   f"| {r['expects_confirmed']} | {r['share_pct']} |")
    out.append("")
    out.append(f"**Итого: {report['confirmed']}/{report['total']} "
               f"= {report['share_pct']}%** (пороги приёмки не калиброваны — v3.1)")
    low = [r for r in report["rules"]
           if r["expects_total"] and r["expects_confirmed"] < r["expects_total"]]
    if low:
        out += ["", "## Неподтверждённые предсказания (кандидаты на возврат в этап 4)", ""]
        for r in low:
            for text in r["unconfirmed"]:
                out.append(f"* {r['rule_id']}: «{text}» — в исходниках не найдено; "
                           "из формулировки правила поведение не восстанавливается?")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Сверка предсказаний этапа 92 с исходниками")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path,
                    help="файл *.predictions.yaml или каталог с ними")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = analyze(args.repo, args.predictions)
    print(json.dumps(report, ensure_ascii=False, indent=1) if args.json
          else render_md(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
