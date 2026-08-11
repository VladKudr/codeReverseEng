"""Общее для механических проверок: формат ошибок и разбор ссылок.

Формат ошибки един для всего детерминированного слоя, потому что его текст
уходит в retry-промпт: LLM исправляет артефакт по тексту конкретной ошибки.
Поэтому каждая ошибка обязана называть файл, путь до поля, ожидание и факт —
без этого модель чинит наугад.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Finding:
    """Одна проблема, найденная механической проверкой."""

    artifact: str          # файл артефакта, в котором проблема
    path: str              # путь до поля внутри артефакта, напр. rules[3].evidence
    expected: str          # что ожидалось
    got: str               # что обнаружено
    hint: str = ""         # как исправить (если известно)
    severity: str = "error"  # error | warning

    def render(self) -> str:
        lines = [
            f"- Артефакт: {self.artifact}",
            f"  Поле: {self.path}",
            f"  Ожидалось: {self.expected}",
            f"  Получено: {self.got}",
        ]
        if self.hint:
            lines.append(f"  Как исправить: {self.hint}")
        return "\n".join(lines)


@dataclass
class CheckResult:
    """Итог одной проверки: имя, находки, произвольная статистика."""

    check: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def render_for_retry(results: list[CheckResult]) -> str:
    """Текст для retry-промпта: только ошибки, по-русски, с путями до полей."""
    blocks: list[str] = []
    for r in results:
        errs = r.errors
        if not errs:
            continue
        blocks.append(f"### Проверка «{r.check}» нашла ошибки ({len(errs)}):\n")
        blocks.extend(f.render() for f in errs)
    return "\n".join(blocks)


SOURCE_RE = re.compile(r"^(?P<file>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$")


def parse_source(src: str) -> tuple[str, int, int] | None:
    """`файл:строка` или `файл:начало-конец` -> (файл, начало, конец) или None."""
    if not isinstance(src, str):
        return None
    m = SOURCE_RE.match(src.strip())
    if not m:
        return None
    start = int(m.group("start"))
    end = int(m.group("end") or start)
    return m.group("file"), start, end


def squash_ws(s: str) -> str:
    """Сжатие пробелов: сверка цитат не должна падать на переносах и отступах."""
    return re.sub(r"\s+", " ", s or "").strip()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()
