"""Выдача свободных диапазонов ID правил.

Этапы 4 идут по одному модулю на запуск, и агент не видит чужой нумерации.
В v2 промпт просил модель «прочитать чужие файлы и посчитать сотни» — это
механическая задача, и она решается здесь: оркестратор сканирует занятые ID
и передаёт агенту начало свободного блока переменной {ID_START}.

Правило блоков — как в v2: первый модуль 001, второй 101, третий 201 и далее;
блок — сотня, внутри блока нумерация подряд.

Известный провал, который эта проверка закрывает: два модуля независимо взяли
R-101…R-104, четыре требования получили одинаковые идентификаторы, и
трассировка перестала сходиться.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from ._common import CheckResult, Finding

BLOCK = 100
ID_RE = re.compile(r"-R-(\d+)$")


def used_ids(rules_dir: Path) -> dict[int, list[str]]:
    """Занятые номера -> файлы, в которых встретились (для диагностики коллизий)."""
    found: dict[int, list[str]] = {}
    if not rules_dir.is_dir():
        return found
    for p in sorted(rules_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue  # битый YAML ловит схемная валидация, не аллокатор
        for rule in data.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            m = ID_RE.search(str(rule.get("id", "")))
            if m:
                found.setdefault(int(m.group(1)), []).append(p.name)
    return found


def next_block_start(rules_dir: Path) -> int:
    """Начало следующего свободного блока-сотни: 1, 101, 201, …"""
    used = used_ids(rules_dir)
    if not used:
        return 1
    top = max(used)
    return ((top - 1) // BLOCK + 1) * BLOCK + 1


def check_collisions(rules_dir: Path) -> CheckResult:
    """Дубликаты номеров между файлами (и внутри файла)."""
    result = CheckResult(check="id_allocator")
    used = used_ids(rules_dir)
    for num, files in sorted(used.items()):
        if len(files) > 1:
            result.findings.append(Finding(
                artifact=", ".join(sorted(set(files))),
                path=f"rules[].id (R-{num:03d})",
                expected="каждый номер правила уникален во всём прогоне",
                got=f"номер {num:03d} встречается {len(files)} раз: "
                    + ", ".join(files),
                hint="перенумеруй правила одного из модулей в его выданный блок",
            ))
    result.stats = {"ids_used": len(used)}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Выдача блоков ID и поиск коллизий")
    ap.add_argument("--rules", required=True, type=Path, help="каталог rules/")
    ap.add_argument("--check", action="store_true",
                    help="только поиск коллизий (без выдачи блока)")
    args = ap.parse_args(argv)

    if args.check:
        result = check_collisions(args.rules)
        for f in result.findings:
            print(f.render())
        print(f"занятых номеров: {result.stats['ids_used']}, "
              f"коллизий: {len(result.errors)}")
        return 1 if result.errors else 0

    print(next_block_start(args.rules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
