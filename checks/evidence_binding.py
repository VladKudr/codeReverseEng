"""Слой 1 как код: привязка правила к коду.

Для каждого правила из rules/*.yaml проверяется механически:
  1. файл из `source` существует в репозитории;
  2. цитата `evidence` присутствует в нём дословно с допуском ±3 строки от
     указанной; окно сверки растягивается на длину самой цитаты — иначе любая
     цитата длиннее четырёх строк «не находится» (промах инструмента,
     проявившийся в прогоне на sqlite-utils, повторять его нельзя);
  3. `enclosing` — функция, реально содержащая первую строку цитаты.
     Для Python — точно, по AST. Для остальных языков — эвристика по
     отступам/скобкам; её результат помечается как эвристика и даёт
     предупреждение, а не ошибку.

Запускается после каждого этапа 4 и повторно перед этапом 90. Провал —
возврат правила на переизвлечение с диагностикой в формате retry.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

from ._common import CheckResult, Finding, parse_source, read_lines, squash_ws

TOLERANCE = 3  # допуск сдвига цитаты в строках


# ── определение объемлющей функции ──────────────────────────────────────────

def enclosing_function_py(source_text: str, line: int) -> str | None:
    """Имя самой внутренней функции/метода, содержащей строку (по AST).

    Для метода возвращается «Класс.метод». Строка вне функций -> None.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None

    best: tuple[int, str] | None = None  # (размер диапазона, имя)

    def walk(node: ast.AST, class_stack: list[str]) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, class_stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min(
                    [child.lineno] + [d.lineno for d in child.decorator_list]
                )
                end = child.end_lineno or child.lineno
                if start <= line <= end:
                    name = ".".join(class_stack + [child.name])
                    size = end - start
                    if best is None or size <= best[0]:
                        best = (size, name)
                    walk(child, class_stack)  # вложенные функции ещё уже
                else:
                    walk(child, class_stack)
            else:
                walk(child, class_stack)

    walk(tree, [])
    return best[1] if best else None


_FUNC_LINE_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|export\s+)*"
    r"(?:def|function|func|fn|sub)\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|^\s*(?:[\w<>\[\],.\s]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?\s*$"
)


def enclosing_function_heuristic(lines: list[str], line: int) -> str | None:
    """Не-Python: ближайшее выше по тексту объявление функции с меньшим отступом.

    Это честная эвристика по отступам/скобкам, а не разбор языка; её результат
    вызывающий код помечает как эвристический.
    """
    if line < 1 or line > len(lines):
        return None
    target_indent = len(lines[line - 1]) - len(lines[line - 1].lstrip())
    for i in range(line - 1, 0, -1):
        text = lines[i - 1]
        if not text.strip():
            continue
        indent = len(text) - len(text.lstrip())
        if indent < target_indent or i == line:
            m = _FUNC_LINE_RE.match(text)
            if m:
                return m.group(1) or m.group(2)
            # поднялись до менее вложенного уровня — ищем объявление выше него
            target_indent = indent
    return None


# ── сверка цитаты ───────────────────────────────────────────────────────────

MASK = "***MASKED***"  # ставит checks/secrets_mask.py


def _contains(haystack: str, needle: str) -> bool:
    """Подстрока с учётом масок секретов.

    После secrets_mask цитата в артефакте содержит ***MASKED*** вместо
    секрета и дословно с кодом уже не совпадает — маска сверяется как
    «дырка» произвольного содержимого без переноса строки.
    """
    if MASK not in needle:
        return needle in haystack
    parts = [re.escape(p) for p in needle.split(MASK)]
    return re.search("[^\\n]*?".join(parts), haystack) is not None


def evidence_in_window(
    lines: list[str], start: int, end: int, evidence: str, tol: int = TOLERANCE
) -> tuple[bool, bool]:
    """(найдена в окне ±tol, найдена точно по указанным строкам)."""
    needle = squash_ws(str(evidence).strip().strip('"'))
    if not needle:
        return True, True
    ev_len = len(str(evidence).strip().splitlines())
    lo = max(0, start - 1 - tol)
    hi = min(len(lines), end + ev_len + tol)
    window = squash_ws("\n".join(lines[lo:hi]))
    exact = _contains(squash_ws("\n".join(lines[start - 1 : end + ev_len - 1])), needle)
    return _contains(window, needle), exact


# ── основная проверка ───────────────────────────────────────────────────────

def check_rules_file(rules_path: Path, repo_root: Path) -> CheckResult:
    """Проверка одного файла rules/<модуль>.yaml."""
    result = CheckResult(check="evidence_binding")
    artifact = str(rules_path)

    try:
        data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        result.findings.append(Finding(
            artifact=artifact, path="(весь файл)",
            expected="валидный YAML", got=f"ошибка разбора: {e}",
            hint="исправь синтаксис YAML и перезапиши файл целиком",
        ))
        return result

    rules = (data or {}).get("rules") or []
    checked = 0
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id", f"rules[{i}]")
        prefix = f"rules[{i}] (id: {rid})"
        src = rule.get("source", "")
        parsed = parse_source(src)
        if parsed is None:
            result.findings.append(Finding(
                artifact=artifact, path=f"{prefix}.source",
                expected="ссылка вида файл:строка или файл:начало-конец",
                got=repr(src),
                hint="укажи файл и номер первой строки цитаты",
            ))
            continue
        rel_file, start, end = parsed
        file_path = repo_root / rel_file
        if not file_path.is_file():
            result.findings.append(Finding(
                artifact=artifact, path=f"{prefix}.source",
                expected=f"существующий файл относительно {repo_root}",
                got=f"файла {rel_file} нет",
                hint="проверь путь: он задаётся от корня репозитория",
            ))
            continue
        lines = read_lines(file_path)
        if start < 1 or start > len(lines):
            result.findings.append(Finding(
                artifact=artifact, path=f"{prefix}.source",
                expected=f"строка в пределах файла (1–{len(lines)})",
                got=f"строка {start}",
                hint="открой файл и укажи фактический номер первой строки цитаты",
            ))
            continue

        checked += 1
        evidence = rule.get("evidence", "")
        found, exact = evidence_in_window(lines, start, end, evidence)
        if not found:
            around = "\n".join(
                f"    {n}: {lines[n - 1]}"
                for n in range(max(1, start - TOLERANCE),
                               min(len(lines), start + TOLERANCE) + 1)
            )
            result.findings.append(Finding(
                artifact=artifact, path=f"{prefix}.evidence",
                expected=f"дословная цитата из {rel_file} у строки {start} (допуск ±{TOLERANCE})",
                got="цитата по ссылке не найдена; фрагмент файла:\n" + around,
                hint="скопируй строки из файла дословно, подряд, без сокращения "
                     "середины многоточием; сокращай границы цитаты, а не её середину",
            ))
            continue

        # enclosing: функция, реально содержащая первую строку цитаты
        declared = rule.get("enclosing")
        if declared in (None, "", "UNCERTAIN"):
            continue
        if file_path.suffix == ".py":
            actual = enclosing_function_py(
                file_path.read_text(encoding="utf-8", errors="replace"), start
            )
            heuristic = False
        else:
            actual = enclosing_function_heuristic(lines, start)
            heuristic = True
        if actual is None:
            # строка вне функций (модульный уровень) — допустимо, если агент
            # так и написал; иначе предупреждение
            if str(declared).lower() not in ("module", "модуль", "module-level"):
                result.findings.append(Finding(
                    artifact=artifact, path=f"{prefix}.enclosing",
                    expected="строка лежит вне функций (модульный уровень)",
                    got=f"заявлено {declared!r}",
                    hint="если строка на уровне модуля — так и напиши",
                    severity="warning",
                ))
            continue
        # сравниваем по последнему сегменту: агент мог написать «метод»,
        # а AST даёт «Класс.метод»
        def last_seg(name: str) -> str:
            return str(name).split(".")[-1].strip()

        if last_seg(declared) != last_seg(actual):
            result.findings.append(Finding(
                artifact=artifact, path=f"{prefix}.enclosing",
                expected=f"функция, содержащая строку {start}: {actual}"
                         + (" (определено эвристикой по отступам)" if heuristic else " (определено по AST)"),
                got=str(declared),
                hint="определи объемлющую функцию по телу кода, а не по соседним "
                     "заголовкам, и переформулируй правило от лица её вызывающих",
                severity="warning" if heuristic else "error",
            ))
    result.stats = {"rules_total": len(rules), "rules_checked": checked}
    return result


def check_rules_dir(rules_dir: Path, repo_root: Path) -> list[CheckResult]:
    return [
        check_rules_file(p, repo_root)
        for p in sorted(rules_dir.glob("*.yaml"))
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Слой 1: привязка evidence к коду")
    ap.add_argument("--repo", required=True, type=Path, help="корень анализируемого репозитория")
    ap.add_argument("--rules", required=True, type=Path, help="файл или каталог rules/*.yaml")
    ap.add_argument("--json", action="store_true", help="вывод в JSON")
    args = ap.parse_args(argv)

    if args.rules.is_dir():
        results = check_rules_dir(args.rules, args.repo)
    else:
        results = [check_rules_file(args.rules, args.repo)]

    errors = [f for r in results for f in r.errors]
    warnings = [f for r in results for f in r.warnings]
    if args.json:
        print(json.dumps({
            "errors": [vars(f) for f in errors],
            "warnings": [vars(f) for f in warnings],
            "stats": [r.stats for r in results],
        }, ensure_ascii=False, indent=1))
    else:
        from ._common import render_for_retry
        text = render_for_retry(results)
        if text:
            print(text)
        for w in warnings:
            print(f"[предупреждение]\n{w.render()}")
        total = sum(r.stats.get("rules_total", 0) for r in results)
        print(f"\nправил всего {total}, ошибок привязки {len(errors)}, предупреждений {len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
