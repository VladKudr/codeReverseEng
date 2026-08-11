"""Метрика полноты извлечения: покрытие ветвлений правилами.

По каждому модулю считается доля веток `raise` / `if` / `match`, на которые
ссылается хоть одно правило (по `source` вида файл:строка). Для Python ветки
и границы функций берутся из AST — точно; для прочих языков — приближение по
regex, и это честно помечается в отчёте (`method: regex`).

Ветка считается покрытой, если строка `source` какого-либо правила по тому же
файлу попадает в диапазон строк ветки (для if/match — вся конструкция, поэтому
правило на внутренний raise покрывает и объемлющий if: это осознанное
приближение в пользу извлечения).

Отчёт «функции без единого правила» — вход для решения о повторной экстракции.
Числа детерминированы: одинаковый вход даёт одинаковый отчёт.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

from ._common import parse_source, read_lines


def rule_ranges_by_file(rules_dir: Path) -> dict[str, list[tuple[int, int]]]:
    """файл (как в source) -> диапазоны строк, на которые ссылаются правила.

    Диапазон — от первой строки цитаты до её конца (по длине evidence):
    правило, цитирующее `if ...:\\n    raise ...`, покрывает обе строки.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for p in sorted(rules_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for rule in data.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            parsed = parse_source(rule.get("source", ""))
            if parsed:
                f, start, end = parsed
                ev_len = len(str(rule.get("evidence") or "").strip().splitlines())
                out.setdefault(f, []).append((start, max(end, start + max(ev_len - 1, 0))))
    return {f: sorted(v) for f, v in out.items()}


# ── Python: точно, по AST ───────────────────────────────────────────────────

def _branches_py(tree: ast.AST) -> list[dict]:
    branches = []
    for node in ast.walk(tree):
        kind = None
        if isinstance(node, ast.Raise):
            kind = "raise"
        elif isinstance(node, ast.If):
            kind = "if"
        elif isinstance(node, ast.Match):
            kind = "match"
        if kind:
            branches.append({
                "kind": kind,
                "line": node.lineno,
                "end": node.end_lineno or node.lineno,
            })
    return sorted(branches, key=lambda b: (b["line"], b["kind"]))


def _functions_py(tree: ast.AST) -> list[dict]:
    funcs = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append({
                    "name": ".".join(stack + [child.name]),
                    "line": child.lineno,
                    "end": child.end_lineno or child.lineno,
                })
                walk(child, stack + [child.name])
            else:
                walk(child, stack)

    walk(tree, [])
    return funcs


# ── прочие языки: приближение по regex ──────────────────────────────────────

_BRANCH_RE = re.compile(r"^\s*(?:\}?\s*else\s+)?(if|match|switch|raise|throw)\b")
_FUNC_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|export\s+)*"
    r"(?:def|function|func|fn|sub)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _branches_regex(lines: list[str]) -> list[dict]:
    out = []
    for i, line in enumerate(lines, 1):
        m = _BRANCH_RE.match(line)
        if m:
            kw = m.group(1)
            kind = {"throw": "raise", "switch": "match"}.get(kw, kw)
            out.append({"kind": kind, "line": i, "end": i})
    return out


REGEX_TOLERANCE = 3  # у regex-веток нет диапазона тела — допуск по строкам


def analyze_module(repo_root: Path, module_rel: str,
                   rules_ranges: dict[str, list[tuple[int, int]]]) -> dict:
    """Отчёт покрытия по одному модулю."""
    path = repo_root / module_rel
    ranges_hit = rules_ranges.get(module_rel, [])
    report: dict = {"module": module_rel, "rules_lines": len(ranges_hit)}

    if not path.is_file():
        return {**report, "error": "не файл (каталог или отсутствует) — пропуск"}

    if path.suffix == ".py":
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            return {**report, "error": f"файл не разбирается AST: {e}"}
        branches = _branches_py(tree)
        functions = _functions_py(tree)
        report["method"] = "ast"

        def covered(b: dict) -> bool:
            return any(s <= b["end"] and e >= b["line"] for s, e in ranges_hit)
    else:
        lines = read_lines(path)
        branches = _branches_regex(lines)
        functions = []
        report["method"] = "regex"

        def covered(b: dict) -> bool:
            return any(s - REGEX_TOLERANCE <= b["line"] <= e + REGEX_TOLERANCE
                       for s, e in ranges_hit)

    cov = [b for b in branches if covered(b)]
    report["branches_total"] = len(branches)
    report["branches_covered"] = len(cov)
    report["coverage_pct"] = (
        round(100 * len(cov) / len(branches), 1) if branches else 100.0)
    report["uncovered"] = [
        {"kind": b["kind"], "line": b["line"]} for b in branches if not covered(b)]
    if functions:
        report["functions_without_rules"] = [
            f["name"] for f in functions
            if not any(s <= f["end"] and e >= f["line"] for s, e in ranges_hit)
        ]
    return report


def analyze(repo_root: Path, rules_dir: Path,
            modules: list[str] | None = None) -> dict:
    """Сводный отчёт: по модулям + итог."""
    rl = rule_ranges_by_file(rules_dir)
    if modules is None:
        modules = sorted(rl)  # файлы, на которые вообще ссылаются правила
    per_module = [analyze_module(repo_root, m, rl) for m in modules]
    total = sum(m.get("branches_total", 0) for m in per_module)
    cov = sum(m.get("branches_covered", 0) for m in per_module)
    return {
        "modules": per_module,
        "branches_total": total,
        "branches_covered": cov,
        "coverage_pct": round(100 * cov / total, 1) if total else 100.0,
    }


def render_md(report: dict) -> str:
    """Markdown-сводка на русском для отчёта прогона."""
    out = ["# Полнота извлечения: покрытие ветвлений правилами", ""]
    out.append("| Модуль | Метод | Веток | Покрыто | % | Функции без правил |")
    out.append("|---|---|---|---|---|---|")
    for m in report["modules"]:
        if "error" in m:
            out.append(f"| `{m['module']}` | — | — | — | — | {m['error']} |")
            continue
        funcs = m.get("functions_without_rules", [])
        out.append(
            f"| `{m['module']}` | {m['method']} | {m['branches_total']} "
            f"| {m['branches_covered']} | {m['coverage_pct']} "
            f"| {', '.join(f'`{f}`' for f in funcs) or '—'} |")
    out.append("")
    out.append(f"**Итого: {report['branches_covered']}/{report['branches_total']} "
               f"= {report['coverage_pct']}%**")
    out.append("")
    out.append("Функции без единого правила — кандидаты на повторную экстракцию "
               "(этап 4 по соответствующему модулю).")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Покрытие ветвлений правилами")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--rules", required=True, type=Path, help="каталог rules/")
    ap.add_argument("--modules", nargs="*", default=None,
                    help="модули (пути от корня репо); по умолчанию — из source правил")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = analyze(args.repo, args.rules, args.modules)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(render_md(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
