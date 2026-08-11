"""Пересчёт полноты: правила ↔ истории ↔ трассировка.

Механическая проверка, изъятая из промптов 6/7 («полнота считается счётом»):
  1. каждый rule_id из rules/*.yaml встречается в таблице покрытия stories.md
     ровно один раз — у истории либо в строке «не покрыто» с причиной;
  2. число уникальных rule_id в traceability.csv равно N (числу правил);
  3. колонки traceability.csv в точности соответствуют шаблону
     (req_id,story_id,rule_id,source,confidence,verdict_stage90) — известный
     провал: агент выдал свой набор колонок без story_id, и связь
     «требование → история» пропала целиком.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

from ._common import CheckResult, Finding

EXPECTED_COLUMNS = ["req_id", "story_id", "rule_id", "source",
                    "confidence", "verdict_stage90"]


def collect_rule_ids(rules_dir: Path) -> list[str]:
    ids: list[str] = []
    for p in sorted(rules_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for rule in data.get("rules") or []:
            if isinstance(rule, dict) and rule.get("id"):
                ids.append(str(rule["id"]))
    return ids


def rule_ids_in_stories(stories_path: Path, rule_ids: list[str]) -> dict[str, int]:
    """Сколько раз каждый rule_id встречается в таблице покрытия stories.md.

    Таблица покрытия — последняя markdown-таблица документа, в строках которой
    упоминаются ID правил. Считаем вхождения по строкам таблиц: критерии
    приёмки (AC1 [X-R-001]) вне таблиц не считаются покрытием.
    """
    counts = {rid: 0 for rid in rule_ids}
    if not stories_path.is_file():
        return counts
    for line in stories_path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        for rid in rule_ids:
            counts[rid] += len(re.findall(re.escape(rid) + r"(?![\w-])", line))
    return counts


def check(rules_dir: Path, stories_path: Path, traceability_path: Path,
          template_path: Path | None = None) -> CheckResult:
    result = CheckResult(check="completeness")
    rule_ids = collect_rule_ids(rules_dir)
    n = len(rule_ids)

    # 1. покрытие в stories.md: ровно один раз
    counts = rule_ids_in_stories(stories_path, rule_ids)
    for rid, c in counts.items():
        if c == 0:
            result.findings.append(Finding(
                artifact=str(stories_path), path=f"таблица покрытия ({rid})",
                expected=f"правило {rid} встречается в таблице покрытия ровно один раз",
                got="не встречается ни разу",
                hint="добавь строку: либо ID истории, либо «не покрыто» с причиной "
                     "(техническое / не относится к пользовательскому поведению)",
            ))
        elif c > 1:
            result.findings.append(Finding(
                artifact=str(stories_path), path=f"таблица покрытия ({rid})",
                expected=f"правило {rid} встречается в таблице покрытия ровно один раз",
                got=f"встречается {c} раза(-з)",
                hint="оставь одну строку на правило",
            ))

    # 2–3. traceability.csv: колонки и счёт
    expected_cols = EXPECTED_COLUMNS
    if template_path and template_path.is_file():
        with template_path.open(encoding="utf-8") as fh:
            expected_cols = next(csv.reader(fh))
    if not traceability_path.is_file():
        result.findings.append(Finding(
            artifact=str(traceability_path), path="(весь файл)",
            expected="файл traceability.csv существует", got="файла нет",
        ))
    else:
        with traceability_path.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        if header != expected_cols:
            result.findings.append(Finding(
                artifact=str(traceability_path), path="заголовок",
                expected=",".join(expected_cols),
                got=",".join(header) or "(пусто)",
                hint="колонки фиксированы шаблоном: не добавляй своих и не "
                     "переименовывай существующие — файл читают другие этапы",
            ))
        else:
            col = {name: i for i, name in enumerate(header)}
            csv_rule_ids = [r[col["rule_id"]].strip() for r in rows
                            if len(r) == len(header)]
            uniq = set(x for x in csv_rule_ids if x)
            if len(uniq) != n:
                result.findings.append(Finding(
                    artifact=str(traceability_path), path="rule_id",
                    expected=f"уникальных rule_id ровно {n} (число правил в rules/*.yaml)",
                    got=f"уникальных rule_id {len(uniq)}",
                    hint="допиши недостающие строки трассировки; не сдавай "
                         "результат, пока числа не сошлись",
                ))
            for rid in rule_ids:
                if rid not in uniq:
                    result.findings.append(Finding(
                        artifact=str(traceability_path), path=f"rule_id ({rid})",
                        expected=f"правило {rid} присутствует в трассировке",
                        got="отсутствует",
                    ))
            for rid in sorted(uniq - set(rule_ids)):
                result.findings.append(Finding(
                    artifact=str(traceability_path), path=f"rule_id ({rid})",
                    expected="каждый rule_id трассировки существует в rules/*.yaml",
                    got=f"{rid} в правилах не найден",
                    hint="убери строку или исправь идентификатор",
                ))
            empty_src = [i + 2 for i, r in enumerate(rows)
                         if len(r) == len(header) and not r[col["source"]].strip()]
            if empty_src:
                result.findings.append(Finding(
                    artifact=str(traceability_path), path="source",
                    expected="нет строк с пустым source",
                    got=f"пустой source в строках: {empty_src}",
                ))

    result.stats = {"rules_total": n}
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Пересчёт полноты: правила ↔ истории ↔ трассировка")
    ap.add_argument("--rules", required=True, type=Path, help="каталог rules/")
    ap.add_argument("--stories", required=True, type=Path, help="stories.md")
    ap.add_argument("--traceability", required=True, type=Path, help="traceability.csv")
    ap.add_argument("--template", type=Path, default=None,
                    help="шаблон traceability_matrix.csv (эталон колонок)")
    args = ap.parse_args(argv)

    result = check(args.rules, args.stories, args.traceability, args.template)
    for f in result.findings:
        print(f.render())
    print(f"\nправил: {result.stats['rules_total']}, ошибок полноты: {len(result.errors)}")
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
