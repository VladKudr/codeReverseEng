#!/usr/bin/env python3
"""Генератор фикстуры «tenacity: 145 цитат, одна сокращённая».

Артефакты реальных прогонов v2 в комплект не вошли, поэтому известный сценарий
(«145 цитат, 144 дословны, единственное расхождение — сокращённая цитата»)
воспроизводится на настоящих исходниках tenacity 9.1.4, вошедших в фикстуру:
из шести модулей прогона Kimi детерминированно извлекаются 145 дословных цитат,
после чего у ПЕРВОГО правила модуля wait.py середина цитаты выбрасывается —
ровно то сокращение, которое запрещено промптом и которое обязан поймать
checks/evidence_binding.py.

Запуск (из каталога req-reverse): python tests/fixtures/gen_tenacity_rules.py
Результат: tests/fixtures/tenacity_artifacts/rules/*.yaml (закоммичены;
перегенерация нужна только при смене исходников фикстуры).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from checks.evidence_binding import enclosing_function_py  # noqa: E402

REPO = HERE / "tenacity"
OUT = HERE / "tenacity_artifacts" / "rules"
TOTAL = 145
# порядок модулей и блоки ID — как в реальном прогоне (первый — 001, далее +100)
MODULES = ["tenacity/__init__.py", "tenacity/retry.py", "tenacity/stop.py",
           "tenacity/wait.py", "tenacity/nap.py", "tenacity/_utils.py"]
TRUNCATED_MODULE = "tenacity/wait.py"   # у первого правила wait.py режем цитату


def candidates(lines: list[str]) -> list[int]:
    """Номера строк (1-based), пригодных как якорь цитаты."""
    return [i for i, line in enumerate(lines, 1)
            if len(line.strip()) >= 12 and not line.lstrip().startswith("#")]


def quota(sizes: list[int], total: int) -> list[int]:
    """Распределение total по модулям пропорционально размеру (метод остатков)."""
    raw = [total * s / sum(sizes) for s in sizes]
    base = [int(x) for x in raw]
    rest = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rest]:
        base[i] += 1
    return base


def three_nonempty(lines: list[str], anchor: int) -> bool:
    return (anchor + 2 <= len(lines)
            and all(lines[anchor - 1 + k].strip() for k in range(3)))


def build_rule(module: str, text: str, lines: list[str], anchor: int,
               rid: str, truncate: bool) -> dict:
    n_lines = 3 if three_nonempty(lines, anchor) else 1
    ev_lines = lines[anchor - 1 : anchor - 1 + n_lines]
    if truncate:
        assert n_lines == 3, "для сокращения нужна трёхстрочная цитата"
        ev_lines = [ev_lines[0], ev_lines[2]]  # выброшена середина — запрещено
    enclosing = enclosing_function_py(text, anchor) or "module"
    return {
        "id": rid,
        "statement": f"Код проверяет фрагмент {module}:{anchor} (фикстура известного прогона).",
        "category": "other",
        "actors_affected": ["UNCERTAIN"],
        "evidence": "\n".join(ev_lines) + "\n",
        "source": f"{module}:{anchor}",
        "enclosing": enclosing,
        "called_by": ["UNCERTAIN"],
        "config_driven": False,
        "confidence": "medium",
    }


def main() -> int:
    texts = {m: (REPO / m).read_text(encoding="utf-8") for m in MODULES}
    lines = {m: texts[m].splitlines() for m in MODULES}
    counts = quota([len(lines[m]) for m in MODULES], TOTAL)

    OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for idx, (module, count) in enumerate(zip(MODULES, counts)):
        cand = candidates(lines[module])
        step = max(1, len(cand) // count)
        anchors = cand[::step][:count]
        assert len(anchors) == count, f"{module}: не хватает кандидатов"
        if module == TRUNCATED_MODULE:
            # якорь сокращённой цитаты обязан иметь три непустых строки
            anchors[0] = next(a for a in cand if three_nonempty(lines[module], a))
            anchors = [anchors[0]] + [a for a in anchors[1:] if a != anchors[0]]
            while len(anchors) < count:
                extra = next(a for a in cand if a not in anchors)
                anchors.append(extra)
            anchors = anchors[:1] + sorted(anchors[1:])
        block = idx * 100 + 1
        rules = [
            build_rule(module, texts[module], lines[module], a,
                       f"tenacity-R-{block + k:03d}",
                       truncate=(module == TRUNCATED_MODULE and k == 0))
            for k, a in enumerate(anchors)
        ]
        doc = {"repo": "tenacity", "module": module, "rules": rules,
               "edge_cases_and_errors": [], "notes_uncertain": []}
        out_file = OUT / (Path(module).stem + ".yaml")
        out_file.write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        made += len(rules)
        print(f"{out_file.name}: {len(rules)} правил (блок {block:03d})")
    print(f"итого правил: {made} (сокращённая цитата: tenacity-R-301)")
    assert made == TOTAL
    return 0


if __name__ == "__main__":
    sys.exit(main())
