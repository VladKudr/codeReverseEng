#!/usr/bin/env python3
"""Мок CLI-агента для тестов runner'а.

Играет роль корпоративного агента: получает файл промпта и целевой артефакт,
«отвечает записью файла» — копирует канонированный артефакт из fixtures/canned.

Поведение для проверки контура ретраев:
  * первый вызов по rules/app.yaml пишет испорченную версию (сокращённая
    цитата + неверная строка source), последующие — исправленную: runner
    обязан дойти до успеха за один ретрай;
  * --always-bad <suffix> — артефакт с этим суффиксом всегда пишется битым:
    runner обязан исчерпать ретраи и пометить задачу эскалацией.

Печатает метрику токенов в stdout — runner собирает стоимость по
agent.metrics_regex.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

CANNED = Path(__file__).resolve().parent / "fixtures" / "canned"


def bump_state(state_path: Path, key: str) -> int:
    """Счётчик вызовов по ключу; возвращает номер текущего вызова (с 1)."""
    state = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state[key] = state.get(key, 0) + 1
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state[key]


def canned_for(out_file: Path) -> list[tuple[Path, Path]]:
    """(источник в canned, куда писать) — этапы с двумя выходами пишут оба."""
    name = out_file.name
    parent = out_file.parent.name
    if parent == "inventory":
        return [(CANNED / "inventory.yaml", out_file)]
    if parent == "modules":
        return [(CANNED / "modules" / name, out_file)]
    if parent == "rules":
        return [(CANNED / "rules" / name, out_file)]
    if name == "domain.yaml":
        return [(CANNED / "domain.yaml", out_file)]
    if parent == "inverse" and name.endswith(".predictions.yaml"):
        return [(CANNED / "inverse" / name, out_file)]
    if name == "api.openapi.yaml":
        # HTTP-границы у фикстуры нет — агент пишет api.cli.yaml
        return [(CANNED / "api.cli.yaml", out_file.parent / "api.cli.yaml")]
    if name == "flows.md":
        return [(CANNED / "flows.md", out_file)]
    if name == "stories.md":
        return [(CANNED / "stories.md", out_file)]
    if name == "SRS.md":
        return [(CANNED / "SRS.md", out_file),
                (CANNED / "traceability.csv", out_file.parent / "traceability.csv")]
    if name == "validation.csv":
        return [(CANNED / "validation.csv", out_file),
                (CANNED / "validation.md", out_file.parent / "validation.md")]
    raise SystemExit(f"мок-агент не знает артефакта {out_file}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--always-bad", default=None,
                    help="суффикс пути артефакта, который всегда пишется битым")
    ap.add_argument("--vendor", default="mock",
                    help="метка вендора: делает команду этапа 90 отличной от этапа 4")
    args = ap.parse_args()

    assert args.prompt.is_file(), "runner обязан передать файл промпта"
    prompt_text = args.prompt.read_text(encoding="utf-8")
    assert "{REPO_PATH}" not in prompt_text and "{OUT_FILE}" not in prompt_text, \
        "в промпте остались неподставленные переменные"

    out = args.out
    rel_key = f"{out.parent.name}/{out.name}"
    call_no = bump_state(args.state, rel_key)

    for src, dst in canned_for(out):
        if args.always_bad and str(dst).endswith(args.always_bad):
            src = CANNED / "bad" / "rules_app_bad.yaml"
        elif dst.parent.name == "rules" and dst.name == "app.yaml" and call_no == 1:
            src = CANNED / "bad" / "rules_app_bad.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    print(json.dumps({"total_tokens": 1000, "vendor": args.vendor}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
