#!/usr/bin/env python3
"""Мок CLI-агента без инструментов (режимы agent.mode: stdout / text).

Эмулирует корпоративный CLI, который НЕ может подтверждать инструментальные
действия: ничего не читает из репозитория и ничего не пишет — получает промпт,
печатает артефакты в stdout FILE-блоками по протоколу runner'а.

Ожидаемые пути артефактов берёт из списка «Ожидаемые артефакты» в промпте,
содержимое — из fixtures/canned (переиспользует mock_agent.canned_for).
С флагом --expect-inputs проверяет, что детерминированный слой действительно
вложил входы INPUT-блоками (режим text).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import mock_agent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, type=Path)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--expect-inputs", action="store_true",
                    help="упасть, если runner не вложил INPUT-блоки (режим text)")
    ap.add_argument("--vendor", default="mock")
    args = ap.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8")
    assert "===FILE:" in prompt or "FILE-блоками" in prompt or "===END FILE===" in prompt, \
        "в промпте нет протокола вывода stdout"
    if args.expect_inputs:
        assert "===INPUT:" in prompt, "режим text: входы не вложены в промпт"

    expected = re.findall(r"^\* `(.+?)`$", prompt, re.MULTILINE)
    assert expected, "в промпте нет списка ожидаемых артефактов"

    outputs: dict[str, Path] = {}
    for p in expected:
        for src, dst in mock_agent.canned_for(Path(p)):
            outputs[str(dst)] = src

    for dst, src in outputs.items():
        mock_agent.bump_state(args.state,
                              f"{Path(dst).parent.name}/{Path(dst).name}")
        body = src.read_text(encoding="utf-8")
        sys.stdout.write(f"===FILE: {dst}===\n")
        sys.stdout.write(body if body.endswith("\n") else body + "\n")
        sys.stdout.write("===END FILE===\n")

    print(json.dumps({"total_tokens": 500, "vendor": args.vendor}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
