"""Схемная валидация артефактов пайплайна.

Каждый YAML-артефакт этапов 0–4 проверяется своей JSON Schema из schemas/;
validation.csv этапа 90 предварительно преобразуется в список объектов.
Ошибки возвращаются в формате, пригодном для retry-промпта: путь до поля,
что ожидалось, что получено.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import jsonschema
import yaml

from ._common import CheckResult, Finding

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

# тип артефакта -> файл схемы
SCHEMA_BY_KIND = {
    "inventory": "inventory.schema.json",
    "module": "module.schema.json",
    "domain": "domain.schema.json",
    "interface": "interface.schema.json",
    "rules": "rules.schema.json",
    "validation": "validation.schema.json",
    "predictions": "predictions.schema.json",
}


def load_schema(kind: str) -> dict:
    path = SCHEMAS_DIR / SCHEMA_BY_KIND[kind]
    return json.loads(path.read_text(encoding="utf-8"))


def _json_path(error: jsonschema.ValidationError) -> str:
    parts = []
    for p in error.absolute_path:
        parts.append(f"[{p}]" if isinstance(p, int) else ("." + str(p) if parts else str(p)))
    return "".join(parts) or "(корень документа)"


def _expected(error: jsonschema.ValidationError) -> str:
    v, s = error.validator, error.validator_value
    if v == "required":
        return f"обязательные поля: {', '.join(s)}"
    if v == "enum":
        return "одно из значений: " + ", ".join(map(str, s))
    if v == "const":
        return f"значение {s!r}"
    if v == "pattern":
        return f"строка по образцу {s}"
    if v == "type":
        return f"тип {s}"
    if v == "minLength":
        return "непустая строка"
    if v == "oneOf":
        return "ровно один из вариантов схемы (OpenAPI с paths либо cli-контракт с openapi_applicable: false)"
    return error.message


def _got(error: jsonschema.ValidationError) -> str:
    inst = error.instance
    text = json.dumps(inst, ensure_ascii=False, default=str)
    if len(text) > 160:
        text = text[:160] + "…"
    return text


def validate_data(data, kind: str, artifact: str) -> CheckResult:
    result = CheckResult(check=f"schema:{kind}")
    validator = jsonschema.Draft202012Validator(load_schema(kind))
    for error in sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path))):
        result.findings.append(Finding(
            artifact=artifact,
            path=_json_path(error),
            expected=_expected(error),
            got=_got(error),
        ))
    return result


def load_artifact(path: Path, kind: str):
    """YAML для этапов 0–4; CSV этапа 90 -> список объектов."""
    if kind == "validation" and path.suffix == ".csv":
        with path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_file(path: Path, kind: str) -> CheckResult:
    artifact = str(path)
    try:
        data = load_artifact(path, kind)
    except yaml.YAMLError as e:
        r = CheckResult(check=f"schema:{kind}")
        r.findings.append(Finding(
            artifact=artifact, path="(весь файл)",
            expected="валидный YAML без markdown-обёрток",
            got=f"ошибка разбора: {str(e)[:200]}",
            hint="перезапиши файл: только YAML, без ``` и пояснений",
        ))
        return r
    if data is None:
        r = CheckResult(check=f"schema:{kind}")
        r.findings.append(Finding(
            artifact=artifact, path="(весь файл)",
            expected="непустой артефакт", got="файл пуст",
        ))
        return r
    return validate_data(data, kind, artifact)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Схемная валидация артефакта")
    ap.add_argument("--kind", required=True, choices=sorted(SCHEMA_BY_KIND))
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    rc = 0
    for path in args.files:
        result = validate_file(path, args.kind)
        if result.errors:
            rc = 1
            for f in result.errors:
                print(f.render())
        else:
            print(f"{path}: схема {args.kind} — ок")
    return rc


if __name__ == "__main__":
    sys.exit(main())
