"""Схемная валидация: валидные артефакты проходят, порча даёт retry-ошибку."""
import shutil

import yaml

from checks import schema_check


def test_all_canned_artifacts_pass(canned):
    cases = [
        (canned / "inventory.yaml", "inventory"),
        (canned / "modules" / "app.yaml", "module"),
        (canned / "modules" / "billing.yaml", "module"),
        (canned / "domain.yaml", "domain"),
        (canned / "api.cli.yaml", "interface"),
        (canned / "rules" / "app.yaml", "rules"),
        (canned / "rules" / "billing.yaml", "rules"),
        (canned / "validation.csv", "validation"),
    ]
    for path, kind in cases:
        result = schema_check.validate_file(path, kind)
        assert result.errors == [], f"{path.name}: {[f.render() for f in result.errors]}"


def test_tenacity_fixture_rules_pass_schema(tenacity_rules):
    for p in sorted(tenacity_rules.glob("*.yaml")):
        assert schema_check.validate_file(p, "rules").errors == [], p.name


def test_deliberate_corruption_yields_retry_error(canned, tmp_path):
    # намеренная порча: у правила удаляем category и ломаем формат source
    data = yaml.safe_load((canned / "rules" / "app.yaml").read_text(encoding="utf-8"))
    del data["rules"][0]["category"]
    data["rules"][1]["source"] = "app.py"          # нет :строки
    data["rules"][2]["confidence"] = "sure"        # вне enum
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    errors = schema_check.validate_file(p, "rules").errors
    rendered = "\n".join(f.render() for f in errors)
    # ошибка внятная: путь до поля, что ожидалось, что получено
    assert "rules[0]" in rendered and "category" in rendered
    assert "rules[1].source" in rendered and "файл:строка" in rendered or \
           "образцу" in rendered
    assert "rules[2].confidence" in rendered and "high" in rendered


def test_broken_yaml_yields_readable_error(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("rules:\n  - id: [unclosed\n", encoding="utf-8")
    errors = schema_check.validate_file(p, "rules").errors
    assert len(errors) == 1
    assert "YAML" in errors[0].expected


def test_validation_csv_bad_verdict(canned, tmp_path):
    p = tmp_path / "validation.csv"
    shutil.copy(canned / "validation.csv", p)
    text = p.read_text(encoding="utf-8").replace("ПОДТВЕРЖДЕНО", "OK", 1)
    p.write_text(text, encoding="utf-8")
    errors = schema_check.validate_file(p, "validation").errors
    assert any("ПОДТВЕРЖДЕНО" in f.expected for f in errors)


def test_interface_schema_rejects_hybrid(tmp_path):
    # ни OpenAPI, ни честный cli-контракт — oneOf обязан отклонить
    p = tmp_path / "api.cli.yaml"
    p.write_text(yaml.safe_dump({"interface": "cli"}), encoding="utf-8")
    assert schema_check.validate_file(p, "interface").errors
