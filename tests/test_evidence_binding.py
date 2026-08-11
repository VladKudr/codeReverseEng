"""Слой 1 на данных tenacity: известный ответ — 1 сокращённая цитата из 145."""
import textwrap

import yaml

from checks import evidence_binding


def test_tenacity_finds_the_one_truncated_quote(tenacity_repo, tenacity_rules):
    results = evidence_binding.check_rules_dir(tenacity_rules, tenacity_repo)
    total = sum(r.stats["rules_total"] for r in results)
    errors = [f for r in results for f in r.errors]

    assert total == 145
    # ровно одна ошибка — сокращённая цитата, без ложных срабатываний на 144
    assert len(errors) == 1
    assert "tenacity-R-301" in errors[0].path
    assert errors[0].path.endswith(".evidence")
    # и ни одного предупреждения по enclosing на дословных цитатах
    assert sum(len(r.warnings) for r in results) == 0


def test_error_text_is_retry_ready(tenacity_repo, tenacity_rules):
    from checks._common import render_for_retry

    results = evidence_binding.check_rules_dir(tenacity_rules, tenacity_repo)
    text = render_for_retry(results)
    # текст ошибки самодостаточен: артефакт, поле, ожидание, как исправить
    assert "wait.yaml" in text
    assert "tenacity-R-301" in text
    assert "Как исправить" in text


def _write_rules(tmp_path, rules, module="app.py"):
    p = tmp_path / "rules"
    p.mkdir(exist_ok=True)
    (p / "x.yaml").write_text(yaml.safe_dump(
        {"repo": "subj", "module": module, "rules": rules},
        allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def test_wrong_enclosing_is_error_for_python(subject, tmp_path):
    rules = [{
        "id": "subj-R-001",
        "statement": "Проверка enclosing.",
        "category": "validation",
        "evidence": 'raise ValueError("amount must be positive")\n',
        "source": "app.py:13",
        "enclosing": "transfer",  # на деле строка лежит в validate_amount
        "confidence": "high",
    }]
    result = evidence_binding.check_rules_file(
        next(_write_rules(tmp_path, rules).glob("*.yaml")), subject)
    errs = result.errors
    assert len(errs) == 1
    assert "validate_amount" in errs[0].expected
    assert "AST" in errs[0].expected


def test_wrong_enclosing_is_warning_for_non_python(tmp_path):
    code = textwrap.dedent("""\
        function realOwner(x) {
            if (x < 0) {
                throw new Error("negative");
            }
        }
    """)
    (tmp_path / "app.js").write_text(code, encoding="utf-8")
    rules = [{
        "id": "subj-R-001",
        "statement": "Проверка эвристики.",
        "category": "validation",
        "evidence": 'throw new Error("negative");\n',
        "source": "app.js:3",
        "enclosing": "someOtherFn",
        "confidence": "high",
    }]
    result = evidence_binding.check_rules_file(
        next(_write_rules(tmp_path, rules, "app.js").glob("*.yaml")), tmp_path)
    # эвристика по отступам/скобкам — честно предупреждение, не ошибка
    assert result.errors == []
    assert len(result.warnings) == 1
    assert "эвристикой" in result.warnings[0].expected


def test_masked_evidence_still_binds(subject, tmp_path):
    # после secrets_mask цитата содержит ***MASKED*** — привязка не ломается
    rules = [{
        "id": "subj-R-103",
        "statement": "Строка подключения.",
        "category": "other",
        "evidence": 'DB_URL = "postgresql://billing:***MASKED***@db.internal:5432/billing"\n',
        "source": "billing.py:3",
        "enclosing": "module",
        "confidence": "medium",
    }]
    result = evidence_binding.check_rules_file(
        next(_write_rules(tmp_path, rules, "billing.py").glob("*.yaml")), subject)
    assert result.errors == []
    assert result.warnings == []


def test_nonexistent_file_and_line_out_of_range(subject, tmp_path):
    rules = [
        {"id": "subj-R-001", "statement": "x", "category": "other",
         "evidence": "x\n", "source": "ghost.py:1", "enclosing": "f",
         "confidence": "low"},
        {"id": "subj-R-002", "statement": "x", "category": "other",
         "evidence": "x\n", "source": "app.py:9999", "enclosing": "f",
         "confidence": "low"},
    ]
    result = evidence_binding.check_rules_file(
        next(_write_rules(tmp_path, rules).glob("*.yaml")), subject)
    notes = [f.got for f in result.errors]
    assert len(result.errors) == 2
    assert any("нет" in n for n in notes)
    assert any("9999" in n for n in notes)
