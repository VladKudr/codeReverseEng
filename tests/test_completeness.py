"""Полнота счётом: правила ↔ истории ↔ трассировка (DoD: внятная ошибка)."""
import shutil

from checks import completeness

TEMPLATE_HEADER = "req_id,story_id,rule_id,source,confidence,verdict_stage90\n"


def _copy_canned(canned, tmp_path):
    rules = tmp_path / "rules"
    shutil.copytree(canned / "rules", rules)
    stories = tmp_path / "stories.md"
    shutil.copy(canned / "stories.md", stories)
    trace = tmp_path / "traceability.csv"
    shutil.copy(canned / "traceability.csv", trace)
    return rules, stories, trace


def test_consistent_artifacts_pass(canned, tmp_path):
    rules, stories, trace = _copy_canned(canned, tmp_path)
    assert completeness.check(rules, stories, trace).errors == []


def test_rule_missing_from_traceability(canned, tmp_path):
    # известный провал: из 125 правил в трассировку попало 121
    rules, stories, trace = _copy_canned(canned, tmp_path)
    lines = trace.read_text(encoding="utf-8").splitlines(keepends=True)
    trace.write_text("".join(l for l in lines if "subj-R-004" not in l),
                     encoding="utf-8")
    errors = completeness.check(rules, stories, trace).errors
    assert any("subj-R-004" in f.path for f in errors)
    assert any("уникальных rule_id" in f.expected for f in errors)


def test_rule_missing_from_stories_coverage(canned, tmp_path):
    rules, stories, trace = _copy_canned(canned, tmp_path)
    text = stories.read_text(encoding="utf-8")
    stories.write_text(text.replace(
        "| subj-R-004 | не покрыто (техническое: размер пачки не виден пользователю) |\n",
        ""), encoding="utf-8")
    errors = completeness.check(rules, stories, trace).errors
    assert any("subj-R-004" in f.path and "не встречается" in f.got for f in errors)


def test_homemade_columns_are_rejected(canned, tmp_path):
    # известный провал: агент выдал свои колонки без story_id
    rules, stories, trace = _copy_canned(canned, tmp_path)
    body = trace.read_text(encoding="utf-8").split("\n", 1)[1]
    trace.write_text("requirement,rule,file\n" + body, encoding="utf-8")
    errors = completeness.check(rules, stories, trace).errors
    assert any(f.path == "заголовок" and "story_id" in f.expected for f in errors)


def test_unknown_rule_id_in_traceability(canned, tmp_path):
    rules, stories, trace = _copy_canned(canned, tmp_path)
    with trace.open("a", encoding="utf-8") as fh:
        fh.write("FR-099,,subj-R-999,app.py:1,low,не проверялось\n")
    errors = completeness.check(rules, stories, trace).errors
    assert any("subj-R-999" in f.path for f in errors)
