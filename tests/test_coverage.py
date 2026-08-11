"""Покрытие ветвлений: числа воспроизводимы, функции без правил находятся."""
import textwrap

from checks import coverage


def test_subject_numbers_are_reproducible(subject, canned):
    r1 = coverage.analyze(subject, canned / "rules", ["app.py", "billing.py"])
    r2 = coverage.analyze(subject, canned / "rules", ["app.py", "billing.py"])
    assert r1 == r2  # детерминированность

    by_module = {m["module"]: m for m in r1["modules"]}
    app = by_module["app.py"]
    assert app["method"] == "ast"
    # app.py: if:12, raise:13, if:14, raise:15, if:21, raise:22, if:35 — 7 веток;
    # правила покрывают всё, кроме ветки audit_note (if:35)
    assert app["branches_total"] == 7
    assert app["branches_covered"] == 6
    assert app["uncovered"] == [{"kind": "if", "line": 35}]
    assert app["functions_without_rules"] == ["audit_note"]

    billing = by_module["billing.py"]
    assert billing["branches_total"] == 3
    assert billing["branches_covered"] == 3
    assert billing["functions_without_rules"] == []

    assert r1["branches_total"] == 10
    assert r1["branches_covered"] == 9
    assert r1["coverage_pct"] == 90.0


def test_report_lists_uncovered_functions_for_reextraction(subject, canned):
    md = coverage.render_md(
        coverage.analyze(subject, canned / "rules", ["app.py", "billing.py"]))
    assert "audit_note" in md
    assert "повторную экстракцию" in md


def test_non_python_uses_regex_and_says_so(tmp_path, canned):
    code = textwrap.dedent("""\
        function f(x) {
            if (x < 0) {
                throw new Error("bad");
            }
            var y = x + 1;
            switch (y) {
            }
        }
    """)
    (tmp_path / "f.js").write_text(code, encoding="utf-8")
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "f.yaml").write_text(textwrap.dedent("""\
        repo: subj
        module: f.js
        rules:
          - id: subj-R-001
            statement: "x"
            category: other
            evidence: |
              if (x < 0) {
            source: f.js:2
            enclosing: f
            confidence: low
    """), encoding="utf-8")
    report = coverage.analyze(tmp_path, rules, ["f.js"])
    m = report["modules"][0]
    assert m["method"] == "regex"        # приближение помечено честно
    assert m["branches_total"] == 3      # if, throw(=raise), switch(=match)
    assert m["branches_covered"] == 2    # if:2 и throw:3 в допуске ±3; switch:6 — нет
