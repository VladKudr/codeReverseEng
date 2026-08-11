"""Этап 92 — инверсная верификация (DoD-10): воспроизводимость и флаг конфига."""
import json

from checks import inverse_match, schema_check
from test_runner import make_config, run


def test_canned_predictions_pass_schema(canned):
    for p in sorted((canned / "inverse").glob("*.predictions.yaml")):
        assert schema_check.validate_file(p, "predictions").errors == [], p.name


def test_shares_per_rule_and_reproducibility(subject, canned):
    r1 = inverse_match.analyze(subject, canned / "inverse")
    r2 = inverse_match.analyze(subject, canned / "inverse")
    assert r1 == r2  # отчёт воспроизводим: LLM-выход зафиксирован фикстурой

    by_rule = {r["rule_id"]: r for r in r1["rules"]}
    # предсказания восстанавливаются из кода…
    assert by_rule["subj-R-001"]["share_pct"] == 100.0
    assert by_rule["subj-R-003"]["share_pct"] == 100.0
    assert by_rule["subj-R-004"]["share_pct"] == 100.0
    assert by_rule["subj-R-101"]["share_pct"] == 100.0
    # …кроме порога R-002: в коде лимит записан константой (10_000), из
    # формулировки буквальное "amount > 10000" не восстанавливается
    assert by_rule["subj-R-002"]["expects_confirmed"] == 1
    assert by_rule["subj-R-002"]["share_pct"] == 50.0
    assert by_rule["subj-R-002"]["unconfirmed"] == ["amount > 10000"]

    assert r1["total"] == 11
    assert r1["confirmed"] == 10
    assert r1["share_pct"] == 90.9


def test_number_placeholder_matches_any_literal(tmp_path):
    (tmp_path / "m.py").write_text("TIMEOUT = 42.5\n", encoding="utf-8")
    idx = inverse_match.SourceIndex(tmp_path)
    assert inverse_match.match_expect(
        idx, {"text": "timeout = <N>"})["confirmed"]
    assert not inverse_match.match_expect(
        idx, {"text": "retries = <N>"})["confirmed"]


def test_normalization_ignores_case_and_spacing(tmp_path):
    (tmp_path / "m.py").write_text('raise ValueError(  "Bad Input"  )\n',
                                   encoding="utf-8")
    idx = inverse_match.SourceIndex(tmp_path)
    assert inverse_match.match_expect(
        idx, {"text": 'raise valueerror( "bad input" )'})["confirmed"]


def test_runner_stage_92_end_to_end(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject, inverse=True)
    assert run(cfg_path, stages="0,1,4,92") == 0

    calls = json.loads(state.read_text())
    assert calls["inverse/app.predictions.yaml"] == 1
    assert calls["inverse/billing.predictions.yaml"] == 1

    # детерминированный слой подготовил формулировки БЕЗ evidence и source
    import yaml
    st = yaml.safe_load((ws / "validation" / "inverse" / "app.statements.yaml")
                        .read_text(encoding="utf-8"))
    assert st["rules"], "формулировки не пусты"
    assert all(set(r.keys()) == {"id", "statement"} for r in st["rules"])

    report = (ws / "reports" / "inverse_match.md").read_text(encoding="utf-8")
    assert "10/11" in report and "90.9%" in report
    assert "не калиброваны" in report
    assert "amount > 10000" in report  # кандидат на возврат в этап 4 назван

    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "[92] подтверждено предсказаний 10/11" in log


def test_stage_92_is_off_by_default(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)  # без validation.inverse
    assert run(cfg_path, stages="0,1,4,92") == 0
    calls = json.loads(state.read_text())
    assert not any(k.startswith("inverse/") for k in calls)
    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "validation.inverse: false" in log


def test_stage_92_cross_vendor_rule(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject, inverse=True,
                                  by_stage_90=False)
    rc = run(cfg_path, stages="0,1,4,92")
    assert rc == 1
    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "92" in summary and "кросс-вендор" in summary
