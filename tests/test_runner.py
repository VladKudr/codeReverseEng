"""E2E оркестратора на мини-репозитории с мок-агентом.

Проверяется контур целиком: план из инвентаризации, фильтр тестов, фан-аут,
выдача блоков ID, маскирование секретов, ретрай по тексту ошибки, эскалация,
идемпотентность повторного запуска, кросс-вендорное правило этапа 90.
"""
import json
import sys
from pathlib import Path

import yaml

import runner as runner_mod

MOCK = Path(__file__).resolve().parent / "mock_agent.py"


def make_config(tmp_path, subject, *, extra_agent_args="", by_stage_90=True,
                tiers=None, inverse=False):
    ws = tmp_path / "ws"
    state = tmp_path / "mock_state.json"
    base_cmd = (f"{sys.executable} {MOCK} --state {state} "
                f"--prompt {{prompt_file}} --out {{out_file}} {extra_agent_args}")
    cfg = {
        "repo": {"path": str(subject), "name": "subj"},
        "workspace": str(ws),
        "agent": {
            "command": base_cmd.strip(),
            "timeout_sec": 60,
            "metrics_regex": r'"total_tokens":\s*([0-9]+)',
        },
        "models": {"0": "cheap", "4": "cheap", "90": "strong", "92": "strong"},
        "module_filter": {"exclude": ["tests/*"], "min_lines": 0},
        "thresholds": {"stage90_confirmed_pct": 97},
        "retries": 2,
    }
    if by_stage_90:
        cfg["agent"]["by_stage"] = {
            "90": (base_cmd + " --vendor other").strip(),
            "92": (base_cmd + " --vendor other").strip(),
        }
    if tiers is not None:
        cfg["tiers"] = tiers
    if inverse:
        cfg["validation"] = {"inverse": True}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path, ws, state


def run(cfg_path, stages="0,1,2,3,4,5,6,7,90", force=False):
    args = ["--config", str(cfg_path), "--stages", stages]
    if force:
        args.append("--force")
    return runner_mod.main(args)


def test_full_pipeline_end_to_end(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)
    rc = run(cfg_path)
    assert rc == 0

    # артефакты всех этапов на месте, без ручной подстановки переменных
    assert (ws / "inventory" / "subj.yaml").is_file()
    assert (ws / "extracts" / "subj" / "modules" / "app.yaml").is_file()
    assert (ws / "extracts" / "subj" / "modules" / "billing.yaml").is_file()
    assert (ws / "extracts" / "subj" / "api.cli.yaml").is_file()
    assert (ws / "extracts" / "subj" / "rules" / "app.yaml").is_file()
    assert (ws / "final" / "SRS.md").is_file()
    assert (ws / "final" / "traceability.csv").is_file()
    assert (ws / "validation" / "validation.csv").is_file()
    assert (ws / "validation" / "layer1_report.md").is_file()
    assert (ws / "run.log").is_file()
    assert (ws / "reports" / "coverage.md").is_file()

    calls = json.loads(state.read_text())
    # фильтр: тестовый модуль не пошёл в фан-аут этапов 1/4
    assert not any("test_app" in k for k in calls)
    # ретрай: первый выход rules/app.yaml был битым, runner дал ошибку и
    # получил исправленный со второй попытки
    assert calls["rules/app.yaml"] == 2
    assert calls["rules/billing.yaml"] == 1

    # маскирование секретов сработало до принятия артефакта
    rules_billing = (ws / "extracts" / "subj" / "rules" / "billing.yaml").read_text(encoding="utf-8")
    assert "S3cr3tPass" not in rules_billing
    assert "***MASKED***" in rules_billing

    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "04_rules_app | ок | 1" in summary          # один ретрай
    assert "НЕ достигнут" in summary                   # 6/7 < 97%
    assert "90.0%" in summary                          # полнота по coverage
    assert "Секретов не найдено" not in summary        # маскирование в сводке

    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "исключён фильтром" in log


def test_second_run_is_idempotent(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)
    assert run(cfg_path) == 0
    calls_before = json.loads(state.read_text())

    assert run(cfg_path) == 0
    calls_after = json.loads(state.read_text())
    # ни одного нового вызова агента: валидные выходы пропущены
    assert calls_after == calls_before

    # --force перегенерирует
    assert run(cfg_path, stages="2", force=True) == 0
    calls_forced = json.loads(state.read_text())
    assert calls_forced["subj/domain.yaml"] == calls_before["subj/domain.yaml"] + 1


def test_escalation_after_exhausted_retries(tmp_path, subject):
    cfg_path, ws, state = make_config(
        tmp_path, subject, extra_agent_args="--always-bad rules/billing.yaml")
    rc = run(cfg_path, stages="0,1,4")
    assert rc == 1  # эскалация — ненулевой код

    calls = json.loads(state.read_text())
    assert calls["rules/billing.yaml"] == 3  # попытка + 2 ретрая

    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "04_rules_billing | эскалация | 2" in summary
    # текст ошибки в сводке внятен: артефакт, поле, ожидание
    assert "evidence" in summary


def test_retry_prompt_contains_error_text(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject)
    run(cfg_path, stages="0,4")
    retry_prompt = (ws / "_scratch" / "prompts" / "04_rules_app.retry1.md")
    assert retry_prompt.is_file()
    text = retry_prompt.read_text(encoding="utf-8")
    assert "ИСПРАВЛЕНИЕ ОШИБОК" in text
    assert "subj-R-001" in text           # конкретное правило
    assert "Как исправить" in text        # диагностика, а не «попробуй ещё»


def test_cross_vendor_rule_blocks_stage_90(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject, by_stage_90=False)
    rc = run(cfg_path)
    assert rc == 1
    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "кросс-вендор" in summary


def test_dry_run_calls_no_agent(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)
    rc = runner_mod.main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert not state.exists()
