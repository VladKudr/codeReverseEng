"""Режимы для CLI, требующих подтверждений на инструменты.

agent.mode: stdout — агент не пишет файлы (снимает подтверждения записи);
agent.mode: text — агент вообще без инструментов: входы вкладывает runner,
выход — FILE-блоки в stdout (снимает подтверждения полностью).
"""
import json
import sys
from pathlib import Path

import yaml

import runner as runner_mod
from test_runner import make_config, run

TEXT_MOCK = Path(__file__).resolve().parent / "text_mock_agent.py"


def make_text_config(tmp_path, subject, mode, *, expect_inputs=False,
                     max_inline_kb=None):
    cfg_path, ws, state = make_config(tmp_path, subject)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = (f"{sys.executable} {TEXT_MOCK} --state {state} "
            f"--prompt {{prompt_file}}" + (" --expect-inputs" if expect_inputs else ""))
    cfg["agent"]["command"] = base
    cfg["agent"]["mode"] = mode
    cfg["agent"]["by_stage"] = {"90": base + " --vendor other",
                                "92": base + " --vendor other"}
    if max_inline_kb is not None:
        cfg["agent"]["max_inline_kb"] = max_inline_kb
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path, ws, state


# ── единицы: протокол FILE-блоков ───────────────────────────────────────────

def test_parse_file_blocks_and_unknown_path_is_skipped(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject)
    r = runner_mod.Runner(runner_mod.Config.load(cfg_path))
    good = ws / "a.yaml"
    stdout = (
        "болтовня CLI до блоков\n"
        f"===FILE: {good}===\nrepo: subj\n===END FILE===\n"
        "===FILE: /etc/passwd===\nвзлом\n===END FILE===\n"
        '{"total_tokens": 5}\n')
    r.write_outputs_from_stdout("t", stdout, [good])
    assert good.read_text(encoding="utf-8") == "repo: subj\n"
    assert not Path("/etc/passwd").read_text().startswith("взлом")
    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "посторонним путём" in log


def test_fallback_whole_stdout_with_fences(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject)
    r = runner_mod.Runner(runner_mod.Config.load(cfg_path))
    out = ws / "b.yaml"
    r.write_outputs_from_stdout("t", "```yaml\nrepo: subj\n```", [out])
    assert out.read_text(encoding="utf-8") == "repo: subj\n"


# ── E2E: text — вообще без инструментов ─────────────────────────────────────

def test_text_mode_end_to_end(tmp_path, subject):
    cfg_path, ws, state = make_text_config(tmp_path, subject, "text",
                                           expect_inputs=True)
    assert run(cfg_path, stages="0,1,4") == 0

    assert (ws / "inventory" / "subj.yaml").is_file()
    assert (ws / "extracts" / "subj" / "rules" / "app.yaml").is_file()
    # маскирование секретов работает и когда файл пишет runner
    billing = (ws / "extracts" / "subj" / "rules" / "billing.yaml").read_text(encoding="utf-8")
    assert "S3cr3tPass" not in billing and "***MASKED***" in billing

    # детерминированный слой вложил входы: исходник модуля целиком в промпте
    prompt = (ws / "_scratch" / "prompts" / "04_rules_app.md").read_text(encoding="utf-8")
    assert "===INPUT:" in prompt
    assert "def validate_amount(amount):" in prompt
    assert "БЕЗ доступа к файловой системе" in prompt

    calls = json.loads(state.read_text())
    assert calls["rules/app.yaml"] == 1 and calls["rules/billing.yaml"] == 1


def test_text_mode_budget_overflow_escalates_before_agent(tmp_path, subject):
    cfg_path, ws, state = make_text_config(tmp_path, subject, "text",
                                           max_inline_kb=0)
    rc = run(cfg_path, stages="0")
    assert rc == 1
    assert not state.exists()  # агент не вызывался
    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "бюджет вложений" in summary and "max_inline_kb" in summary


def test_text_mode_skips_stage_89(tmp_path, subject):
    cfg_path, ws, state = make_text_config(tmp_path, subject, "text")
    assert run(cfg_path, stages="89") == 0
    assert not state.exists()
    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "требует исполнения" in log


# ── E2E: stdout — агент читает сам, но не пишет ─────────────────────────────

def test_stdout_mode_full_pipeline(tmp_path, subject):
    cfg_path, ws, state = make_text_config(tmp_path, subject, "stdout")
    assert run(cfg_path) == 0  # этапы 0–7 и 90

    for rel in ["inventory/subj.yaml", "extracts/subj/api.cli.yaml",
                "integration/flows.md", "final/stories.md", "final/SRS.md",
                "final/traceability.csv", "validation/validation.csv",
                "validation/validation.md"]:
        assert (ws / rel).is_file(), rel

    # многофайловые этапы собраны из блоков одного вызова
    calls = json.loads(state.read_text())
    assert calls["final/SRS.md"] == 1 and calls["final/traceability.csv"] >= 1

    summary = (ws / "run_summary.md").read_text(encoding="utf-8")
    assert "эскалация" not in summary.split("## Эскалации")[1].split("##")[0] \
        or "Нет." in summary
