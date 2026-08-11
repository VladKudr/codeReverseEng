"""Способы передачи промпта агенту: {prompt_file}, {prompt_text}, stdin.

Контракт под CLI вида Qwen Code (`qwen -m <модель> -p <текст>`): текст промпта
уходит одним аргументом argv, кавычки и переносы внутри него не ломают разбор
команды; модель этапа подставляется в {model}.
"""
import json
import sys
import textwrap

import runner as runner_mod
from test_runner import make_config

PROMPT = 'Строка с "кавычками", переносом\nи {фигурными} скобками'


def _echo_agent(tmp_path):
    """Скрипт-агент: пишет свой argv и stdin в файл, переданный первым аргументом."""
    script = tmp_path / "echo_agent.py"
    script.write_text(textwrap.dedent("""\
        import json, sys
        out = sys.argv[1]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"argv": sys.argv[2:], "stdin": sys.stdin.read()
                       if not sys.stdin.isatty() else ""}, fh, ensure_ascii=False)
    """), encoding="utf-8")
    return script


def _runner_with_command(tmp_path, subject, command, stdin=False):
    cfg_path, ws, _ = make_config(tmp_path, subject)
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"]["command"] = command
    cfg["agent"]["stdin"] = stdin
    cfg["models"] = {"0": "qwen3-coder-flash"}
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return runner_mod.Runner(runner_mod.Config.load(cfg_path)), ws


def _read_echo(out):
    return json.loads(out.read_text(encoding="utf-8"))


def test_prompt_text_is_single_argv_element(tmp_path, subject):
    script = _echo_agent(tmp_path)
    out = tmp_path / "echo.json"
    r, _ = _runner_with_command(
        tmp_path, subject,
        f"{sys.executable} {script} {out} -m {{model}} -p {{prompt_text}}")
    r.call_agent("0", "t_text", PROMPT, out)
    got = _read_echo(out)
    # текст промпта дошёл целиком одним аргументом, кавычки/переносы/скобки целы
    assert got["argv"] == ["-m", "qwen3-coder-flash", "-p", PROMPT]


def test_prompt_file_contains_rendered_prompt(tmp_path, subject):
    script = _echo_agent(tmp_path)
    out = tmp_path / "echo.json"
    r, ws = _runner_with_command(
        tmp_path, subject,
        f"{sys.executable} {script} {out} --prompt-file {{prompt_file}}")
    r.call_agent("0", "t_file", PROMPT, out)
    got = _read_echo(out)
    prompt_path = got["argv"][got["argv"].index("--prompt-file") + 1]
    assert prompt_path == str(ws / "_scratch" / "prompts" / "t_file.md")
    from pathlib import Path
    assert Path(prompt_path).read_text(encoding="utf-8") == PROMPT


def test_stdin_mode_feeds_prompt_to_agent(tmp_path, subject):
    script = _echo_agent(tmp_path)
    out = tmp_path / "echo.json"
    r, _ = _runner_with_command(
        tmp_path, subject,
        f"{sys.executable} {script} {out} -m {{model}}", stdin=True)
    r.call_agent("0", "t_stdin", PROMPT, out)
    got = _read_echo(out)
    assert got["stdin"] == PROMPT
    assert got["argv"] == ["-m", "qwen3-coder-flash"]


def test_empty_command_stops_early_with_hint(tmp_path, subject):
    # поставка нейтральна к инструменту: без команды — внятный останов
    # до первого вызова, а не шторм ретраев
    cfg_path, ws, state = make_config(tmp_path, subject)
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"]["command"] = ""
    cfg["agent"].pop("by_stage", None)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    rc = runner_mod.main(["--config", str(cfg_path), "--stages", "0,1,4"])
    assert rc == 2
    assert not state.exists()  # ни одного вызова агента
    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "agent.command" in log and "RUNBOOK" in log


def test_empty_by_stage_falls_back_to_default_command(tmp_path, subject):
    # пустое переопределение by_stage = «не задано»: кросс-вендорная проверка
    # этапа 90 обязана сработать, а не получить пустую команду
    cfg_path, ws, _ = make_config(tmp_path, subject)
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"]["by_stage"] = {"90": "", "92": ""}
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    r = runner_mod.Runner(runner_mod.Config.load(cfg_path))
    assert r.cfg.command_for("90") == r.cfg.command_for("4")
