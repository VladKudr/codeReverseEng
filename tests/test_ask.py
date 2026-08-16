"""--ask: вопрос к базе знаний через промпт 93 — без сканирования репозитория."""
import json

import yaml

import runner as runner_mod
from test_runner import make_config, run
from test_text_mode import make_text_config


def test_ask_renders_question_and_orientation_rules(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)
    assert run(cfg_path, stages="0,1,4") == 0  # база построена

    rc = runner_mod.main(["--config", str(cfg_path),
                          "--ask", "Какой лимит перевода?"])
    assert rc == 0
    answer = ws / "answers" / "answer_001.md"
    assert answer.is_file()
    assert "amount exceeds transfer limit" in answer.read_text(encoding="utf-8")

    prompt = (ws / "_scratch" / "prompts" / "93_ask_001.md").read_text(encoding="utf-8")
    # вопрос подставлен, ориентация на артефакты и запрет сканирования на месте
    assert "Какой лимит перевода?" in prompt
    assert "{QUESTION}" not in prompt
    assert "источник истины" in prompt
    assert "Сканировать репозиторий" in prompt
    assert "traceability.csv" in prompt

    calls = json.loads(state.read_text())
    assert calls["answers/answer_001.md"] == 1


def test_ask_numbers_answers_sequentially(tmp_path, subject):
    cfg_path, ws, _ = make_config(tmp_path, subject)
    assert run(cfg_path, stages="0") == 0
    assert runner_mod.main(["--config", str(cfg_path), "--ask", "Вопрос один?"]) == 0
    assert runner_mod.main(["--config", str(cfg_path), "--ask", "Вопрос два?"]) == 0
    assert (ws / "answers" / "answer_001.md").is_file()
    assert (ws / "answers" / "answer_002.md").is_file()


def test_ask_in_text_mode_inlines_knowledge_base(tmp_path, subject):
    # режим text: агенту физически некуда «бегать» — база вложена в промпт
    cfg_path, ws, _ = make_text_config(tmp_path, subject, "text")
    assert run(cfg_path, stages="0,1,4") == 0

    rc = runner_mod.main(["--config", str(cfg_path),
                          "--ask", "Что происходит при отрицательной сумме?"])
    assert rc == 0
    prompt = (ws / "_scratch" / "prompts" / "93_ask_001.md").read_text(encoding="utf-8")
    assert "===INPUT:" in prompt
    assert "subj-R-102" in prompt          # правила вложены
    assert "БЕЗ доступа к файловой системе" in prompt


def test_ask_without_agent_command_fails_fast(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"]["command"] = ""
    cfg["agent"].pop("by_stage", None)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    rc = runner_mod.main(["--config", str(cfg_path), "--ask", "Вопрос?"])
    assert rc == 2
    assert not state.exists()
