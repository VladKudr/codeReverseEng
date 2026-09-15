# codeReverseEng

Два независимых подпроекта, язык кода и общения — русский.

* **req-reverse v3** (корень: `runner.py`, `checks/`, `prompts/`, `schemas/`,
  `tests/`) — обратная разработка требований из кода. Подробности — `README.md`.
  Тесты: `python -m pytest tests/ -q` (нужны `pyyaml`, `jsonschema`).
* **player_tracker** (`player_tracker/`) — слежение за одним футболистом на
  видео с iPhone 15 Pro Max с повторным захватом, метриками, журналом ошибок
  и веб-приложением. Пользовательская документация —
  `player_tracker/README.md`; **контекст решений, параметры, что проверено и
  что дальше — `player_tracker/HANDOFF.md` (читать перед продолжением
  работы над трекером)**. Тесты: `cd player_tracker && python -m pytest tests/ -q`.
  Ядро не должно требовать torch; тесты — без нейросетей.
