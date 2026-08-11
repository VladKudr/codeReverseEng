"""Ярусная глубина реверса (DoD-9): фильтр плана по карте, --raise-tier."""
import json

import runner as runner_mod
from test_runner import make_config, run


def calls(state):
    return json.loads(state.read_text()) if state.exists() else {}


def test_l2_zone_gets_full_stages_others_l0(tmp_path, subject):
    # аналог DoD-9: default L0, одна зона L2 — полные этапы только для неё
    cfg_path, ws, state = make_config(
        tmp_path, subject,
        tiers={"default": "L0", "map": [{"path": "app.py", "tier": "L2"}]})
    assert run(cfg_path, stages="0,1,2,3,4") == 0

    c = calls(state)
    # этап 1 (ярус L0) — по всем модулям
    assert "modules/app.yaml" in c and "modules/billing.yaml" in c
    # этап 4 — только для зоны L2
    assert "rules/app.yaml" in c
    assert "rules/billing.yaml" not in c
    assert not (ws / "extracts" / "subj" / "rules" / "billing.yaml").exists()
    # этапы 2–3 выполняются: в прогоне есть зона L1 и выше
    assert "subj/domain.yaml" in c


def test_all_l0_skips_stages_2_and_above(tmp_path, subject):
    cfg_path, ws, state = make_config(tmp_path, subject, tiers={"default": "L0"})
    assert run(cfg_path, stages="0,1,2,3,4,5,6,7,90") == 0

    c = calls(state)
    assert "modules/app.yaml" in c            # L0 = этапы 0–1
    assert "subj/domain.yaml" not in c        # этап 2 требует L1
    assert not any(k.startswith("rules/") for k in c)
    log = (ws / "run.log").read_text(encoding="utf-8")
    assert "этап 2 пропущен: требует яруса L1" in log
    assert "этап 4 пропущен: требует яруса L2" in log


def test_raise_tier_catches_up_without_regenerating(tmp_path, subject):
    # DoD-9: --raise-tier на другой зоне догоняет недостающие этапы,
    # не перегенерируя валидные артефакты нижних ярусов
    tiers = {"default": "L0", "map": [{"path": "app.py", "tier": "L2"}]}
    cfg_path, ws, state = make_config(tmp_path, subject, tiers=tiers)
    assert run(cfg_path, stages="0,1,4") == 0
    before = calls(state)
    assert "rules/billing.yaml" not in before

    # в холодную зону пришёл change — поднимаем по требованию
    rc = runner_mod.main(["--config", str(cfg_path), "--stages", "0,1,4",
                          "--raise-tier", "billing.py", "L2"])
    assert rc == 0
    after = calls(state)
    # догнан только недостающий этап 4 новой зоны
    assert after["rules/billing.yaml"] == 1
    # готовые артефакты не перегенерированы: счётчики прочих вызовов не выросли
    assert {k: v for k, v in after.items() if k != "rules/billing.yaml"} == before
    # поднятие сохранено в workspace и переживает следующий прогон
    assert (ws / "tiers.yaml").is_file()
    assert run(cfg_path, stages="0,1,4") == 0
    assert calls(state) == after


def test_no_tiers_section_means_full_run(tmp_path, subject):
    # обратная совместимость с базовой постановкой: без секции tiers — всё L2
    cfg_path, _, state = make_config(tmp_path, subject)
    assert run(cfg_path, stages="0,1,4") == 0
    c = calls(state)
    assert "rules/app.yaml" in c and "rules/billing.yaml" in c


def test_tier_of_longest_prefix_wins(tmp_path, subject):
    cfg_path, _, _ = make_config(
        tmp_path, subject,
        tiers={"default": "L0", "map": [
            {"path": "src/", "tier": "L1"},
            {"path": "src/payments/", "tier": "L2"},
        ]})
    r = runner_mod.Runner(runner_mod.Config.load(cfg_path))
    assert r.tier_of("src/payments/core.py") == "L2"
    assert r.tier_of("src/other/x.py") == "L1"
    assert r.tier_of("lib/x.py") == "L0"
    # каталог src2 не должен ловиться префиксом src/
    assert r.tier_of("src2/x.py") == "L0"
