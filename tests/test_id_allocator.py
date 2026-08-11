"""Воспроизведение известного провала: два модуля независимо взяли R-101…R-104."""
import yaml

from checks import id_allocator


def _rules_file(path, ids, module="m.py"):
    path.write_text(yaml.safe_dump({
        "repo": "subj", "module": module,
        "rules": [{"id": i, "statement": "x", "category": "other",
                   "evidence": "x\n", "source": "m.py:1", "enclosing": "f",
                   "confidence": "low"} for i in ids],
    }, allow_unicode=True), encoding="utf-8")


def test_known_collision_r101_r104_is_detected(tmp_path):
    # сценарий провала v2: оба модуля начали свой блок с 101
    _rules_file(tmp_path / "alpha.yaml",
                [f"subj-R-{n}" for n in range(101, 105)], "alpha.py")
    _rules_file(tmp_path / "beta.yaml",
                [f"subj-R-{n}" for n in range(101, 105)], "beta.py")
    result = id_allocator.check_collisions(tmp_path)
    assert len(result.errors) == 4
    assert all("alpha.yaml" in f.artifact and "beta.yaml" in f.artifact
               for f in result.errors)


def test_sequential_allocation_prevents_collision(tmp_path):
    # «параллельные» модули получают блоки от аллокатора ДО запуска — коллизия
    # невозможна: второй блок начинается со следующей свободной сотни
    start_a = id_allocator.next_block_start(tmp_path)
    assert start_a == 1
    _rules_file(tmp_path / "alpha.yaml",
                [f"subj-R-{n:03d}" for n in range(start_a, start_a + 4)], "alpha.py")

    start_b = id_allocator.next_block_start(tmp_path)
    assert start_b == 101
    _rules_file(tmp_path / "beta.yaml",
                [f"subj-R-{n:03d}" for n in range(start_b, start_b + 4)], "beta.py")

    assert id_allocator.check_collisions(tmp_path).errors == []
    # следующий модуль — 201, даже если блок 101 занят не целиком
    assert id_allocator.next_block_start(tmp_path) == 201


def test_canned_rules_have_no_collisions(canned):
    result = id_allocator.check_collisions(canned / "rules")
    assert result.errors == []
    assert result.stats["ids_used"] == 7
