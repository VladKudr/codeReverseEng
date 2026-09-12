"""Сценарии автомата слежения на синтетических треках.

Дескрипторы — единичные векторы в 16-мерном пространстве; «тот же игрок» —
базовый вектор с шумом, «одноклубник» — вектор с заметной, но не полной
общей компонентой (одинаковая форма даёт высокое базовое сходство).
"""
import numpy as np
import pytest

from _synth import feature_family, make_track, unit
from tracker.jersey import NumberRead
from tracker.target import TargetConfig, TargetFollower, TargetState

W, H = 1280, 720
DIM = 16


def base_vectors(rng, teammate_overlap: float = 0.75):
    """Цель и одноклубник: общая часть (форма) + индивидуальная часть."""
    shared = unit(rng.normal(size=DIM))
    t_own = unit(rng.normal(size=DIM))
    m_own = unit(rng.normal(size=DIM))
    target = unit(teammate_overlap * shared + (1 - teammate_overlap) * t_own)
    mate = unit(teammate_overlap * shared + (1 - teammate_overlap) * m_own)
    return target, mate


def walk(x0, y0, vx, vy, i, w=60, h=150):
    return [x0 + vx * i, y0 + vy * i, x0 + vx * i + w, y0 + vy * i + h]


def warmup(follower, rng, target_vec, mate_vec, frames=60, noise=0.05):
    """Цель на треке 1 рядом с одноклубником на треке 2: модель и калибровка накапливаются."""
    for i in range(frames):
        t1 = make_track(1, walk(300, 200, 3, 0, i), feature_family(target_vec, rng, noise), i)
        t2 = make_track(2, walk(800, 300, -2, 0, i), feature_family(mate_vec, rng, noise), i)
        if i == 0:
            follower.lock(t1, t1.last_feature, 0)
            continue
        obs = follower.step(i, [t1, t2], {1: t1.last_feature, 2: t2.last_feature})
        assert obs.state == TargetState.ACTIVE and obs.track_id == 1
    return frames


def test_exit_and_reenter_with_lookalike_present(rng):
    """Цель уходит из кадра; пока её нет, появляется одноклубник — захватить его нельзя.
    Когда цель возвращается под новым id, захват должен произойти на ней."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    # цель исчезла; одноклубник на треке 2 остаётся, ещё один одноклубник приходит треком 3
    locked_on = []
    for i in range(fi, fi + 90):
        t2 = make_track(2, walk(800, 300, -2, 0, i), feature_family(mate_vec, rng, 0.05), i)
        t3 = make_track(3, walk(200, 100, 2, 1, i - fi), feature_family(mate_vec, rng, 0.05), i)
        obs = f.step(i, [t2, t3], {2: t2.last_feature, 3: t3.last_feature})
        if obs.track_id is not None:
            locked_on.append(obs.track_id)
    assert not locked_on, f"ложный захват одноклубника: {locked_on}"
    assert f.state == TargetState.LOST
    # цель вернулась треком 4
    got = None
    for i in range(fi + 90, fi + 130):
        t2 = make_track(2, walk(800, 300, -2, 0, i), feature_family(mate_vec, rng, 0.05), i)
        t4 = make_track(4, walk(50, 400, 4, 0, i - fi - 90), feature_family(target_vec, rng, 0.05), i)
        obs = f.step(i, [t2, t4], {2: t2.last_feature, 4: t4.last_feature})
        if obs.event == "reacquired":
            got = (i, obs.track_id)
            break
    assert got is not None and got[1] == 4
    assert got[0] - (fi + 90) < 10


def test_ambiguous_when_two_lookalikes_and_no_number(rng):
    """Два игрока с почти одинаковым дескриптором цели — автомат сообщает ambiguous, не захватывает."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    f.step(fi, [], {})
    assert f.state == TargetState.LOST
    ambiguous = 0
    for i in range(fi + 1, fi + 40):
        a = make_track(7, walk(100, 300, 2, 0, i - fi), feature_family(target_vec, rng, 0.05), i)
        b = make_track(8, walk(900, 300, -2, 0, i - fi), feature_family(target_vec, rng, 0.05), i)
        obs = f.step(i, [a, b], {7: a.last_feature, 8: b.last_feature})
        ambiguous += obs.ambiguous
        assert obs.track_id is None
    assert ambiguous > 20


def test_number_breaks_the_tie(rng):
    """Тот же случай, но у кандидатов прочитаны номера: цель — 10, кандидат с 10 захватывается быстро."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    f.model.numbers.set_known("10")
    f.step(fi, [], {})
    got = None
    for i in range(fi + 1, fi + 40):
        a = make_track(7, walk(100, 300, 2, 0, i - fi), feature_family(target_vec, rng, 0.05), i)
        b = make_track(8, walk(900, 300, -2, 0, i - fi), feature_family(target_vec, rng, 0.05), i)
        reads = {7: NumberRead("10", 0.8), 8: NumberRead("4", 0.8)}
        obs = f.step(i, [a, b], {7: a.last_feature, 8: b.last_feature}, number_reads=reads)
        if obs.event == "reacquired":
            got = (i, obs.track_id)
            break
    assert got is not None and got[1] == 7
    assert got[0] - fi <= 8


def test_number_contradiction_releases_track(rng):
    """Цель на треке, но на нём стабильно читается чужой номер — трек отпускается."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    f.model.numbers.set_known("10")
    events = []
    for i in range(fi, fi + 10):
        t1 = make_track(1, walk(300, 200, 3, 0, i), feature_family(target_vec, rng, 0.05), i)
        obs = f.step(i, [t1], {1: t1.last_feature}, number_reads={1: NumberRead("9", 0.9)})
        events.append(obs.event)
    assert "released_number" in events
    assert 1 in f._cooldown


def test_team_mismatch_rejects_candidate(rng):
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    for i in range(fi, fi + 10):  # команда цели = 0
        t1 = make_track(1, walk(300, 200, 3, 0, i), feature_family(target_vec, rng, 0.05), i)
        f.step(i, [t1], {1: t1.last_feature}, team_labels={1: 0})
    assert f.model.team == 0
    f.step(fi + 10, [], {})
    for i in range(fi + 11, fi + 40):
        c = make_track(5, walk(300, 200, 3, 0, i), feature_family(target_vec, rng, 0.05), i)
        obs = f.step(i, [c], {5: c.last_feature}, team_labels={5: 1})
        if i > fi + 11 + 5:
            assert obs.track_id is None
            assert obs.candidates and obs.candidates[0].rejected == "team"


def test_quick_relock_after_short_detection_gap(rng):
    """Пропуск детекции на несколько кадров: тот же id мультитрекера возвращается — захват за 1 кадр."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec)
    for i in range(fi, fi + 4):
        t1 = make_track(1, walk(300, 200, 3, 0, i), None, i, detected=False)
        obs = f.step(i, [t1], {})
        assert obs.state == TargetState.LOST
    t1 = make_track(1, walk(300, 200, 3, 0, fi + 4), feature_family(target_vec, rng, 0.05), fi + 4)
    obs = f.step(fi + 4, [t1], {1: t1.last_feature})
    assert obs.event == "reacquired" and obs.track_id == 1


def test_id_swap_during_contest_is_detected(rng):
    """Столкновение: после перекрытия трек 1 «уносит» одноклубника, а цель продолжает треком 9.
    Автомат должен отпустить трек 1 по несходству и захватить трек 9."""
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec, frames=80)
    # сближение и перекрытие: 10 кадров рамки почти совпадают
    for i in range(fi, fi + 10):
        box = walk(600, 250, 0, 0, 0)
        t1 = make_track(1, box, feature_family(target_vec, rng, 0.15), i)
        t2 = make_track(2, [box[0] + 10, box[1], box[2] + 10, box[3]], feature_family(mate_vec, rng, 0.15), i)
        obs = f.step(i, [t1, t2], {1: t1.last_feature, 2: t2.last_feature})
        assert obs.state == TargetState.CONTESTED
    # после перекрытия трек 1 несёт одноклубника, цель — новый трек 9
    events = []
    ids = []
    for i in range(fi + 10, fi + 60):
        k = i - fi - 10
        t1 = make_track(1, walk(600, 250, -3, 0, k), feature_family(mate_vec, rng, 0.05), i)
        t9 = make_track(9, walk(620, 250, 3, 0, k), feature_family(target_vec, rng, 0.05), i)
        obs = f.step(i, [t1, t9], {1: t1.last_feature, 9: t9.last_feature})
        events.append(obs.event)
        ids.append(obs.track_id)
    assert "released_swap" in events
    assert "reacquired" in events
    assert ids[-1] == 9


def test_gallery_is_bounded_and_diverse(rng):
    cfg = TargetConfig(gallery_size=5, gallery_update_every=1, gallery_novelty=0.02)
    f = TargetFollower((W, H), cfg)
    target_vec, mate_vec = base_vectors(rng)
    warmup(f, rng, target_vec, mate_vec, frames=100, noise=0.2)
    assert len(f.model.gallery) == 5
    assert f.model.centroid is not None


def test_idle_until_locked():
    f = TargetFollower((W, H))
    obs = f.step(0, [], {})
    assert obs.state == TargetState.IDLE and obs.track_id is None


def test_manual_release_and_events(rng):
    f = TargetFollower((W, H))
    target_vec, mate_vec = base_vectors(rng)
    fi = warmup(f, rng, target_vec, mate_vec, frames=10)
    f.release(fi, "released_manual")
    assert f.state == TargetState.LOST
    assert [e for _, e in f.events] == ["locked", "released_manual"]
