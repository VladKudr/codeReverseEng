"""Состав: пометка относится к человеку — переносится на фрагменты его трека и на похожих по внешности."""
import numpy as np

from tracker.roster import (PLAY, SIDELINE, LinkConfig, Mark, TrackInfo, is_off, link_tracks, off_intervals, resolve)

FPS = 30.0


def tr(tid, f0, f1, x0, x1, y=500.0, h=100.0, color=(120, 180, 150)):
    f = np.arange(f0, f1 + 1)
    return TrackInfo(tid, f, np.linspace(x0, x1, len(f)), np.full(len(f), y), np.full(len(f), h),
                     np.array(color, np.float32))


def test_continuation_links_fragments_of_one_runner():
    """Игрок бежит вправо, трек рвётся на полсекунды — два фрагмента становятся одним человеком."""
    info = {1: tr(1, 0, 60, 100, 400), 2: tr(2, 75, 140, 475, 800), 3: tr(3, 0, 140, 900, 900, y=300)}
    person, stats = link_tracks(info, FPS)
    assert person[1] == person[2] != person[3] and stats["links"] == 1


def test_no_link_across_colour_height_or_distance():
    base = tr(1, 0, 60, 100, 400)
    far = tr(2, 75, 140, 1200, 1300)                       # далеко от прогноза
    tall = tr(3, 75, 140, 475, 800, h=190.0)               # взрослый
    dark = tr(4, 75, 140, 475, 800, color=(20, 128, 128))  # другая форма
    for other in (far, tall, dark):
        person, _ = link_tracks({1: base, other.tid: other}, FPS)
        assert person[1] != person[other.tid]


def test_standing_person_is_one_person_across_long_gap():
    """Зритель стоит на месте; трек рвётся на минуту (камера уходила) — тот же человек."""
    info = {1: tr(1, 0, 90, 1000, 1004), 2: tr(2, 2000, 2100, 1010, 1006), 3: tr(3, 2000, 2100, 1300, 1300)}
    person, stats = link_tracks(info, FPS)
    assert person[1] == person[2] != person[3] and stats["static_links"] == 1
    # без точной привязки кадров место «уплывает» — длинный разрыв не склеивается
    person, _ = link_tracks(info, FPS, LinkConfig(static_max_gap_sec=10))
    assert person[1] != person[2]


def test_two_people_in_frame_at_once_are_never_one_person():
    info = {1: tr(1, 0, 100, 1000, 1002), 2: tr(2, 50, 150, 1005, 1003)}
    person, _ = link_tracks(info, FPS)
    assert person[1] != person[2]


def test_mark_follows_person_and_substitution_from_frame():
    info = {1: tr(1, 0, 60, 100, 400), 2: tr(2, 75, 140, 475, 800), 3: tr(3, 0, 140, 900, 900, y=300)}
    person, _ = link_tracks(info, FPS)
    off = off_intervals(info, person, [Mark(1, SIDELINE)])
    assert off[1] == [[0, 60]] and off[2] == [[75, 140]] and off[3] == []        # выключен весь человек
    # замена: «ушёл с поля с кадра 100» — до этого в игре; продолжение трека тоже вне игры
    off = off_intervals(info, person, [Mark(2, SIDELINE, at=100)])
    assert off[1] == [] and off[2] == [[100, 140]]
    assert not is_off(off[2], 99) and is_off(off[2], 100)
    # режим «только отмеченные»: «вышел на поле с кадра 30» — раньше вне игры
    off = off_intervals(info, person, [Mark(1, PLAY, at=30)], default=SIDELINE)
    assert off[1] == [[0, 29]] and off[2] == [] and off[3] == [[0, 140]]
    # поздняя пометка «в игре» на продолжении отменяет раннюю «вне игры» — с начала этого фрагмента
    off = off_intervals(info, person, [Mark(1, SIDELINE), Mark(2, PLAY)])
    assert off[1] == [[0, 60]] and off[2] == []


def test_similar_people_follow_the_marks():
    """Выключен зритель в тёмном; через минуту он же в другом месте (склейки нет) — выключается по внешности.
    Игроки в красном остаются: остальные люди кадра разметки — примеры «в игре»."""
    red, dark = np.array([1.0, 0.0, 0.2]), np.array([0.0, 1.0, 0.1])
    info = {1: tr(1, 0, 90, 100, 400), 2: tr(2, 0, 90, 1000, 1001), 3: tr(3, 2000, 2100, 300, 600),
            4: tr(4, 2000, 2100, 700, 760)}
    feats = {1: red / np.linalg.norm(red), 2: dark / np.linalg.norm(dark),
             3: np.array([0.95, 0.1, 0.25]) / np.linalg.norm([0.95, 0.1, 0.25]),
             4: np.array([0.1, 0.9, 0.1]) / np.linalg.norm([0.1, 0.9, 0.1])}
    r = resolve(info, [Mark(2, SIDELINE)], FPS, PLAY, feats, mark_frames=[40])
    assert r["why"] == {1: "default", 2: "manual", 3: "default", 4: "similar"}
    assert r["off"][4] == [[2000, 2100]] and r["off"][3] == [] and r["off"][1] == []
    # «только отмеченные»: отмечен игрок в красном — другой красный тоже в игре, тёмные — нет
    r = resolve(info, [Mark(1, PLAY)], FPS, SIDELINE, feats, mark_frames=[40])
    assert r["off"][3] == [] and r["why"][3] == "similar" and r["off"][4] == [[2000, 2100]] and r["off"][2] == [[0, 90]]
    # без кадра разметки примеров обратного нет — по внешности никто не переносится
    r = resolve(info, [Mark(2, SIDELINE)], FPS, PLAY, feats, mark_frames=[])
    assert r["why"][4] == "default" and r["off"][4] == []
