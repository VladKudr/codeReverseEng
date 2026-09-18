"""Плоскость поля по игрокам и траектория игрового мяча: размер, запасные мячи, разрывы."""
import numpy as np

from tracker.ball import DETECTED, ESTIMATED, BallConfig, plausible, track_ball
from tracker.ground import GroundConfig, GroundModel
from tracker.pitch import PitchSetup

FPS, W = 30.0, 1280
# «камера»: рост человека у строки ног v равен A * (v - V0)
A, V0, H = 0.8, 400.0, 1.5


def person(v, x=640.0):
    h = A * (v - V0)
    return [x - h / 4, v - h, x + h / 4, v, 0.9]


def people_frames(n, rows=(440, 480, 520, 560, 600, 640)):
    return [np.array([person(v, 200 + 150 * k) for k, v in enumerate(rows)], np.float32) for _ in range(n)]


def ball_box(u, v, size):
    return [u - size / 2, v - size, u + size / 2, v, 0.6]


def ground_for(people):
    return GroundModel.fit(people, [None] * len(people), W, GroundConfig(player_height_m=H))


def test_ground_recovers_plane_and_depth():
    people = people_frames(10)
    g = ground_for(people)
    assert abs(g.height_at(0, 600) - A * (600 - V0)) < 1.0
    assert abs(g.horizon(0) - V0) < 2.0
    X, Z = g.ground(0, 640.0, np.array([500.0, 600.0]))
    assert Z[0] > Z[1] > 0            # ближе к горизонту — дальше от камеры
    assert np.isnan(g.ground(0, 640.0, np.array([V0 - 20]))[1]).all()


def test_ball_size_and_head_filters():
    people = people_frames(1)
    g = ground_for(people)
    cfg = BallConfig()
    d = A * (600 - V0) * cfg.diameter_m / H
    balls = np.array([ball_box(640, 600, d),                 # мяч нужного размера
                      ball_box(640, 600, d * 6),             # слишком крупный «мяч» — голова зрителя
                      ball_box(640, V0 - 30, d)], np.float32)  # выше горизонта — фонарь
    ok = plausible(balls, people[0], g, 0, cfg)
    assert list(ok) == [True, False, False]


def test_ball_inside_head_of_person_box_rejected():
    people = people_frames(1)
    g = ground_for(people)
    p = people[0][0]
    head_u, head_v = (p[0] + p[2]) / 2, p[1] + 0.2 * (p[3] - p[1])
    d = A * (head_v - V0) * BallConfig().diameter_m / H
    ok = plausible(np.array([ball_box(head_u, head_v, d)], np.float32), people[0], g, 0, BallConfig())
    assert not ok.any()


def test_moving_ball_wins_over_static_spare_balls():
    """У кромки лежат два запасных мяча, по полю катится игровой — выбирается катящийся."""
    n = 120
    people = people_frames(n)
    g = ground_for(people)
    balls = []
    for i in range(n):
        d = A * (600 - V0) * BallConfig().diameter_m / H
        row = [ball_box(200, 450, A * 50 * 0.14), ball_box(260, 450, A * 50 * 0.14)]   # лежат неподвижно
        row.append(ball_box(500 + 3 * i, 600, d))                                      # катится вправо
        balls.append(np.array(row, np.float32))
    track = track_ball(balls, people, g, FPS)
    assert track.summary()["detected_share"] > 0.9
    assert all(abs(u - (500 + 3 * i)) < 2 for i, u in enumerate(track.u) if not np.isnan(u))
    assert track.summary()["static_candidates"] > 100


def test_gap_is_estimated_then_unknown():
    n = 150
    people = people_frames(n)
    g = ground_for(people)
    d = A * (600 - V0) * BallConfig().diameter_m / H
    balls = []
    for i in range(n):
        hidden = 40 <= i < 55 or 80 <= i < 140          # короткий разрыв и длинный
        balls.append(np.zeros((0, 5), np.float32) if hidden
                     else np.array([ball_box(500 + 2 * i, 600, d)], np.float32))
    track = track_ball(balls, people, g, FPS, BallConfig(interp_max_sec=1.0))
    assert track.state[45] == ESTIMATED and abs(track.u[45] - (500 + 2 * 45)) < 6
    assert track.state[100] is None                      # разрыв 2 с — мяч неизвестен
    assert track.state[30] == DETECTED


def test_ball_outside_pitch_is_ignored():
    """Мяч у зрителей за дальней бровкой отбрасывается по схеме поля."""
    n = 60
    people = people_frames(n)
    g = ground_for(people)
    d = A * (600 - V0) * BallConfig().diameter_m / H
    balls = [np.array([ball_box(500 + 2 * i, 600, d)], np.float32) for i in range(n)]
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    with_pitch = track_ball(balls, people, g, FPS, BallConfig(), pitch)
    far = PitchSetup((0.5, -3.0), (0.5, -2.0), "left", 40.0, 25.0)   # поле «за спиной» камеры
    without = track_ball(balls, people, g, FPS, BallConfig(), far)
    assert with_pitch.summary()["detected_share"] > 0.8
    assert without.summary()["known_share"] == 0.0


def test_tile_offsets_cover_frame_below_sky():
    from tracker.detection import tile_offsets

    offs = tile_offsets(1280, 720, 640)
    assert offs == [(0, 80), (480, 80), (640, 80)]           # один ряд ниже неба, перекрытие 25 %
    assert tile_offsets(640, 360, 640) == []                 # кадр меньше плитки — плиток нет
    assert len(tile_offsets(1920, 1080, 640)) == 8


def test_merge_points_keeps_strongest_of_overlapping_tiles():
    from tracker.detection import BALL_CLASS, Detection, merge_points

    dets = [Detection(np.array([100.0, 100.0, 106.0, 106.0]), 0.2, BALL_CLASS),
            Detection(np.array([102.0, 101.0, 108.0, 107.0]), 0.6, BALL_CLASS),   # тот же мяч с соседней плитки
            Detection(np.array([300.0, 200.0, 306.0, 206.0]), 0.1, BALL_CLASS)]
    out = merge_points(dets)
    assert len(out) == 2 and out[0].score == 0.6


def test_pitch_fit_recovers_stretched_depth():
    """Игроки равномерно по полю 45×30, камера за ближней бровкой; глубина растянута в 1.6 раза (зум), поперёк
    сжато в 0.8 (рост) — подгонка возвращает их в поле и находит поправки."""
    from tracker.pitch import fit_to_occupancy

    rng = np.random.default_rng(0)
    L, W = 45.0, 30.0
    true = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", L, W)
    C, f, r = true._basis()
    P = np.stack([rng.uniform(1, L - 1, 5000), rng.uniform(1, W - 1, 5000)], axis=1)
    X = (P - C) @ r * 0.8
    Z = (P - C) @ f * 1.6
    base = PitchSetup.default("6x6")
    assert base.outside_share(X, Z, 1.0) > 0.3
    fitted, info = fit_to_occupancy(base, X, Z)
    assert info["fitted"] and info["outside_after"] < 0.08
    # поправка минимальная, но достаточная: глубина сжата заметно (истина 0.625), поперёк почти не тронут
    assert 0.55 < fitted.kz < 0.8 and 0.9 < fitted.kx < 1.3
    px, py = fitted.to_pitch(X, Z)
    assert np.median(np.abs(px - P[:, 0])) < 4 and np.median(np.abs(py - P[:, 1])) < 4


def test_pitch_fit_does_not_stretch_a_drill():
    """Упражнение на части площадки: все и так в поле — схема не растягивается на всё поле."""
    from tracker.pitch import fit_to_occupancy

    rng = np.random.default_rng(2)
    base = PitchSetup.default("6x6")
    C, f, r = base._basis()
    P = np.stack([rng.uniform(15, 28, 2000), rng.uniform(12, 20, 2000)], axis=1)
    X, Z = (P - C) @ r, (P - C) @ f
    fitted, info = fit_to_occupancy(base, X, Z)
    assert info["outside_after"] == 0.0
    assert 0.85 < fitted.kx < 1.15 and 0.85 < fitted.kz < 1.15


def test_pitch_fit_keeps_marked_camera_when_only_part_filmed():
    """Снята только левая половина поля: по длине охват мал — камера не «переезжает» в центр."""
    from tracker.pitch import fit_to_occupancy

    rng = np.random.default_rng(1)
    marked = PitchSetup((0.25, 1.1), (0.25, 0.5), "left", 45.0, 30.0)
    C, f, r = marked._basis()
    P = np.stack([rng.uniform(2, 20, 3000), rng.uniform(1, 29, 3000)], axis=1)
    X, Z = (P - C) @ r, (P - C) @ f
    fitted, info = fit_to_occupancy(marked, X, Z)
    assert info["coverage"][0] < 0.8
    px, _ = fitted.to_pitch(X, Z)
    assert np.percentile(px, 95) < 30                       # игроки остались в левой половине
    assert fitted.outside_share(X, Z, 1.0) < 0.05
