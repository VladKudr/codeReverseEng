"""Разметка поля на кадре: углы площадки -> метры поля на любом кадре с учётом движения камеры."""
import numpy as np

from tracker.field import FieldMarks, FieldProjector, apply_h, homography, pitch_corners

L, W = 45.0, 30.0
# углы площадки на опорном кадре 1280×720: ближние — внизу (левый уходит за край кадра), дальние — у горизонта
CORNERS = np.array([[-150.0, 700.0], [1400.0, 690.0], [980.0, 420.0], [260.0, 425.0]])


def projector(inv=None, scale=1.0, rotation=0, frame=0):
    inv = inv or [np.eye(3)] * 5
    return FieldProjector(FieldMarks(frame, CORNERS / scale, rotation), scale, inv, L, W)


def test_corners_map_to_pitch_corners():
    pr = projector()
    px, py = pr(0, CORNERS[:, 0], CORNERS[:, 1])
    assert np.allclose(np.stack([px, py], 1), pitch_corners(L, W), atol=1e-6)
    # центр поля — где-то между ближней и дальней линией, по центру кадра
    cx, cy = apply_h(np.linalg.inv(pr.H), np.array([L / 2]), np.array([W / 2]))[:2]
    assert 500 < cx[0] < 800 and 430 < cy[0] < 690


def test_rotation_and_scale():
    pr = projector(rotation=1)
    px, py = pr(0, CORNERS[0, 0], CORNERS[0, 1])
    assert np.allclose([px, py], pitch_corners(L, W, 1)[0])
    # углы хранятся в координатах исходного кадра (4K), обработка — в 1280: точка кадра обработки
    half = projector(scale=0.5)                  # углы в исходнике — CORNERS·2, в кадре обработки — CORNERS
    x, y = half(0, CORNERS[2, 0], CORNERS[2, 1])
    assert np.allclose([x, y], [L, 0.0], atol=1e-6)


def test_camera_pan_is_compensated():
    """Камера сдвинулась на 100 px вправо: точка поля на кадре 3 левее на 100 px — метры те же."""
    shift = np.array([[1.0, 0, -100.0], [0, 1.0, 0], [0, 0, 1.0]])
    # inv[i] — кадр i -> координаты сцены (первого кадра): на кадре 3 сдвиг уже накоплен
    inv = [np.eye(3), np.eye(3), np.eye(3), np.linalg.inv(shift), np.linalg.inv(shift)]
    pr = projector(inv=inv)
    a = pr(0, 640.0, 600.0)
    b = pr(3, 540.0, 600.0)
    assert np.allclose(a, b, atol=1e-6)


def test_points_beyond_horizon_are_rejected():
    pr = projector()
    px, py = pr(0, np.array([640.0, 640.0]), np.array([100.0, 600.0]))     # небо и поле
    assert np.isnan(px[0]) and not np.isnan(px[1])


def test_homography_is_exact_on_four_points():
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    dst = np.array([[1, 2], [20, 3], [18, 25], [0, 22]], float)
    H = homography(src, dst)
    x, y, _ = apply_h(H, src[:, 0], src[:, 1])
    assert np.allclose(np.stack([x, y], 1), dst)


def test_marks_legacy_and_list_forms():
    from tracker.field import parse_marks

    one = {"frame": 7, "corners": CORNERS.tolist(), "rotation": 1, "own_goal": "left"}
    assert [m.frame for m in parse_marks(one)] == [7] and parse_marks(one)[0].rotation == 1
    many = {"marks": [{"frame": 90, "corners": CORNERS.tolist()}, {"frame": 5, "corners": CORNERS.tolist(), "rotation": 2}]}
    assert [m.frame for m in parse_marks(many)] == [5, 90]           # по порядку кадров
    assert parse_marks(None) == [] and parse_marks({"marks": [{"frame": 1, "corners": [[0, 0]]}]}) == []


def test_nearest_mark_is_used_for_each_frame():
    """Оператор перешёл: вторая разметка сдвинута на 200 px — кадры рядом с ней считаются по ней."""
    from tracker.field import MultiProjector

    inv = [np.eye(3)] * 100
    a = FieldProjector(FieldMarks(10, CORNERS, 0), 1.0, inv, L, W)
    b = FieldProjector(FieldMarks(90, CORNERS + [200.0, 0.0], 0), 1.0, inv, L, W)
    multi = MultiProjector([b, a])
    u, v = CORNERS[2]
    assert np.allclose(multi(20, u, v), [L, 0.0], atol=1e-6)         # ближе разметка кадра 10
    assert np.allclose(multi(80, u + 200, v), [L, 0.0], atol=1e-6)   # ближе разметка кадра 90
    assert not np.allclose(multi(80, u, v), [L, 0.0], atol=0.5)


def test_convex_check():
    from tracker.field import convex

    assert convex(CORNERS)
    assert not convex(CORNERS[[0, 2, 1, 3]])                         # «бабочка»: углы не по кругу
    assert not convex(np.array([[0, 0], [10, 0], [20, 0], [30, 0.0]]))


def test_expand_moves_only_sides_players_cross():
    """Игроки выходят за дальнюю бровку на 6 м — переносится только она; остальные стороны на месте."""
    from tracker.field import expand_to_points

    rng = np.random.default_rng(0)
    Hi = np.linalg.inv(homography(CORNERS, pitch_corners(L, W)))
    px, py = rng.uniform(2, L - 2, 4000), rng.uniform(-6, W - 8, 4000)       # y < 0 — за дальней бровкой
    u, v, _ = apply_h(Hi, px, py)
    new = expand_to_points(CORNERS, 0, L, W, u, v)
    nx, ny, _ = apply_h(homography(CORNERS, pitch_corners(L, W)), new[:, 0], new[:, 1])
    assert np.allclose(nx, pitch_corners(L, W)[:, 0], atol=0.05)             # по длине не тронуто
    assert np.allclose(ny[:2], W, atol=0.05)                                 # ближняя бровка на месте
    assert -7.5 < ny[2] < -5.5 and -7.5 < ny[3] < -5.5                       # дальняя ушла к игрокам (+1 м запаса)
    # все внутри — ничего не меняется
    u2, v2, _ = apply_h(Hi, rng.uniform(5, L - 5, 500), rng.uniform(5, W - 5, 500))
    assert np.allclose(expand_to_points(CORNERS, 0, L, W, u2, v2), CORNERS, atol=1e-6)


def test_corners_from_ground_orders_corners_and_rotation():
    """Точки ног — прямоугольник 40×20 м перед камерой, длинной стороной к ней: углы по кругу, rotation 0."""
    from tracker.field import convex, corners_from_ground

    rng = np.random.default_rng(1)
    X, Z = rng.uniform(-20, 20, 3000), rng.uniform(15, 35, 3000)
    to_image = lambda X, Z: (640 + 900 * X / Z, 300 + 2000 / Z)              # noqa: E731 — простая перспектива
    corners, rotation, clipped = corners_from_ground(X, Z, to_image)
    assert not clipped
    assert rotation == 0 and convex(corners)
    assert corners[0, 1] > corners[3, 1] and corners[1, 1] > corners[2, 1]   # 1, 2 — ближние (ниже в кадре)
    assert corners[0, 0] < corners[1, 0] and corners[3, 0] < corners[2, 0]   # 1, 4 — левые
    # камера за воротами: к ней обращена короткая сторона
    _, rot, _ = corners_from_ground(rng.uniform(-10, 10, 3000), rng.uniform(10, 50, 3000), to_image)
    assert rot == 1
    assert corners_from_ground(X[:50], Z[:50], to_image) is None             # мало данных
    # играют у самых ног оператора: ближние точки сдвигаются по боковым линиям на глубину 3 м
    near, _, clipped = corners_from_ground(rng.uniform(-20, 20, 3000), rng.uniform(0.5, 30, 3000), to_image)
    assert clipped and np.isfinite(near).all()


def _camera_view(f, hc, v0, cx, theta, tx, ty, pts):
    """Кадр точки поля (метры) для камеры без крена — обратное к модели `fit_sides`."""
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, s], [s, -c]])
    g = (np.asarray(pts, float) - [tx, ty]) @ np.linalg.inv(A).T          # (Xc, Zc)
    return np.stack([cx + f * g[:, 0] / g[:, 1], v0 + f * hc / g[:, 1]], axis=1)


def test_sides_mode_recovers_pitch_from_far_corners_and_side_points():
    """Камера у угла поля: видны два дальних угла и куски боковых линий; ближние углы — далеко за кадром."""
    from tracker.field import SIDES, CameraPrior, fit_sides

    f, hc, v0, cx = 1100.0, 1.7, 400.0, 640.0
    theta, tx, ty = np.radians(25), 12.0, W + 3.0                         # камера в 3 м за ближней бровкой, под углом
    pc = pitch_corners(L, W)
    far = _camera_view(f, hc, v0, cx, theta, tx, ty, pc[[3, 2]])          # углы 4 и 3
    side = _camera_view(f, hc, v0, cx, theta, tx, ty, [[0.0, 0.55 * W], [L, 0.4 * W]])   # точки на боковых линиях
    near = _camera_view(f, hc, v0, cx, theta, tx, ty, pc[[0, 1]])
    assert np.abs(near).max() > 3000                                       # ближние углы мышью не поставить
    marks = np.array([side[0], side[1], far[1], far[0]])
    prior = CameraPrior(v0=v0 + 3, hc=1.6, cx=cx, f0=914.0)                # горизонт и высота камеры — с ошибкой
    H, info = fit_sides(marks, 0, L, W, prior)
    assert info["rms_m"] < 0.3
    probe = np.array([[10.0, 5.0], [30.0, 12.0], [22.0, 20.0], [40.0, 3.0]])
    img = _camera_view(f, hc, v0, cx, theta, tx, ty, probe)
    x, y, _ = apply_h(H, img[:, 0], img[:, 1])
    assert np.abs(np.stack([x, y], 1) - probe).max() < 2.0                 # в пределах пары метров при ошибке горизонта
    pr = FieldProjector(FieldMarks(0, marks, 0, SIDES), 1.0, [np.eye(3)] * 3, L, W, camera=prior)
    assert np.allclose(pr(0, far[0, 0], far[0, 1]), [0.0, 0.0], atol=0.3)


def test_scene_horizon_pools_people_over_frames():
    from tracker.field import scene_horizon

    rng = np.random.default_rng(3)
    n = 4000
    f = rng.integers(0, 50, n)
    v = rng.uniform(430, 700, n)
    h = 0.85 * (v - 402.0) * rng.normal(1.0, 0.12, n)                      # дети разного роста
    got = scene_horizon(f, rng.uniform(0, 1280, n), v, h, [np.eye(3)] * 50, 0, 1.45)
    assert abs(got[0] - 402.0) < 4 and abs(got[1] - 1.45 / 0.85) < 0.1
    assert scene_horizon(f[:50], v[:50], v[:50], h[:50], [np.eye(3)] * 50, 0, 1.45) is None
