"""Компенсация движения камеры: оценка сдвига кадра и перенос состояний треков."""
import numpy as np

from tracker.camera import IDENTITY, CameraMotion, warp_box
from tracker.detection import Detection
from tracker.multitracker import MultiTracker, MultiTrackerConfig


def textured(w=640, h=360, seed=3):
    """Фон с деталями (стена, сетка): угловые точки для оптического потока."""
    import cv2

    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.uint8), (0, 0), 2.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def shifted(img, dx, dy):
    import cv2

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_estimates_pan_and_ignores_players():
    base = textured()
    cam = CameraMotion()
    assert np.allclose(cam.estimate(base), IDENTITY)          # первый кадр — сравнивать не с чем
    nxt = shifted(base, -7.0, 2.0)
    # «игрок» движется по-своему — его точки маскируются рамкой
    nxt[100:220, 300:340] = 255
    A = cam.estimate(nxt, boxes=[(290, 90, 350, 230)])
    assert cam.last_inliers >= cam.min_inliers
    assert abs(A[0, 2] + 7.0) < 0.6 and abs(A[1, 2] - 2.0) < 0.6
    assert abs(A[0, 0] - 1.0) < 0.02


def test_flat_frame_falls_back_to_identity():
    cam = CameraMotion()
    flat = np.full((200, 300, 3), 90, np.uint8)
    cam.estimate(flat)
    assert np.allclose(cam.estimate(flat), IDENTITY)


def test_warp_box_moves_center_and_scales_size():
    A = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]])
    b = warp_box(A, [10, 10, 20, 30])
    assert np.allclose(b, [30, 15, 50, 55])   # центр (15, 20) -> (40, 35), размер x2


def test_camera_pan_keeps_track_id():
    """Игрок стоит, камера ведёт кадр на 40 пикселей за кадр (больше ширины рамки): без компенсации
    рамки соседних кадров не пересекаются и трек рвётся, с компенсацией id один."""
    def created_tracks(with_camera: bool) -> int:
        mt = MultiTracker(MultiTrackerConfig(min_hits=2))
        for i in range(12):
            box = np.array([400 - 40 * i, 100, 430 - 40 * i, 180], float)
            A = np.array([[1, 0, -40.0], [0, 1, 0]]) if (with_camera and i > 0) else None
            mt.update([Detection(box, 0.9)], camera=A)
        return mt._next_id - 1

    assert created_tracks(True) == 1
    assert created_tracks(False) > 1
