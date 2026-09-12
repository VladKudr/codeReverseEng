"""Синтетические помощники для тестов (треки, дескрипторы, рисованные игроки)."""
import numpy as np

from tracker.kalman import KalmanBoxFilter
from tracker.multitracker import Track, TrackState  # noqa: E402
from tracker.geometry import xyxy_to_xyah  # noqa: E402


def make_track(track_id: int, box, feature=None, frame_idx: int = 0, detected: bool = True,
               velocity=(0.0, 0.0)) -> Track:
    """Готовый подтверждённый трек мультитрекера для тестов автомата цели."""
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xyxy_to_xyah(box))
    mean[4:6] = velocity
    t = Track(track_id, mean, cov, 0.9, frame_idx, state=TrackState.CONFIRMED, hits=10, last_frame=frame_idx,
              last_box=np.asarray(box, dtype=np.float64))
    t.time_since_update = 0 if detected else 1
    if feature is not None:
        f = np.asarray(feature, dtype=np.float32)
        f = f / np.linalg.norm(f)
        t.feature = f
        t.last_feature = f.copy()
    return t


def unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def feature_family(base: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Дескриптор «того же человека» в другом кадре: базовый вектор + шум."""
    return unit(base + rng.normal(0, noise, size=base.shape).astype(np.float32))



def draw_player(frame, box, shirt, shorts, socks, hair, skin=(150, 190, 230)):
    """Рисует «игрока» полосами: голова, торс, шорты, гетры, бутсы (BGR)."""
    import cv2

    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    bands = [
        (0.00, 0.08, hair),
        (0.08, 0.16, skin),
        (0.16, 0.48, shirt),
        (0.48, 0.64, shorts),
        (0.64, 0.72, skin),
        (0.72, 0.88, socks),
        (0.88, 1.00, (20, 20, 20)),
    ]
    for top, bottom, color in bands:
        cv2.rectangle(frame, (x1, y1 + int(top * h)), (x2, y1 + int(bottom * h)), color, -1)
    return frame


def grass_frame(w: int = 640, h: int = 360) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (60, 140, 60)
    return frame
