import numpy as np

from tracker.geometry import xyxy_to_xyah
from tracker.kalman import KalmanBoxFilter


def test_kalman_tracks_linear_motion():
    kf = KalmanBoxFilter()
    box = np.array([100, 100, 140, 200], dtype=float)
    mean, cov = kf.initiate(xyxy_to_xyah(box))
    for i in range(1, 30):
        mean, cov = kf.predict(mean, cov)
        obs = xyxy_to_xyah(box + np.array([5 * i, 0, 5 * i, 0]))
        mean, cov = kf.update(mean, cov, obs)
    mean, cov = kf.predict(mean, cov)
    # после 30 шагов скорость выучена: предсказание близко к следующему положению
    assert abs(mean[4] - 5.0) < 0.5
    assert abs(mean[0] - (120 + 5 * 30)) < 3


def test_gating_distance_prefers_near():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(xyxy_to_xyah([100, 100, 140, 200]))
    mean, cov = kf.predict(mean, cov)
    meas = np.stack([xyxy_to_xyah([102, 101, 142, 201]), xyxy_to_xyah([300, 100, 340, 200])])
    d = kf.gating_distance(mean, cov, meas)
    assert d[0] < d[1]
    assert d[0] < 9.49  # квантиль 95% для df=4


def test_camera_zoom_keeps_covariance_positive_definite():
    """Резкий «отъезд» зума (масштаб 0.93 за кадр) у движущегося трека: ковариация остаётся положительно
    определённой, стробирование не падает (ролик IMG_7462, кадр 98)."""
    from tracker.detection import Detection
    from tracker.multitracker import MultiTracker

    mot = MultiTracker()
    A = np.array([[0.93, 0.0, 9.0], [0.0, 0.93, 4.3]])
    for i in range(60):
        x = 300 + 8 * i
        dets = [Detection(np.array([x, 300.0, x + 20, 360.0]), 0.9), Detection(np.array([700.0 - 3 * i, 320, 716 - 3 * i, 370]), 0.9)]
        cam = A if 30 <= i < 40 else None          # десять кадров подряд камера отдаляет
        if cam is not None:
            dets = []                              # и детекции в эти кадры пропадают — треки только предсказываются
        mot.update(dets, None, frame_idx=i, camera=cam)
        for t in mot.tracks:
            assert np.linalg.eigvalsh((t.cov + t.cov.T) / 2).min() > 0
            assert np.allclose(t.cov, t.cov.T)


def test_update_keeps_covariance_positive_definite_for_tiny_boxes():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 3.0]))
    for i in range(200):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + i, 100.0, 0.5, 3.0 - 0.01 * i]))
        assert np.linalg.eigvalsh(cov).min() > 0
    assert np.isfinite(kf.gating_distance(mean, cov, np.array([[100.0, 100.0, 0.5, 1.0]]))).all()
