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
