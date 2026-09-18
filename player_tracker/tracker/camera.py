"""Компенсация движения камеры (CMC/GMC, как в BoT-SORT).

Съёмка с рук: камера ведёт игру и за секунду смещает кадр на десятки
пикселей. Без компенсации фильтр Калмана и модель движения цели считают,
что это игроки прыгнули, — ассоциация рвётся, а после потери цель ищется
не там. Поэтому на каждом кадре оценивается глобальное преобразование
«предыдущий кадр -> текущий» (сдвиг + поворот + масштаб) по точкам фона, и
им переносятся предсказания треков и последняя позиция цели.

Точки берутся вне рамок людей (игроки движутся сами по себе), кадр
уменьшается для скорости, выбросы отсекаются RANSAC. Если оценка не
удалась (однотонный кадр, мало точек), возвращается тождественное
преобразование — поведение как без компенсации.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def warp_points(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Применяет аффинное 2x3 к точкам (N, 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ A[:, :2].T + A[:, 2]


def warp_box(A: np.ndarray, box) -> np.ndarray:
    """Переносит рамку (x1, y1, x2, y2): центр — преобразованием, размер — его масштабом."""
    b = np.asarray(box, dtype=np.float64)
    c = warp_points(A, [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])[0]
    s = scale_of(A)
    hw, hh = (b[2] - b[0]) / 2 * s, (b[3] - b[1]) / 2 * s
    return np.array([c[0] - hw, c[1] - hh, c[0] + hw, c[1] + hh])


def scale_of(A: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(A[:, :2]))))


class CameraMotion:
    """Оценка преобразования предыдущий кадр -> текущий по разреженному оптическому потоку.

    downscale   — во сколько раз уменьшать кадр (точность субпиксельная и так);
    max_corners — сколько угловых точек искать;
    min_inliers — меньше согласованных точек — считаем, что оценки нет.
    """

    def __init__(self, downscale: float = 2.0, max_corners: int = 400, min_inliers: int = 12,
                 ransac_thresh: float = 1.5):
        self.downscale = max(downscale, 1.0)
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self._prev: Optional[np.ndarray] = None
        self.last_inliers = 0

    def reset(self) -> None:
        self._prev = None

    def estimate(self, frame: np.ndarray, boxes=None) -> np.ndarray:
        import cv2

        k = 1.0 / self.downscale
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if k != 1.0:
            gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        prev, self._prev = self._prev, gray
        self.last_inliers = 0
        if prev is None or prev.shape != gray.shape:
            return IDENTITY.copy()
        mask = np.full(prev.shape, 255, dtype=np.uint8)
        if boxes is not None:
            for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
                x1, y1, x2, y2 = (b * k).astype(int)
                mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 6, mask=mask)
        if p0 is None or len(p0) < self.min_inliers:
            return IDENTITY.copy()
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.reshape(-1) == 1
        if ok.sum() < self.min_inliers:
            return IDENTITY.copy()
        M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_thresh)
        if M is None or inl is None or int(inl.sum()) < self.min_inliers:
            return IDENTITY.copy()
        self.last_inliers = int(inl.sum())
        M = M.astype(np.float64)
        M[:, 2] /= k   # сдвиг — обратно в пиксели исходного кадра
        return M
