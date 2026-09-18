"""Фильтр Калмана с постоянной скоростью для рамки (cx, cy, aspect, h).

Состояние — 8 чисел: положение (cx, cy, a, h) и его скорости. Шум модели
масштабируется высотой рамки, как в SORT/ByteTrack, поэтому один и тот же
фильтр работает и для игрока на дальнем плане, и для крупного плана.
"""
from __future__ import annotations

import numpy as np

# Квантили хи-квадрат для стробирования по расстоянию Махаланобиса (df=4).
CHI2_95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


MIN_HEIGHT = 1.0      # рамка ниже пикселя — шум модели по ней вырождается


def sanitize_cov(cov: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Симметричная положительно определённая матрица: отрицательные собственные числа поднимаются до floor."""
    cov = (cov + cov.T) / 2
    try:
        np.linalg.cholesky(cov)
        return cov
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        return (V * np.maximum(w, floor)) @ V.T


class KalmanBoxFilter:
    def __init__(self, std_weight_position: float = 1 / 20, std_weight_velocity: float = 1 / 160):
        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Новый трек из измерения (cx, cy, a, h): скорости нулевые, ковариация широкая."""
        mean = np.zeros(8)
        mean[:4] = measurement
        h = measurement[3]
        std = [
            2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
            10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], MIN_HEIGHT)
        std = [
            self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
            self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h,
        ]
        Q = np.diag(np.square(std))
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + Q
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = max(mean[3], MIN_HEIGHT)
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        R = np.diag(np.square(std))
        return self.H @ mean, self.H @ cov @ self.H.T + R

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj_mean, proj_cov = self.project(mean, cov)
        R = proj_cov - self.H @ cov @ self.H.T
        K = np.linalg.solve(proj_cov, self.H @ cov).T
        innovation = measurement - proj_mean
        new_mean = mean + K @ innovation
        # форма Джозефа: ковариация остаётся симметричной и положительно определённой при любом округлении
        # (упрощённая cov − K·S·Kᵀ «уходила» в отрицательные собственные числа на ролике с резким зумом)
        I_KH = np.eye(8) - K @ self.H
        new_cov = I_KH @ cov @ I_KH.T + K @ R @ K.T
        return new_mean, sanitize_cov(new_cov)

    def gating_distance(self, mean: np.ndarray, cov: np.ndarray, measurements: np.ndarray, only_position: bool = False) -> np.ndarray:
        """Квадрат расстояния Махаланобиса от предсказания до каждого измерения (N, 4)."""
        proj_mean, proj_cov = self.project(mean, cov)
        measurements = np.asarray(measurements, dtype=np.float64).reshape(-1, 4)
        if only_position:
            proj_mean, proj_cov = proj_mean[:2], proj_cov[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - proj_mean
        try:
            L = np.linalg.cholesky(proj_cov)
        except np.linalg.LinAlgError:
            # страховка: испорченная ковариация одного трека не должна ронять весь прогон
            L = np.linalg.cholesky(sanitize_cov(proj_cov))
        z = np.linalg.solve(L, d.T)
        return np.sum(z * z, axis=0)
