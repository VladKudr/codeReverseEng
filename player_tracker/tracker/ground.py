"""Плоскость поля по самим игрокам: глубина любой точки земли в кадре, метры для мяча и всех игроков.

Камера не откалибрована, но на каждом кадре есть 10–20 человек на одной плоскости. У камеры над
плоской землёй рост рамки линейно зависит от строки ног: h ≈ a · (v − v0), v0 — горизонт. Робастная
прямая по всем рамкам кадра даёт «сколько пикселей рост человека у точки земли со строкой v»:

  * глубина точки земли Z = f · H / h(v) — и для мяча, у которого своего роста нет;
  * ожидаемый диаметр мяча в пикселях: h(v) · D / H — отсекает «мячи» не того размера (головы
    зрителей у края кадра, фонари над горизонтом);
  * у всех игроков глубина берётся по строке ног, а не по росту своей рамки: рамка частично
    закрытого игрока ниже — по росту он «улетел бы» вдаль.

Рост H — масштаб метров (как в `player_metrics`): дети разного роста и взрослые у бровки дают
разброс, прямая по всем рамкам его усредняет. Поперечная координата — X = (u − cx) · H / h, u — в
координатах сцены (панорамирование камеры убрано накопленным движением кадра).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .camera import scale_of


@dataclass
class GroundConfig:
    min_score: float = 0.35
    min_height: float = 10.0
    min_people: int = 6
    smooth_frames: int = 31          # медиана параметров прямой по соседним кадрам (панорама и наклон камеры — плавные)
    hfov_deg: float = 70.0
    player_height_m: float = 1.5


def _robust_line(v: np.ndarray, h: np.ndarray, iters: int = 12) -> Optional[tuple[float, float]]:
    """h = a·v + c, веса Хьюбера (взрослые у бровки и обрезанные рамки — выбросы)."""
    A = np.vstack([v, np.ones_like(v)]).T
    w = np.ones_like(v)
    coef = None
    for _ in range(iters):
        coef = np.linalg.lstsq(A * w[:, None], h * w, rcond=None)[0]
        r = h - A @ coef
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-6
        w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9)))
    if coef is None or coef[0] <= 1e-3:
        return None
    return float(coef[0]), float(coef[1])


class GroundModel:
    """Прямая «рост от строки ног» на каждом кадре + накопленное движение камеры."""

    def __init__(self, a: np.ndarray, c: np.ndarray, cameras: Sequence[Optional[np.ndarray]], width: int,
                 cfg: GroundConfig):
        self.a, self.c = a, c
        self.cfg = cfg
        self.cx = width / 2
        self.f = (width / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)
        self.H = cfg.player_height_m
        self.inv: list[np.ndarray] = []
        M = np.eye(3)
        for A in cameras:
            if A is not None:
                M = np.vstack([A, [0.0, 0.0, 1.0]]) @ M
            self.inv.append(np.linalg.inv(M))

    @classmethod
    def fit(cls, people: Sequence[np.ndarray], cameras: Sequence[Optional[np.ndarray]], width: int,
            cfg: GroundConfig | None = None) -> "GroundModel":
        """people[i] — (N, 5) рамки людей кадра i (x1, y1, x2, y2, score)."""
        cfg = cfg or GroundConfig()
        n = len(people)
        a = np.full(n, np.nan)
        c = np.full(n, np.nan)
        for i, p in enumerate(people):
            p = np.asarray(p, float).reshape(-1, 5)
            h = p[:, 3] - p[:, 1]
            m = (p[:, 4] >= cfg.min_score) & (h >= cfg.min_height)
            if m.sum() < cfg.min_people:
                continue
            line = _robust_line(p[m, 3], h[m])
            if line is not None:
                a[i], c[i] = line
        ok = ~np.isnan(a)
        if not ok.any():                       # людей не нашлось: условная камера (горизонт на трети кадра)
            a[:], c[:] = 0.5, -0.5 * width * 0.2
        else:
            idx = np.arange(n)
            a = np.interp(idx, idx[ok], a[ok])
            c = np.interp(idx, idx[ok], c[ok])
            half = cfg.smooth_frames // 2
            if half and n > 2:
                a = np.array([np.median(a[max(0, i - half):i + half + 1]) for i in range(n)])
                c = np.array([np.median(c[max(0, i - half):i + half + 1]) for i in range(n)])
        return cls(a, c, cameras, width, cfg)

    def __len__(self) -> int:
        return len(self.a)

    def height_at(self, i: int, v) -> np.ndarray:
        """Рост человека (px) у точки земли со строкой v; ≤ 0 — выше горизонта."""
        return self.a[i] * np.asarray(v, float) + self.c[i]

    def horizon(self, i: int) -> float:
        return float(-self.c[i] / self.a[i])

    def to_scene_u(self, i: int, u, v) -> np.ndarray:
        u, v = np.asarray(u, float), np.asarray(v, float)
        M = self.inv[i]
        return M[0, 0] * u + M[0, 1] * v + M[0, 2]

    def scene_xy(self, i: int, u, v) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(u, float), np.asarray(v, float)
        M = self.inv[i]
        return M[0, 0] * u + M[0, 1] * v + M[0, 2], M[1, 0] * u + M[1, 1] * v + M[1, 2]

    def scene_scale(self, i: int) -> float:
        return scale_of(self.inv[i][:2])

    def ground(self, i: int, u, v, min_height: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Точка земли (u, v кадра i) -> (X поперёк, Z вглубь), метры. Выше горизонта — NaN."""
        h = self.height_at(i, v)
        h = np.where(h >= min_height, h, np.nan)
        us = self.to_scene_u(i, u, v)
        # рост в координатах сцены — как в player_metrics (масштаб накопленного движения камеры)
        hs = h * self.scene_scale(i)
        return (us - self.cx) * self.H / hs, self.f * self.H / hs

    def unground(self, i: int, X, Z) -> tuple[np.ndarray, np.ndarray]:
        """Обратное к `ground`: (X, Z) в метрах -> точка (u, v) кадра i. Z ≤ 0 (за камерой) — NaN."""
        X, Z = np.asarray(X, float), np.asarray(Z, float)
        Z = np.where(Z > 1e-6, Z, np.nan)
        hs = self.f * self.H / Z
        v = (hs / self.scene_scale(i) - self.c[i]) / self.a[i]
        us = self.cx + X * hs / self.H
        M = self.inv[i]
        return (us - M[0, 1] * v - M[0, 2]) / M[0, 0], v

    def metres_per_px(self, i: int, v) -> np.ndarray:
        h = self.height_at(i, v)
        return np.where(h > 1.0, self.H / np.maximum(h, 1e-6), np.nan)
