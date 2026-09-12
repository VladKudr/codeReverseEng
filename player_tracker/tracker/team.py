"""Разделение игроков по цвету формы.

Не пытаемся узнать игрока по форме — она у одноклубников одинакова. Задача
проще: отсечь кандидатов из ДРУГОЙ команды и судью. Цвет торса каждой
детекции переводится в Lab, по накопленной выборке строятся два кластера
(k-means); всё далёкое от обоих центров считается «прочим» (судья, вратарь).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

UNKNOWN = -1
OTHER = 2


def torso_color(frame: np.ndarray, box, top: float = 0.18, bottom: float = 0.48, center_frac: float = 0.5) -> Optional[np.ndarray]:
    """Медианный цвет торса в Lab (робастнее среднего к номеру и рукам)."""
    import cv2

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    pad = (1 - center_frac) / 2 * w
    xa, xb = int(max(0, x1 + pad)), int(min(W, x2 - pad))
    ya, yb = int(max(0, y1 + top * h)), int(min(H, y1 + bottom * h))
    if xb - xa < 2 or yb - ya < 2:
        return None
    lab = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


@dataclass
class TeamClassifier:
    """Двухкластерная модель цветов формы.

    Копит цвета торсов (`observe`), после `min_samples` строит центры
    (`fit`). `predict` возвращает 0/1 (команды), 2 — прочие (далеко от обоих
    центров), -1 — модель ещё не обучена. Порог «прочих» — кратный
    внутрикластерному разбросу, поэтому не зависит от конкретных цветов.
    """

    min_samples: int = 60
    max_samples: int = 2000
    outlier_factor: float = 2.5
    min_outlier_dist: float = 18.0
    samples: list[np.ndarray] = field(default_factory=list)
    centers: Optional[np.ndarray] = None
    spread: float = 0.0

    def observe(self, color: Optional[np.ndarray]) -> None:
        if color is None:
            return
        if len(self.samples) < self.max_samples:
            self.samples.append(np.asarray(color, dtype=np.float32))

    @property
    def ready(self) -> bool:
        return self.centers is not None

    def fit(self, force: bool = False) -> bool:
        if len(self.samples) < self.min_samples and not force:
            return False
        if len(self.samples) < 2:
            return False
        data = np.stack(self.samples)
        centers, labels = _kmeans2(data)
        d = np.linalg.norm(data - centers[labels], axis=1)
        self.centers = centers
        self.spread = float(np.median(d) + 1e-6)
        return True

    def predict(self, color: Optional[np.ndarray]) -> int:
        if color is None or self.centers is None:
            return UNKNOWN
        d = np.linalg.norm(self.centers - np.asarray(color, dtype=np.float32), axis=1)
        k = int(np.argmin(d))
        thr = max(self.outlier_factor * self.spread, self.min_outlier_dist)
        return OTHER if d[k] > thr else k


def _kmeans2(data: np.ndarray, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means на два кластера без внешних зависимостей; старт — две самые далёкие точки."""
    rng = np.random.default_rng(seed)
    if len(data) < 2:
        return data[:1].copy(), np.zeros(len(data), dtype=int)
    i0 = int(rng.integers(len(data)))
    i1 = int(np.argmax(np.linalg.norm(data - data[i0], axis=1)))
    i0 = int(np.argmax(np.linalg.norm(data - data[i1], axis=1)))
    centers = np.stack([data[i0], data[i1]])
    labels = np.zeros(len(data), dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for k in range(2):
            if np.any(labels == k):
                centers[k] = data[labels == k].mean(axis=0)
    return centers, labels


def teams_compatible(a: int, b: int) -> Optional[bool]:
    """True — одна команда, False — разные, None — хотя бы одна метка неизвестна."""
    if a == UNKNOWN or b == UNKNOWN:
        return None
    return a == b
