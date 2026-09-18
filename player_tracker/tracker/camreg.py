"""Привязка кадров к общей сцене гомографией по опорным кадрам (без накопления ошибки).

Покадровое движение камеры (`camera.CameraMotion`) — подобие «предыдущий кадр -> текущий». Для рамок игроков этого
достаточно, но цепочка из тысяч подобий уплывает (замер 17.09.2026: 15–25 px к концу ролика), а поворот камеры на
20–30° подобием не описывается вовсе (перспектива у краёв кадра). Разметке поля на кадре нужен точный перенос точки
между далёкими по времени кадрами.

Здесь каждый step-й кадр сопоставляется по особым точкам (SIFT) не с предыдущим кадром, а с ближайшим ОПОРНЫМ кадром,
гомографией. Опорные кадры добавляются, когда камера ушла от имеющихся (перекрытие < min_overlap); ошибка растёт с
числом опорных кадров на пути к кадру 0 (единицы), а не с числом кадров. Люди (подвижны) и небо (облака плывут)
маскируются. Между обработанными кадрами — подобие из покадрового движения (ошибка за 2–3 кадра ничтожна).

Оператор, стоящий на месте (поворот + зум), описывается гомографией точно для всей сцены. Если оператор ходит, у
ближней земли появляется параллакс — гомография следует за дальним планом (трибуны, щиты у дальней бровки), где и идёт
игра; после перехода на другое место нужна новая разметка поля (`field.FieldMarks` на нескольких кадрах).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


@dataclass
class RegConfig:
    step: int = 5                   # сопоставляется каждый step-й кадр
    features: int = 1500
    ratio: float = 0.75             # тест Лоу
    ransac_px: float = 2.5
    min_inliers: int = 40
    min_overlap: float = 0.6        # меньше — новый опорный кадр
    sky_above_horizon: float = 0.18 # доля высоты кадра над горизонтом, выше которой — небо (маска)
    max_keyframes: int = 60


def _overlap(H: np.ndarray, w: int, h: int) -> float:
    """Доля кадра, попадающая в опорный кадр после переноса H (кадр -> опорный), по сетке точек."""
    gx, gy = np.meshgrid(np.linspace(0, w, 12), np.linspace(0, h, 8))
    P = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    Q = H @ P
    ok = Q[2] > 1e-9
    x, y = Q[0] / np.where(ok, Q[2], 1), Q[1] / np.where(ok, Q[2], 1)
    return float(np.mean(ok & (x >= 0) & (x <= w) & (y >= 0) & (y <= h)))


def _sane(H: Optional[np.ndarray], w: int, h: int) -> bool:
    """Гомография похожа на движение камеры: углы кадра остаются выпуклым четырёхугольником разумного размера."""
    if H is None or not np.isfinite(H).all():
        return False
    P = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    Q = H @ P
    if (Q[2] <= 1e-9).any():
        return False
    q = (Q[:2] / Q[2]).T
    d = np.roll(q, -1, axis=0) - q
    cross = d[:, 0] * np.roll(d, -1, axis=0)[:, 1] - d[:, 1] * np.roll(d, -1, axis=0)[:, 0]
    area = 0.5 * abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    return bool((cross > 0).all() and 0.1 < area / (w * h) < 10)


def sim3(A: Optional[np.ndarray]) -> np.ndarray:
    if A is None or np.isnan(A).any():
        return np.eye(3)
    return np.vstack([A, [0.0, 0.0, 1.0]])


def register(frames: Iterable[np.ndarray], cameras: Sequence[Optional[np.ndarray]],
             boxes: Sequence[np.ndarray], horizon: Optional[Sequence[float]] = None,
             cfg: RegConfig | None = None, progress: Optional[Callable[[int], None]] = None,
             stop: Optional[Callable[[], bool]] = None) -> tuple[np.ndarray, dict]:
    """G[i] (3×3): точка кадра i -> сцена (координаты кадра 0), пиксели кадра обработки.

    frames — кадры по порядку (BGR); cameras[i] — подобие «кадр i-1 -> кадр i» (2×3 или None);
    boxes[i] — рамки людей кадра i (N, 4+); horizon[i] — строка горизонта (или None)."""
    import cv2

    cfg = cfg or RegConfig()
    sift = cv2.SIFT_create(nfeatures=cfg.features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    n = len(cameras)
    G = np.tile(np.eye(3), (n, 1, 1))
    keys: list[dict] = []                     # опорные кадры: G, точки, дескрипторы
    stats = {"frames": n, "matched": 0, "fallback": 0, "keyframes": 0, "inliers": []}
    quality = np.zeros(n, np.float32)         # доля совпавших точек, согласных с гомографией (0 — кадр не сопоставлен)
    last = 0                                  # последний обработанный кадр
    chain = np.eye(3)                         # подобие «кадр last -> текущий кадр»
    w = h = 0

    def describe(img, i):
        nonlocal w, h
        h, w = img.shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        if horizon is not None and i < len(horizon) and np.isfinite(horizon[i]):
            mask[:max(int(horizon[i] - cfg.sky_above_horizon * h), 0)] = 0
        else:
            mask[:int(0.2 * h)] = 0
        for b in np.asarray(boxes[i], float).reshape(-1, 4):
            x1, y1, x2, y2 = b
            px, py = 0.15 * (x2 - x1) + 4, 0.05 * (y2 - y1) + 4
            mask[max(int(y1 - py), 0):int(y2 + py), max(int(x1 - px), 0):int(x2 + px)] = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, mask)
        pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        return pts, des

    def match(pts, des, key):
        if des is None or key["des"] is None or len(des) < 8 or len(key["des"]) < 8:
            return None, 0, 0.0
        pairs = matcher.knnMatch(des, key["des"], k=2)
        good = [m for m, s in (p for p in pairs if len(p) == 2) if m.distance < cfg.ratio * s.distance]
        if len(good) < cfg.min_inliers:
            return None, len(good), 0.0
        src = pts[[m.queryIdx for m in good]]
        dst = key["pts"][[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.ransac_px)
        k = int(inl.sum()) if inl is not None else 0
        return (H if k >= cfg.min_inliers and _sane(H, w, h) else None), k, k / len(good)

    for i, img in enumerate(frames):
        if i >= n or (stop is not None and stop()):
            break
        if i > 0:
            chain = sim3(cameras[i]) @ chain               # last -> i
        guess = G[last] @ np.linalg.inv(chain)             # i -> сцена по подобию от последнего обработанного
        if i % cfg.step and i != n - 1:
            G[i], quality[i] = guess, quality[last]
            continue
        pts, des = describe(img, i)
        if not keys:
            keys.append({"G": np.eye(3), "pts": pts, "des": des, "frame": 0})
            last, chain = i, np.eye(3)
            continue
        # опорные кадры по убыванию ожидаемого перекрытия; пробуем два лучших
        order = sorted(range(len(keys)), key=lambda k: -_overlap(np.linalg.inv(keys[k]["G"]) @ guess, w, h))
        best, best_in, best_k, best_q = None, 0, -1, 0.0
        for k in order[:2]:
            H, inl, q = match(pts, des, keys[k])
            if H is not None and inl > best_in:
                best, best_in, best_k, best_q = keys[k]["G"] @ H, inl, k, q
            if best_in >= 3 * cfg.min_inliers:
                break
        if best is None:
            G[i] = guess
            stats["fallback"] += 1
        else:
            G[i] = best / best[2, 2]
            stats["matched"] += 1
            stats["inliers"].append(best_in)
            quality[i] = best_q
            ov = _overlap(np.linalg.inv(keys[best_k]["G"]) @ G[i], w, h)
            if ov < cfg.min_overlap and len(keys) < cfg.max_keyframes and best_in >= 2 * cfg.min_inliers:
                keys.append({"G": G[i].copy(), "pts": pts, "des": des, "frame": i})
        last, chain = i, np.eye(3)
        if progress is not None:
            progress(i)
    stats["keyframes"] = len(keys)
    stats["keyframe_frames"] = [k["frame"] for k in keys]
    inl = stats.pop("inliers")
    stats["median_inliers"] = int(np.median(inl)) if inl else 0
    stats["quality"] = quality
    return G, stats
