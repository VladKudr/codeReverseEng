"""Геометрия рамок. Формат рамки — (x1, y1, x2, y2) в пикселях кадра."""
from __future__ import annotations

import numpy as np


def as_boxes(boxes) -> np.ndarray:
    """Приводит список/массив рамок к float-массиву формы (N, 4)."""
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return arr.reshape(-1, 4)


def box_area(boxes) -> np.ndarray:
    b = as_boxes(boxes)
    w = np.clip(b[:, 2] - b[:, 0], 0, None)
    h = np.clip(b[:, 3] - b[:, 1], 0, None)
    return w * h


def box_center(boxes) -> np.ndarray:
    b = as_boxes(boxes)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def box_wh(boxes) -> np.ndarray:
    b = as_boxes(boxes)
    return np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def iou_matrix(a, b) -> np.ndarray:
    """Матрица IoU формы (N, M) между рамками a (N, 4) и b (M, 4)."""
    a = as_boxes(a)
    b = as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def iou_pair(a, b) -> float:
    return float(iou_matrix(as_boxes(a)[:1], as_boxes(b)[:1])[0, 0])


def clip_box(box, width: int, height: int) -> np.ndarray:
    """Обрезает рамку по границам кадра (width, height)."""
    b = np.asarray(box, dtype=np.float64).copy()
    b[0::2] = np.clip(b[0::2], 0, width)
    b[1::2] = np.clip(b[1::2], 0, height)
    return b


def scale_boxes(boxes, factor: float) -> np.ndarray:
    """Масштабирует рамки (например, из уменьшенного кадра детекции в исходный)."""
    return as_boxes(boxes) * float(factor)


def xyxy_to_xyah(box) -> np.ndarray:
    """(x1, y1, x2, y2) -> (cx, cy, aspect=w/h, h) — состояние фильтра Калмана."""
    b = np.asarray(box, dtype=np.float64)
    w = b[2] - b[0]
    h = max(b[3] - b[1], 1e-6)
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h], dtype=np.float64)


def xyah_to_xyxy(xyah) -> np.ndarray:
    cx, cy, a, h = np.asarray(xyah, dtype=np.float64)[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def touches_border(box, width: int, height: int, margin: float = 0.02) -> bool:
    """Касается ли рамка края кадра (объект уходит или входит в кадр)."""
    b = np.asarray(box, dtype=np.float64)
    mx, my = width * margin, height * margin
    return bool(b[0] <= mx or b[1] <= my or b[2] >= width - mx or b[3] >= height - my)


def point_in_box(point, box) -> bool:
    x, y = point
    b = np.asarray(box, dtype=np.float64)
    return bool(b[0] <= x <= b[2] and b[1] <= y <= b[3])


def click_rank(point, box, score: float) -> float:
    """Чем меньше, тем вероятнее, что клик — по этому игроку.

    Игроки часто стоят вплотную: точка попадает сразу в несколько рамок, в том
    числе в край соседа или в слитную рамку двоих. Клик ставят в фигуру, поэтому
    решает удалённость точки от центра рамки (в долях её размеров), а
    уверенность детекции разводит близкие случаи."""
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = (point[0] - (x1 + x2) / 2) / max(x2 - x1, 1.0)
    dy = (point[1] - (y1 + y2) / 2) / max(y2 - y1, 1.0)
    return (dx * dx + 0.25 * dy * dy) ** 0.5 - 0.3 * float(score)
