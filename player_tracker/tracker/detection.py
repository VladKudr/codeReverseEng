"""Детекция людей на кадре.

Протокол `Detector.detect(frame) -> list[Detection]` — единственная точка
контакта с нейросетью. Основная реализация — YOLO через `ultralytics`
(импортируется лениво, чтобы ядро и тесты работали без torch). Для тестов и
разметки — `ScriptedDetector`, отдающий заранее известные рамки.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from .geometry import as_boxes, clip_box

PERSON_CLASS = 0  # индекс класса «person» в COCO


@dataclass
class Detection:
    box: np.ndarray            # (x1, y1, x2, y2) в пикселях исходного кадра
    score: float
    cls: int = PERSON_CLASS
    extra: dict = field(default_factory=dict)

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloDetector:
    """Детектор людей на базе ultralytics YOLO (v8/11).

    Параметры:
      weights — файл весов (`yolov8n.pt` скачается сам при первом запуске;
                для 4K с мелкими фигурами лучше `yolov8s.pt`/`yolo11s.pt`);
      imgsz   — сторона входа сети; для футбола с дальнего плана 1280;
      conf    — нижний порог уверенности. Оставляем низким (0.1): ByteTrack
                использует слабые детекции на втором проходе ассоциации;
      device  — "cpu", "mps" (Apple Silicon), "cuda:0" или None (авто).
    """

    def __init__(self, weights: str = "yolov8n.pt", imgsz: int = 1280, conf: float = 0.1,
                 iou: float = 0.6, device: str | None = None, classes: Iterable[int] = (PERSON_CLASS,),
                 half: bool = False):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для YoloDetector нужен пакет ultralytics: pip install ultralytics"
            ) from exc
        self._model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = list(classes)
        self.half = half

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=self.classes,
            device=self.device, verbose=False,
        )
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        h, w = frame.shape[:2]
        for b, s, c in zip(xyxy, conf, cls):
            out.append(Detection(clip_box(b, w, h), float(s), int(c)))
        return out


class ScriptedDetector:
    """Детектор по сценарию: {frame_idx: [(x1, y1, x2, y2, score), ...]}."""

    def __init__(self, script: dict[int, list], noise: float = 0.0, seed: int = 0):
        self.script = script
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self.frame_idx = -1

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.frame_idx += 1
        dets = []
        for item in self.script.get(self.frame_idx, []):
            box = as_boxes(item[:4])[0]
            if self.noise:
                box = box + self._rng.normal(0, self.noise, size=4)
            score = float(item[4]) if len(item) > 4 else 0.9
            dets.append(Detection(box, score))
        return dets


def filter_detections(dets: list[Detection], min_height: float = 0.0, max_aspect: float = 1.2,
                      min_score: float = 0.0) -> list[Detection]:
    """Отсев мусора: слишком низкие рамки, «лежачие» рамки (не человек стоя/в беге), слабые."""
    out = []
    for d in dets:
        w = float(d.box[2] - d.box[0])
        h = float(d.box[3] - d.box[1])
        if h < min_height or h <= 0 or d.score < min_score:
            continue
        if w / h > max_aspect:
            continue
        out.append(d)
    return out
