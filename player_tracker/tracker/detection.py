"""Детекция людей на кадре.

Протокол `Detector.detect(frame) -> list[Detection]` — единственная точка
контакта с нейросетью. Основная реализация — YOLO через `ultralytics`
(импортируется лениво, чтобы ядро и тесты работали без torch). Для тестов и
разметки — `ScriptedDetector`, отдающий заранее известные рамки.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .geometry import as_boxes, clip_box

PERSON_CLASS = 0  # индекс класса «person» в COCO
BALL_CLASS = 32   # «sports ball»: мяч мелкий, находится не на каждом кадре — используется только для метрик

# Нейросеть на GPU — по одному вызову на процесс: одновременные вызовы из двух потоков (слежение одной задачи и
# фоновый поиск мяча другой) роняют процесс внутри Metal/PyTorch (SIGSEGV в arange_range_fill_mps, 16.09.2026).
GPU_LOCK = threading.RLock()


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
      device  — "cpu", "mps" (Apple Silicon), "cuda:0" или None (авто: cuda -> mps -> cpu;
                ultralytics сам выбирает только cuda, и на Mac без явного "mps" считает на CPU в разы медленнее).
    """

    def __init__(self, weights: str = "yolo11s.pt", imgsz: int = 1280, conf: float = 0.1,
                 iou: float = 0.6, device: str | None = None, classes: Iterable[int] = (PERSON_CLASS, BALL_CLASS),
                 half: bool = False, ball_tiles: bool = True, ball_tile: int = 640, ball_conf: float = 0.03,
                 ball_hint_frames: int = 0, ball_hint_conf: float = 0.25):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для YoloDetector нужен пакет ultralytics: pip install ultralytics"
            ) from exc
        with GPU_LOCK:
            self._model = YOLO(resolve_weights(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device or best_device()
        self.classes = list(classes)
        self.half = half
        self.ball_tiles = ball_tiles and BALL_CLASS in self.classes
        self.ball_tile = ball_tile
        self.ball_conf = ball_conf
        # ball_hint_frames > 0: пока есть свежая уверенная детекция, считается одна плитка вокруг неё,
        # полный обход — только когда мяч потерян дольше стольких кадров. Вдвое быстрее (198 против 403 мс
        # на кадр 1280×720), но кандидат мяча теряется на каждом пятом кадре (0.89 -> 0.69) — по умолчанию выключено
        self.ball_hint_frames = ball_hint_frames
        self.ball_hint_conf = ball_hint_conf
        self._hint: tuple[float, float] | None = None
        self._hint_age = 0
        self.tile_calls = 0            # сколько плиток посчитано (для замера)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        with GPU_LOCK:
            return self._detect(frame)

    def _detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=self.classes,
            device=self.device, verbose=False,
        )
        out: list[Detection] = []
        h, w = frame.shape[:2]
        r = results[0] if results else None
        if r is not None and r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, conf, cls):
                out.append(Detection(clip_box(b, w, h), float(s), int(c)))
        if self.ball_tiles:
            balls = merge_points(self._detect_balls(frame) + [d for d in out if d.cls == BALL_CLASS])
            out = [d for d in out if d.cls != BALL_CLASS] + balls
            best = max(balls, key=lambda d: d.score, default=None)
            if best is not None and best.score >= self.ball_hint_conf:
                self._hint, self._hint_age = ((best.box[0] + best.box[2]) / 2, (best.box[1] + best.box[3]) / 2), 0
            else:
                self._hint_age += 1
        return out

    def reset(self) -> None:
        self._hint, self._hint_age = None, 0

    def detect_balls(self, frame: np.ndarray) -> list[Detection]:
        with GPU_LOCK:
            return self._detect_balls(frame)

    def _detect_balls(self, frame: np.ndarray) -> list[Detection]:
        """Мяч на увеличенных плитках кадра: сеть видит мяч в 2 раза крупнее (4–8 px -> 8–16 px).

        На кадре 1280×720 целиком мяч с дальнего плана находится на ~16 % кадров, по плиткам 640 px
        со входом 1280 — примерно вдвое чаще (замер на игровом ролике)."""
        H, W = frame.shape[:2]
        if (self.ball_hint_frames > 0 and self._hint is not None and self._hint_age <= self.ball_hint_frames
                and W >= self.ball_tile and H >= self.ball_tile):
            offs = [(int(np.clip(self._hint[0] - self.ball_tile / 2, 0, W - self.ball_tile)),
                     int(np.clip(self._hint[1] - self.ball_tile / 2, 0, H - self.ball_tile)))]
        else:
            offs = tile_offsets(W, H, self.ball_tile)
        if not offs:
            return []
        self.tile_calls += len(offs)
        crops = [np.ascontiguousarray(frame[y:y + self.ball_tile, x:x + self.ball_tile]) for x, y in offs]
        results = self._model.predict(crops, imgsz=self.imgsz, conf=self.ball_conf, iou=self.iou, classes=[BALL_CLASS],
                                       device=self.device, verbose=False, batch=len(crops))
        out = []
        for (x, y), r in zip(offs, results):
            if r.boxes is None or not len(r.boxes):
                continue
            for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append(Detection(clip_box(b + np.array([x, y, x, y]), W, H), float(s), BALL_CLASS))
        return out

    def detect_balls_roi(self, frame: np.ndarray, center: tuple[float, float], size: int = 256, imgsz: int = 1024,
                         conf: float = 0.15) -> list[Detection]:
        """Мяч на вырезке size×size вокруг ожидаемого положения, увеличенной до imgsz (мяч 4 px -> 16 px)."""
        H, W = frame.shape[:2]
        x0 = int(np.clip(center[0] - size / 2, 0, max(W - size, 0)))
        y0 = int(np.clip(center[1] - size / 2, 0, max(H - size, 0)))
        crop = np.ascontiguousarray(frame[y0:y0 + size, x0:x0 + size])
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            return []
        with GPU_LOCK:
            r = self._model.predict(crop, imgsz=imgsz, conf=conf, iou=self.iou, classes=[BALL_CLASS],
                                    device=self.device, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        return [Detection(clip_box(b + np.array([x0, y0, x0, y0]), W, H), float(s), BALL_CLASS)
                for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


def tile_offsets(width: int, height: int, tile: int, overlap: float = 0.25, skip_top: float = 0.1) -> list[tuple[int, int]]:
    """Левые верхние углы плиток, покрывающих кадр ниже `skip_top` (небо) с перекрытием."""
    if width < tile or height < tile:
        return []
    step = max(int(tile * (1 - overlap)), 1)

    def axis(size: int, start: int) -> list[int]:
        if start > 0 and start >= size - tile - step // 4:   # под небом осталось меньше четверти шага — один ряд
            return [size - tile]
        pos = list(range(start, size - tile + 1, step))
        if not pos or pos[-1] != size - tile:
            pos.append(size - tile)
        return sorted(set(p for p in pos if p >= 0))

    top = min(int(height * skip_top), height - tile)
    return [(x, y) for y in axis(height, top) for x in axis(width, 0)]


def merge_points(dets: list[Detection], radius: float = 6.0) -> list[Detection]:
    """Слияние детекций мелкого объекта из перекрывающихся плиток и полного кадра: по центрам, сильнейшая остаётся."""
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        c = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
        if all(np.hypot(c[0] - (k.box[0] + k.box[2]) / 2, c[1] - (k.box[1] + k.box[3]) / 2) > radius for k in keep):
            keep.append(d)
    return keep


WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def resolve_weights(weights: str) -> str:
    """Голое имя весов («yolo11s.pt») -> player_tracker/weights/<имя>.

    ultralytics скачивает недостающие веса в текущий каталог процесса — у сервера это может быть
    чей угодно рабочий каталог. Явный путь (с каталогом) и уже существующий файл не трогаем."""
    p = Path(weights)
    if p.parent != Path(".") or p.exists() or p.suffix != ".pt":
        return weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / p.name)


def best_device() -> str:
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - зависит от окружения
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


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
            cls = int(item[5]) if len(item) > 5 else PERSON_CLASS
            dets.append(Detection(box, score, cls))
        return dets


def filter_detections(dets: list[Detection], min_height: float = 0.0, max_aspect: float = 1.2,
                      min_score: float = 0.0) -> list[Detection]:
    """Люди без мусора: слишком низкие рамки, «лежачие» рамки (не человек стоя/в беге), слабые."""
    out = []
    for d in dets:
        if d.cls != PERSON_CLASS:
            continue
        w = float(d.box[2] - d.box[0])
        h = float(d.box[3] - d.box[1])
        if h < min_height or h <= 0 or d.score < min_score:
            continue
        if w / h > max_aspect:
            continue
        out.append(d)
    return out
