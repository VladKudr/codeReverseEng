"""Дескрипторы внешности игрока.

Задача дескриптора — отличать игроков в одинаковой форме, поэтому важен не
цвет футболки (он у всех один), а то, что у игроков различается: волосы и
цвет кожи (голова), гетры/бутсы, телосложение (соотношение сторон рамки).

Две реализации одного протокола `encode(frame, boxes) -> (N, dim)`:
  * `PartColorEncoder` — полосовые HSV-гистограммы с весами по частям тела;
    без нейросетей, работает на CPU в реальном времени. Базовый вариант.
  * `TorchReidEncoder` — OSNet из пакета `torchreid` (опционально): устойчивее
    к позе и освещению; при наличии — предпочтителен.
  * `CompositeEncoder` — конкатенация нескольких дескрипторов с весами.

Все векторы L2-нормированы, сходство — косинусное.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .geometry import as_boxes


class AppearanceEncoder(Protocol):
    dim: int

    def encode(self, frame: np.ndarray, boxes) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Косинусная близость между (N, d) и (M, d) -> (N, M). Векторы уже нормированы."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return a @ b.T


@dataclass
class BodyStripe:
    """Горизонтальная полоса тела: доля высоты рамки [top, bottom) и вес в дескрипторе."""

    name: str
    top: float
    bottom: float
    weight: float


# Разбиение фигуры сверху вниз. Торс и шорты у одноклубников одинаковы —
# вес мал, но не ноль: он отделяет соперников и судью без отдельной модели.
DEFAULT_STRIPES: tuple[BodyStripe, ...] = (
    BodyStripe("head", 0.00, 0.16, 1.00),
    BodyStripe("torso", 0.16, 0.48, 0.35),
    BodyStripe("shorts", 0.48, 0.64, 0.30),
    BodyStripe("legs", 0.64, 0.88, 0.80),
    BodyStripe("boots", 0.88, 1.00, 0.60),
)


class PartColorEncoder:
    """Полосовые HSV-гистограммы с весами по частям тела.

    Для каждой полосы считается двумерная гистограмма (H x S) и гистограмма
    яркости V по центральной части рамки (края рамки — трава и соседи).
    Пиксели с низкой насыщенностью попадают только в V-гистограмму: у них
    тон не определён (белая форма, бледная кожа, чёрные бутсы).
    """

    def __init__(self, stripes: Sequence[BodyStripe] = DEFAULT_STRIPES, h_bins: int = 12, s_bins: int = 4,
                 v_bins: int = 8, center_frac: float = 0.6, min_saturation: int = 40, min_value: int = 30,
                 value_weight: float = 0.5):
        self.stripes = tuple(stripes)
        self.h_bins, self.s_bins, self.v_bins = h_bins, s_bins, v_bins
        self.value_weight = value_weight  # яркость информативна меньше тона: освещение меняется
        self.center_frac = center_frac
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.stripe_dim = h_bins * s_bins + v_bins
        self.dim = self.stripe_dim * len(self.stripes)

    def _stripe_hist(self, hsv: np.ndarray) -> np.ndarray:
        if hsv.size == 0:
            return np.zeros(self.stripe_dim, dtype=np.float32)
        h = hsv[..., 0].reshape(-1).astype(np.int32)
        s = hsv[..., 1].reshape(-1).astype(np.int32)
        v = hsv[..., 2].reshape(-1).astype(np.int32)
        chroma = (s >= self.min_saturation) & (v >= self.min_value)
        hs = np.zeros(self.h_bins * self.s_bins, dtype=np.float32)
        if chroma.any():
            hi = np.clip(h[chroma] * self.h_bins // 180, 0, self.h_bins - 1)
            si = np.clip(s[chroma] * self.s_bins // 256, 0, self.s_bins - 1)
            np.add.at(hs, hi * self.s_bins + si, 1.0)
        vh = np.zeros(self.v_bins, dtype=np.float32)
        vi = np.clip(v * self.v_bins // 256, 0, self.v_bins - 1)
        np.add.at(vh, vi, 1.0)
        # тон/насыщенность и яркость нормируются порознь: иначе у слабонасыщенных
        # частей (белая форма, чёрные бутсы) яркость подавляет всё остальное
        return np.concatenate([l2_normalize(hs), l2_normalize(vh) * self.value_weight])

    def encode(self, frame: np.ndarray, boxes) -> np.ndarray:
        import cv2

        boxes = as_boxes(boxes)
        H, W = frame.shape[:2]
        out = np.zeros((len(boxes), self.dim), dtype=np.float32)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            w = x2 - x1
            pad = (1 - self.center_frac) / 2 * w
            xa, xb = int(max(0, x1 + pad)), int(min(W, x2 - pad))
            ya, yb = int(max(0, y1)), int(min(H, y2))
            if xb - xa < 2 or yb - ya < 2:
                continue
            crop = frame[ya:yb, xa:xb]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            ch = hsv.shape[0]
            parts = []
            for st in self.stripes:
                sa, sb = int(st.top * ch), int(st.bottom * ch)
                hist = self._stripe_hist(hsv[sa:max(sb, sa + 1)])
                parts.append(l2_normalize(hist) * st.weight)
            out[i] = np.concatenate(parts)
        return l2_normalize(out)


class TorchReidEncoder:
    """Нейросетевой ReID-дескриптор (OSNet) из пакета torchreid — опционально."""

    def __init__(self, model_name: str = "osnet_x0_25", device: str | None = None, batch: int = 32):
        try:
            import torch  # type: ignore
            import torchreid  # type: ignore
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError("Для TorchReidEncoder нужны torch и torchreid") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torchreid.models.build_model(name=model_name, num_classes=1000, pretrained=True)
        self._model.eval().to(self.device)
        self.batch = batch
        self.dim = 512
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def encode(self, frame: np.ndarray, boxes) -> np.ndarray:  # pragma: no cover - требует torch
        import cv2

        boxes = as_boxes(boxes)
        H, W = frame.shape[:2]
        crops = []
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(cv2.resize(crop, (128, 256)))
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = (np.stack(crops).astype(np.float32) / 255.0 - self._mean) / self._std
        tensor = self._torch.from_numpy(arr).permute(0, 3, 1, 2)
        feats = []
        with self._torch.no_grad():
            for i in range(0, len(tensor), self.batch):
                feats.append(self._model(tensor[i:i + self.batch].to(self.device)).cpu().numpy())
        return l2_normalize(np.concatenate(feats))


@dataclass
class CompositeEncoder:
    """Конкатенация дескрипторов с весами: например, OSNet (1.0) + цвет полос (0.5)."""

    encoders: list[tuple[AppearanceEncoder, float]] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return sum(e.dim for e, _ in self.encoders)

    def encode(self, frame: np.ndarray, boxes) -> np.ndarray:
        parts = [enc.encode(frame, boxes) * w for enc, w in self.encoders]
        if not parts:
            return np.zeros((len(as_boxes(boxes)), 0), dtype=np.float32)
        return l2_normalize(np.concatenate(parts, axis=1))


class ConstantEncoder:
    """Заглушка для тестов: дескриптор берётся из словаря по идентификатору рамки."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.by_box: dict[tuple, np.ndarray] = {}

    def encode(self, frame: np.ndarray, boxes) -> np.ndarray:
        boxes = as_boxes(boxes)
        out = np.zeros((len(boxes), self.dim), dtype=np.float32)
        for i, b in enumerate(boxes):
            key = tuple(np.round(b).astype(int).tolist())
            out[i] = self.by_box.get(key, np.ones(self.dim, dtype=np.float32))
        return l2_normalize(out)
