"""Номер на футболке: чтение и накопление голосов.

Номер — единственный признак, который надёжно различает игроков в одной
форме, но читается он редко (спина к камере, разрешение, размытие). Поэтому
чтение отделено от решения: `NumberReader` выдаёт отдельные ненадёжные
прочтения, `NumberVotes` копит их и выносит вердикт только при перевесе.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

_DIGITS = re.compile(r"^\d{1,2}$")


@dataclass
class NumberRead:
    text: str
    confidence: float


class NumberReader(Protocol):
    def read(self, crop: np.ndarray) -> Optional[NumberRead]: ...


def torso_crop(frame: np.ndarray, box, top: float = 0.14, bottom: float = 0.52, center_frac: float = 0.8,
               min_side: int = 24) -> Optional[np.ndarray]:
    """Вырезка области номера (спина/грудь). None, если вырезка слишком мала для OCR."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    pad = (1 - center_frac) / 2 * w
    xa, xb = int(max(0, x1 + pad)), int(min(W, x2 - pad))
    ya, yb = int(max(0, y1 + top * h)), int(min(H, y1 + bottom * h))
    if xb - xa < min_side or yb - ya < min_side:
        return None
    return frame[ya:yb, xa:xb]


class EasyOcrReader:
    """Чтение цифр пакетом easyocr (опционально; тяжёлый, вызывать редко)."""

    def __init__(self, gpu: bool = False, min_confidence: float = 0.3, upscale_to: int = 96):
        try:
            import easyocr  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Для EasyOcrReader нужен пакет easyocr") from exc
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        self.min_confidence = min_confidence
        self.upscale_to = upscale_to

    def read(self, crop: np.ndarray) -> Optional[NumberRead]:  # pragma: no cover - требует easyocr
        import cv2

        h = crop.shape[0]
        if h < self.upscale_to:
            k = self.upscale_to / h
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        results = self._reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False)
        best = None
        for _, text, conf in results:
            text = text.strip()
            if _DIGITS.match(text) and conf >= self.min_confidence and (best is None or conf > best.confidence):
                best = NumberRead(text, float(conf))
        return best


class ScriptedNumberReader:
    """Для тестов: словарь frame_idx -> {track_id: NumberRead}; читается через `read_for`."""

    def __init__(self, script: dict[int, dict[int, NumberRead]]):
        self.script = script

    def read(self, crop: np.ndarray) -> Optional[NumberRead]:
        return None

    def read_for(self, frame_idx: int, track_id: int) -> Optional[NumberRead]:
        return self.script.get(frame_idx, {}).get(track_id)


@dataclass
class NumberVotes:
    """Накопитель голосов за номер.

    Вердикт `best()` выдаётся, когда суммарный вес лидера не меньше
    `min_weight` и он превосходит второго не менее чем в `dominance` раз.
    Так одно ошибочное прочтение «8» вместо «6» не переопределяет номер.
    """

    min_weight: float = 1.2
    dominance: float = 2.0
    weights: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    reads: int = 0

    def add(self, read: Optional[NumberRead]) -> None:
        if read is None or not _DIGITS.match(read.text):
            return
        self.weights[read.text] += float(read.confidence)
        self.reads += 1

    def set_known(self, number: str, weight: float = 10.0) -> None:
        """Номер задан пользователем — фиксируем с большим весом."""
        self.weights[str(int(number))] += weight

    def best(self, min_weight: Optional[float] = None) -> Optional[tuple[str, float]]:
        if not self.weights:
            return None
        ranked = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)
        lead, w = ranked[0]
        if w < (self.min_weight if min_weight is None else min_weight):
            return None
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if second > 0 and w / second < self.dominance:
            return None
        total = sum(self.weights.values())
        return lead, float(w / total)

    def relation(self, other: "NumberVotes", other_min_weight: Optional[float] = None) -> Optional[bool]:
        """True — номера совпадают, False — противоречат, None — хотя бы один неизвестен.

        `other_min_weight` — более строгий порог для чужого вердикта: когда от
        ответа зависит отказ от трека, двух прочтений мало.
        """
        a, b = self.best(), other.best(other_min_weight)
        if a is None or b is None:
            return None
        return a[0] == b[0]
