"""Отрисовка результата на кадре."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .multitracker import Track
from .target import TargetObservation, TargetState

COLOR_ACTIVE = (40, 200, 40)
COLOR_CONTESTED = (0, 200, 255)
COLOR_LOST = (0, 0, 255)
COLOR_OTHER = (160, 160, 160)
COLOR_CANDIDATE = (255, 160, 0)


def _label(out: np.ndarray, text: str, x: int, y: int, color, scale: float = 0.5) -> None:
    """Номер на плашке над рамкой: читается на траве и на форме любого цвета."""
    import cv2

    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(y, h + base + 2)
    x = int(min(max(x, 0), out.shape[1] - w - 4))
    cv2.rectangle(out, (x, y - h - base - 2), (x + w + 4, y), (20, 20, 20), -1)
    cv2.putText(out, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, obs: TargetObservation, tracks: Optional[list[Track]] = None,
         show_others: bool = True, thickness: int = 2, numbers: Optional[dict] = None) -> np.ndarray:
    """numbers — короткие номера треков (`board.display_numbers`): те же, что на тактической доске.
    Без них подписывается id трека."""
    import cv2

    out = frame.copy()
    if show_others and tracks:
        for t in tracks:
            if obs.track_id == t.track_id or not t.detected_now:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_OTHER, 1)
            num = numbers.get(t.track_id) if numbers is not None else t.track_id
            if num is not None:
                _label(out, str(num), x1, y1 - 2, (255, 255, 255))
    if obs.state == TargetState.LOST:
        for c in obs.candidates:
            if c.rejected:
                continue
            x1, y1, x2, y2 = [int(v) for v in c.box]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_CANDIDATE, 1)
            cv2.putText(out, f"{c.score:.2f}", (x1, min(y2 + 14, out.shape[0] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CANDIDATE, 1)
        label = f"LOST {obs.lost_frames}f" + (" ambiguous" if obs.ambiguous else "")
        cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LOST, 2)
    elif obs.box is not None:
        color = COLOR_CONTESTED if obs.state == TargetState.CONTESTED else COLOR_ACTIVE
        x1, y1, x2, y2 = [int(v) for v in obs.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness + 1)
        num = numbers.get(obs.track_id) if numbers is not None else obs.track_id
        _label(out, f"{num if num is not None else ''} TARGET".strip(), x1, y1 - 3, color, 0.55)
        cv2.putText(out, obs.state.value.upper(), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if obs.event:
        cv2.putText(out, obs.event, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
