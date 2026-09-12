"""Мультитрекер ByteTrack-типа с учётом внешности.

Каждый кадр:
  1. предсказание всех треков фильтром Калмана;
  2. первая ассоциация: подтверждённые и недавно потерянные треки <-> сильные
     детекции; стоимость = (1 - IoU), при наличии дескрипторов смешивается с
     косинусным расстоянием внешности и стробируется расстоянием Махаланобиса;
  3. вторая ассоциация: неприкаянные активные треки <-> слабые детекции
     (частично перекрытые игроки) только по IoU;
  4. новые треки из оставшихся сильных детекций; треки без обновлений дольше
     `lost_max` кадров удаляются.

Мультитрекер не знает, кто из игроков «наш». Его задача — стабильные
идентификаторы на коротких отрезках; длинные разрывы и перекрытия решает
`target.TargetFollower` поверх него.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .appearance import cosine_similarity, l2_normalize
from .detection import Detection
from .geometry import as_boxes, iou_matrix, xyah_to_xyxy, xyxy_to_xyah
from .kalman import CHI2_95, KalmanBoxFilter


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    start_frame: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    last_frame: int = 0
    feature: Optional[np.ndarray] = None       # сглаженный дескриптор
    last_feature: Optional[np.ndarray] = None  # дескриптор с последней детекции
    last_box: Optional[np.ndarray] = None      # рамка последней детекции (без Калмана)
    extra: dict = field(default_factory=dict)

    @property
    def box(self) -> np.ndarray:
        """Текущая оценка рамки (предсказание, если в этом кадре детекции не было)."""
        return xyah_to_xyxy(self.mean[:4])

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[4:6].copy()

    @property
    def height(self) -> float:
        return float(self.mean[3])

    @property
    def detected_now(self) -> bool:
        return self.time_since_update == 0

    def is_active(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Венгерский алгоритм (scipy) с порогом стоимости; без scipy — жадный перебор."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    except ImportError:  # pragma: no cover
        pairs = []
        used_r, used_c = set(), set()
        for r, c in sorted(np.ndindex(*cost.shape), key=lambda rc: cost[rc]):
            if cost[r, c] > thresh:
                break
            if r in used_r or c in used_c:
                continue
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    mr = {r for r, _ in pairs}
    mc = {c for _, c in pairs}
    return pairs, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]


@dataclass
class MultiTrackerConfig:
    det_high: float = 0.5        # порог «сильной» детекции
    det_low: float = 0.1         # порог «слабой» детекции для второго прохода
    match_iou_thresh: float = 0.8   # максимум стоимости (1 - IoU) первого прохода
    second_iou_thresh: float = 0.5  # второй проход
    new_track_iou_thresh: float = 0.7  # новый трек vs. неподтверждённые
    lost_max: int = 45           # кадров хранить потерянный трек (1.5 с при 30 fps)
    min_hits: int = 3            # подтверждение трека
    appearance_weight: float = 0.3   # доля косинусного расстояния в стоимости
    feature_momentum: float = 0.9    # EMA сглаженного дескриптора
    gate_chi2: float = CHI2_95[4]    # строб Махаланобиса (df=4)
    gate_lost_only_position: bool = True


class MultiTracker:
    def __init__(self, config: MultiTrackerConfig | None = None):
        self.cfg = config or MultiTrackerConfig()
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_idx = -1

    # --- вспомогательное -------------------------------------------------
    def _new_track(self, det: Detection, feat: Optional[np.ndarray], frame_idx: int) -> Track:
        mean, cov = self.kf.initiate(xyxy_to_xyah(det.box))
        t = Track(self._next_id, mean, cov, det.score, frame_idx, last_frame=frame_idx,
                  last_box=np.asarray(det.box, dtype=np.float64))
        if feat is not None:
            t.feature = l2_normalize(feat)
            t.last_feature = t.feature.copy()
        self._next_id += 1
        return t

    def _update_track(self, t: Track, det: Detection, feat: Optional[np.ndarray], frame_idx: int) -> None:
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(det.box))
        t.score = det.score
        t.hits += 1
        t.time_since_update = 0
        t.last_frame = frame_idx
        t.last_box = np.asarray(det.box, dtype=np.float64)
        if feat is not None:
            f = l2_normalize(feat)
            t.last_feature = f
            if t.feature is None:
                t.feature = f
            else:
                m = self.cfg.feature_momentum
                t.feature = l2_normalize(m * t.feature + (1 - m) * f)
        if t.state == TrackState.LOST:
            t.state = TrackState.CONFIRMED
        elif t.state == TrackState.TENTATIVE and t.hits >= self.cfg.min_hits:
            t.state = TrackState.CONFIRMED

    def _cost(self, tracks: list[Track], dets: list[Detection], feats: Optional[np.ndarray], use_appearance: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = as_boxes([t.box for t in tracks])
        db = as_boxes([d.box for d in dets])
        cost = 1.0 - iou_matrix(tb, db)
        if use_appearance and feats is not None and self.cfg.appearance_weight > 0:
            tf = np.stack([t.feature if t.feature is not None else np.zeros(feats.shape[1]) for t in tracks])
            app = 1.0 - cosine_similarity(tf, feats)
            w = self.cfg.appearance_weight
            cost = (1 - w) * cost + w * app
        # строб Махаланобиса: невероятные пары исключаем
        meas = np.stack([xyxy_to_xyah(d.box) for d in dets])
        for i, t in enumerate(tracks):
            only_pos = self.cfg.gate_lost_only_position and t.state == TrackState.LOST
            g = self.kf.gating_distance(t.mean, t.cov, meas, only_position=only_pos)
            chi = CHI2_95[2] if only_pos else self.cfg.gate_chi2
            # потерянным трекам разрешаем шире: неопределённость растёт со временем
            chi *= 1.0 + 0.5 * t.time_since_update
            cost[i, g > chi] = 1e6
        return cost

    # --- основной шаг ------------------------------------------------------
    def update(self, detections: list[Detection], features: Optional[np.ndarray] = None,
               frame_idx: Optional[int] = None) -> list[Track]:
        """Обновляет треки детекциями кадра, возвращает активные (подтверждённые) треки."""
        self.frame_idx = self.frame_idx + 1 if frame_idx is None else frame_idx
        fi = self.frame_idx
        feats = None if features is None else np.asarray(features, dtype=np.float32)

        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            t.age += 1
            t.time_since_update += 1

        hi = [i for i, d in enumerate(detections) if d.score >= self.cfg.det_high]
        lo = [i for i, d in enumerate(detections) if self.cfg.det_low <= d.score < self.cfg.det_high]
        det_hi = [detections[i] for i in hi]
        det_lo = [detections[i] for i in lo]
        f_hi = feats[hi] if feats is not None and len(hi) else None
        f_lo = feats[lo] if feats is not None and len(lo) else None

        confirmed = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        # 1. сильные детекции <-> подтверждённые/потерянные
        cost = self._cost(confirmed, det_hi, f_hi, use_appearance=True)
        pairs, un_tr, un_det = linear_assignment(cost, self.cfg.match_iou_thresh)
        for r, c in pairs:
            self._update_track(confirmed[r], det_hi[c], None if f_hi is None else f_hi[c], fi)

        # 2. слабые детекции <-> оставшиеся активные подтверждённые (не потерянные)
        remaining = [confirmed[r] for r in un_tr if confirmed[r].state == TrackState.CONFIRMED and confirmed[r].time_since_update == 1]
        cost = self._cost(remaining, det_lo, None, use_appearance=False)
        pairs2, un_tr2, _ = linear_assignment(cost, self.cfg.second_iou_thresh)
        for r, c in pairs2:
            self._update_track(remaining[r], det_lo[c], None if f_lo is None else f_lo[c], fi)
        for r in un_tr2:
            remaining[r].state = TrackState.LOST
        for r in un_tr:
            t = confirmed[r]
            if t.state == TrackState.CONFIRMED and t.time_since_update > 0 and t not in remaining:
                t.state = TrackState.LOST

        # 3. неподтверждённые треки <-> оставшиеся сильные детекции
        left_hi = [det_hi[c] for c in un_det]
        left_f = None if f_hi is None else f_hi[un_det]
        cost = self._cost(tentative, left_hi, None, use_appearance=False)
        pairs3, un_tent, un_left = linear_assignment(cost, self.cfg.new_track_iou_thresh)
        for r, c in pairs3:
            self._update_track(tentative[r], left_hi[c], None if left_f is None else left_f[c], fi)
        for r in un_tent:
            tentative[r].state = TrackState.REMOVED

        # 4. новые треки
        for c in un_left:
            self.tracks.append(self._new_track(left_hi[c], None if left_f is None else left_f[c], fi))

        # 5. чистка
        alive = []
        for t in self.tracks:
            if t.state == TrackState.REMOVED:
                continue
            if t.state == TrackState.LOST and t.time_since_update > self.cfg.lost_max:
                continue
            alive.append(t)
        self.tracks = alive
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def get(self, track_id: int) -> Optional[Track]:
        for t in self.tracks:
            if t.track_id == track_id and t.state != TrackState.REMOVED:
                return t
        return None
