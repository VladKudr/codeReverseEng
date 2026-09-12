"""Конвейер: кадр -> детекции -> дескрипторы -> мультитрекер -> цель -> вывод.

Один объект `Pipeline` обрабатывает поток кадров (любой итератор numpy BGR).
Детектор, кодировщик внешности и читатель номеров передаются извне — так
конвейер тестируется на синтетике без нейросетей, а в бою получает YOLO.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from .appearance import AppearanceEncoder
from .detection import Detection, Detector, filter_detections
from .geometry import iou_matrix, point_in_box
from .jersey import NumberRead, NumberReader, NumberVotes, ScriptedNumberReader, torso_crop
from .multitracker import MultiTracker, MultiTrackerConfig, Track, TrackState
from .target import TargetConfig, TargetFollower, TargetObservation, TargetState
from .team import TeamClassifier, torso_color


@dataclass
class InitSpec:
    """Как выбрать цель: рамка или точка на кадре `frame`, необязательный номер."""

    frame: int = 0
    box: Optional[tuple[float, float, float, float]] = None   # x1, y1, x2, y2
    point: Optional[tuple[float, float]] = None
    number: Optional[str] = None
    max_wait_frames: int = 90   # сколько кадров ждать появления трека в рамке/точке


@dataclass
class PipelineConfig:
    tracker: MultiTrackerConfig = field(default_factory=MultiTrackerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    init: InitSpec = field(default_factory=InitSpec)
    min_det_height: float = 24.0        # пиксели в кадре обработки
    max_det_aspect: float = 1.2
    use_team: bool = True
    number_every: int = 5               # OCR цели раз в N кадров в состоянии ACTIVE
    number_lost_every: int = 2          # OCR кандидатов раз в N кадров в состоянии LOST
    number_max_candidates: int = 6
    pass_lost_tracks: bool = True       # отдавать цели и потерянные треки мультитрекера


@dataclass
class FrameResult:
    frame_idx: int
    observation: TargetObservation
    tracks: list[Track]
    detections: list[Detection]
    elapsed_ms: float


class Pipeline:
    def __init__(self, frame_size: tuple[int, int], detector: Detector, encoder: AppearanceEncoder,
                 number_reader: Optional[NumberReader] = None, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.width, self.height = frame_size
        self.detector = detector
        self.encoder = encoder
        self.number_reader = number_reader
        self.mot = MultiTracker(self.cfg.tracker)
        self.follower = TargetFollower(frame_size, self.cfg.target)
        self.teams = TeamClassifier() if self.cfg.use_team else None
        self.frame_idx = -1
        self._init_deadline: Optional[int] = None

    # --- выбор цели --------------------------------------------------------
    def _try_init(self, tracks: list[Track], features: dict[int, np.ndarray]) -> Optional[TargetObservation]:
        spec = self.cfg.init
        if self.frame_idx < spec.frame:
            return None
        if self._init_deadline is None:
            self._init_deadline = spec.frame + spec.max_wait_frames
        cands = [t for t in tracks if t.detected_now]
        pick: Optional[Track] = None
        if spec.box is not None and cands:
            ious = iou_matrix([spec.box], [t.box for t in cands])[0]
            k = int(np.argmax(ious))
            if ious[k] > 0.3:
                pick = cands[k]
        elif spec.point is not None:
            inside = [t for t in cands if point_in_box(spec.point, t.box)]
            if inside:
                # при вложенных рамках берём меньшую: клик точнее попадает в ближнего игрока
                pick = min(inside, key=lambda t: t.height)
        if pick is None:
            if self.frame_idx > self._init_deadline:
                raise RuntimeError(
                    f"Цель не найдена: ни один трек не совпал с init.box/init.point за {spec.max_wait_frames} кадров"
                )
            return None
        return self.follower.lock(pick, features.get(pick.track_id), self.frame_idx, number=spec.number)

    # --- номера и команды ----------------------------------------------------
    def _read_numbers(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, NumberRead]:
        reader = self.number_reader
        if reader is None:
            return {}
        if isinstance(reader, ScriptedNumberReader):
            return {t.track_id: r for t in tracks if (r := reader.read_for(self.frame_idx, t.track_id)) is not None}
        state = self.follower.state
        targets: list[Track] = []
        if state in (TargetState.ACTIVE, TargetState.CONTESTED):
            if self.frame_idx % self.cfg.number_every == 0:
                targets = [t for t in tracks if t.track_id == self.follower.track_id]
        elif state == TargetState.LOST and self.frame_idx % self.cfg.number_lost_every == 0:
            targets = sorted((t for t in tracks if t.detected_now), key=lambda t: -t.height)[: self.cfg.number_max_candidates]
        out = {}
        for t in targets:
            crop = torso_crop(frame, t.last_box if t.last_box is not None else t.box)
            if crop is None:
                continue
            r = reader.read(crop)
            if r is not None:
                out[t.track_id] = r
        return out

    def _team_labels(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, int]:
        if self.teams is None:
            return {}
        labels = {}
        for t in tracks:
            if not t.detected_now:
                continue
            color = torso_color(frame, t.last_box if t.last_box is not None else t.box)
            self.teams.observe(color)
            if self.teams.ready:
                labels[t.track_id] = self.teams.predict(color)
        if not self.teams.ready:
            self.teams.fit()
        return labels

    # --- шаг ---------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_idx += 1
        dets = filter_detections(self.detector.detect(frame), self.cfg.min_det_height, self.cfg.max_det_aspect)
        feats = self.encoder.encode(frame, [d.box for d in dets]) if dets else np.zeros((0, self.encoder.dim), np.float32)
        confirmed = self.mot.update(dets, feats, frame_idx=self.frame_idx)
        tracks = [t for t in self.mot.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)] if self.cfg.pass_lost_tracks else confirmed
        features = {t.track_id: t.last_feature for t in tracks if t.detected_now and t.last_feature is not None}

        obs: Optional[TargetObservation] = None
        if self.follower.state == TargetState.IDLE:
            obs = self._try_init(tracks, features)
        team_labels = self._team_labels(frame, tracks)
        numbers = self._read_numbers(frame, tracks)
        if obs is None:
            obs = self.follower.step(self.frame_idx, tracks, features, numbers, team_labels)
        else:
            # в кадре захвата тоже учитываем номер/команду
            self.follower.step(self.frame_idx, tracks, features, numbers, team_labels)
        return FrameResult(self.frame_idx, obs, tracks, dets, (time.perf_counter() - t0) * 1000)

    def run(self, frames: Iterable[np.ndarray], on_frame: Optional[Callable[[np.ndarray, FrameResult], None]] = None) -> list[FrameResult]:
        results = []
        for frame in frames:
            res = self.process(frame)
            results.append(res)
            if on_frame is not None:
                on_frame(frame, res)
        return results

    # --- ручное вмешательство ---------------------------------------------------
    def reselect(self, track_id: int) -> TargetObservation:
        """Оператор указал верный трек (например, после ambiguous)."""
        t = self.mot.get(track_id)
        if t is None:
            raise KeyError(f"Трек {track_id} не существует")
        feat = t.last_feature
        return self.follower.lock(t, feat, self.frame_idx)
