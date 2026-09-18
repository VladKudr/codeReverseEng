"""Конвейер: кадр -> детекции -> дескрипторы -> мультитрекер -> цель -> вывод.

Один объект `Pipeline` обрабатывает поток кадров (любой итератор numpy BGR).
Детектор, кодировщик внешности и читатель номеров передаются извне — так
конвейер тестируется на синтетике без нейросетей, а в бою получает YOLO.

Кадр обрабатывается в два этапа: `read_inputs` (детекции, дескрипторы, цвет торса,
движение камеры — всё, что требует пикселей) и шаг слежения по этим данным.
Данные кадров можно сохранить (`record_inputs`) и прогнать слежение заново
(`process_inputs`) без декодирования и детектора — так после поправки оператора
(`Correction`) ролик перестраивается за секунды.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Optional

import numpy as np

from .appearance import AppearanceEncoder
from .camera import CameraMotion
from .detection import BALL_CLASS, Detection, Detector, filter_detections
from .geometry import click_rank, iou_matrix, point_in_box
from .jersey import NumberRead, NumberReader, ScriptedNumberReader, torso_crop
from .metrics import ErrorLog, RunMetrics
from .multitracker import MultiTracker, MultiTrackerConfig, Track, TrackState
from .offline import RefineConfig, TrackLog, refine_online
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
class Correction:
    """Поправка оператора на кадре `frame`: цель — игрок в точке/рамке, или (`absent`) цели в кадре нет."""

    frame: int
    point: Optional[tuple[float, float]] = None
    box: Optional[tuple[float, float, float, float]] = None
    absent: bool = False


@dataclass
class FrameInputs:
    """Всё, что шаг слежения берёт из пикселей кадра."""

    detections: list[Detection]
    features: np.ndarray                 # (N, d) дескрипторы детекций
    colors: np.ndarray                   # (N, 3) медианный Lab торса, NaN — не определён
    camera: Optional[np.ndarray]         # 2x3 предыдущий кадр -> текущий
    balls: np.ndarray = field(default_factory=lambda: np.zeros((0, 5), np.float32))   # (M, 5) x1,y1,x2,y2,score


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
    camera_motion: bool = True          # компенсация движения камеры (съёмка с рук)
    refine: bool = True                 # после прогона уточнить результат по всему ролику (offline.refine_online)
    refine_cfg: RefineConfig = field(default_factory=RefineConfig)
    fps: float = 30.0                   # для времени в метриках и журнале ошибок
    tolerate_errors: bool = True        # исключение детектора/кодировщика/OCR -> запись в журнал,
                                        # кадр обрабатывается как пустой; False — исключение наружу
    corrections: list[Correction] = field(default_factory=list)
    record_inputs: bool = False         # сохранять FrameInputs каждого кадра (для перестроения после поправок)


# записи журнала, не зависящие от решений о цели: при пересчёте метрик после уточнения сохраняются
_RUN_LEVEL_ERRORS = {"exception", "init_failed", "ocr_miss", "correction_failed"}


def snapshot_tracks(tracks: list[Track]) -> list[SimpleNamespace]:
    return [SimpleNamespace(track_id=t.track_id, detected_now=t.detected_now, box=t.box.copy()) for t in tracks]


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
        self.camera = CameraMotion() if self.cfg.camera_motion else None
        self.frame_idx = -1
        self._init_deadline: Optional[int] = None
        self.errors = ErrorLog(self.cfg.fps)
        self.metrics = RunMetrics(frame_size, self.cfg.fps, self.errors, det_high=self.cfg.tracker.det_high)
        self._init_failed = False
        self.log: Optional[TrackLog] = TrackLog() if self.cfg.refine else None
        self.online: list[TargetObservation] = []      # решения автомата по кадрам
        self._frames: list[tuple] = []                  # (кадр, детекции, лёгкие снимки треков, мс) для пересчёта метрик
        self._frame_dets: Optional[tuple[list[Detection], np.ndarray]] = None
        self.last_camera: Optional[np.ndarray] = None
        self.inputs: list[FrameInputs] = []            # при record_inputs
        self._corrections = {c.frame: c for c in self.cfg.corrections}

    # --- выбор цели --------------------------------------------------------
    def _try_init(self, tracks: list[Track], features: dict[int, np.ndarray]) -> Optional[TargetObservation]:
        spec = self.cfg.init
        if self.frame_idx < spec.frame:
            return None
        if self._init_deadline is None:
            self._init_deadline = spec.frame + spec.max_wait_frames
        pick = self._pick_track(tracks, spec.point, spec.box)
        if pick is None:
            if self.frame_idx > self._init_deadline and not self._init_failed:
                self._init_failed = True
                msg = f"Цель не найдена: ни один трек не совпал с init.box/init.point за {spec.max_wait_frames} кадров"
                self.errors.add(self.frame_idx, "init_failed", msg, box=spec.box, point=spec.point)
                if not self.cfg.tolerate_errors:
                    raise RuntimeError(msg)
            return None
        feat = features.get(pick.track_id, pick.last_feature)
        return self.follower.lock(pick, feat, self.frame_idx, number=spec.number)

    def _pick_track(self, tracks: list[Track], point, box) -> Optional[Track]:
        """Трек игрока, указанного точкой или рамкой на текущем кадре (выбор цели, поправка оператора).

        Годится и неподтверждённый трек: пользователь сам указал игрока. Если игрок найден только слабой
        детекцией, не ставшей треком, трек заводится сразу."""
        cands = [t for t in tracks if t.detected_now and t.state != TrackState.REMOVED]
        pick: Optional[Track] = None
        if box is not None and cands:
            ious = iou_matrix([box], [t.box for t in cands])[0]
            k = int(np.argmax(ious))
            if ious[k] > 0.3:
                pick = cands[k]
        elif point is not None:
            inside = [t for t in cands if point_in_box(point, t.box)]
            if inside:
                pick = min(inside, key=lambda t: click_rank(point, t.box, t.score))
        if pick is None and self._frame_dets:
            dets, feats = self._frame_dets
            idx = None
            if box is not None and dets:
                ious = iou_matrix([box], [d.box for d in dets])[0]
                k = int(np.argmax(ious))
                idx = k if ious[k] > 0.3 else None
            elif point is not None:
                inside = [i for i, d in enumerate(dets) if point_in_box(point, d.box)]
                idx = min(inside, key=lambda i: click_rank(point, dets[i].box, dets[i].score)) if inside else None
            if idx is not None:
                pick = self.mot.spawn(dets[idx], feats[idx] if len(feats) else None, self.frame_idx)
        if pick is not None and pick.state == TrackState.TENTATIVE:
            pick.state = TrackState.CONFIRMED
        return pick

    def _apply_correction(self, corr: Correction) -> Optional[TargetObservation]:
        everyone = [t for t in self.mot.tracks if t.detected_now]
        if corr.absent:
            return self.follower.mark_absent(self.frame_idx, everyone)
        pick = self._pick_track(everyone, corr.point, corr.box)
        if pick is None:
            self.errors.add(self.frame_idx, "correction_failed", "в указанной точке/рамке нет игрока",
                            point=corr.point, box=corr.box)
            return None
        return self.follower.correct(pick, pick.last_feature, self.frame_idx)

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
        attempted = 0
        for t in targets:
            crop = torso_crop(frame, t.last_box if t.last_box is not None else t.box)
            if crop is None:
                continue
            attempted += 1
            r = self._guard("ocr", lambda: reader.read(crop), None)
            if r is not None:
                out[t.track_id] = r
            else:
                self.errors.add(self.frame_idx, "ocr_miss", "номер не прочитан", track_id=t.track_id)
        self.metrics.observe_ocr(attempted, len(out))
        return out

    def _guard(self, where: str, fn, fallback):
        """Вызов нейросетевого компонента с перехватом исключений в журнал."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - любое исключение компонента фиксируем
            self.errors.exception(self.frame_idx, where, exc)
            if not self.cfg.tolerate_errors:
                raise
            return fallback

    def _team_labels(self, colors: np.ndarray, tracks: list[Track]) -> dict[int, int]:
        if self.teams is None:
            return {}
        labels = {}
        for t in tracks:
            i = t.extra.get("det_index")
            if not t.detected_now or i is None or i >= len(colors) or np.isnan(colors[i]).any():
                continue
            color = colors[i]
            self.teams.observe(color)
            if self.teams.ready:
                labels[t.track_id] = self.teams.predict(color)
        if not self.teams.ready:
            self.teams.fit()
        return labels

    # --- шаг ---------------------------------------------------------------------
    def read_inputs(self, frame: np.ndarray) -> FrameInputs:
        """Детекции, дескрипторы, цвет торса и движение камеры кадра (всё, что требует пикселей)."""
        raw = self._guard("detector", lambda: self.detector.detect(frame), [])
        dets = filter_detections(raw, self.cfg.min_det_height, self.cfg.max_det_aspect)
        if dets:
            feats = self._guard("encoder", lambda: self.encoder.encode(frame, [d.box for d in dets]), None)
            if feats is None:
                feats = np.zeros((len(dets), self.encoder.dim), np.float32)
        else:
            feats = np.zeros((0, self.encoder.dim), np.float32)
        if self.cfg.record_inputs:
            # сохраняются в float16 — чтобы перестроение по сохранённому шло ровно так же, считаем на тех же числах
            feats = np.asarray(feats, np.float16).astype(np.float32)
        colors = np.full((len(dets), 3), np.nan, np.float32)
        if self.teams is not None:
            for i, d in enumerate(dets):
                c = torso_color(frame, d.box)
                if c is not None:
                    colors[i] = c
        motion = None
        if self.camera is not None:
            motion = self._guard("camera", lambda: self.camera.estimate(frame, [d.box for d in raw]), None)
        balls = np.array([[*d.box, d.score] for d in raw if d.cls == BALL_CLASS], np.float32).reshape(-1, 5)
        return FrameInputs(dets, np.asarray(feats, np.float32), colors, motion, balls)

    def process(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_idx += 1
        return self._step(self.read_inputs(frame), frame, t0)

    def process_inputs(self, inputs: FrameInputs) -> FrameResult:
        """Шаг слежения по сохранённым данным кадра (без пикселей: номера не читаются)."""
        t0 = time.perf_counter()
        self.frame_idx += 1
        return self._step(inputs, None, t0)

    def _step(self, inputs: FrameInputs, frame: Optional[np.ndarray], t0: float) -> FrameResult:
        if self.cfg.record_inputs:
            self.inputs.append(inputs)
        dets, feats, motion = inputs.detections, inputs.features, inputs.camera
        for i, d in enumerate(dets):
            d.extra["index"] = i
        self.last_camera = motion
        if motion is not None:
            self.follower.apply_camera_motion(motion)
        self.mot.update(dets, feats, frame_idx=self.frame_idx, camera=motion)

        obs: Optional[TargetObservation] = None
        self._frame_dets = (dets, feats)
        corr = self._corrections.get(self.frame_idx)
        if corr is not None:
            obs = self._apply_correction(corr)
        if obs is None and self.follower.state == TargetState.IDLE:
            everyone = [t for t in self.mot.tracks if t.detected_now]
            obs = self._try_init(everyone, {t.track_id: t.last_feature for t in everyone if t.last_feature is not None})
        tracks = self._visible_tracks()
        features = {t.track_id: t.last_feature for t in tracks if t.detected_now and t.last_feature is not None}
        team_labels = self._team_labels(inputs.colors, tracks)
        numbers = self._read_numbers(frame, tracks) if frame is not None else {}
        if obs is None:
            obs = self.follower.step(self.frame_idx, tracks, features, numbers, team_labels)
        elif not (corr is not None and corr.absent):
            # в кадре захвата/поправки тоже учитываем номер/команду; «цели нет» — в этом кадре не ищем
            self.follower.step(self.frame_idx, tracks, features, numbers, team_labels)
        res = FrameResult(self.frame_idx, obs, tracks, dets, (time.perf_counter() - t0) * 1000)
        self.metrics.observe(self.frame_idx, dets, tracks, obs, res.elapsed_ms)
        self.online.append(obs)
        if self.log is not None:
            self.log.add(self.frame_idx, self.mot.tracks, motion)
            self._frames.append((self.frame_idx, dets, snapshot_tracks(tracks), res.elapsed_ms))
        return res

    def _visible_tracks(self) -> list[Track]:
        if self.cfg.pass_lost_tracks:
            return [t for t in self.mot.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        return [t for t in self.mot.tracks if t.state == TrackState.CONFIRMED]

    def run(self, frames: Iterable[np.ndarray], on_frame: Optional[Callable[[np.ndarray, FrameResult], None]] = None) -> list[FrameResult]:
        results = []
        for frame in frames:
            res = self.process(frame)
            results.append(res)
            if on_frame is not None:
                on_frame(frame, res)
        self.finalize()
        return results

    def final_observations(self) -> list[TargetObservation]:
        """Итог прогона: уточнённые по всему ролику решения (или онлайн, если уточнение выключено).

        Метрики и журнал ошибок, касающиеся цели, пересчитываются по итоговым решениям."""
        self.finalize()
        if self.log is None or not self.online:
            return list(self.online)
        refined = refine_online(self.online, self.log, self.cfg.refine_cfg,
                                protected={c.frame: c.absent for c in self.cfg.corrections})
        errors = ErrorLog(self.cfg.fps)
        errors.records = [r for r in self.errors.records if r.kind in _RUN_LEVEL_ERRORS]
        metrics = RunMetrics((self.width, self.height), self.cfg.fps, errors, det_high=self.cfg.tracker.det_high)
        metrics.ocr_attempts, metrics.ocr_reads = self.metrics.ocr_attempts, self.metrics.ocr_reads
        for (frame_idx, dets, tracks, elapsed), obs in zip(self._frames, refined):
            metrics.observe(frame_idx, dets, tracks, obs, elapsed)
        self.errors, self.metrics = errors, metrics
        return refined

    def tracks_at(self, i: int) -> list:
        """Лёгкие снимки треков i-го обработанного кадра (для отрисовки после прогона)."""
        return self._frames[i][2] if i < len(self._frames) else []

    def finalize(self) -> None:
        """Конец ролика: если цель так и не была выбрана, это ошибка прогона."""
        if self.follower.state == TargetState.IDLE and not self._init_failed:
            self._init_failed = True
            self.errors.add(self.frame_idx, "init_failed", "ролик закончился, а цель по стартовым координатам не найдена",
                            box=self.cfg.init.box, point=self.cfg.init.point)

    # --- ручное вмешательство ---------------------------------------------------
    def reselect(self, track_id: int) -> TargetObservation:
        """Оператор указал верный трек (например, после ambiguous)."""
        t = self.mot.get(track_id)
        if t is None:
            raise KeyError(f"Трек {track_id} не существует")
        feat = t.last_feature
        return self.follower.lock(t, feat, self.frame_idx)


# --- сохранённые данные кадров -----------------------------------------------------------------------

def save_inputs(path, inputs: list[FrameInputs], feature_dim: int, ball_method: str = "frame") -> None:
    """Данные кадров прогона в .npz (дескрипторы — float16: ~100 МБ на 80 с игры при 20 игроках в кадре).

    ball_method — как искался мяч: «frame» (только на кадре целиком) или «tiles» (ещё и на увеличенных плитках)."""
    counts = np.array([len(x.detections) for x in inputs], dtype=np.int32)
    boxes = [np.array([d.box for d in x.detections], dtype=np.float32).reshape(-1, 4) for x in inputs]
    feats = [np.asarray(x.features, dtype=np.float16).reshape(-1, feature_dim) for x in inputs]
    np.savez(
        path,
        counts=counts,
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
        scores=np.array([d.score for x in inputs for d in x.detections], dtype=np.float32),
        features=np.concatenate(feats) if feats else np.zeros((0, feature_dim), np.float16),
        colors=np.concatenate([x.colors.reshape(-1, 3) for x in inputs]) if inputs else np.zeros((0, 3), np.float32),
        camera=np.stack([x.camera if x.camera is not None else np.full((2, 3), np.nan) for x in inputs])
        if inputs else np.zeros((0, 2, 3)),
        ball_counts=np.array([len(x.balls) for x in inputs], dtype=np.int32),
        balls=np.concatenate([x.balls.reshape(-1, 5) for x in inputs]).astype(np.float32) if inputs else np.zeros((0, 5), np.float32),
        ball_method=np.array(ball_method),
    )


def inputs_ball_method(path) -> str:
    with np.load(path) as data:
        return str(data["ball_method"]) if "ball_method" in data else "frame"


def replace_balls(path, balls: list[np.ndarray], ball_method: str) -> None:
    """Перезаписать мяч в сохранённых данных кадров (после поиска мяча на плитках по копии кадров)."""
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["ball_counts"] = np.array([len(b) for b in balls], dtype=np.int32)
    arrays["balls"] = (np.concatenate([np.asarray(b, np.float32).reshape(-1, 5) for b in balls])
                       if balls else np.zeros((0, 5), np.float32))
    arrays["ball_method"] = np.array(ball_method)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_inputs(path) -> list[FrameInputs]:
    data = np.load(path)
    counts, boxes, scores = data["counts"], data["boxes"], data["scores"]
    feats, colors, camera = data["features"], data["colors"], data["camera"]
    ball_counts = data["ball_counts"] if "ball_counts" in data else np.zeros(len(counts), np.int32)
    balls = data["balls"] if "balls" in data else np.zeros((0, 5), np.float32)
    out, pos, bpos = [], 0, 0
    for i, n in enumerate(counts):
        sl = slice(pos, pos + int(n))
        dets = [Detection(boxes[j].astype(np.float64), float(scores[j])) for j in range(pos, pos + int(n))]
        cam = camera[i]
        nb = int(ball_counts[i])
        out.append(FrameInputs(dets, feats[sl].astype(np.float32), colors[sl].astype(np.float32),
                               None if np.isnan(cam).any() else cam.astype(np.float64), balls[bpos:bpos + nb].copy()))
        pos += int(n)
        bpos += nb
    return out
