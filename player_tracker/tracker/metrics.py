"""Метрики прогона и журнал ошибок.

Два потребителя:
  * без разметки — «здоровье» прогона: сколько людей находит детектор, как
    часто пропадают детекции цели, сколько длятся потери, сколько раз
    пришлось отпустить трек, сколько кадров неоднозначны, скорость;
  * с разметкой (`truth`: кадр -> рамка цели) — качество: IoU по кадрам,
    доля верно отслеженных кадров, ложные захваты (следили не за тем),
    пропуски (цель в кадре, а состояние LOST), переключения личности.

Каждая проблема записывается в `ErrorLog` отдельной строкой с кадром, видом,
серьёзностью и подробностями — это сырьё для разбора: почему потеряли, что
показывал детектор, какие были кандидаты.
"""
from __future__ import annotations

import json
import statistics
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .detection import Detection
from .geometry import iou_pair, touches_border
from .multitracker import Track
from .target import TargetObservation, TargetState

# Виды ошибок. Серьёзность: error — слежение сломано, warning — подозрительно, info — для статистики.
ERROR_KINDS = {
    "exception": "error",          # исключение в детекторе/кодировщике/OCR
    "init_failed": "error",        # цель не найдена по стартовой рамке/точке
    "wrong_target": "error",       # по разметке: следим не за тем игроком
    "missed_target": "error",      # по разметке: цель в кадре, а состояние LOST
    "released_swap": "warning",    # трек отпущен по несходству после перекрытия
    "released_number": "warning",
    "released_team": "warning",
    "target_lost": "warning",      # мультитрекер потерял цель
    "ambiguous": "warning",        # двойники: захват отложен
    "no_detections": "warning",    # детектор ничего не нашёл в кадре
    "detection_dropout": "info",   # цель на треке, но в этом кадре детекции не было
    "low_confidence": "info",      # уверенность в цели ниже порога
    "target_at_border": "info",    # цель частично за краем — дескриптор ненадёжен
    "ocr_miss": "info",            # OCR ничего не прочёл на вырезке
    "long_loss": "warning",        # потеря дольше порога
}


@dataclass
class ErrorRecord:
    frame: int
    kind: str
    severity: str
    message: str = ""
    time: Optional[float] = None
    details: dict = field(default_factory=dict)


class ErrorLog:
    """Журнал ошибок: копит записи, пишет JSONL и краткую сводку в Markdown."""

    def __init__(self, fps: float = 30.0, dedupe_consecutive: Iterable[str] = ("no_detections", "detection_dropout",
                                                                              "low_confidence", "target_at_border", "ambiguous",
                                                                              "missed_target", "wrong_target", "ocr_miss")):
        self.fps = fps
        self.records: list[ErrorRecord] = []
        self._dedupe = set(dedupe_consecutive)
        self._last_frame_by_kind: dict[str, int] = {}
        self._runs: dict[str, int] = defaultdict(int)   # длина серии подряд идущих одинаковых записей

    def add(self, frame: int, kind: str, message: str = "", **details) -> Optional[ErrorRecord]:
        severity = ERROR_KINDS.get(kind, "warning")
        # серии однотипных кадровых замечаний схлопываются: одна запись на серию + длина
        if kind in self._dedupe and self._last_frame_by_kind.get(kind) == frame - 1:
            self._last_frame_by_kind[kind] = frame
            self._runs[kind] += 1
            if self.records:
                for r in reversed(self.records):
                    if r.kind == kind:
                        r.details["run_length"] = self._runs[kind]
                        r.details["last_frame"] = frame
                        break
            return None
        self._last_frame_by_kind[kind] = frame
        self._runs[kind] = 1
        rec = ErrorRecord(frame, kind, severity, message, round(frame / self.fps, 3) if self.fps else None, dict(details))
        self.records.append(rec)
        return rec

    def exception(self, frame: int, where: str, exc: BaseException) -> ErrorRecord:
        return self.add(frame, "exception", f"{where}: {exc!r}", where=where,
                        traceback=traceback.format_exc(limit=5))

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.kind for r in self.records))

    def by_severity(self) -> dict[str, int]:
        return dict(Counter(r.severity for r in self.records))

    def sorted_records(self) -> list[ErrorRecord]:
        """Записи по кадрам: ошибки оценки по разметке добавляются после прогона, но читать удобно по времени."""
        return sorted(self.records, key=lambda r: (r.frame, r.kind))

    def write_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for r in self.sorted_records():
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def render_markdown(self, limit: int = 200) -> str:
        lines = ["# Ошибки прогона", "", "| Вид | Серьёзность | Сколько |", "|---|---|---|"]
        for kind, n in sorted(self.counts().items(), key=lambda kv: (ERROR_KINDS.get(kv[0], "z"), -kv[1])):
            lines.append(f"| {kind} | {ERROR_KINDS.get(kind, '?')} | {n} |")
        lines += ["", "## Журнал", ""]
        for r in self.sorted_records()[:limit]:
            extra = ""
            if "run_length" in r.details:
                extra = f" (серия {r.details['run_length']} кадров, до {r.details['last_frame']})"
            t = f"{r.time:8.2f}s" if r.time is not None else ""
            lines.append(f"- кадр {r.frame:>6} {t} **{r.kind}** {r.message}{extra}")
        if len(self.records) > limit:
            lines.append(f"- … ещё {len(self.records) - limit} записей в errors.jsonl")
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> None:
        Path(path).write_text(self.render_markdown(), encoding="utf-8")


@dataclass
class FrameStats:
    frame: int
    n_dets: int
    n_dets_high: int
    max_score: float
    mean_height: float
    n_tracks: int
    state: str
    confidence: float
    elapsed_ms: float
    target_detected: bool


class RunMetrics:
    """Сборщик метрик прогона. Вызывать `observe` на каждом кадре, `summary()` в конце."""

    def __init__(self, frame_size: tuple[int, int], fps: float = 30.0, errors: Optional[ErrorLog] = None,
                 det_high: float = 0.5, low_confidence: float = 0.6, long_loss_sec: float = 3.0):
        self.width, self.height = frame_size
        self.fps = fps
        self.errors = errors if errors is not None else ErrorLog(fps)
        self.det_high = det_high
        self.low_confidence = low_confidence
        self.long_loss_frames = int(long_loss_sec * fps) if fps else 90
        self.frames: list[FrameStats] = []
        self.det_scores: list[float] = []
        self.det_heights: list[float] = []
        self.track_ids_seen: set[int] = set()
        self.target_track_ids: list[int] = []      # последовательность id, на которых была цель
        self.loss_episodes: list[dict] = []        # {start, end, frames, reacquired}
        self._loss_start: Optional[int] = None
        self._long_loss_reported = False
        self.events: Counter = Counter()
        self.ocr_attempts = 0
        self.ocr_reads = 0

    # --- сбор ---------------------------------------------------------------------
    def observe(self, frame_idx: int, detections: list[Detection], tracks: list[Track],
                obs: TargetObservation, elapsed_ms: float) -> None:
        scores = [d.score for d in detections]
        heights = [d.height for d in detections]
        self.det_scores.extend(scores)
        self.det_heights.extend(heights)
        for t in tracks:
            self.track_ids_seen.add(t.track_id)
        target_detected = False
        if obs.track_id is not None:
            if not self.target_track_ids or self.target_track_ids[-1] != obs.track_id:
                self.target_track_ids.append(obs.track_id)
            t = next((t for t in tracks if t.track_id == obs.track_id), None)
            target_detected = bool(t and t.detected_now)
        self.frames.append(FrameStats(
            frame_idx, len(detections), sum(1 for s in scores if s >= self.det_high),
            max(scores) if scores else 0.0, float(np.mean(heights)) if heights else 0.0,
            sum(1 for t in tracks if t.detected_now), obs.state.value, float(obs.confidence), elapsed_ms, target_detected,
        ))
        self._observe_errors(frame_idx, detections, obs, target_detected)

    def _observe_errors(self, frame_idx: int, detections: list[Detection], obs: TargetObservation, target_detected: bool) -> None:
        e = self.errors
        if obs.event:
            self.events[obs.event] += 1
            if obs.event == "lost":
                e.add(frame_idx, "target_lost", "мультитрекер потерял цель", track_id=obs.track_id)
            elif obs.event.startswith("released_"):
                e.add(frame_idx, obs.event, "трек отпущен", track_id=obs.track_id)
            elif obs.event == "init_failed":
                e.add(frame_idx, "init_failed", "цель не найдена по стартовым координатам")
        if not detections and obs.state != TargetState.IDLE:
            e.add(frame_idx, "no_detections", "детектор не нашёл ни одного человека")
        if obs.state == TargetState.LOST:
            if self._loss_start is None:
                self._loss_start = frame_idx
                self._long_loss_reported = False
            if obs.ambiguous:
                e.add(frame_idx, "ambiguous", "двойники, захват отложен",
                      candidates=[(c.track_id, c.score) for c in obs.candidates if c.rejected is None][:4])
            if not self._long_loss_reported and frame_idx - self._loss_start >= self.long_loss_frames:
                self._long_loss_reported = True
                e.add(frame_idx, "long_loss", f"цель не найдена дольше {self.long_loss_frames} кадров", since=self._loss_start)
        else:
            if self._loss_start is not None:
                self.loss_episodes.append({"start": self._loss_start, "end": frame_idx, "frames": frame_idx - self._loss_start,
                                           "reacquired": obs.event == "reacquired"})
                self._loss_start = None
            if obs.state in (TargetState.ACTIVE, TargetState.CONTESTED):
                if not target_detected:
                    e.add(frame_idx, "detection_dropout", "цель на треке без детекции в кадре", track_id=obs.track_id)
                if obs.confidence < self.low_confidence:
                    e.add(frame_idx, "low_confidence", f"уверенность {obs.confidence:.2f}", track_id=obs.track_id)
                if obs.box is not None and touches_border(obs.box, self.width, self.height):
                    e.add(frame_idx, "target_at_border", "цель у края кадра", track_id=obs.track_id)

    def observe_ocr(self, attempted: int, read: int) -> None:
        self.ocr_attempts += attempted
        self.ocr_reads += read

    # --- итог -----------------------------------------------------------------------------
    def summary(self) -> dict:
        n = len(self.frames)
        if n == 0:
            return {"frames": 0}
        states = Counter(f.state for f in self.frames)
        tracked = states.get("active", 0) + states.get("contested", 0)
        if self._loss_start is not None:   # потеря не закончилась к концу ролика
            self.loss_episodes.append({"start": self._loss_start, "end": self.frames[-1].frame + 1,
                                       "frames": self.frames[-1].frame + 1 - self._loss_start, "reacquired": False})
            self._loss_start = None
        loss_lengths = [ep["frames"] for ep in self.loss_episodes]
        elapsed = [f.elapsed_ms for f in self.frames]
        tracked_frames = [f for f in self.frames if f.state in ("active", "contested")]
        return {
            "frames": n,
            "duration_sec": round(n / self.fps, 2) if self.fps else None,
            "detection": {
                "per_frame_mean": round(statistics.fmean(f.n_dets for f in self.frames), 2),
                "per_frame_high_mean": round(statistics.fmean(f.n_dets_high for f in self.frames), 2),
                "frames_without_detections": sum(1 for f in self.frames if f.n_dets == 0),
                "score_mean": round(statistics.fmean(self.det_scores), 3) if self.det_scores else None,
                "score_p10": round(float(np.percentile(self.det_scores, 10)), 3) if self.det_scores else None,
                "height_px_median": round(float(np.median(self.det_heights)), 1) if self.det_heights else None,
                "height_px_p10": round(float(np.percentile(self.det_heights, 10)), 1) if self.det_heights else None,
            },
            "tracks": {
                "unique_ids": len(self.track_ids_seen),
                "per_frame_mean": round(statistics.fmean(f.n_tracks for f in self.frames), 2),
            },
            "target": {
                "tracked_frames": tracked,
                "tracked_share": round(tracked / n, 3),
                "contested_frames": states.get("contested", 0),
                "lost_frames": states.get("lost", 0),
                "idle_frames": states.get("idle", 0),
                "detection_dropouts": sum(1 for f in tracked_frames if not f.target_detected),
                "confidence_mean": round(statistics.fmean(f.confidence for f in tracked_frames), 3) if tracked_frames else None,
                "track_ids_used": len(set(self.target_track_ids)),
                "id_changes": max(len(self.target_track_ids) - 1, 0),
                "loss_episodes": len(self.loss_episodes),
                "loss_frames_mean": round(statistics.fmean(loss_lengths), 1) if loss_lengths else 0,
                "loss_frames_max": max(loss_lengths) if loss_lengths else 0,
                "reacquired": sum(1 for ep in self.loss_episodes if ep["reacquired"]),
                "ambiguous_frames": sum(1 for f in self.frames if f.state == "lost") and self.errors.counts().get("ambiguous", 0),
                "events": dict(self.events),
            },
            "ocr": {"attempts": self.ocr_attempts, "reads": self.ocr_reads,
                    "read_rate": round(self.ocr_reads / self.ocr_attempts, 3) if self.ocr_attempts else None},
            "speed": {
                "ms_per_frame_mean": round(statistics.fmean(elapsed), 1),
                "ms_per_frame_p90": round(float(np.percentile(elapsed, 90)), 1),
                "fps": round(1000 / statistics.fmean(elapsed), 1) if statistics.fmean(elapsed) > 0 else None,
            },
            "errors": {"by_kind": self.errors.counts(), "by_severity": self.errors.by_severity()},
        }

    def write(self, out_dir: str | Path) -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(out / "frame_stats.csv", "w", encoding="utf-8") as fh:
            fh.write("frame,n_dets,n_dets_high,max_score,mean_height,n_tracks,state,confidence,elapsed_ms,target_detected\n")
            for f in self.frames:
                fh.write(f"{f.frame},{f.n_dets},{f.n_dets_high},{f.max_score:.3f},{f.mean_height:.1f},{f.n_tracks},"
                         f"{f.state},{f.confidence:.3f},{f.elapsed_ms:.1f},{int(f.target_detected)}\n")
        self.errors.write_jsonl(out / "errors.jsonl")
        self.errors.write_markdown(out / "errors.md")
        return summary


# --- оценка по разметке -------------------------------------------------------------------

def load_truth(path: str | Path) -> dict[int, Optional[list[float]]]:
    """Разметка цели: JSON {"frames": {"12": [x1,y1,x2,y2] | null, ...}} или {"12": [...]}
    либо CSV `frame,x1,y1,x2,y2` (пустые координаты — цели в кадре нет). Координаты исходного кадра."""
    p = Path(path)
    truth: dict[int, Optional[list[float]]] = {}
    if p.suffix.lower() == ".csv":
        import csv

        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vals = [row.get(k, "") for k in ("x1", "y1", "x2", "y2")]
                truth[int(row["frame"])] = None if any(v == "" for v in vals) else [float(v) for v in vals]
        return truth
    data = json.loads(p.read_text(encoding="utf-8"))
    frames = data.get("frames", data) if isinstance(data, dict) else {}
    for k, v in frames.items():
        truth[int(k)] = None if v is None else [float(x) for x in v]
    return truth


def evaluate(records: list[dict], truth: dict[int, Optional[list[float]]], errors: Optional[ErrorLog] = None,
             iou_thr: float = 0.5) -> dict:
    """Сравнение выхода (`export.observation_to_dict`, координаты исходного кадра) с разметкой.

    На каждом размеченном кадре:
      TP — цель отслежена и IoU >= порога; FP (wrong_target) — отслежена, но рамка не на цели;
      FN (missed_target) — цель в кадре, а слежение в LOST/IDLE; TN — цели нет и слежения нет;
      false_track — цели нет в кадре, а мы за кем-то следим.
    Переключение личности — смена track_id между двумя TP-кадрами с FP между ними или без.
    """
    tp = fp = fn = tn = false_track = 0
    ious: list[float] = []
    switches = 0
    last_ok_id: Optional[int] = None
    by_frame = {r["frame"]: r for r in records}
    for frame in sorted(truth):
        r = by_frame.get(frame)
        if r is None:
            continue
        gt = truth[frame]
        tracked = r["state"] in ("active", "contested") and r["box"] is not None
        if gt is None:
            if tracked:
                false_track += 1
                if errors:
                    errors.add(frame, "wrong_target", "цели нет в кадре, а слежение активно", track_id=r["track_id"])
            else:
                tn += 1
            continue
        if not tracked:
            fn += 1
            if errors:
                errors.add(frame, "missed_target", "цель в кадре, слежение в состоянии " + r["state"],
                           candidates=[(c["track_id"], c["score"]) for c in r.get("candidates", [])][:4])
            continue
        iou = iou_pair(r["box"], gt)
        ious.append(iou)
        if iou >= iou_thr:
            tp += 1
            if last_ok_id is not None and r["track_id"] != last_ok_id:
                switches += 1
            last_ok_id = r["track_id"]
        else:
            fp += 1
            if errors:
                errors.add(frame, "wrong_target", f"IoU {iou:.2f} с разметкой", track_id=r["track_id"],
                           box=r["box"], truth=gt)
    evaluated = tp + fp + fn + tn + false_track
    present = tp + fp + fn
    return {
        "frames_evaluated": evaluated,
        "frames_with_target": present,
        "tp": tp, "wrong_target": fp, "missed_target": fn, "true_absent": tn, "false_track": false_track,
        "recall": round(tp / present, 3) if present else None,
        "precision": round(tp / (tp + fp + false_track), 3) if (tp + fp + false_track) else None,
        "iou_mean": round(statistics.fmean(ious), 3) if ious else None,
        "id_switches": switches,
        "mota": round(1 - (fn + fp + false_track + switches) / present, 3) if present else None,
    }
