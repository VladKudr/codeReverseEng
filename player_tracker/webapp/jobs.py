"""Задачи слежения: запуск в фоновом потоке, прогресс, отмена, результаты на диске.

Хранилище — каталог `data/`:
  data/videos/<video_id>/source.<ext> + info.json
  data/jobs/<job_id>/state.json + артефакты прогона (track.json, metrics.json,
  errors.md, annotated.mp4, evaluation.json, ...)

Состояние задачи живёт в state.json, поэтому список задач переживает
перезапуск сервера; незавершённые на момент перезапуска помечаются `failed`.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from tracker import export
from tracker.appearance import CompositeEncoder, PartColorEncoder
from tracker.metrics import evaluate, load_truth
from tracker.pipeline import InitSpec, Pipeline, PipelineConfig
from tracker.render import draw
from tracker.video import FrameSource, FrameWriter, VideoInfo, probe

ALLOWED_VIDEO_EXT = {".mov", ".mp4", ".m4v", ".hevc", ".avi", ".mkv", ".webm"}


class JobCancelled(Exception):
    pass


@dataclass
class JobOptions:
    width: int = 1280               # ширина кадра обработки (0 — исходная)
    weights: str = "yolov8n.pt"
    imgsz: int = 1280
    device: Optional[str] = None
    encoder: str = "color"          # color | osnet
    ocr: str = "none"               # none | easyocr
    team: bool = True
    tonemap: Optional[bool] = None  # None — авто
    start_sec: float = 0.0
    max_frames: Optional[int] = None
    render: bool = True


@dataclass
class JobInit:
    t_sec: float = 0.0                              # момент кадра выбора (от начала фрагмента)
    point: Optional[tuple[float, float]] = None     # в координатах исходного кадра
    box: Optional[tuple[float, float, float, float]] = None
    number: Optional[str] = None


@dataclass
class JobState:
    id: str
    video_id: str
    status: str = "queued"          # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    progress: int = 0               # обработано кадров
    total: int = 0                  # всего кадров (оценка)
    fps_proc: float = 0.0
    state_now: str = "idle"         # состояние цели на последнем кадре
    error: Optional[str] = None
    events: list[dict] = field(default_factory=list)
    summary: Optional[dict] = None
    metrics: Optional[dict] = None
    evaluation: Optional[dict] = None
    options: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)
    scale: float = 1.0
    process_size: list[int] = field(default_factory=list)


class JobManager:
    def __init__(self, data_dir: str | Path, detector_factory: Optional[Callable[[JobOptions], object]] = None,
                 encoder_factory: Optional[Callable[[JobOptions], object]] = None,
                 reader_factory: Optional[Callable[[JobOptions], object]] = None):
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.jobs_dir = self.data_dir / "jobs"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._detector_factory = detector_factory or _default_detector
        self._encoder_factory = encoder_factory or _default_encoder
        self._reader_factory = reader_factory or _default_reader
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._load()

    # --- видео ---------------------------------------------------------------------------
    def add_video(self, filename: str, stream) -> dict:
        ext = Path(filename).suffix.lower() or ".mp4"
        if ext not in ALLOWED_VIDEO_EXT:
            raise ValueError(f"Неподдерживаемое расширение {ext}; ожидается одно из {sorted(ALLOWED_VIDEO_EXT)}")
        video_id = uuid.uuid4().hex[:12]
        vdir = self.videos_dir / video_id
        vdir.mkdir(parents=True)
        dst = vdir / f"source{ext}"
        with open(dst, "wb") as fh:
            shutil.copyfileobj(stream, fh, length=1 << 20)
        try:
            info = probe(dst)
        except Exception as exc:
            shutil.rmtree(vdir, ignore_errors=True)
            raise ValueError(f"Файл не распознан как видео: {exc}") from exc
        rec = {"id": video_id, "name": filename, "path": str(dst), "uploaded": time.time(), "info": asdict(info),
               "is_hdr": info.is_hdr}
        (vdir / "info.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    def list_videos(self) -> list[dict]:
        out = []
        for p in sorted(self.videos_dir.glob("*/info.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(out, key=lambda r: r["uploaded"], reverse=True)

    def get_video(self, video_id: str) -> dict:
        p = self.videos_dir / video_id / "info.json"
        if not p.exists():
            raise KeyError(video_id)
        return json.loads(p.read_text(encoding="utf-8"))

    def delete_video(self, video_id: str) -> None:
        vdir = self.videos_dir / video_id
        if not vdir.exists():
            raise KeyError(video_id)
        shutil.rmtree(vdir)

    # --- задачи ---------------------------------------------------------------------------
    def _load(self) -> None:
        for p in self.jobs_dir.glob("*/state.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                st = JobState(**d)
            except (json.JSONDecodeError, TypeError):
                continue
            if st.status in ("queued", "running"):
                st.status, st.error = "failed", "сервер перезапущен во время выполнения"
                self._save(st)
            self._jobs[st.id] = st

    def _save(self, st: JobState) -> None:
        d = self.jobs_dir / st.id
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(asdict(st), ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / "state.json")

    def list_jobs(self) -> list[JobState]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def get_job(self, job_id: str) -> JobState:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def create_job(self, video_id: str, init: JobInit, options: JobOptions) -> JobState:
        video = self.get_video(video_id)
        if init.point is None and init.box is None:
            raise ValueError("Нужна точка или рамка цели")
        st = JobState(id=uuid.uuid4().hex[:12], video_id=video_id, options=asdict(options), init=asdict(init))
        with self._lock:
            self._jobs[st.id] = st
            self._cancel[st.id] = threading.Event()
        self._save(st)
        th = threading.Thread(target=self._run, args=(st, video, init, options), name=f"job-{st.id}", daemon=True)
        self._threads[st.id] = th
        th.start()
        return st

    def cancel_job(self, job_id: str) -> JobState:
        st = self.get_job(job_id)
        ev = self._cancel.get(job_id)
        if ev is not None:
            ev.set()
        return st

    def delete_job(self, job_id: str) -> None:
        st = self.get_job(job_id)
        if st.status in ("queued", "running"):
            raise ValueError("Сначала остановите задачу")
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> JobState:
        th = self._threads.get(job_id)
        if th is not None:
            th.join(timeout)
        return self.get_job(job_id)

    # --- выполнение -------------------------------------------------------------------------
    def _run(self, st: JobState, video: dict, init: JobInit, opt: JobOptions) -> None:
        out = self.job_dir(st.id)
        cancel = self._cancel[st.id]
        st.status, st.started = "running", time.time()
        self._save(st)
        writer = None
        try:
            source = FrameSource(video["path"], max_width=opt.width or None, tonemap=opt.tonemap,
                                 start_sec=opt.start_sec, max_frames=opt.max_frames)
            scale = source.scale
            st.scale, st.process_size = scale, [source.width, source.height]
            info: VideoInfo = source.info
            total = info.nb_frames
            if opt.start_sec and info.fps:
                total = max(total - int(opt.start_sec * info.fps), 0)
            if opt.max_frames:
                total = min(total, opt.max_frames) if total else opt.max_frames
            st.total = total
            spec = InitSpec(frame=int(round(init.t_sec * source.fps)), number=init.number)
            if init.box is not None:
                spec.box = tuple(v * scale for v in init.box)
            if init.point is not None:
                spec.point = (init.point[0] * scale, init.point[1] * scale)
            cfg = PipelineConfig(init=spec, use_team=opt.team, fps=source.fps)
            pipe = Pipeline((source.width, source.height), self._detector_factory(opt), self._encoder_factory(opt),
                            self._reader_factory(opt), cfg)
            self._pipelines[st.id] = pipe
            if opt.render:
                writer = FrameWriter(out / "annotated.mp4", source.width, source.height, source.fps)
            records: list[dict] = []
            t0 = time.perf_counter()
            last_save = t0

            for frame in source:
                if cancel.is_set():
                    raise JobCancelled()
                res = pipe.process(frame)
                obs = res.observation
                records.append(export.observation_to_dict(obs, scale, source.fps))
                if writer is not None:
                    writer.write(draw(frame, obs, res.tracks))
                st.progress = res.frame_idx + 1
                st.state_now = obs.state.value
                if obs.event:
                    st.events.append({"frame": res.frame_idx, "time": round(res.frame_idx / source.fps, 2),
                                      "event": obs.event, "track_id": obs.track_id})
                now = time.perf_counter()
                st.fps_proc = round(st.progress / max(now - t0, 1e-6), 1)
                if now - last_save > 1.0:
                    self._save(st)
                    last_save = now
            st.total = st.progress
            self._finish(st, out, records, source, video, init, opt)
            st.status = "done"
        except JobCancelled:
            st.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 — любая ошибка задачи уходит в state.json
            st.status, st.error = "failed", f"{exc!r}\n{traceback.format_exc(limit=8)}"
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    st.error = (st.error or "") + f"\nannotated.mp4: {exc}"
            st.finished = time.time()
            self._save(st)
            self._pipelines.pop(st.id, None)

    def _finish(self, st: JobState, out: Path, records: list[dict], source: FrameSource, video: dict,
                init: JobInit, opt: JobOptions) -> None:
        pipe = self._pipelines[st.id]
        pipe.finalize()
        meta = {"video": video["name"], "video_id": video["id"], "fps": source.fps,
                "frame_size": [source.info.width, source.info.height],
                "process_size": [source.width, source.height], "scale": source.scale,
                "init": asdict(init), "options": asdict(opt)}
        export.write_json(out / "track.json", records, meta)
        export.write_csv(out / "track.csv", records)
        export.write_events(out / "events.log", records, source.fps)
        st.summary = export.summarize(records)
        (out / "summary.json").write_text(json.dumps(st.summary, ensure_ascii=False, indent=1), encoding="utf-8")
        st.metrics = pipe.metrics.write(out)

    # --- оценка по разметке после завершения ---------------------------------------------------
    def evaluate_job(self, job_id: str, truth_path: Path) -> dict:
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Оценка возможна только для завершённой задачи")
        out = self.job_dir(job_id)
        data = json.loads((out / "track.json").read_text(encoding="utf-8"))
        truth = load_truth(truth_path)
        # журнал ошибок прогона дополняется ошибками по разметке и перезаписывается
        from tracker.metrics import ErrorLog, ErrorRecord

        log = ErrorLog(data["meta"]["fps"])
        jsonl = out / "errors.jsonl"
        if jsonl.exists():
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    log.records.append(ErrorRecord(**d))
        log.records = [r for r in log.records if r.kind not in ("wrong_target", "missed_target")]
        ev = evaluate(data["frames"], truth, log)
        log.write_jsonl(jsonl)
        log.write_markdown(out / "errors.md")
        (out / "evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
        st.evaluation = ev
        if st.metrics is not None:
            st.metrics["errors"] = {"by_kind": log.counts(), "by_severity": log.by_severity()}
        self._save(st)
        return ev


def _default_detector(opt: JobOptions):
    from tracker.detection import YoloDetector

    return YoloDetector(opt.weights, imgsz=opt.imgsz, device=opt.device)


def _default_encoder(opt: JobOptions):
    if opt.encoder == "osnet":
        from tracker.appearance import TorchReidEncoder

        return CompositeEncoder([(TorchReidEncoder(), 1.0), (PartColorEncoder(), 0.6)])
    return PartColorEncoder()


def _default_reader(opt: JobOptions):
    if opt.ocr == "easyocr":
        from tracker.jersey import EasyOcrReader

        return EasyOcrReader()
    return None
