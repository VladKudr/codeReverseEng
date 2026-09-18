"""Задачи слежения: запуск в фоновом потоке, прогресс, отмена, результаты на диске.

Прогон идёт в два этапа:
  1. слежение без поиска мяча на плитках (~10 кадр/с): сразу готовы видео с номерами, доска и метрики
     игроков — задача получает статус `done`;
  2. фоном (`ball_stage`): мяч ищется на увеличенных плитках по сохранённой копии кадров (frames.mp4),
     кандидаты дописываются в inputs.npz, доска и метрики пересчитываются уже с мячом. Этапы 2 разных
     задач идут по одному (делят GPU); прерванный перезапуском сервера этап продолжается при старте.

Хранилище — каталог `data/`:
  data/videos/<video_id>/source.<ext> + info.json
  data/jobs/<job_id>/state.json + артефакты прогона (track.json, metrics.json,
  errors.md, annotated.mp4, evaluation.json, ...)

Состояние задачи живёт в state.json, поэтому список задач переживает
перезапуск сервера; незавершённые на момент перезапуска помечаются `failed`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tracker import board as board_mod
from tracker import export, player_metrics
from tracker.analyst import Analyst, DeepSeekClient, PlayerContext, load_api_key
from tracker.ball import BallConfig, accept_roi, predict_gaps, track_ball
from tracker.detection import BALL_CLASS, Detection, merge_points
from tracker.ground import GroundConfig, GroundModel
from tracker import roster as roster_mod
from tracker.camreg import register
from tracker.field import (SIDES, CameraPrior, FieldMarks, FieldProjector, MultiProjector, apply_h, convex,
                           corners_from_ground, expand_to_points, inside_share, parse_marks, scene_horizon,
                           visible_side_points)
from tracker.pitch import PitchSetup, default_field, fit_to_occupancy
from tracker.profile import PlayerProfile, speed_zones
from tracker.appearance import CompositeEncoder, PartColorEncoder, default_color_encoder
from tracker.metrics import evaluate, load_truth
from tracker.pipeline import (Correction, InitSpec, Pipeline, PipelineConfig, inputs_ball_method, load_inputs,
                              replace_balls, save_inputs)
from tracker.render import draw
from tracker.video import FrameSource, FrameWriter, VideoInfo, probe, read_frame_at

ALLOWED_VIDEO_EXT = {".mov", ".mp4", ".m4v", ".hevc", ".avi", ".mkv", ".webm"}


class JobCancelled(Exception):
    pass


@dataclass
class JobOptions:
    width: int = 1280               # ширина кадра обработки (0 — исходная)
    weights: str = "yolo11s.pt"
    imgsz: int = 1280
    device: Optional[str] = None
    encoder: str = "color"          # color | osnet
    ocr: str = "none"               # none | easyocr
    team: bool = True
    tonemap: Optional[bool] = None  # None — авто
    start_sec: float = 0.0
    max_frames: Optional[int] = None
    render: bool = True
    refine: bool = True             # уточнить результат по всему ролику после прогона
    ball_tiles: bool = True         # искать мяч ещё и на увеличенных плитках кадра (медленнее, но мяч виден вдвое чаще)
    ball_roi: bool = False          # повторная детекция в разрывах цепочки вокруг предсказания: на IMG_7463 дала +0.3 п.п.
                                    # (мяч в разрыве почти никогда не там, где предсказано) — выключено по умолчанию


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
    fps: float = 0.0
    stage: str = ""                 # ball | tracking | rendering — что делается сейчас
    corrections: list[dict] = field(default_factory=list)   # {frame, point|null, absent} в координатах исходного кадра
    # опрос перед слежением (profile.PlayerProfile: age, position, game_format, заметки, height_m) + угол обзора
    # камеры и поля ИИ-разбора (team_color, focus)
    player: dict = field(default_factory=lambda: {"hfov_deg": 70.0})
    revision: int = 0               # сколько раз результат перестраивался по поправкам
    # ручные пометки людей на доске: id трека -> {"role": "play" | "sideline", "f": первый кадр, "box": рамка на нём}
    # (по началу трека пометка находит того же человека после перенумерации треков, см. board.resolve_roles)
    track_roles: dict = field(default_factory=dict)
    # состав: "all" — в игре все, кроме выключенных; "marked" — в игре только отмеченные (и похожие на них)
    roster_mode: str = "all"
    ball_pass: dict = field(default_factory=dict)   # итог повторной детекции мяча в разрывах: {gap_frames, added}
    # второй этап — поиск мяча в фоне: "" (не нужен/не начат) | queued | running | done | failed | cancelled | off
    ball_stage: str = ""
    ball_progress: int = 0
    ball_total: int = 0
    ball_error: Optional[str] = None
    # фоновая привязка кадров к сцене гомографией (tracker.camreg) — для разметки поля на кадре:
    # "" | queued | running | done | failed | cancelled
    reg_stage: str = ""
    reg_progress: int = 0
    reg_error: Optional[str] = None


class JobManager:
    def __init__(self, data_dir: str | Path, detector_factory: Optional[Callable[[JobOptions], object]] = None,
                 encoder_factory: Optional[Callable[[JobOptions], object]] = None,
                 reader_factory: Optional[Callable[[JobOptions], object]] = None,
                 ai_client_factory: Optional[Callable[[str], object]] = None):
        self.data_dir = Path(data_dir)
        self._ai_client_factory = ai_client_factory or (lambda key: DeepSeekClient(key))
        self.videos_dir = self.data_dir / "videos"
        self.jobs_dir = self.data_dir / "jobs"
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._detector_factory = detector_factory or _default_detector
        self._encoder_factory = encoder_factory or _default_encoder
        self._reader_factory = reader_factory or _default_reader
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._deleted: set[str] = set()
        self._jobs: dict[str, JobState] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pipelines: dict[str, Pipeline] = {}
        self._ball_sem = threading.Semaphore(1)            # поиск мяча — по одной задаче: делят GPU
        self._ball_cancel: dict[str, threading.Event] = {}
        self._ball_threads: dict[str, threading.Thread] = {}
        self._reg_sem = threading.Semaphore(1)             # привязка кадров — по одной задаче: считается на CPU
        self._reg_cancel: dict[str, threading.Event] = {}
        self._reg_threads: dict[str, threading.Thread] = {}
        self._feet_cache: dict[str, tuple] = {}
        self._job_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._tracking_now = 0                              # сколько задач сейчас на этапе 1 (им — приоритет)
        self._tracking_cv = threading.Condition()
        self._load()
        self.merge_duplicate_videos()
        # поиск мяча, прерванный перезапуском сервера, продолжается
        for st in list(self._jobs.values()):
            if st.status == "done" and st.ball_stage in ("queued", "running", "waiting"):
                self._start_ball_stage(st)
            if st.status == "done" and st.reg_stage in ("queued", "running"):
                self._start_reg_stage(st)

    # --- видео ---------------------------------------------------------------------------
    def add_video(self, filename: str, stream) -> dict:
        ext = Path(filename).suffix.lower() or ".mp4"
        if ext not in ALLOWED_VIDEO_EXT:
            raise ValueError(f"Неподдерживаемое расширение {ext}; ожидается одно из {sorted(ALLOWED_VIDEO_EXT)}")
        video_id = uuid.uuid4().hex[:12]
        vdir = self.videos_dir / video_id
        vdir.mkdir(parents=True)
        dst = vdir / f"source{ext}"
        digest = hashlib.sha256()
        size = 0
        with open(dst, "wb") as fh:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
                fh.write(chunk)
        sha = digest.hexdigest()
        # тот же файл уже загружали — отдаём прежний ролик со всеми его задачами и разборами
        for old in self.list_videos():
            if old.get("size", size) == size and self._video_hash(old) == sha:
                shutil.rmtree(vdir, ignore_errors=True)
                return {**old, "duplicate": True}
        try:
            info = probe(dst)
        except Exception as exc:
            shutil.rmtree(vdir, ignore_errors=True)
            raise ValueError(f"Файл не распознан как видео: {exc}") from exc
        rec = {"id": video_id, "name": filename, "path": str(dst), "uploaded": time.time(), "info": asdict(info),
               "is_hdr": info.is_hdr, "sha256": sha, "size": size}
        (vdir / "info.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    def _video_hash(self, rec: dict) -> Optional[str]:
        """SHA-256 файла ролика; для роликов, загруженных до появления хеша, считается один раз и сохраняется."""
        if rec.get("sha256"):
            return rec["sha256"]
        path = Path(rec["path"])
        if not path.exists():
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                digest.update(chunk)
        rec["sha256"], rec["size"] = digest.hexdigest(), path.stat().st_size
        (self.videos_dir / rec["id"] / "info.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec["sha256"]

    def merge_duplicate_videos(self) -> int:
        """Один и тот же файл, загруженный несколько раз (до появления проверки по хешу): задачи переносятся
        на самую раннюю загрузку, лишние копии файла удаляются. Возвращает число слитых копий."""
        by_hash: dict[str, list[dict]] = {}
        for rec in self.list_videos():
            h = self._video_hash(rec)
            if h:
                by_hash.setdefault(h, []).append(rec)
        merged = 0
        for recs in by_hash.values():
            if len(recs) < 2:
                continue
            recs.sort(key=lambda r: r["uploaded"])
            keep, dups = recs[0], recs[1:]
            for dup in dups:
                with self._lock:
                    jobs = [j for j in self._jobs.values() if j.video_id == dup["id"]]
                if any(j.status in ("queued", "running") for j in jobs):
                    continue
                for j in jobs:
                    j.video_id = keep["id"]
                    self._save(j)
                shutil.rmtree(self.videos_dir / dup["id"], ignore_errors=True)
                merged += 1
        return merged

    def video_jobs(self, video_id: str) -> list[dict]:
        """Что уже сделано по ролику: задачи слежения (игрок, поправки, опрос) и сохранённые разборы."""
        self.get_video(video_id)
        out = []
        for st in self.list_jobs():
            if st.video_id != video_id:
                continue
            sessions = self._analysis_sessions(st.id)
            out.append({
                "id": st.id, "status": st.status, "created": st.created, "init": st.init, "revision": st.revision,
                "corrections": len(st.corrections), "tracked_share": (st.summary or {}).get("tracked_share"),
                "player": {k: st.player.get(k) for k in ("age", "position", "game_format", "height_m") if st.player.get(k)},
                "pitch": _analysis_pitch(st.player) is not None, "analyses": len(sessions),
                "analysis_current": any(x.get("revision") == st.revision for x in sessions),
            })
        return out

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
        # состояние пишут несколько потоков (слежение, мяч, привязка кадров): общий временный файл — под замком;
        # удалённая задача не воскресает от запоздавшего сохранения фонового потока
        with self._save_lock:
            if st.id in self._deleted:
                return
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

    def create_job(self, video_id: str, init: JobInit, options: JobOptions, player: Optional[dict] = None) -> JobState:
        video = self.get_video(video_id)
        if init.point is None and init.box is None:
            raise ValueError("Нужна точка или рамка цели")
        st = JobState(id=uuid.uuid4().hex[:12], video_id=video_id, options=asdict(options), init=asdict(init))
        st.player.update({k: v for k, v in (player or {}).items() if v not in (None, "")})
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
        self._stop_ball_stage(job_id)
        self._stop_reg_stage(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
        with self._save_lock:
            self._deleted.add(job_id)
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def wait(self, job_id: str, timeout: Optional[float] = None) -> JobState:
        th = self._threads.get(job_id)
        if th is not None:
            th.join(timeout)
        return self.get_job(job_id)

    # --- выполнение -------------------------------------------------------------------------
    def _run(self, st: JobState, video: dict, init: JobInit, opt: JobOptions) -> None:
        with self._tracking_cv:
            self._tracking_now += 1
        try:
            self._run_tracking(st, video, init, opt)
        finally:
            with self._tracking_cv:
                self._tracking_now -= 1
                self._tracking_cv.notify_all()
        if st.status == "done":
            self._start_ball_stage(st)
            self._start_reg_stage(st)

    def _run_tracking(self, st: JobState, video: dict, init: JobInit, opt: JobOptions) -> None:
        out = self.job_dir(st.id)
        cancel = self._cancel[st.id]
        st.status, st.started, st.error, st.stage = "running", time.time(), None, "tracking"
        st.progress, st.evaluation = 0, None
        self._save(st)
        try:
            source = FrameSource(video["path"], max_width=opt.width or None, tonemap=opt.tonemap,
                                 start_sec=opt.start_sec, max_frames=opt.max_frames)
            scale = source.scale
            st.scale, st.process_size, st.fps = scale, [source.width, source.height], source.fps
            cache = out / "inputs.npz"
            # перестроение по поправкам: детекции и дескрипторы уже посчитаны — без детектора и декодирования
            # (номера на футболках читаются по пикселям, поэтому с OCR ролик прогоняется целиком)
            replay = cache.exists() and opt.ocr == "none"
            inputs = load_inputs(cache) if replay else None
            info: VideoInfo = source.info
            total = info.nb_frames
            if opt.start_sec and info.fps:
                total = max(total - int(opt.start_sec * info.fps), 0)
            if opt.max_frames:
                total = min(total, opt.max_frames) if total else opt.max_frames
            st.total = len(inputs) if replay else total
            spec = InitSpec(frame=int(round(init.t_sec * source.fps)), number=init.number)
            if init.box is not None:
                spec.box = tuple(v * scale for v in init.box)
            if init.point is not None:
                spec.point = (init.point[0] * scale, init.point[1] * scale)
            corrections = [Correction(int(c["frame"]), None if c.get("point") is None else
                                      (c["point"][0] * scale, c["point"][1] * scale), None, bool(c.get("absent")))
                           for c in st.corrections]
            cfg = PipelineConfig(init=spec, use_team=opt.team, fps=source.fps, refine=opt.refine,
                                 corrections=corrections, record_inputs=not replay)
            encoder = self._encoder_factory(opt)
            # этап 1: детекция людей (и мяча только на кадре целиком); плитки мяча — во втором этапе, в фоне
            detector = None if replay else self._detector_factory(replace(opt, ball_tiles=False))
            pipe = Pipeline((source.width, source.height), detector, encoder,
                            None if replay else self._reader_factory(opt), cfg)
            self._pipelines[st.id] = pipe
            t0 = time.perf_counter()
            last_save = t0
            stream = inputs if replay else source
            # копия обработанных кадров: точный кадр по номеру для поправок и быстрый второй проход отрисовки
            frames_copy = None if replay else FrameWriter(out / "frames.part.mp4", source.width, source.height,
                                                          source.fps, crf=20, gop=15)
            try:
                for item in stream:
                    if cancel.is_set():
                        raise JobCancelled()
                    if frames_copy is not None:
                        frames_copy.write(item)
                    res = pipe.process_inputs(item) if replay else pipe.process(item)
                    st.progress = res.frame_idx + 1
                    st.state_now = res.observation.state.value
                    now = time.perf_counter()
                    st.fps_proc = round(st.progress / max(now - t0, 1e-6), 1)
                    if now - last_save > 1.0:
                        self._save(st)
                        last_save = now
            finally:
                if frames_copy is not None:
                    frames_copy.close()
            if frames_copy is not None:
                (out / "frames.part.mp4").replace(out / "frames.mp4")
            st.total = st.progress
            if not replay:
                save_inputs(cache, pipe.inputs, encoder.dim, "tiles" if getattr(detector, "ball_tiles", False) else "frame")
                st.ball_stage, st.ball_progress, st.ball_error = "", 0, None
            final = pipe.final_observations()
            if pipe.log is not None:
                board_mod.save_tracks(out / "tracks.npz", pipe.log)
            records = [export.observation_to_dict(o, scale, source.fps) for o in final]
            st.events = [{"frame": o.frame_idx, "time": round(o.frame_idx / source.fps, 2), "event": o.event,
                          "track_id": o.track_id} for o in final if o.event]
            if final:
                st.state_now = final[-1].state.value
            if opt.render:
                st.stage = "rendering"
                self._save(st)
                self._render(out, video, opt, source, pipe, final, cancel)
            self._finish(st, out, records, source, video, init, opt)
            st.status = "done"
        except JobCancelled:
            st.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 — любая ошибка задачи уходит в state.json
            st.status, st.error = "failed", f"{exc!r}\n{traceback.format_exc(limit=8)}"
        finally:
            st.stage = ""
            st.finished = time.time()
            self._save(st)
            self._pipelines.pop(st.id, None)

    # --- привязка кадров к сцене (для разметки поля) — в фоне, на CPU ------------------------------------------
    def _reg_stamp(self, st: JobState) -> Optional[int]:
        copy = self.job_dir(st.id) / "frames.mp4"
        return copy.stat().st_mtime_ns if copy.exists() else None

    def _load_reg(self, st: JobState) -> Optional[np.ndarray]:
        """G[i] «кадр i -> сцена» (пиксели кадра обработки) или None, если привязка не посчитана / устарела."""
        path = self.job_dir(st.id) / "camreg.npz"
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                if int(z["stamp"]) != self._reg_stamp(st):
                    return None
                return z["G"]
        except Exception:  # noqa: BLE001 — битый файл: считаем, что привязки нет
            return None

    def _start_reg_stage(self, st: JobState) -> None:
        th = self._reg_threads.get(st.id)
        if th is not None and th.is_alive():
            return
        out = self.job_dir(st.id)
        if not ((out / "frames.mp4").exists() and (out / "inputs.npz").exists() and (out / "tracks.npz").exists()):
            return
        if self._load_reg(st) is not None:
            if st.reg_stage != "done":
                st.reg_stage = "done"
                self._save(st)
            return
        self._reg_cancel[st.id] = threading.Event()
        st.reg_stage, st.reg_progress, st.reg_error = "queued", 0, None
        self._save(st)
        th = threading.Thread(target=self._reg_stage, args=(st.id,), name=f"reg-{st.id}", daemon=True)
        self._reg_threads[st.id] = th
        th.start()

    def _stop_reg_stage(self, job_id: str, timeout: float = 30.0) -> None:
        ev = self._reg_cancel.get(job_id)
        if ev is not None:
            ev.set()
        th = self._reg_threads.get(job_id)
        if th is not None and th.is_alive():
            th.join(timeout)

    def _reg_stage(self, job_id: str) -> None:
        cancel = self._reg_cancel[job_id]
        with self._reg_sem:
            try:
                st = self.get_job(job_id)
            except KeyError:
                return
            try:
                if cancel.is_set():
                    raise JobCancelled()
                out = self.job_dir(job_id)
                st.reg_stage = "running"
                self._save(st)
                stamp = self._reg_stamp(st)
                with np.load(out / "inputs.npz") as z:
                    cameras = list(z["camera"])
                tracks = board_mod.load_tracks(out / "tracks.npz")
                boxes = [np.asarray(b, float).reshape(-1, 4) for b in tracks.boxes]
                boxes += [np.zeros((0, 4))] * (len(cameras) - len(boxes))
                last = [time.perf_counter()]

                def progress(i: int) -> None:
                    st.reg_progress = i + 1
                    if time.perf_counter() - last[0] > 1.0:
                        self._save(st)
                        last[0] = time.perf_counter()

                G, stats = register(FrameSource(out / "frames.mp4", cfr=False), cameras, boxes,
                                    progress=progress, stop=cancel.is_set)
                if cancel.is_set() or stamp != self._reg_stamp(st):
                    raise JobCancelled()
                quality = stats.pop("quality")
                np.savez_compressed(out / "camreg.npz", G=G, quality=quality, stamp=np.int64(stamp),
                                    stats=json.dumps(stats))
                st.reg_stage = "done"
                self._save(st)
                st = self.get_job(job_id)
                if st.status == "done" and parse_marks(st.player.get("field")):
                    self._compute_player_metrics(st)      # разметка поля есть — положения пересчитываются точнее
            except JobCancelled:
                st.reg_stage = "cancelled"
            except Exception as exc:  # noqa: BLE001 — без привязки остаётся цепочка подобий
                st.reg_stage, st.reg_error = "failed", f"{exc!r}\n{traceback.format_exc(limit=6)}"
            finally:
                self._save(st)

    def _scene_inv(self, st: JobState, ground: GroundModel) -> tuple[list, bool]:
        """«Кадр -> сцена» для разметки поля: точная привязка, если посчитана, иначе цепочка подобий."""
        G = self._load_reg(st)
        if G is not None and len(G) >= len(ground.inv):
            return list(G), True
        return ground.inv, False

    # --- этап 2: мяч в фоне -------------------------------------------------------------------------------
    def _ball_needed(self, st: JobState) -> bool:
        opt = JobOptions(**st.options)
        cache = self.job_dir(st.id) / "inputs.npz"
        return (opt.ball_tiles and cache.exists() and (self.job_dir(st.id) / "frames.mp4").exists()
                and "tiles" not in inputs_ball_method(cache))

    def _start_ball_stage(self, st: JobState) -> None:
        """Поставить поиск мяча в очередь, если он нужен и ещё не идёт."""
        th = self._ball_threads.get(st.id)
        if th is not None and th.is_alive():
            return
        if not self._ball_needed(st):
            if st.ball_stage == "waiting":
                st.ball_stage = "running"
            # статус явный: мяч уже найден — done, поиск на плитках выключен в параметрах — off
            final = "off" if not JobOptions(**st.options).ball_tiles else "done"
            if st.ball_stage != final and st.ball_stage not in ("failed",):
                st.ball_stage = final
                self._save(st)
            return
        self._ball_cancel[st.id] = threading.Event()
        st.ball_stage, st.ball_progress, st.ball_error = "queued", 0, None
        self._save(st)
        th = threading.Thread(target=self._ball_stage, args=(st.id,), name=f"ball-{st.id}", daemon=True)
        self._ball_threads[st.id] = th
        th.start()

    def _yield_to_tracking(self, st: JobState, cancel: threading.Event) -> None:
        """Пока какая-то задача на этапе 1, поиск мяча ждёт: человек ждёт результат по игрокам, а мяч — фоновый."""
        if not self._tracking_now:
            return
        st.ball_stage = "waiting"
        self._save(st)
        with self._tracking_cv:
            while self._tracking_now and not cancel.is_set():
                self._tracking_cv.wait(timeout=1.0)
        if cancel.is_set():
            raise JobCancelled()
        st.ball_stage = "running"
        self._save(st)

    def _stop_ball_stage(self, job_id: str, timeout: float = 30.0) -> None:
        ev = self._ball_cancel.get(job_id)
        if ev is not None:
            ev.set()
        th = self._ball_threads.get(job_id)
        if th is not None and th.is_alive():
            th.join(timeout)

    def _ball_stage(self, job_id: str) -> None:
        cancel = self._ball_cancel[job_id]
        with self._ball_sem:
            try:
                st = self.get_job(job_id)
            except KeyError:
                return
            if cancel.is_set():
                st.ball_stage = "cancelled"
                self._save(st)
                return
            out = self.job_dir(job_id)
            cache = out / "inputs.npz"
            try:
                opt = JobOptions(**st.options)
                detector = self._detector_factory(opt)
                if not hasattr(detector, "detect_balls"):
                    st.ball_stage = "off"             # детектор не умеет искать мяч на плитках (тесты, чужой детектор)
                    self._save(st)
                    return
                st.ball_stage = "running"
                self._save(st)
                stamp = cache.stat().st_mtime_ns
                inputs = load_inputs(cache)
                st.ball_total = len(inputs)
                # продолжение после перезапуска сервера: найденное раньше лежит в balls.part.npz
                part = out / "balls.part.npz"
                balls = _load_ball_part(part, stamp)
                start = len(balls)
                last = time.perf_counter()
                for i, frame in enumerate(FrameSource(out / "frames.mp4", cfr=False)):
                    if cancel.is_set():
                        raise JobCancelled()
                    if i >= len(inputs):
                        break
                    if i < start:                     # уже найдено до перезапуска
                        continue
                    self._yield_to_tracking(st, cancel)
                    prev = [Detection(np.asarray(b[:4], float), float(b[4]), BALL_CLASS) for b in inputs[i].balls]
                    found = merge_points(detector.detect_balls(frame) + prev)
                    balls.append(np.array([[*d.box, d.score] for d in found], np.float32).reshape(-1, 5))
                    st.ball_progress = i + 1
                    if (i + 1) % BALL_CHECKPOINT_FRAMES == 0:
                        _save_ball_part(part, balls, stamp)
                    if time.perf_counter() - last > 1.0:
                        self._save(st)
                        last = time.perf_counter()
                balls += [np.zeros((0, 5), np.float32)] * (len(inputs) - len(balls))
                # данные кадров могли смениться (полный перепрогон) — тогда найденное не к ним
                if not cache.exists() or cache.stat().st_mtime_ns != stamp:
                    raise JobCancelled()
                with self._job_locks[job_id]:
                    replace_balls(cache, balls, "tiles")
                part.unlink(missing_ok=True)
                if opt.ball_roi:
                    for x, b in zip(inputs, balls):
                        x.balls = b
                    self._roi_pass(st, out, opt, inputs, cancel, detector)
                st.ball_stage = "done"
                self._save(st)
                st = self.get_job(job_id)
                if st.status == "done":               # идёт перестроение — оно само возьмёт новый мяч
                    self._compute_player_metrics(st)
            except JobCancelled:
                st.ball_stage = "cancelled"
            except Exception as exc:  # noqa: BLE001 — ошибка этапа 2 не портит готовый результат этапа 1
                st.ball_stage, st.ball_error = "failed", f"{exc!r}\n{traceback.format_exc(limit=6)}"
            finally:
                self._save(st)

    def _roi_pass(self, st: JobState, out: Path, opt: JobOptions, inputs: list, cancel: threading.Event,
                  detector=None) -> None:
        """Повторная детекция мяча в разрывах цепочки: вырезка вокруг предсказанного положения, увеличенная в 4 раза.
        Один раз на задачу; результат — в inputs.npz (метод «tiles+roi»)."""
        detector = detector or self._detector_factory(opt)
        if not hasattr(detector, "detect_balls_roi"):
            return
        profile = PlayerProfile.from_dict(st.player)
        height, _ = profile.effective_height()
        pitch = PitchSetup.from_dict(st.player.get("pitch"), profile.game_format)
        width = st.process_size[0] if st.process_size else 1280
        fps = st.fps or 30.0
        people = [np.array([[*d.box, d.score] for d in x.detections], np.float32).reshape(-1, 5) for x in inputs]
        ground = GroundModel.fit(people, [x.camera for x in inputs], width,
                                 GroundConfig(hfov_deg=float(st.player.get("hfov_deg", 70.0)), player_height_m=height))
        cfg = BallConfig()
        track = track_ball([x.balls for x in inputs], people, ground, fps, cfg, pitch)
        todo = predict_gaps(track, ground, fps, cfg)
        st.ball_progress, st.ball_total = 0, len(inputs)
        self._save(st)
        added, last = 0, time.perf_counter()
        for i, frame in enumerate(FrameSource(out / "frames.mp4", cfr=False)):
            if cancel.is_set():
                raise JobCancelled()
            if i >= len(inputs):
                break
            if i in todo:
                found = detector.detect_balls_roi(frame, todo[i], cfg.roi_size, cfg.roi_imgsz, cfg.roi_conf)
                cands = accept_roi(np.array([[*d.box, d.score] for d in found], np.float32), todo[i], people[i], ground, i,
                                   cfg, pitch)
                if len(cands):
                    merged = merge_points([Detection(np.asarray(b[:4], float), float(b[4]), BALL_CLASS)
                                           for b in np.vstack([inputs[i].balls.reshape(-1, 5), cands])])
                    inputs[i].balls = np.array([[*d.box, d.score] for d in merged], np.float32).reshape(-1, 5)
                    added += 1
            st.ball_progress = i + 1
            if time.perf_counter() - last > 1.0:
                self._save(st)
                last = time.perf_counter()
        with self._job_locks[st.id]:
            replace_balls(out / "inputs.npz", [x.balls for x in inputs], "tiles+roi")
        st.ball_pass = {"gap_frames": len(todo), "added": added}

    # --- поправки оператора ----------------------------------------------------------------------------
    def add_correction(self, job_id: str, frame: int, point: Optional[tuple[float, float]] = None,
                       absent: bool = False) -> JobState:
        """Поправка на кадре `frame` (от начала обработанного фрагмента): цель — игрок в точке (координаты
        исходного кадра) или цели нет. Результат перестраивается с учётом всех поправок."""
        st = self.get_job(job_id)
        if st.status in ("queued", "running"):
            raise ValueError("Задача ещё выполняется — поправку можно внести после завершения")
        if frame < 0 or (st.total and frame >= st.total):
            raise ValueError(f"Кадр {frame} вне ролика (0..{max(st.total - 1, 0)})")
        if not absent and point is None:
            raise ValueError("Нужна точка на игроке или отметка «цели нет»")
        corr = {"frame": int(frame), "point": None if absent else [float(point[0]), float(point[1])], "absent": bool(absent)}
        st.corrections = sorted([c for c in st.corrections if c["frame"] != corr["frame"]] + [corr],
                                key=lambda c: c["frame"])
        return self._rerun(st)

    def remove_correction(self, job_id: str, frame: int) -> JobState:
        st = self.get_job(job_id)
        if st.status in ("queued", "running"):
            raise ValueError("Задача ещё выполняется")
        before = len(st.corrections)
        st.corrections = [c for c in st.corrections if c["frame"] != frame]
        if len(st.corrections) == before:
            raise KeyError(frame)
        return self._rerun(st)

    def _rerun(self, st: JobState) -> JobState:
        video = self.get_video(st.video_id)
        st.revision += 1
        st.status = "queued"
        self._cancel[st.id] = threading.Event()
        self._save(st)
        init, opt = JobInit(**st.init), JobOptions(**st.options)
        th = threading.Thread(target=self._run, args=(st, video, init, opt), name=f"job-{st.id}-r{st.revision}", daemon=True)
        self._threads[st.id] = th
        th.start()
        return st

    def frame_detections(self, job_id: str, frame: int) -> dict:
        """Детекции кадра в координатах исходного кадра и рамка цели — для выбора игрока при поправке."""
        st = self.get_job(job_id)
        out = self.job_dir(job_id)
        cache = out / "inputs.npz"
        boxes: list[list[float]] = []
        if cache.exists():
            data = np.load(cache)
            counts = data["counts"]
            if 0 <= frame < len(counts):
                pos = int(counts[:frame].sum())
                k = 1.0 / st.scale if st.scale else 1.0
                boxes = [[round(float(v) * k, 1) for v in b] for b in data["boxes"][pos:pos + int(counts[frame])]]
        target = None
        track = out / "track.json"
        if track.exists():
            frames = json.loads(track.read_text(encoding="utf-8"))["frames"]
            if 0 <= frame < len(frames):
                target = {k: frames[frame][k] for k in ("state", "box", "track_id", "event")}
        return {"frame": frame, "boxes": boxes, "target": target,
                "correction": next((c for c in st.corrections if c["frame"] == frame), None)}

    def frame_image(self, job_id: str, frame: int) -> tuple[np.ndarray, float]:
        """Кадр `frame` обработанного фрагмента (для выбора игрока при поправке)."""
        st = self.get_job(job_id)
        if st.total and not 0 <= frame < st.total:
            raise ValueError(f"Кадр {frame} вне ролика (0..{st.total - 1})")
        copy = self.job_dir(job_id) / "frames.mp4"
        if copy.exists() and st.process_size:
            w, h = st.process_size
            return read_frame_at(copy, frame, st.fps or 30.0, w, h), st.scale
        # задача до появления копии кадров: номер кадра по времени у роликов iPhone не находится — читаем подряд
        opt = JobOptions(**st.options)
        video = self.get_video(st.video_id)
        src = FrameSource(video["path"], max_width=opt.width or None, tonemap=opt.tonemap,
                          start_sec=opt.start_sec, max_frames=frame + 1)
        img = None
        for img in src:
            pass
        if img is None or src.frames_read <= frame:
            raise ValueError(f"Кадр {frame} не прочитан")
        return img, src.scale

    def _render(self, out: Path, video: dict, opt: JobOptions, source: FrameSource, pipe: Pipeline,
                final: list, cancel: threading.Event) -> None:
        copy = out / "frames.mp4"
        again = FrameSource(copy, cfr=False) if copy.exists() else FrameSource(
            video["path"], max_width=opt.width or None, tonemap=opt.tonemap, start_sec=opt.start_sec, max_frames=opt.max_frames)
        writer = FrameWriter(out / "annotated.mp4", source.width, source.height, source.fps)
        # номера людей — те же, что на тактической доске
        tracks_path = out / "tracks.npz"
        numbers = board_mod.display_numbers(board_mod.load_tracks(tracks_path)) if tracks_path.exists() else None
        try:
            for i, frame in enumerate(again):
                if cancel.is_set():
                    raise JobCancelled()
                if i >= len(final):
                    break
                writer.write(draw(frame, final[i], pipe.tracks_at(i), numbers=numbers))
        finally:
            writer.close()

    def _finish(self, st: JobState, out: Path, records: list[dict], source: FrameSource, video: dict,
                init: JobInit, opt: JobOptions) -> None:
        pipe = self._pipelines[st.id]
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
        if (out / "inputs.npz").exists():
            self._compute_player_metrics(st, records)

    # --- метрики игрока ------------------------------------------------------------------------------
    def _compute_player_metrics(self, st: JobState, records: Optional[list[dict]] = None) -> dict:
        with self._job_locks[st.id]:
            return self._compute_player_metrics_locked(st, records)

    def _compute_player_metrics_locked(self, st: JobState, records: Optional[list[dict]] = None) -> dict:
        out = self.job_dir(st.id)
        cache = out / "inputs.npz"
        if not cache.exists():
            raise ValueError("Для метрик нужны сохранённые данные кадров — нажмите «Пересчитать задачу» (ролик прогонится заново один раз)")
        if records is None:
            records = json.loads((out / "track.json").read_text(encoding="utf-8"))["frames"]
        profile = PlayerProfile.from_dict(st.player)
        height, height_src = profile.effective_height()
        cfg = player_metrics.MetricsConfig(player_height_m=height, hfov_deg=float(st.player.get("hfov_deg", 70.0)),
                                           zones=speed_zones(profile.age))
        width = st.process_size[0] if st.process_size else 1280
        inputs = load_inputs(cache)
        pitch = PitchSetup.from_dict(st.player.get("pitch"), profile.game_format)
        ball, board, game, ignore, ground, pitch = self._game_data(st, inputs, records, height, pitch)
        m = player_metrics.compute(records, inputs, st.scale or 1.0, st.fps or 30.0, width, cfg,
                                   corrections=len(st.corrections), pitch=pitch, ball=ball, game=game,
                                   ignore_boxes=ignore, ground=ground)
        m["revision"] = st.revision
        m["config"].update(height_source=height_src, age=profile.age, game_format=profile.format_label(),
                           position=profile.position_label())
        if height_src != "указан":
            m["quality"]["notes"].insert(1, f"Рост игрока не указан — взят {height:.2f} м ({height_src}); укажите рост, "
                                            "метры станут точнее.")
        if m.get("pitch") and board is not None and (board.get("calibration") or {}).get("method") == "field":
            m["pitch"]["description"] = ("Поле отмечено оператором на кадре (углы площадки): положения на поле — по "
                                         "этой разметке, точнее оценки по росту. " + m["pitch"]["description"])
        if board is not None and board.get("ball_pending"):
            m["ball"] = {"pending": True}
            m["quality"]["notes"] = [n for n in m["quality"]["notes"] if "мяч" not in n.lower()]
            m["quality"]["notes"].insert(1, "Мяч ещё ищется в фоне — «мяч рядом», владение и дистанция до мяча "
                                            "появятся, когда поиск закончится (страница обновится сама).")
        (out / "player_metrics.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        return m

    def _game_data(self, st: JobState, inputs: list, records: list[dict], height: float, pitch: Optional[PitchSetup]):
        """Мяч по всему ролику, тактическая доска и игровой контекст (board.json). Без журнала треков — только мяч."""
        out = self.job_dir(st.id)
        width = st.process_size[0] if st.process_size else 1280
        fps = st.fps or 30.0
        people = [np.array([[*d.box, d.score] for d in x.detections], np.float32).reshape(-1, 5) for x in inputs]
        ground = GroundModel.fit(people, [x.camera for x in inputs], width,
                                 GroundConfig(hfov_deg=float(st.player.get("hfov_deg", 70.0)), player_height_m=height))
        # мяч ищется в фоне — до конца поиска результат «только игроки», без мяча с кадра целиком
        pending = JobOptions(**st.options).ball_tiles and "tiles" not in inputs_ball_method(out / "inputs.npz")
        balls = [np.zeros((0, 5), np.float32) for _ in inputs] if pending else [x.balls for x in inputs]
        board = game = None
        ignore: list = []
        tracks_path = out / "tracks.npz"
        tracks = board_mod.load_tracks(tracks_path) if tracks_path.exists() else None
        # схема поля, подогнанная по самим игрокам: глубина по росту в кадре растянута или сжата неизвестным
        # зумом телефона, без подгонки большая часть игроков оказывалась за полем
        marked = pitch is not None
        calibration = None
        roster = self._roster(st, tracks, inputs, ground, fps) if tracks is not None else None
        field_pitch = self._field_pitch(st, ground)
        if field_pitch is not None:
            # поле отмечено на кадре: метры — по разметке, подгонка по игрокам не нужна
            pitch, calibration = field_pitch
            marked = bool(pitch.own_goal_known)
        elif tracks is not None:
            base = pitch or PitchSetup.default(PlayerProfile.from_dict(st.player).game_format)
            # подгонка — только по тем, кто в игре (состав) и двигается: зрители у камеры и люди на дальних полях
            # растягивают облако, и чтобы «вместить» их, подгонка сжимала схему — игроки слипались в кучу (17.09.2026)
            feet = board_mod.play_feet(tracks, ground, st.track_roles)
            mask = board_mod.active_feet(feet, roster["off"], MOBILE_SPREAD_M)
            pitch, calibration = fit_to_occupancy(base, feet["X"][mask], feet["Z"][mask])
            calibration["fit_points"] = int(mask.sum())
            if not calibration.get("fitted"):
                pitch = pitch if marked else None
        ball = track_ball(balls, people, ground, fps, BallConfig(), pitch)
        if tracks is not None:
            if any(isinstance(v, str) for v in st.track_roles.values()):
                # пометки старого вида {id: роль}: id ещё соответствуют журналу — дописываем начало трека,
                # чтобы пометка пережила перенумерацию после следующей поправки
                sigs = board_mod.track_signatures(tracks)
                st.track_roles = {k: ({"role": v, "f": sigs[int(k)][0], "box": sigs[int(k)][1]} if isinstance(v, str) else v)
                                  for k, v in st.track_roles.items() if not isinstance(v, str) or int(k) in sigs}
                self._save(st)
            board = board_mod.build_board(tracks, [x.colors for x in inputs], ground, ball, records, st.scale or 1.0, fps,
                                          pitch, manual_roles=st.track_roles, pitch_marked=marked,
                                          roster=roster)
            board["roster_mode"] = st.roster_mode
            if calibration is not None and pitch is not None:
                pts = [(x, y) for fr in board["frames"] for t, x, y in fr["p"]
                       if board["tracks"].get(str(t), {}).get("role") != "sideline"]
                if pts:
                    P = np.array(pts, float)
                    calibration["outside"] = round(1.0 - inside_share(P[:, 0], P[:, 1], pitch.length_m, pitch.width_m), 3)
            board["calibration"] = calibration
            # люди вне игры (запасные, тренеры, зрители у бровки) не считаются соседями в метриках
            meta = board["tracks"]
            ignore = [np.array([b for t, b in zip(tracks.ids[i], np.asarray(tracks.boxes[i], float).reshape(-1, 4))
                                if board_mod.track_off(meta.get(str(int(t)), {}), i)], float).reshape(-1, 4)
                      for i in range(len(tracks))]
            board["ball_method"] = inputs_ball_method(out / "inputs.npz")
            board["ball_pending"] = pending
            board["revision"] = st.revision
            if marked:                                # владение и «своя/чужая» — только при отмеченных воротах
                game = board_mod.game_context(board, pitch)
            (out / "board.json").write_text(json.dumps(board, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return ball, board, game, ignore, ground, (pitch if marked else None)

    def _roster(self, st: JobState, tracks, inputs: list, ground: GroundModel, fps: float) -> dict:
        """Кто вне игры по кадрам: пометки оператора -> люди (склейка фрагментов) -> похожие по внешности."""
        inv, registered = self._scene_inv(st, ground)
        info = roster_mod.track_tracks(tracks, [x.colors for x in inputs], inv)
        feats = roster_mod.track_features_frames(tracks, [x.features for x in inputs])
        stored = board_mod.resolve_marks(st.track_roles, board_mod.track_signatures(tracks))
        marks = [roster_mod.Mark(t, v["role"], v.get("at")) for t, v in stored.items() if v.get("role") in ("play", "sideline")]
        frames = sorted({int(v["frame"]) for v in stored.values() if v.get("frame") is not None})
        default = roster_mod.SIDELINE if st.roster_mode == "marked" else roster_mod.PLAY
        # без точной привязки кадров место в сцене «уплывает» — стоящих склеиваем только через короткий разрыв
        link_cfg = roster_mod.LinkConfig(static_max_gap_sec=None if registered else 10.0)
        return roster_mod.resolve(info, marks, fps, default, feats, frames, link_cfg)

    def _field_dims(self, st: JobState) -> tuple[float, float, Optional[str]]:
        fd = st.player.get("field") or {}
        pd = st.player.get("pitch") or {}
        L, W = default_field(PlayerProfile.from_dict(st.player).game_format)
        return (float(fd.get("length_m") or pd.get("length_m") or L), float(fd.get("width_m") or pd.get("width_m") or W),
                fd.get("own_goal") or pd.get("own_goal"))

    def _camera_prior(self, st: JobState, ref: int, inv: list, ground: Optional[GroundModel] = None) -> CameraPrior:
        """Горизонт и высота камеры на кадре `ref` — по росту людей всего ролика; мало людей — по самому кадру."""
        feet, g = self._feet(st)
        ground = ground or g
        ref = int(min(max(ref, 0), len(ground) - 1))
        got = scene_horizon(feet["f"], feet["u"], feet["v"], feet["h"], inv, ref, ground.H)
        v0, hc = got if got is not None else (ground.horizon(ref), ground.H / max(float(ground.a[ref]), 1e-6))
        return CameraPrior(v0=float(v0), hc=float(hc), cx=float(ground.cx), f0=float(ground.f))

    def _projector(self, st: JobState, m: FieldMarks, inv: list, L: float, W: float,
                   ground: Optional[GroundModel] = None) -> FieldProjector:
        cam = self._camera_prior(st, m.frame, inv, ground) if m.mode == SIDES else None
        return FieldProjector(m, st.scale or 1.0, inv, L, W, camera=cam)

    def _field_pitch(self, st: JobState, ground: GroundModel):
        """Схема поля из разметки на кадре (st.player["field"]) или None."""
        marks = parse_marks(st.player.get("field"))
        if not marks or not st.scale:
            return None
        L, W, own = self._field_dims(st)
        inv, registered = self._scene_inv(st, ground)
        proj = MultiProjector([self._projector(st, m, inv, L, W, ground) for m in marks])
        size = st.process_size or [1280, 720]
        pitch = proj.pitch_setup(own or "left", (size[0], size[1]))
        pitch.project = proj
        pitch.own_goal_known = own in ("left", "right")
        return pitch, {"fitted": False, "method": "field", "frame": marks[0].frame, "frames": [m.frame for m in marks],
                       "registered": registered, "length_m": L, "width_m": W}

    def board(self, job_id: str) -> dict:
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Доска строится по завершённой задаче")
        if st.ball_stage in ("", "cancelled"):
            # задача старше двухэтапного прогона: мяч догоняется фоном (или статус становится явным done/off)
            self._start_ball_stage(st)
        if st.reg_stage in ("", "cancelled"):
            self._start_reg_stage(st)
        path = self.job_dir(job_id) / "board.json"
        if not path.exists():
            if not (self.job_dir(job_id) / "tracks.npz").exists():
                raise FileNotFoundError("Задача прогнана до появления доски — нажмите «Построить доску» (слежение "
                                        "перестроится по сохранённым данным, мяч будет найден заново)")
            self._compute_player_metrics(st)
        return json.loads(path.read_text(encoding="utf-8"))

    def track_boxes(self, job_id: str) -> dict:
        """Рамки треков по кадрам для разметки поверх чистой копии кадров: frames[i] — плоский список
        [id, x1, y1, x2, y2, id, …] в координатах исходного кадра (как рамки цели в track.json)."""
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Разметка доступна по завершённой задаче")
        out = self.job_dir(job_id)
        if not (out / "tracks.npz").exists():
            raise FileNotFoundError("Журнал треков не сохранён — нажмите «Построить доску»")
        tracks = board_mod.load_tracks(out / "tracks.npz")
        k = 1.0 / st.scale if st.scale else 1.0
        frames = []
        for ids, boxes in zip(tracks.ids, tracks.boxes):
            row: list[int] = []
            for t, b in zip(ids, np.asarray(boxes, float).reshape(-1, 4)):
                row += [int(t), *[int(round(v * k)) for v in b]]
            frames.append(row)
        # накопленное движение камеры «кадр -> сцена (первый кадр)» в координатах исходного кадра —
        # для линий разметки поля поверх видео (как в field.FieldProjector)
        # с точной привязкой (camreg) — гомография, 8 чисел (h22 = 1); без неё — подобие, 6 чисел
        cams = []
        S, Si = np.diag([st.scale or 1.0, st.scale or 1.0, 1.0]), np.diag([k, k, 1.0])
        G = self._load_reg(st)
        if G is not None and len(G) >= len(frames):
            for Gi in G:
                H = Si @ Gi @ S
                cams.append([float(f"{v:.6g}") for v in (H / H[2, 2]).ravel()[:8]])
        else:
            with np.load(out / "inputs.npz") as data:
                M = np.eye(3)
                for A in data["camera"]:
                    if not np.isnan(A).any():
                        M = np.vstack([A, [0.0, 0.0, 1.0]]) @ M
                    inv = Si @ np.linalg.inv(M) @ S
                    cams.append([round(float(v), 5) for v in inv[:2].ravel()])
        return {"fps": st.fps or 30.0, "total": len(frames), "frames": frames, "cams": cams,
                "registered": G is not None and len(G) >= len(frames), "field": self._field_with_h(st),
                "clean_video": (out / "frames.mp4").exists(), "revision": st.revision}

    def _field_with_h(self, st: JobState) -> Optional[dict]:
        """Разметка для линий поверх видео: у каждой разметки H — «исходный кадр -> метры поля»."""
        fd = _field_norm(st.player.get("field"))
        if fd is None:
            return None
        try:
            _, ground = self._feet(st)
            inv, _ = self._scene_inv(st, ground)
            L, W, _ = self._field_dims(st)
            S = np.diag([st.scale or 1.0, st.scale or 1.0, 1.0])
            for m, d in zip(parse_marks(fd), fd["marks"]):
                H = self._projector(st, m, inv, L, W, ground).H @ S
                d["H"] = [[float(f"{v:.8g}") for v in row] for row in H / np.abs(H).max()]
            fd.setdefault("length_m", L)
            fd.setdefault("width_m", W)
        except Exception:  # noqa: BLE001 — без H линии в режиме corners фронтенд построит сам
            pass
        return fd

    # --- разметка поля: подсказки редактору --------------------------------------------------------------
    def _feet(self, st: JobState):
        """Точки ног участников (board.play_feet) и плоскость поля; кэш — до смены журнала треков или пометок."""
        out = self.job_dir(st.id)
        tp = out / "tracks.npz"
        if not tp.exists() or not (out / "inputs.npz").exists():
            raise FileNotFoundError("Журнал треков не сохранён — нажмите «Построить доску»")
        key = (tp.stat().st_mtime_ns, json.dumps(st.track_roles, sort_keys=True), st.player.get("height_m"))
        hit = self._feet_cache.get(st.id)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        inputs = load_inputs(out / "inputs.npz")
        people = [np.array([[*d.box, d.score] for d in x.detections], np.float32).reshape(-1, 5) for x in inputs]
        width = st.process_size[0] if st.process_size else 1280
        height = PlayerProfile.from_dict(st.player).effective_height()[0]
        ground = GroundModel.fit(people, [x.camera for x in inputs], width,
                                 GroundConfig(hfov_deg=float(st.player.get("hfov_deg", 70.0)), player_height_m=height))
        feet = board_mod.play_feet(board_mod.load_tracks(tp), ground, st.track_roles)
        self._feet_cache[st.id] = (key, feet, ground)
        return feet, ground

    def _feet_on_frame(self, st: JobState, frame: int):
        """Точки ног всего ролика, перенесённые на кадр `frame` (пиксели кадра обработки)."""
        feet, ground = self._feet(st)
        inv, registered = self._scene_inv(st, ground)
        frame = int(min(max(frame, 0), len(inv) - 1))
        back = np.linalg.inv(inv[frame])
        u, v = np.full(len(feet["f"]), np.nan), np.full(len(feet["f"]), np.nan)
        for i in np.unique(feet["f"]):
            if i >= len(inv):
                continue
            m = feet["f"] == i
            T = back @ inv[i]
            P = T @ np.stack([feet["u"][m], feet["v"][m], np.ones(m.sum())])
            ok = P[2] > 1e-9
            u[m], v[m] = np.where(ok, P[0] / P[2], np.nan), np.where(ok, P[1] / P[2], np.nan)
        return feet, ground, frame, u, v, registered

    def field_occupancy(self, job_id: str, frame: int, limit: int = 3000) -> dict:
        """«Где играли»: точки ног участников за весь ролик на кадре `frame`, координаты исходного кадра."""
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Подсказка доступна по завершённой задаче")
        _, _, frame, u, v, registered = self._feet_on_frame(st, frame)
        k = 1.0 / (st.scale or 1.0)
        size = st.process_size or [1280, 720]
        ok = np.isfinite(u) & np.isfinite(v) & (u > -size[0]) & (u < 2 * size[0]) & (v > -size[1]) & (v < 2 * size[1])
        idx = np.flatnonzero(ok)
        if len(idx) > limit:
            idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
        return {"frame": frame, "registered": registered, "reg_stage": st.reg_stage,
                "points": [[int(round(u[i] * k)), int(round(v[i] * k))] for i in idx]}

    def _suggest_points(self, st: JobState, frame: int):
        feet, ground, frame, u, v, _ = self._feet_on_frame(st, frame)
        # углы ставятся по тем, кто двигался: стоящие у камеры зрители и запасные поле не растягивают
        ok = np.isfinite(u) & np.isfinite(v) & (feet["spread"] >= MOBILE_SPREAD_M)
        if ok.sum() < 200:
            ok = np.isfinite(u) & np.isfinite(v)
        return feet, ground, frame, u, v, ok

    def field_preview(self, job_id: str, frame: int, corners: list, rotation: int = 0, mode: str = "corners",
                      length_m: Optional[float] = None, width_m: Optional[float] = None) -> dict:
        """Как ляжет поле при такой разметке: H «исходный кадр -> метры поля», согласие точек с моделью камеры
        (режим `sides`), доля положений игроков за полем."""
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Разметка доступна по завершённой задаче")
        feet, ground, frame, u, v, ok = self._suggest_points(st, frame)
        scale = st.scale or 1.0
        L0, W0, _ = self._field_dims(st)
        L, W = float(length_m or L0), float(width_m or W0)
        if not convex(corners):
            raise ValueError("Точки 1–4 должны идти по кругу и образовывать выпуклый четырёхугольник")
        inv, _ = self._scene_inv(st, ground)
        pr = self._projector(st, FieldMarks(frame, np.asarray(corners, float), rotation, mode), inv, L, W, ground)
        px, py, _ = apply_h(pr.H, u[ok], v[ok])
        H = pr.H @ np.diag([scale, scale, 1.0])
        return {"frame": frame, "H": [[float(f"{x:.8g}") for x in row] for row in H / np.abs(H).max()],
                "rms_m": pr.fit.get("rms_m"), "outside": round(1.0 - inside_share(px, py, L, W), 3)}

    def field_suggest(self, job_id: str, frame: int, corners: Optional[list] = None, rotation: int = 0,
                      length_m: Optional[float] = None, width_m: Optional[float] = None, mode: str = "corners") -> dict:
        """Углы поля по игрокам. Без `corners` — заготовка с нуля (прямоугольник вокруг точек ног на плоскости
        поля); с `corners` — разметка оператора раздвигается туда, где игроки выходят за неё."""
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Подсказка доступна по завершённой задаче")
        feet, ground, frame, u, v, ok = self._suggest_points(st, frame)
        scale = st.scale or 1.0
        L0, W0, _ = self._field_dims(st)
        L, W = float(length_m or L0), float(width_m or W0)
        inv, _ = self._scene_inv(st, ground)

        def outside(c, rot, md) -> float:
            pr = self._projector(st, FieldMarks(frame, np.asarray(c, float) / scale, rot, md), inv, L, W, ground)
            px, py, _ = apply_h(pr.H, u[ok], v[ok])
            return round(1.0 - inside_share(px, py, L, W), 3)

        if corners is None:
            res = corners_from_ground(feet["X"][ok], feet["Z"][ok], lambda X, Z: ground.unground(frame, X, Z))
            if res is None:
                raise ValueError("Мало данных об игроках — отметьте углы вручную")
            new, guess, clipped = res
            size = st.process_size or [1280, 720]
            new, mode = visible_side_points(new, (size[0], size[1]))     # ближние углы за кадром — режим «боковые линии»
            mode = SIDES if clipped else mode
            # какая сторона обращена к камере — длинная (0) или короткая (1): по тому, при каком соответствии
            # игроки лучше помещаются в поле L×W; при равенстве — по форме облака
            share = {r: outside(new, r, mode) for r in (0, 1)}
            rotation = guess if abs(share[0] - share[1]) < 0.03 else min(share, key=share.get)
            kind, before = "scratch", None
        else:
            cur = np.asarray(corners, float) * scale
            pr = self._projector(st, FieldMarks(frame, np.asarray(corners, float), rotation, mode), inv, L, W, ground)
            new = expand_to_points(cur, rotation, L, W, u[ok], v[ok], H=pr.H, mode=mode)
            kind, before = "expand", outside(cur, rotation, mode)
        return {"frame": frame, "mode": kind, "marks_mode": mode, "rotation": int(rotation),
                "corners": [[round(float(x) / scale, 1), round(float(y) / scale, 1)] for x, y in new],
                "outside_before": before, "outside_after": outside(new, rotation, mode)}

    def set_track_role(self, job_id: str, track_id: int, role: str, frame: Optional[int] = None,
                       scope: str = "all", defer: bool = False) -> dict:
        """Пометка человека «в игре / вне игры» (auto — снять пометку с этого трека). Пометка относится к человеку:
        переносится на все фрагменты его трека и на похожих по внешности (`tracker.roster`).

        frame — кадр видео, на котором щёлкнули: остальные люди этого кадра становятся примерами обратного;
        scope="from" — пометка действует с кадра `frame` (замена: ушёл с поля / вышел на поле).
        defer — только сохранить (оператор щёлкает по нескольким людям подряд); пересчёт — `apply_roster`."""
        if role not in ("play", "sideline", "auto"):
            raise ValueError("role: play | sideline | auto")
        if scope not in ("all", "from") or (scope == "from" and frame is None):
            raise ValueError("scope: all | from (для from нужен frame)")
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Пометки ставятся по завершённой задаче")
        if not (self.job_dir(job_id) / "tracks.npz").exists():
            raise FileNotFoundError("Доска ещё не построена")
        roles = dict(st.track_roles)
        roles.pop(str(track_id), None)
        if role != "auto":
            sig = board_mod.track_signatures(board_mod.load_tracks(self.job_dir(job_id) / "tracks.npz")).get(int(track_id))
            if sig is None:
                raise KeyError(track_id)
            mark = {"role": role, "f": sig[0], "box": sig[1]}
            if frame is not None:
                mark["frame"] = int(frame)
            if scope == "from":
                mark["at"] = int(frame)
            roles[str(track_id)] = mark
        st.track_roles = roles
        self._save(st)
        if defer:
            return {"deferred": True, "marks": len(roles)}
        return self.apply_roster(job_id)

    def apply_roster(self, job_id: str) -> dict:
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Пометки ставятся по завершённой задаче")
        self._compute_player_metrics(st)
        self._save(st)
        return self.board(job_id)

    def set_roster_mode(self, job_id: str, mode: str) -> dict:
        if mode not in ("all", "marked"):
            raise ValueError("mode: all | marked")
        st = self.get_job(job_id)
        st.roster_mode = mode
        self._save(st)
        return self.apply_roster(job_id)

    def clear_roster(self, job_id: str) -> dict:
        st = self.get_job(job_id)
        st.track_roles = {}
        self._save(st)
        return self.apply_roster(job_id)

    def build_board(self, job_id: str) -> JobState:
        """Перестроение по сохранённым данным кадров: журнал треков для доски и мяч на плитках (если его ещё не было)."""
        st = self.get_job(job_id)
        if st.status in ("queued", "running"):
            raise ValueError("Задача ещё выполняется")
        if not (self.job_dir(job_id) / "inputs.npz").exists():
            return self.rerun(job_id)
        (self.job_dir(job_id) / "board.json").unlink(missing_ok=True)
        return self._rerun(st)

    def player_metrics(self, job_id: str, height_m: Optional[float] = None, hfov_deg: Optional[float] = None) -> dict:
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Метрики считаются по завершённой задаче")
        path = self.job_dir(job_id) / "player_metrics.json"
        if height_m is None and hfov_deg is None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if height_m is not None:
            st.player["height_m"] = float(height_m)
        if hfov_deg is not None:
            st.player["hfov_deg"] = float(hfov_deg)
        m = self._compute_player_metrics(st)
        self._save(st)
        return m

    def update_player(self, job_id: str, player: dict) -> JobState:
        """Правка опроса после прогона: сохраняется в задаче, метрики пересчитываются (слежение не трогается)."""
        st = self.get_job(job_id)
        if player.get("field"):
            player = {**player, "field": _field_norm(player["field"])}
            for m in parse_marks(player["field"]):
                if not convex(m.corners):
                    raise ValueError(f"Разметка на кадре {m.frame}: углы 1–4 должны идти по кругу и образовывать "
                                     "выпуклый четырёхугольник")
        for k, v in player.items():
            if v in (None, ""):
                st.player.pop(k, None)
            else:
                st.player[k] = v
        if st.status == "done" and (self.job_dir(job_id) / "inputs.npz").exists():
            self._compute_player_metrics(st)
        self._save(st)
        return st

    def rerun(self, job_id: str) -> JobState:
        """Полный повторный прогон (детектор заново): для задач без сохранённых данных кадров."""
        st = self.get_job(job_id)
        if st.status in ("queued", "running"):
            raise ValueError("Задача ещё выполняется")
        self._stop_ball_stage(job_id)
        self._stop_reg_stage(job_id)
        for name in ("inputs.npz", "frames.mp4", "camreg.npz"):
            (self.job_dir(job_id) / name).unlink(missing_ok=True)
        return self._rerun(st)

    # --- ИИ-аналитик -------------------------------------------------------------------------------------------
    def ai_configured(self) -> bool:
        return load_api_key(self.data_dir) is not None

    def _analysis_sessions(self, job_id: str) -> list[dict]:
        path = self.job_dir(job_id) / "analysis.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if "sessions" not in data:          # файл до появления версий разбора: один разговор
            data = {"sessions": [{"id": 1, "created": path.stat().st_mtime, "revision": data.get("revision"),
                                  "player": data.get("player", {}), "messages": data.get("messages", [])}]}
        return data["sessions"]

    def _save_sessions(self, job_id: str, sessions: list[dict]) -> None:
        path = self.job_dir(job_id) / "analysis.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def analysis(self, job_id: str) -> dict:
        """Сохранённые разборы задачи (все версии) — показываются без повторного запроса к модели."""
        st = self.get_job(job_id)
        sessions = [{"id": x["id"], "created": x["created"], "revision": x.get("revision"),
                     "stale": x.get("revision") != st.revision, "player": x.get("player", {}),
                     "messages": _visible(x.get("messages", []))} for x in self._analysis_sessions(job_id)]
        return {"sessions": sessions, "revision": st.revision}

    def reset_analysis(self, job_id: str, session_id: Optional[int] = None) -> None:
        self.get_job(job_id)
        if session_id is None:
            (self.job_dir(job_id) / "analysis.json").unlink(missing_ok=True)
            return
        sessions = [x for x in self._analysis_sessions(job_id) if x["id"] != session_id]
        self._save_sessions(job_id, sessions)

    def analyze(self, job_id: str, question: Optional[str], player: dict, new: bool = False,
                session_id: Optional[int] = None) -> dict:
        """Вопрос модели. Без `new` — продолжение последнего (или указанного) разбора по текущему слежению;
        `new` — новая версия разбора. Первый разбор по неизменившимся данным при уже сохранённом не
        запрашивается повторно — возвращается сохранённый (`cached`)."""
        st = self.get_job(job_id)
        if st.status != "done":
            raise ValueError("Разбор возможен по завершённой задаче")
        if player:
            self.update_player(job_id, player)
        if _analysis_pitch(st.player) is None:
            raise ValueError("Сначала отметьте поле: на схеме (камера, взгляд, свои ворота) или углы поля на кадре "
                             "вместе со своими воротами")
        sessions = self._analysis_sessions(job_id)
        profile_now = _profile_key(st.player)
        same_data = [x for x in sessions if x.get("revision") == st.revision and _profile_key(x.get("player", {})) == profile_now]
        if not question and same_data and not new:
            last = same_data[-1]
            return {"session_id": last["id"], "cached": True, "answer": _visible(last["messages"])[-1]["content"]
                    if last["messages"] else "", **self.analysis(job_id)}
        key = load_api_key(self.data_dir)
        if not key:
            raise ValueError("Не задан ключ DeepSeek: переменная DEEPSEEK_API_KEY или data/secrets.json")
        metrics = self.player_metrics(job_id)
        target = None
        if not new:
            cands = [x for x in sessions if x.get("revision") == st.revision]
            target = next((x for x in cands if x["id"] == session_id), None) if session_id else (cands[-1] if cands else None)
        if target is None:
            target = {"id": max((x["id"] for x in sessions), default=0) + 1, "created": time.time(),
                      "revision": st.revision, "player": dict(st.player), "messages": []}
            sessions.append(target)
        ctx = PlayerContext.from_dict(st.player)
        question = (question or "").strip() or "Сделай разбор действий игрока по этим данным."
        board_path = self.job_dir(job_id) / "board.json"
        board = json.loads(board_path.read_text(encoding="utf-8")) if board_path.exists() else None
        analyst = Analyst(self._ai_client_factory(key), metrics, asdict(st), board=board,
                          pitch=_analysis_pitch(st.player))
        answer, target["messages"] = analyst.ask(target["messages"], question, ctx)
        target["player"] = dict(st.player)
        self._save_sessions(job_id, sessions)
        return {"session_id": target["id"], "cached": False, "answer": answer, **self.analysis(job_id)}

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


BALL_CHECKPOINT_FRAMES = 150
MOBILE_SPREAD_M = 6.0            # размах положений трека (м), с которого человек считается участником игры (подсказки разметки)
# BALL_CHECKPOINT_FRAMES — как часто сохранять найденный мяч (≈50 с работы): перезапуск сервера не теряет больше


def _save_ball_part(path: Path, balls: list, stamp: int) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, stamp=np.array(stamp, dtype=np.int64), counts=np.array([len(b) for b in balls], np.int32),
             balls=np.concatenate(balls).astype(np.float32) if balls else np.zeros((0, 5), np.float32))
    tmp.replace(path)


def _load_ball_part(path: Path, stamp: int) -> list:
    """Промежуточный результат поиска мяча — только если он для этих же данных кадров."""
    if not path.exists():
        return []
    try:
        with np.load(path) as d:
            if int(d["stamp"]) != stamp:
                return []
            counts, flat = d["counts"], d["balls"]
    except (OSError, ValueError, KeyError):
        return []
    pos = np.concatenate([[0], np.cumsum(counts)])
    return [flat[a:b].reshape(-1, 5) for a, b in zip(pos[:-1], pos[1:])]


def _analysis_pitch(player: dict) -> Optional[PitchSetup]:
    """Схема для разбора: отмеченная на схеме, либо по разметке поля на кадре со своими воротами
    (разбору нужны размеры поля и сторона своих ворот — трети, фланги, «к чужим воротам»)."""
    pitch = PitchSetup.from_dict(player.get("pitch"), player.get("game_format"))
    if pitch is not None:
        return pitch
    fd = player.get("field") or {}
    own = fd.get("own_goal") or (player.get("pitch") or {}).get("own_goal")
    if not parse_marks(fd) or own not in ("left", "right"):
        return None
    L, W = default_field(player.get("game_format"))
    return PitchSetup((0.5, 1.1), (0.5, 0.5), own, float(fd.get("length_m") or L), float(fd.get("width_m") or W))


def _field_norm(fd: Optional[dict]) -> Optional[dict]:
    """Разметка поля в едином виде: {"marks": [{frame, corners, rotation}], own_goal, length_m, width_m}."""
    marks = parse_marks(fd)
    if not marks:
        return None
    out = {"marks": [{"frame": m.frame, "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in m.corners],
                      "rotation": m.rotation, "mode": m.mode} for m in marks]}
    for k in ("own_goal", "length_m", "width_m"):
        if fd.get(k) is not None:
            out[k] = fd[k]
    return out


def _profile_key(player: dict) -> str:
    """Данные, от которых зависит разбор (опрос, схема поля, рост) — для решения «разбор по этим данным уже есть»."""
    keys = ("age", "position", "position_note", "game_format", "format_note", "height_m", "hfov_deg", "pitch", "field",
            "team_color", "focus")
    return json.dumps({k: player.get(k) for k in keys}, sort_keys=True, ensure_ascii=False)


def _visible(messages: list[dict]) -> list[dict]:
    """Сообщения для показа: вопросы человека и ответы модели (без выжимки данных и вызовов инструментов)."""
    out = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and not (i == 0 and m["content"].startswith("Данные анализа:")):
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
            out.append({"role": "assistant", "content": m["content"]})
    return out


def _default_detector(opt: JobOptions):
    from tracker.detection import YoloDetector

    # device=None -> cuda/mps/cpu
    return YoloDetector(opt.weights, imgsz=opt.imgsz, device=opt.device, ball_tiles=opt.ball_tiles)


def _default_encoder(opt: JobOptions):
    if opt.encoder == "osnet":
        from tracker.appearance import TorchReidEncoder

        return CompositeEncoder([(TorchReidEncoder(), 1.0), (PartColorEncoder(), 0.6)])
    return default_color_encoder()


def _default_reader(opt: JobOptions):
    if opt.ocr == "easyocr":
        from tracker.jersey import EasyOcrReader

        return EasyOcrReader()
    return None
