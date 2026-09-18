"""FastAPI-сервер player_tracker.

API:
  POST /api/videos                      загрузка ролика (multipart, поле file)
  GET  /api/videos                      список
  GET  /api/videos/{id}/frame?t=&width= кадр JPEG для выбора игрока (заголовок X-Scale)
  DELETE /api/videos/{id}
  POST /api/jobs                        {video_id, init:{t_sec, point|box, number}, options:{...}}
  GET  /api/jobs, GET /api/jobs/{id}    состояние, прогресс, события, метрики
  POST /api/jobs/{id}/cancel, DELETE /api/jobs/{id}
  GET  /api/jobs/{id}/files/{name}      track.json, track.csv, errors.md, errors.jsonl,
                                        metrics.json, frame_stats.csv, annotated.mp4, evaluation.json
  GET  /api/jobs/{id}/frames?start=&end= срез записей по кадрам
  POST /api/jobs/{id}/truth             загрузка разметки -> evaluation
Статика фронтенда — из webapp/static.

Аутентификации нет: инструмент для локального запуска или доверенной сети.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from tracker.pitch import default_field
from tracker.profile import GAME_FORMATS, POSITIONS

from .jobs import JobInit, JobManager, JobOptions

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
SERVABLE = {"track.json", "track.csv", "events.log", "summary.json", "metrics.json", "frame_stats.csv", "player_metrics.json",
            "errors.jsonl", "errors.md", "annotated.mp4", "evaluation.json", "frames.mp4"}
MEDIA = {".json": "application/json", ".csv": "text/csv", ".log": "text/plain", ".md": "text/markdown",
         ".jsonl": "application/x-ndjson", ".mp4": "video/mp4"}


class InitBody(BaseModel):
    t_sec: float = 0.0
    point: Optional[tuple[float, float]] = None
    box: Optional[tuple[float, float, float, float]] = None
    number: Optional[str] = None


class OptionsBody(BaseModel):
    width: int = 1280
    weights: str = "yolo11s.pt"
    imgsz: int = 1280
    device: Optional[str] = None
    encoder: str = Field("color", pattern="^(color|osnet)$")
    ocr: str = Field("none", pattern="^(none|easyocr)$")
    team: bool = True
    tonemap: Optional[bool] = None
    start_sec: float = 0.0
    max_frames: Optional[int] = None
    render: bool = True
    refine: bool = True
    ball_tiles: bool = True
    ball_roi: bool = False


class TrackRoleBody(BaseModel):
    """Пометка человека на доске: играет / вне игры (запасной, тренер, зритель); auto — снять ручную пометку."""
    track_id: int
    role: str = Field(pattern="^(play|sideline|auto)$")
    frame: Optional[int] = Field(None, ge=0)          # кадр видео, на котором щёлкнули
    scope: str = Field("all", pattern="^(all|from)$")  # from — с этого кадра (замена)
    defer: bool = False                                # только сохранить; пересчёт — POST /board/roster/apply


class RosterModeBody(BaseModel):
    mode: str = Field(pattern="^(all|marked)$")


class CorrectionBody(BaseModel):
    frame: int = Field(ge=0)
    point: Optional[tuple[float, float]] = None   # координаты исходного кадра
    absent: bool = False                          # цели в этом кадре нет


class PlayerMetricsBody(BaseModel):
    height_m: Optional[float] = Field(None, gt=0.8, lt=2.3)
    hfov_deg: Optional[float] = Field(None, gt=5, lt=130)


class AnalysisBody(BaseModel):
    question: Optional[str] = None
    player: Optional[PlayerBody] = None
    new: bool = False                 # новая версия разбора (иначе — продолжение или сохранённый)
    session_id: Optional[int] = None  # продолжить конкретную версию


class PitchBody(BaseModel):
    """Схема поля: x вдоль длины от левой линии ворот (0..1), y поперёк от дальней бровки (0..1); вне поля — за пределами."""
    camera: tuple[float, float]
    look: tuple[float, float]
    own_goal: str = Field(pattern="^(left|right)$")
    length_m: Optional[float] = Field(None, gt=10, lt=130)
    width_m: Optional[float] = Field(None, gt=5, lt=90)


class FieldMarkBody(BaseModel):
    """Одна разметка: углы 1–4 (левый ближний, правый ближний, правый дальний, левый дальний) в координатах
    исходного кадра, могут быть за краями кадра; rotation — поворот соответствия углам схемы."""
    frame: int = Field(ge=0)
    corners: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    rotation: int = Field(0, ge=0, le=3)
    # corners — четыре угла; sides — 3, 4 в дальних углах, 1, 2 — на боковых линиях (ближние углы за кадром)
    mode: str = Field("corners", pattern="^(corners|sides)$")


class FieldBody(BaseModel):
    """Разметка поля на кадрах: `marks` — одна или несколько разметок (оператор переходил — по одной на место
    съёмки; кадр ролика считается по ближайшей по времени). Прежний вид с одной разметкой (frame, corners,
    rotation прямо в теле) тоже принимается."""
    marks: Optional[list[FieldMarkBody]] = Field(None, min_length=1, max_length=12)
    frame: Optional[int] = Field(None, ge=0)
    corners: Optional[list[tuple[float, float]]] = Field(None, min_length=4, max_length=4)
    rotation: int = Field(0, ge=0, le=3)
    mode: str = Field("corners", pattern="^(corners|sides)$")
    own_goal: Optional[str] = Field(None, pattern="^(left|right)$")
    length_m: Optional[float] = Field(None, gt=10, lt=130)
    width_m: Optional[float] = Field(None, gt=5, lt=90)


class FieldSuggestBody(BaseModel):
    frame: int = Field(ge=0)
    corners: Optional[list[tuple[float, float]]] = Field(None, min_length=4, max_length=4)
    rotation: int = Field(0, ge=0, le=3)
    mode: str = Field("corners", pattern="^(corners|sides)$")
    length_m: Optional[float] = Field(None, gt=10, lt=130)
    width_m: Optional[float] = Field(None, gt=5, lt=90)


class PlayerBody(BaseModel):
    """Опрос перед слежением; ключи позиций и форматов — `GET /api/profile/options`."""
    age: Optional[float] = Field(None, ge=4, le=60)
    position: Optional[str] = None
    position_note: Optional[str] = None
    game_format: Optional[str] = None
    format_note: Optional[str] = None
    height_m: Optional[float] = Field(None, gt=0.8, lt=2.3)
    team_color: Optional[str] = None
    focus: Optional[str] = None
    pitch: Optional[PitchBody] = None
    field: Optional[FieldBody] = None


class JobBody(BaseModel):
    video_id: str
    init: InitBody
    options: OptionsBody = OptionsBody()
    player: Optional[PlayerBody] = None


def create_app(manager: Optional[JobManager] = None, data_dir: str | Path | None = None) -> FastAPI:
    manager = manager or JobManager(data_dir or os.environ.get("PLAYER_TRACKER_DATA", HERE.parent / "data"))
    app = FastAPI(title="player_tracker", version="0.1")
    app.state.manager = manager
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=4096)     # рамки всех треков по кадрам — сотни КБ JSON

    @app.get("/", response_class=HTMLResponse)
    def index():
        # фронтенд правится на месте: к app.js/style.css добавляется версия по времени файла,
        # иначе браузер держит старую версию после обновления инструмента
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "style.css"):
            path = STATIC / name
            if path.exists():
                html = html.replace(f"/static/{name}", f"/static/{name}?v={int(path.stat().st_mtime)}")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    # --- видео ------------------------------------------------------------------------------
    @app.post("/api/videos")
    async def upload_video(file: UploadFile = File(...)):
        try:
            rec = manager.add_video(file.filename or "video.mp4", file.file)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return rec

    @app.get("/api/videos")
    def list_videos():
        return manager.list_videos()

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str):
        try:
            return manager.get_video(video_id)
        except KeyError:
            raise HTTPException(404, "видео не найдено")

    @app.delete("/api/videos/{video_id}")
    def delete_video(video_id: str):
        try:
            manager.delete_video(video_id)
        except KeyError:
            raise HTTPException(404, "видео не найдено")
        return {"ok": True}

    @app.get("/api/videos/{video_id}/jobs")
    def video_jobs(video_id: str):
        try:
            return manager.video_jobs(video_id)
        except KeyError:
            raise HTTPException(404, "видео не найдено")

    @app.get("/api/videos/{video_id}/frame")
    def video_frame(video_id: str, t: float = Query(0.0, ge=0), width: int = Query(960, ge=160, le=3840),
                    tonemap: Optional[bool] = None):
        import cv2

        from tracker.video import extract_frame

        try:
            video = manager.get_video(video_id)
        except KeyError:
            raise HTTPException(404, "видео не найдено")
        try:
            frame, scale = extract_frame(video["path"], at_sec=t, max_width=width, tonemap=tonemap)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise HTTPException(500, "не удалось закодировать кадр")
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={"X-Scale": f"{scale:.6f}", "X-Width": str(frame.shape[1]), "X-Height": str(frame.shape[0]),
                                 "Cache-Control": "no-store"})

    # --- задачи ------------------------------------------------------------------------------------
    @app.post("/api/jobs")
    def create_job(body: JobBody):
        try:
            _check_profile(body.player)
            st = manager.create_job(body.video_id, JobInit(**body.init.model_dump()), JobOptions(**body.options.model_dump()),
                                    body.player.model_dump() if body.player else None)
        except KeyError:
            raise HTTPException(404, "видео не найдено")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return asdict(st)

    @app.get("/api/jobs")
    def list_jobs():
        return [_brief(j) for j in manager.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return asdict(manager.get_job(job_id))
        except KeyError:
            raise HTTPException(404, "задача не найдена")

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            return asdict(manager.cancel_job(job_id))
        except KeyError:
            raise HTTPException(404, "задача не найдена")

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            manager.delete_job(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        if name not in SERVABLE:
            raise HTTPException(404, "нет такого файла")
        try:
            manager.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        path = manager.job_dir(job_id) / name
        if not path.exists():
            raise HTTPException(404, "файл ещё не создан")
        return FileResponse(path, media_type=MEDIA.get(path.suffix, "application/octet-stream"), filename=name)

    @app.get("/api/jobs/{job_id}/frames")
    def job_frames(job_id: str, start: int = Query(0, ge=0), end: Optional[int] = None, step: int = Query(1, ge=1)):
        try:
            manager.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        path = manager.job_dir(job_id) / "track.json"
        if not path.exists():
            raise HTTPException(404, "результат ещё не готов")
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = data["frames"][start:end:step]
        return {"meta": data["meta"], "frames": frames, "total": len(data["frames"])}

    # --- поправки оператора ---------------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}/frame/{frame}")
    def job_frame_image(job_id: str, frame: int):
        import cv2

        try:
            img, scale = manager.frame_image(job_id, frame)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "не удалось закодировать кадр")
        return Response(buf.tobytes(), media_type="image/jpeg",
                        headers={"X-Scale": f"{scale:.6f}", "X-Width": str(img.shape[1]), "X-Height": str(img.shape[0]),
                                 "Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/detections/{frame}")
    def job_frame_detections(job_id: str, frame: int):
        try:
            return manager.frame_detections(job_id, frame)
        except KeyError:
            raise HTTPException(404, "задача не найдена")

    @app.post("/api/jobs/{job_id}/corrections")
    def add_correction(job_id: str, body: CorrectionBody):
        try:
            return asdict(manager.add_correction(job_id, body.frame, body.point, body.absent))
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.delete("/api/jobs/{job_id}/corrections/{frame}")
    def remove_correction(job_id: str, frame: int):
        try:
            return asdict(manager.remove_correction(job_id, frame))
        except KeyError:
            raise HTTPException(404, "задача или поправка не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    # --- метрики игрока и ИИ-разбор -------------------------------------------------------------------
    @app.get("/api/jobs/{job_id}/player-metrics")
    def get_player_metrics(job_id: str):
        try:
            return manager.player_metrics(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/player-metrics")
    def recompute_player_metrics(job_id: str, body: PlayerMetricsBody):
        try:
            return manager.player_metrics(job_id, body.height_m, body.hfov_deg)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str):
        try:
            return asdict(manager.rerun(job_id))
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/jobs/{job_id}/tracks")
    def get_track_boxes(job_id: str):
        """Рамки всех треков по кадрам (координаты исходного кадра) — для разметки поверх видео."""
        try:
            return manager.track_boxes(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/jobs/{job_id}/board")
    def get_board(job_id: str):
        try:
            return manager.board(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/board/tracks")
    def set_track_role(job_id: str, body: TrackRoleBody):
        try:
            return manager.set_track_role(job_id, body.track_id, body.role, body.frame, body.scope, body.defer)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/board/roster/{action}")
    def roster_action(job_id: str, action: str, body: Optional[dict] = None):
        try:
            if action == "apply":
                return manager.apply_roster(job_id)
            if action == "clear":
                return manager.clear_roster(job_id)
            if action == "mode":
                return manager.set_roster_mode(job_id, RosterModeBody(**(body or {})).mode)
            raise HTTPException(404, "действие: apply | clear | mode")
        except ValidationError as exc:
            raise HTTPException(422, str(exc))
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/board/build")
    def build_board(job_id: str):
        try:
            return asdict(manager.build_board(job_id))
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/profile/options")
    def profile_options():
        return {"positions": POSITIONS, "game_formats": {k: v["label"] for k, v in GAME_FORMATS.items()},
                "fields": {k: list(default_field(k)) for k in GAME_FORMATS}}

    @app.get("/api/jobs/{job_id}/field/occupancy")
    def field_occupancy(job_id: str, frame: int = 0):
        try:
            return manager.field_occupancy(job_id, frame)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/field/suggest")
    def field_suggest(job_id: str, body: FieldSuggestBody):
        try:
            return manager.field_suggest(job_id, body.frame, body.corners, body.rotation, body.length_m, body.width_m,
                                         body.mode)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/jobs/{job_id}/field/preview")
    def field_preview(job_id: str, body: FieldSuggestBody):
        if body.corners is None:
            raise HTTPException(422, "нужны точки разметки (corners)")
        try:
            return manager.field_preview(job_id, body.frame, body.corners, body.rotation, body.mode, body.length_m,
                                         body.width_m)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc))

    @app.put("/api/jobs/{job_id}/player")
    def put_player(job_id: str, body: PlayerBody):
        _check_profile(body)
        try:
            return asdict(manager.update_player(job_id, body.model_dump(exclude_unset=True)))
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/ai/status")
    def ai_status():
        return {"configured": manager.ai_configured()}

    @app.get("/api/jobs/{job_id}/analysis")
    def get_analysis(job_id: str):
        try:
            return manager.analysis(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")

    @app.post("/api/jobs/{job_id}/analysis")
    def post_analysis(job_id: str, body: AnalysisBody):
        try:
            _check_profile(body.player)
            return manager.analyze(job_id, body.question, body.player.model_dump(exclude_unset=True) if body.player else {},
                                   new=body.new, session_id=body.session_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(502, str(exc))

    @app.delete("/api/jobs/{job_id}/analysis")
    def delete_analysis(job_id: str, session_id: Optional[int] = None):
        try:
            manager.reset_analysis(job_id, session_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/truth")
    async def upload_truth(job_id: str, file: UploadFile = File(...)):
        try:
            manager.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "задача не найдена")
        suffix = Path(file.filename or "truth.json").suffix.lower() or ".json"
        if suffix not in (".json", ".csv"):
            raise HTTPException(400, "разметка — JSON или CSV")
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            return manager.evaluate_job(job_id, tmp_path)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except (json.JSONDecodeError, KeyError) as exc:
            raise HTTPException(400, f"разметка не разобрана: {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    class _NoCacheStatic(StaticFiles):
        """Фронтенд без сборки правится на месте — браузер не должен держать старый app.js."""

        def file_response(self, *args, **kwargs):
            resp = super().file_response(*args, **kwargs)
            resp.headers["Cache-Control"] = "no-store"
            return resp

    app.mount("/static", _NoCacheStatic(directory=str(STATIC)), name="static")
    return app


def _check_profile(p) -> None:
    if p is None:
        return
    if p.position and p.position not in POSITIONS:
        raise HTTPException(422, f"позиция: одно из {list(POSITIONS)}")
    if p.game_format and p.game_format not in GAME_FORMATS:
        raise HTTPException(422, f"формат игры: одно из {list(GAME_FORMATS)}")
    fd = getattr(p, "field", None)
    if fd is not None and not fd.marks and (fd.corners is None or fd.frame is None):
        raise HTTPException(422, "разметка поля: нужен список marks или frame и corners")


def _brief(j) -> dict:
    d = asdict(j)
    d.pop("metrics", None)
    d["events"] = d["events"][-5:]
    return d
