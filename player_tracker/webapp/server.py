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
from pydantic import BaseModel, Field

from .jobs import JobInit, JobManager, JobOptions

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
SERVABLE = {"track.json", "track.csv", "events.log", "summary.json", "metrics.json", "frame_stats.csv",
            "errors.jsonl", "errors.md", "annotated.mp4", "evaluation.json"}
MEDIA = {".json": "application/json", ".csv": "text/csv", ".log": "text/plain", ".md": "text/markdown",
         ".jsonl": "application/x-ndjson", ".mp4": "video/mp4"}


class InitBody(BaseModel):
    t_sec: float = 0.0
    point: Optional[tuple[float, float]] = None
    box: Optional[tuple[float, float, float, float]] = None
    number: Optional[str] = None


class OptionsBody(BaseModel):
    width: int = 1280
    weights: str = "yolov8n.pt"
    imgsz: int = 1280
    device: Optional[str] = None
    encoder: str = Field("color", pattern="^(color|osnet)$")
    ocr: str = Field("none", pattern="^(none|easyocr)$")
    team: bool = True
    tonemap: Optional[bool] = None
    start_sec: float = 0.0
    max_frames: Optional[int] = None
    render: bool = True


class JobBody(BaseModel):
    video_id: str
    init: InitBody
    options: OptionsBody = OptionsBody()


def create_app(manager: Optional[JobManager] = None, data_dir: str | Path | None = None) -> FastAPI:
    manager = manager or JobManager(data_dir or os.environ.get("PLAYER_TRACKER_DATA", HERE.parent / "data"))
    app = FastAPI(title="player_tracker", version="0.1")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

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
            st = manager.create_job(body.video_id, JobInit(**body.init.model_dump()), JobOptions(**body.options.model_dump()))
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

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    return app


def _brief(j) -> dict:
    d = asdict(j)
    d.pop("metrics", None)
    d["events"] = d["events"][-5:]
    return d
