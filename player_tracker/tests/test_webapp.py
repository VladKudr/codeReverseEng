"""API веб-приложения на синтетическом ролике и сценарном детекторе (без YOLO)."""
import json
import time

import numpy as np
import pytest

from _synth import draw_player, grass_frame
from tracker.appearance import PartColorEncoder
from tracker.detection import ScriptedDetector
from tracker.video import FrameWriter, ffmpeg_exe

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from webapp.jobs import JobManager  # noqa: E402
from webapp.server import create_app  # noqa: E402

W, H, N = 320, 180, 60
RED, WHITE, BLOND = (0, 0, 255), (255, 255, 255), (120, 200, 230)


def build_clip(path):
    """Игрок идёт вправо, уходит за край на кадрах 25..40 и возвращается слева."""
    script = {}
    with FrameWriter(path, W, H, 24.0) as wr:
        for i in range(N):
            f = grass_frame(W, H)
            if i < 25:
                box = (20 + 10 * i, 30, 50 + 10 * i, 130)
            elif i >= 40:
                box = (10 + 8 * (i - 40), 40, 40 + 8 * (i - 40), 140)
            else:
                box = None
            dets = []
            if box is not None:
                cb = (max(box[0], 0), box[1], min(box[2], W), box[3])
                if cb[2] - cb[0] > 12:
                    draw_player(f, cb, RED, WHITE, RED, BLOND)
                    dets.append((*cb, 0.9))
            script[i] = dets
            wr.write(f)
    return script


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    clip = root / "clip.mp4"
    script = build_clip(clip)
    manager = JobManager(root / "data", detector_factory=lambda opt: ScriptedDetector(script),
                         encoder_factory=lambda opt: PartColorEncoder(), reader_factory=lambda opt: None)
    app = create_app(manager)
    with TestClient(app) as c:
        c.clip = clip
        c.manager = manager
        yield c


def wait_done(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError("задача не завершилась")


def test_index_and_static(client):
    assert "player_tracker" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_upload_and_frame(client):
    with open(client.clip, "rb") as fh:
        r = client.post("/api/videos", files={"file": ("IMG_0001.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["info"]["width"] == W and abs(v["info"]["fps"] - 24) < 0.5
    assert any(x["id"] == v["id"] for x in client.get("/api/videos").json())
    r = client.get(f"/api/videos/{v['id']}/frame", params={"t": 0.2, "width": 160})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert abs(float(r.headers["X-Scale"]) - 0.5) < 1e-6
    assert client.get("/api/videos/nope/frame").status_code == 404
    r = client.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_job_lifecycle_and_results(client):
    v = client.get("/api/videos").json()[0]
    body = {"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
            "options": {"width": 0, "render": ffmpeg_exe() is not None}}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    j = wait_done(client, job_id)
    assert j["status"] == "done", j["error"]
    assert j["progress"] == N and j["total"] == N
    events = [e["event"] for e in j["events"]]
    assert "locked" in events and "lost" in events and "reacquired" in events
    assert j["summary"]["reacquired"] == 1
    assert j["metrics"]["target"]["loss_episodes"] == 1
    # файлы
    for name in ("track.json", "metrics.json", "errors.md", "errors.jsonl", "frame_stats.csv"):
        assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 200, name
    if ffmpeg_exe():
        r = client.get(f"/api/jobs/{job_id}/files/annotated.mp4")
        assert r.status_code == 200 and r.headers["content-type"] == "video/mp4" and len(r.content) > 1000
    assert client.get(f"/api/jobs/{job_id}/files/../state.json").status_code in (404, 422)
    fr = client.get(f"/api/jobs/{job_id}/frames", params={"start": 0, "end": 10}).json()
    assert fr["total"] == N and len(fr["frames"]) == 10 and fr["frames"][2]["state"] == "active"
    # оценка по разметке: рамки цели в исходных координатах
    truth = {"frames": {}}
    for i in range(N):
        if i < 25:
            b = (20 + 10 * i, 30, 50 + 10 * i, 130)
        elif i >= 40:
            b = (10 + 8 * (i - 40), 40, 40 + 8 * (i - 40), 140)
        else:
            b = None
        truth["frames"][str(i)] = None if b is None or min(b[2], W) - max(b[0], 0) <= 12 else [max(b[0], 0), b[1], min(b[2], W), b[3]]
    r = client.post(f"/api/jobs/{job_id}/truth", files={"file": ("truth.json", json.dumps(truth), "application/json")})
    assert r.status_code == 200, r.text
    ev = r.json()
    assert ev["wrong_target"] == 0 and ev["false_track"] == 0 and ev["recall"] > 0.7
    j = client.get(f"/api/jobs/{job_id}").json()
    assert j["evaluation"]["tp"] == ev["tp"]
    assert "missed_target" in j["metrics"]["errors"]["by_kind"] or ev["missed_target"] == 0
    assert client.get(f"/api/jobs/{job_id}/files/evaluation.json").status_code == 200
    # список и удаление
    assert any(x["id"] == job_id for x in client.get("/api/jobs").json())
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_job_validation(client):
    v = client.get("/api/videos").json()[0]
    assert client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 0}}).status_code == 400
    assert client.post("/api/jobs", json={"video_id": "nope", "init": {"t_sec": 0, "point": [1, 1]}}).status_code == 404
    assert client.post("/api/jobs", json={"video_id": v["id"], "init": {"point": [1, 1]}, "options": {"encoder": "magic"}}).status_code == 422


def test_init_failure_reported(client):
    v = client.get("/api/videos").json()[0]
    r = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 0, "point": [300, 170]},
                                       "options": {"width": 0, "render": False, "max_frames": 20}})
    j = wait_done(client, r.json()["id"])
    assert j["status"] == "done"
    assert j["metrics"]["errors"]["by_kind"].get("init_failed") == 1
    assert j["summary"]["tracked_frames"] == 0


def test_cancel(client, tmp_path):
    v = client.get("/api/videos").json()[0]
    slow = client.manager._detector_factory

    class Slow:
        def __init__(self, inner):
            self.inner = inner

        def detect(self, frame):
            time.sleep(0.05)
            return self.inner.detect(frame)

    client.manager._detector_factory = lambda opt: Slow(slow(opt))
    try:
        r = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 0, "point": [35, 80]}, "options": {"width": 0, "render": False}})
        job_id = r.json()["id"]
        time.sleep(0.3)
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        j = wait_done(client, job_id)
        assert j["status"] == "cancelled" and 0 < j["progress"] < N
    finally:
        client.manager._detector_factory = slow


def test_state_survives_restart(client, tmp_path_factory):
    jobs_before = client.get("/api/jobs").json()
    m2 = JobManager(client.manager.data_dir)
    assert {j["id"] for j in jobs_before} == {j.id for j in m2.list_jobs()}
    assert len(m2.list_videos()) == len(client.get("/api/videos").json())
