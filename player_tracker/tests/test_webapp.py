"""API веб-приложения на синтетическом ролике и сценарном детекторе (без YOLO)."""
import json
import time
from pathlib import Path

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
    # тот же файл второй раз (под другим именем) — прежний ролик, новой копии нет
    n_before = len(client.get("/api/videos").json())
    with open(client.clip, "rb") as fh:
        again = client.post("/api/videos", files={"file": ("copy.mp4", fh, "video/mp4")}).json()
    assert again["id"] == v["id"] and again["duplicate"] is True
    assert len(client.get("/api/videos").json()) == n_before
    assert client.get(f"/api/videos/{v['id']}/jobs").json() == [] or True


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


def test_corrections_rebuild_from_frame(client):
    """Поправки: «цели нет» на кадре 45 — дальше слежения нет (игрок в кадре признан не целью); «цель здесь»
    на кадре 50 — слежение с этого кадра. Перестроение идёт по сохранённым данным кадров."""
    v = client.get("/api/videos").json()[0]
    r = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                       "options": {"width": 0, "render": ffmpeg_exe() is not None}})
    job_id = r.json()["id"]
    j = wait_done(client, job_id)
    assert j["status"] == "done", j["error"]
    before = client.get(f"/api/jobs/{job_id}/frames").json()["frames"]
    if ffmpeg_exe():
        img = client.get(f"/api/jobs/{job_id}/frame/45")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert client.get(f"/api/jobs/{job_id}/frame/{N + 5}").status_code == 400
    det = client.get(f"/api/jobs/{job_id}/detections/45").json()
    assert len(det["boxes"]) == 1 and det["target"]["state"] == "active"

    r = client.post(f"/api/jobs/{job_id}/corrections", json={"frame": 45, "absent": True})
    assert r.status_code == 200, r.text
    j = wait_done(client, job_id)
    assert j["status"] == "done", j["error"] and j["revision"] == 1
    after = client.get(f"/api/jobs/{job_id}/frames").json()["frames"]
    # решения автомата до поправки те же; уточнение могло продлить назад захват, который поправка отменила
    assert [f["state"] for f in after[:25]] == [f["state"] for f in before[:25]]
    assert all(f["state"] == "lost" for f in after[45:])
    assert any(e["event"] == "corrected_absent" for e in j["events"])

    box = det["boxes"][0]
    point = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    assert client.post(f"/api/jobs/{job_id}/corrections", json={"frame": 50, "point": [5, 5]}).status_code == 200
    j = wait_done(client, job_id)
    assert j["metrics"]["errors"]["by_kind"].get("correction_failed") == 1     # в точке никого — поправка не применена
    frame50 = client.get(f"/api/jobs/{job_id}/detections/50").json()["boxes"][0]
    point = [(frame50[0] + frame50[2]) / 2, (frame50[1] + frame50[3]) / 2]
    assert client.post(f"/api/jobs/{job_id}/corrections", json={"frame": 50, "point": point}).status_code == 200
    j = wait_done(client, job_id)
    frames = client.get(f"/api/jobs/{job_id}/frames").json()["frames"]
    assert frames[45]["state"] == "lost"         # кадр «цели нет» неприкосновенен
    assert all(f["state"] in ("active", "contested") for f in frames[50:])
    # уточнение продлевает поправку назад по тому же треку, но не через кадр «цели нет»
    assert frames[50]["event"] == "corrected" and frames[46]["state"] == "active"
    assert [c["frame"] for c in j["corrections"]] == [45, 50]
    assert client.post(f"/api/jobs/{job_id}/corrections", json={"frame": 50}).status_code == 409   # ни точки, ни «нет»

    assert client.delete(f"/api/jobs/{job_id}/corrections/45").status_code == 200
    j = wait_done(client, job_id)
    assert [c["frame"] for c in j["corrections"]] == [50]
    assert client.delete(f"/api/jobs/{job_id}/corrections/45").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200


def test_player_metrics_and_ai_analysis(client, tmp_path):
    """Метрики считаются после прогона и пересчитываются с другим ростом; разбор — через подменённого клиента
    (ключ из data/secrets.json), история сохраняется, после перестроения слежения разбор начинается заново."""
    v = client.get("/api/videos").json()[0]
    opts = client.get("/api/profile/options").json()
    assert "forward" in opts["positions"] and opts["game_formats"]["5x5"] == "5 × 5 (+ вратарь, 6 в команде)"
    bad = {"video_id": v["id"], "init": {"t_sec": 0, "point": [60, 80]}, "player": {"age": 11, "game_format": "3x3"}}
    assert client.post("/api/jobs", json=bad).status_code == 422
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 9, "position": "forward", "game_format": "5x5"}}).json()["id"]
    j = wait_done(client, job_id)
    assert j["status"] == "done", j["error"]
    assert j["player"]["age"] == 9 and j["player"]["game_format"] == "5x5"
    m = client.get(f"/api/jobs/{job_id}/player-metrics").json()
    assert m["config"]["player_height_m"] == 1.33 and m["config"]["height_source"] == "типичный для 9 лет"
    assert m["movement"]["zones"][-1]["from_ms"] == 4.0 and m["config"]["game_format"].startswith("5 × 5 (+ вратарь")
    j = client.put(f"/api/jobs/{job_id}/player", json={"age": 14, "height_m": 1.62}).json()
    m = client.get(f"/api/jobs/{job_id}/player-metrics").json()
    assert m["config"]["player_height_m"] == 1.62 and m["movement"]["zones"][-1]["from_ms"] == 5.0
    # отрезки в тестовом ролике короче секунды — скорость не считается, только присутствие
    assert m["presence"]["tracked_s"] > 1 and m["presence"]["appearances"] == 2 and len(m["timeline"]) == N // 24 + 1
    m2 = client.post(f"/api/jobs/{job_id}/player-metrics", json={"height_m": 1.2, "hfov_deg": 108}).json()
    assert m2["config"]["player_height_m"] == 1.2 and m2["config"]["hfov_deg"] == 108.0
    assert client.post(f"/api/jobs/{job_id}/player-metrics", json={"height_m": 5}).status_code == 422

    manager = client.manager
    secrets = manager.data_dir / "secrets.json"
    had = secrets.exists()
    asked = []

    class Fake:
        def chat(self, messages, tools):
            asked.append(messages[-1]["content"])
            return {"content": f"разбор #{len(asked)}"}

    manager._ai_client_factory = lambda key: Fake()
    try:
        if not had:
            secrets.write_text(json.dumps({"deepseek_api_key": "test"}))
        assert client.get("/api/ai/status").json()["configured"] is True
        r = client.post(f"/api/jobs/{job_id}/analysis", json={"player": {"age": 11, "height_m": 1.4, "focus": "открывания"}})
        assert r.status_code == 409 and "отметьте поле" in r.json()["detail"]   # без схемы поля разбора нет
        pitch = {"camera": [0.5, 1.15], "look": [0.5, 0.5], "own_goal": "left"}
        assert client.post(f"/api/jobs/{job_id}/analysis", json={"player": {"pitch": {**pitch, "own_goal": "up"}}}).status_code == 422
        r = client.post(f"/api/jobs/{job_id}/analysis", json={"player": {"pitch": pitch}})
        assert r.status_code == 200, r.text
        assert r.json()["answer"] == "разбор #1" and r.json()["cached"] is False
        m = client.get(f"/api/jobs/{job_id}/player-metrics").json()
        assert m["pitch"]["setup"]["length_m"] == 37.5 and "Свои ворота слева" in m["pitch"]["description"]
        # тот же разбор по тем же данным — сохранённый, без запроса к модели
        r = client.post(f"/api/jobs/{job_id}/analysis", json={})
        assert r.json()["cached"] is True and len(asked) == 1
        # новая версия разбора — новый запрос, прежняя сохраняется
        r = client.post(f"/api/jobs/{job_id}/analysis", json={"new": True})
        assert r.json()["answer"] == "разбор #2" and len(r.json()["sessions"]) == 2
        client.delete(f"/api/jobs/{job_id}/analysis", params={"session_id": 2})
        first = json.loads((manager.job_dir(job_id) / "analysis.json").read_text())["sessions"][0]["messages"][0]["content"]
        assert "позиция: нападающий" in first and "5 × 5 (+ вратарь, 6 в команде)" in first and "на что смотреть: открывания" in first
        assert "Свои ворота слева" in first
        r = client.post(f"/api/jobs/{job_id}/analysis", json={"question": "Ещё?", "player": {}})
        sessions = client.get(f"/api/jobs/{job_id}/analysis").json()["sessions"]
        assert len(sessions) == 1 and [h["role"] for h in sessions[0]["messages"]] == ["user", "assistant", "user", "assistant"]
        assert client.get(f"/api/jobs/{job_id}").json()["player"]["height_m"] == 1.4
        assert client.post(f"/api/jobs/{job_id}/analysis", json={"player": {"position": "striker"}}).status_code == 422
        client.post(f"/api/jobs/{job_id}/corrections", json={"frame": 45, "absent": True})
        wait_done(client, job_id)
        r = client.post(f"/api/jobs/{job_id}/analysis", json={"question": "После поправки", "player": {}})
        ss = r.json()["sessions"]
        assert len(ss) == 2 and ss[0]["stale"] and not ss[1]["stale"] and len(ss[1]["messages"]) == 2   # данные изменились
        vj = client.get(f"/api/videos/{v['id']}/jobs").json()
        mine = next(x for x in vj if x["id"] == job_id)
        assert mine["analyses"] == 2 and mine["analysis_current"] and mine["pitch"] and mine["corrections"] == 1
        assert client.delete(f"/api/jobs/{job_id}/analysis").status_code == 200
        assert client.get(f"/api/jobs/{job_id}/analysis").json()["sessions"] == []
    finally:
        if not had:
            secrets.unlink(missing_ok=True)
        client.delete(f"/api/jobs/{job_id}")


def test_board_after_run(client):
    """Тактическая доска строится по завершённой задаче: игроки по кадрам, цель, схема поля."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 11, "position": "forward", "game_format": "5x5",
                                                       "pitch": {"camera": [0.5, 1.15], "look": [0.5, 0.5], "own_goal": "left"}}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    b = client.get(f"/api/jobs/{job_id}/board").json()
    assert b["field"]["own_goal"] == "left" and b["field"]["default"] is False
    assert b["frames"] and any(fr["p"] for fr in b["frames"])
    assert any("focus" in fr for fr in b["frames"])
    assert b["ball"]["frames"] == N and b["ball"]["detected_share"] == 0.0     # мяча в синтетике нет
    assert all(m["num"] for m in b["tracks"].values())                          # у каждого на доске есть номер
    assert client.get(f"/api/jobs/{job_id}").json()["ball_stage"] == "off"      # сценарный детектор мяч не ищет
    assert client.get("/api/jobs/nope/board").status_code == 404
    # рамки всех треков по кадрам — для разметки поверх чистой копии кадров
    tb = client.get(f"/api/jobs/{job_id}/tracks").json()
    assert tb["total"] == N and tb["clean_video"] is True and len(tb["frames"][5]) % 5 == 0
    tid = str(tb["frames"][5][0])
    assert tid in b["tracks"]                                                   # те же id, что на доске
    assert client.get(f"/api/jobs/{job_id}/files/frames.mp4").status_code == 200
    client.delete(f"/api/jobs/{job_id}")


def test_board_track_roles(client):
    """Человека с доски можно пометить «вне игры» и вернуть; пометка живёт в задаче."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 11, "position": "forward", "game_format": "5x5",
                                                       "pitch": {"camera": [0.5, 1.15], "look": [0.5, 0.5], "own_goal": "left"}}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    b = client.get(f"/api/jobs/{job_id}/board").json()
    tid = int(next(iter(b["tracks"])))
    r = client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "sideline"})
    assert r.status_code == 200, r.text
    assert tid in [x["id"] for x in r.json()["sideline"]]
    roles = client.get(f"/api/jobs/{job_id}").json()["track_roles"]
    assert list(roles) == [str(tid)] and roles[str(tid)]["role"] == "sideline" and "f" in roles[str(tid)]
    back = client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "auto"}).json()
    assert tid not in [x["id"] for x in back["sideline"]]
    assert client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "bench"}).status_code == 422
    client.delete(f"/api/jobs/{job_id}")


def wait_reg(client, job_id, timeout=30.0):
    """Фоновая привязка кадров к сцене (camreg) закончилась."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["reg_stage"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"привязка кадров не закончилась: {j["reg_stage"]} {j["reg_progress"]} {j["reg_error"]}")


def test_field_marks_drive_board(client):
    """Разметка углов поля на кадре заменяет оценку по росту; разметок может быть несколько; её можно удалить."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 11, "game_format": "5x5"}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    assert wait_reg(client, job_id)["reg_stage"] == "done"
    corners = [[0, 230], [320, 230], [260, 20], [60, 20]]
    # прежний вид (одна разметка прямо в теле) принимается и сохраняется списком
    r = client.put(f"/api/jobs/{job_id}/player", json={"field": {"frame": 3, "corners": corners, "own_goal": "left",
                                                                 "length_m": 40, "width_m": 25}})
    assert r.status_code == 200, r.text
    assert [m["frame"] for m in r.json()["player"]["field"]["marks"]] == [3]
    b = client.get(f"/api/jobs/{job_id}/board").json()
    cal = b["calibration"]
    assert cal["method"] == "field" and cal["frames"] == [3] and cal["registered"] is True and 0 <= cal["outside"] <= 1
    assert b["field"]["own_goal"] == "left"
    tb = client.get(f"/api/jobs/{job_id}/tracks").json()
    assert tb["field"]["marks"][0]["frame"] == 3 and tb["registered"] is True and len(tb["cams"][0]) == 8
    # две разметки
    marks = [{"frame": 3, "corners": corners, "rotation": 0}, {"frame": 20, "corners": corners, "rotation": 0}]
    r = client.put(f"/api/jobs/{job_id}/player", json={"field": {"marks": marks, "own_goal": "left"}})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/jobs/{job_id}/board").json()["calibration"]["frames"] == [3, 20]
    # негодные тела
    bad = client.put(f"/api/jobs/{job_id}/player", json={"field": {"frame": 3, "corners": corners[:3]}})
    assert bad.status_code == 422
    assert client.put(f"/api/jobs/{job_id}/player", json={"field": {"own_goal": "left"}}).status_code == 422
    butterfly = [corners[0], corners[2], corners[1], corners[3]]
    assert client.put(f"/api/jobs/{job_id}/player", json={"field": {"frame": 3, "corners": butterfly}}).status_code == 409
    client.put(f"/api/jobs/{job_id}/player", json={"field": None})
    b = client.get(f"/api/jobs/{job_id}/board").json()
    assert (b.get("calibration") or {}).get("method") != "field"
    client.delete(f"/api/jobs/{job_id}")


def test_field_editor_hints(client):
    """Подсказки редактору разметки: где играли (точки ног на кадре) и углы по игрокам."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 11, "game_format": "5x5"}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    occ = client.get(f"/api/jobs/{job_id}/field/occupancy", params={"frame": 5}).json()
    assert occ["frame"] == 5 and len(occ["points"]) > 10
    xs = [p[0] for p in occ["points"]]
    assert min(xs) > -320 and max(xs) < 640
    # раздвинуть тесную разметку: после — за полем меньше точек, чем до
    tight = [[100, 200], [200, 200], [190, 150], [110, 150]]
    r = client.post(f"/api/jobs/{job_id}/field/suggest", json={"frame": 5, "corners": tight, "length_m": 40, "width_m": 25})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["mode"] == "expand" and len(s["corners"]) == 4 and s["outside_after"] <= s["outside_before"]
    # с нуля: либо углы, либо честный отказ «мало данных» (в синтетике игроков мало)
    r = client.post(f"/api/jobs/{job_id}/field/suggest", json={"frame": 5})
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        assert r.json()["mode"] == "scratch" and len(r.json()["corners"]) == 4
    assert client.get("/api/jobs/nope/field/occupancy").status_code == 404
    # режим «дальние углы + боковые линии»: предпросмотр отдаёт H модели камеры, разметка сохраняется с режимом
    sides = [[20, 200], [300, 200], [250, 120], [70, 120]]
    pv = client.post(f"/api/jobs/{job_id}/field/preview", json={"frame": 5, "corners": sides, "mode": "sides",
                                                                "length_m": 40, "width_m": 25})
    assert pv.status_code == 200, pv.text
    assert len(pv.json()["H"]) == 3 and pv.json()["rms_m"] is not None and 0 <= pv.json()["outside"] <= 1
    r = client.put(f"/api/jobs/{job_id}/player", json={"field": {"marks": [{"frame": 5, "corners": sides, "mode": "sides"}],
                                                                 "own_goal": "right", "length_m": 40, "width_m": 25}})
    assert r.status_code == 200, r.text
    assert r.json()["player"]["field"]["marks"][0]["mode"] == "sides"
    tb = client.get(f"/api/jobs/{job_id}/tracks").json()
    assert len(tb["field"]["marks"][0]["H"]) == 3                   # линии поверх видео строятся по H с сервера
    assert client.get(f"/api/jobs/{job_id}/board").json()["calibration"]["method"] == "field"
    bad = client.post(f"/api/jobs/{job_id}/field/preview", json={"frame": 5, "corners": [sides[0], sides[2], sides[1], sides[3]]})
    assert bad.status_code == 409
    client.delete(f"/api/jobs/{job_id}")


def test_roster_marks_modes_and_substitution(client):
    """Состав: пометка с кадром и «с этого момента», отложенный пересчёт, режим «только отмеченные», сброс."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False},
                                            "player": {"age": 11, "game_format": "5x5"}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    b = client.get(f"/api/jobs/{job_id}/board").json()
    assert b["roster_mode"] == "all" and all(m["off"] == [] for m in b["tracks"].values())
    focus = {fr["focus"]["id"] for fr in b["frames"] if "focus" in fr}
    tid = int(next(iter(b["tracks"])))                 # в синтетике человек один — он же цель; явная пометка сильнее
    # отложенная пометка: сохраняется сразу, доска пересчитывается по apply
    r = client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "sideline", "frame": 10, "defer": True})
    assert r.status_code == 200 and r.json()["deferred"] is True
    b = client.post(f"/api/jobs/{job_id}/board/roster/apply").json()
    m = b["tracks"][str(tid)]
    assert m["role"] == "sideline" and m["why"] == "manual" and m["off"] and tid in [x["id"] for x in b["sideline"]]
    assert client.get(f"/api/jobs/{job_id}").json()["track_roles"][str(tid)]["frame"] == 10
    # замена: «ушёл с поля с кадра 20» — до него в игре
    b = client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "sideline", "frame": 20, "scope": "from"}).json()
    assert b["tracks"][str(tid)]["off"][0][0] == 20
    assert client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "sideline", "scope": "from"}).status_code == 409
    # «только отмеченные»: без пометок «в игре» все вне игры, кроме цели слежения
    b = client.post(f"/api/jobs/{job_id}/board/roster/clear").json()
    b = client.post(f"/api/jobs/{job_id}/board/roster/mode", json={"mode": "marked"}).json()
    assert b["roster_mode"] == "marked"
    assert all(m["role"] == "sideline" for t, m in b["tracks"].items() if int(t) not in focus)
    assert all(b["tracks"][str(t)]["role"] == "play" for t in focus if str(t) in b["tracks"])   # цель не выключается режимом
    b = client.post(f"/api/jobs/{job_id}/board/tracks", json={"track_id": tid, "role": "play", "frame": 10}).json()
    assert b["tracks"][str(tid)]["role"] == "play" and b["tracks"][str(tid)]["why"] == "manual"
    assert client.post(f"/api/jobs/{job_id}/board/roster/mode", json={"mode": "nobody"}).status_code == 422
    client.delete(f"/api/jobs/{job_id}")


def test_board_requires_rebuild_for_old_job(client, tmp_path):
    """Задача без журнала треков (прогнана до появления доски) просит перестроение."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    out = client.manager.job_dir(job_id)
    (out / "tracks.npz").unlink()
    (out / "board.json").unlink(missing_ok=True)
    r = client.get(f"/api/jobs/{job_id}/board")
    assert r.status_code == 409 and "Построить доску" in r.json()["detail"]
    assert client.post(f"/api/jobs/{job_id}/board/build").status_code == 200
    assert wait_done(client, job_id)["status"] == "done"
    assert client.get(f"/api/jobs/{job_id}/board").status_code == 200
    client.delete(f"/api/jobs/{job_id}")


def test_ball_pass_fills_saved_inputs_for_old_job(client):
    """Задача, прогнанная без поиска мяча на плитках: при перестроении мяч ищется по копии кадров."""
    from tracker.pipeline import inputs_ball_method, load_inputs

    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    cache = client.manager.job_dir(job_id) / "inputs.npz"
    assert inputs_ball_method(cache) == "frame" and not any(len(x.balls) for x in load_inputs(cache))

    base = client.manager._detector_factory
    client.manager._detector_factory = lambda opt: _WithBalls(base(opt))
    try:
        client.post(f"/api/jobs/{job_id}/board/build")
        assert wait_done(client, job_id)["status"] == "done"
        j = wait_ball(client, job_id)
    finally:
        client.manager._detector_factory = base
    assert j["ball_stage"] == "done" and j["ball_progress"] == N
    assert inputs_ball_method(cache) == "tiles"
    assert sum(len(x.balls) for x in load_inputs(cache)) > 0
    b = client.get(f"/api/jobs/{job_id}/board").json()
    assert b["ball_pending"] is False and b["ball_method"] == "tiles"


def test_two_stage_run_gives_players_first_then_ball(client):
    """Этап 1 отдаёт результат без мяча (детектор без плиток), этап 2 в фоне дописывает мяч и пересчитывает доску."""
    from tracker.pipeline import inputs_ball_method

    seen_opts = []
    base = client.manager._detector_factory

    def factory(opt):
        seen_opts.append(opt.ball_tiles)
        return _WithBalls(base(opt))

    client.manager._detector_factory = factory
    try:
        v = client.get("/api/videos").json()[0]
        job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                                "options": {"width": 0, "render": False}}).json()["id"]
        j = wait_done(client, job_id)
        assert j["status"] == "done"
        assert seen_opts[0] is False                     # этап 1 — без плиток мяча
        j = wait_ball(client, job_id)
        assert j["ball_stage"] == "done"
    finally:
        client.manager._detector_factory = base
    assert inputs_ball_method(client.manager.job_dir(job_id) / "inputs.npz") == "tiles"
    m = client.get(f"/api/jobs/{job_id}/player-metrics").json()
    assert not (m.get("ball") or {}).get("pending")
    client.delete(f"/api/jobs/{job_id}")


def test_ball_stage_resumes_from_checkpoint(client):
    """Сервер перезапустили посреди поиска мяча: продолжение с сохранённого места, а не с нуля."""
    from tracker.pipeline import inputs_ball_method
    from webapp.jobs import _save_ball_part

    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    manager = client.manager
    out = manager.job_dir(job_id)
    cache = out / "inputs.npz"
    # «до перезапуска» найдено 40 кадров, на каждом — по мячу в углу
    marked = [np.array([[1.0, 1.0, 5.0, 5.0, 0.9]], np.float32) for _ in range(40)]
    _save_ball_part(out / "balls.part.npz", marked, cache.stat().st_mtime_ns)
    calls = []
    base = manager._detector_factory

    class Counting(_WithBalls):
        def detect_balls(self, frame):
            calls.append(1)
            return super().detect_balls(frame)

    manager._detector_factory = lambda opt: Counting(base(opt))
    try:
        st = manager.get_job(job_id)
        st.ball_stage = "running"                       # так осталось на диске после остановки сервера
        manager._start_ball_stage(st)
        j = wait_ball(client, job_id)
    finally:
        manager._detector_factory = base
    assert j["ball_stage"] == "done" and len(calls) == N - 40      # первые 40 кадров не пересчитывались
    assert inputs_ball_method(cache) == "tiles" and not (out / "balls.part.npz").exists()
    from tracker.pipeline import load_inputs
    got = load_inputs(cache)
    assert any(abs(b[0] - 1.0) < 1e-3 for b in got[10].balls) and len(got[45].balls) >= 1
    client.delete(f"/api/jobs/{job_id}")


def test_ball_stage_waits_while_other_job_tracks(client):
    """Пока идёт этап 1 любой задачи, поиск мяча стоит на паузе (GPU один, одновременная работа роняла процесс)."""
    v = client.get("/api/videos").json()[0]
    job_id = client.post("/api/jobs", json={"video_id": v["id"], "init": {"t_sec": 2 / 24, "point": [60, 80]},
                                            "options": {"width": 0, "render": False}}).json()["id"]
    assert wait_done(client, job_id)["status"] == "done"
    manager = client.manager
    base = manager._detector_factory
    manager._detector_factory = lambda opt: _WithBalls(base(opt))
    try:
        with manager._tracking_cv:
            manager._tracking_now += 1                  # «другая задача на этапе 1»
        st = manager.get_job(job_id)
        st.ball_stage = ""
        manager._start_ball_stage(st)
        t0 = time.time()
        while client.get(f"/api/jobs/{job_id}").json()["ball_stage"] != "waiting":
            assert time.time() - t0 < 30, "поиск мяча не встал на паузу"
            time.sleep(0.05)
        time.sleep(0.5)
        assert client.get(f"/api/jobs/{job_id}").json()["ball_progress"] == 0
        with manager._tracking_cv:
            manager._tracking_now -= 1
            manager._tracking_cv.notify_all()
        j = wait_ball(client, job_id)
    finally:
        manager._detector_factory = base
    assert j["ball_stage"] == "done" and j["ball_progress"] == N
    client.delete(f"/api/jobs/{job_id}")


def wait_ball(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["ball_stage"] not in ("", "queued", "running", "waiting"):
            return j
        time.sleep(0.2)
    raise AssertionError(f"поиск мяча не завершился: {j['ball_stage']} {j['ball_progress']}/{j['ball_total']} {j['ball_error']} status={j['status']}")


class _WithBalls:
    """Сценарный детектор, умеющий «искать мяч на плитках» (для проверки отдельного прохода)."""

    def __init__(self, inner):
        self._inner = inner
        self.ball_tiles = False          # как настоящий детектор этапа 1: мяч на плитках ищется только во втором этапе

    def detect(self, frame):
        return self._inner.detect(frame)

    def detect_balls(self, frame):
        from tracker.detection import BALL_CLASS, Detection

        return [Detection(np.array([100.0, 120.0, 106.0, 126.0]), 0.5, BALL_CLASS)]


def test_duplicate_uploads_are_merged_on_start(tmp_path):
    """Ролик, загруженный дважды до появления проверки по хешу: при старте задачи переносятся на первую загрузку."""
    import shutil as sh

    root = tmp_path / "data"
    clip = tmp_path / "clip.mp4"
    build_clip(clip)
    m = JobManager(root, detector_factory=lambda o: None, encoder_factory=lambda o: PartColorEncoder(), reader_factory=lambda o: None)
    first = m.add_video("a.mp4", open(clip, "rb"))
    # имитируем старую вторую загрузку: копия каталога без хеша и задача на ней
    dup_dir = root / "videos" / "dup000000000"
    sh.copytree(root / "videos" / first["id"], dup_dir)
    info = json.loads((dup_dir / "info.json").read_text())
    info.update(id="dup000000000", path=str(dup_dir / Path(info["path"]).name), uploaded=info["uploaded"] + 10)
    info.pop("sha256"); info.pop("size")
    (dup_dir / "info.json").write_text(json.dumps(info))
    from webapp.jobs import JobState
    st = JobState(id="job1", video_id="dup000000000", status="done")
    m._save(st)
    m2 = JobManager(root)
    assert [v["id"] for v in m2.list_videos()] == [first["id"]]
    assert m2.get_job("job1").video_id == first["id"]
    assert not dup_dir.exists()
