"""Метрики игрока: масштаб по росту, компенсация камеры, зоны скорости, соседи и мяч."""
import numpy as np

from tracker.detection import Detection
from tracker.pipeline import FrameInputs
from tracker.player_metrics import MetricsConfig, compute

FPS, W = 30.0, 1280


def build(n, target_box, others=(), balls=(), camera=None):
    records, inputs = [], []
    for i in range(n):
        tb = target_box(i)
        dets = [Detection(np.asarray(tb, float), 0.9)] if tb is not None else []
        dets += [Detection(np.asarray(o(i), float), 0.9) for o in others]
        bl = np.array([b(i) for b in balls], np.float32).reshape(-1, 5)
        records.append({"frame": i, "state": "active" if tb is not None else "lost", "box": None if tb is None else list(tb),
                        "event": None})
        cam = None if camera is None or i == 0 else camera
        inputs.append(FrameInputs(dets, np.zeros((len(dets), 4), np.float32), np.zeros((len(dets), 3), np.float32), cam, bl))
    return records, inputs


def test_lateral_run_distance_speed_and_sprint():
    """Игрок ростом 1.5 м, рамка 100 px (1.5 см/px), бежит поперёк 10 px/кадр = 4.5 м/с 3 секунды."""
    records, inputs = build(90, lambda i: (300 + 10 * i, 300, 330 + 10 * i, 400))
    m = compute(records, inputs, scale=1.0, fps=FPS, width=W, cfg=MetricsConfig(player_height_m=1.5))
    mv = m["movement"]
    assert abs(mv["distance_m"] - 13.4) < 1.0
    assert abs(mv["max_speed_ms"] - 4.5) < 0.3
    assert m["presence"]["tracked_share"] == 1.0 and m["presence"]["appearances"] == 1
    assert len(m["timeline"]) == 3 and m["timeline"][1]["zone"] in ("бег", "спринт")


def test_camera_pan_is_not_player_movement():
    """Игрок стоит, камера панорамирует на 10 px/кадр: на экране рамка едет, дистанция ~0."""
    pan = np.array([[1.0, 0.0, -10.0], [0.0, 1.0, 0.0]])
    records, inputs = build(90, lambda i: (600 - 10 * i, 300, 630 - 10 * i, 400), camera=pan)
    m = compute(records, inputs, scale=1.0, fps=FPS, width=W)
    assert m["movement"]["distance_m"] < 0.5
    assert m["movement"]["zones"][0]["seconds"] > 2.5          # всё время — «шаг» (стоит)


def test_contact_and_ball_near():
    records, inputs = build(
        60, lambda i: (600, 300, 630, 400),
        others=[lambda i: (640, 300, 670, 400), lambda i: (100, 300, 130, 400)],   # соперник вплотную и дальний
        balls=[lambda i: (618, 392, 628, 402, 0.6)],                                 # мяч у ног
    )
    m = compute(records, inputs, scale=1.0, fps=FPS, width=W)
    inv = m["involvement"]
    assert inv["avg_nearest_m"] < 1.5 and inv["close_contact_s"] > 1.5
    assert inv["ball_near_s"] > 1.5 and len(inv["ball_episodes"]) == 1
    assert m["position"]["thirds"] is not None


def test_gaps_split_segments_and_absent_frames():
    records, inputs = build(120, lambda i: None if 40 <= i < 80 else (300 + 2 * i, 300, 330 + 2 * i, 400))
    m = compute(records, inputs, scale=1.0, fps=FPS, width=W)
    assert m["presence"]["appearances"] == 2
    assert abs(m["presence"]["tracked_s"] - 80 / FPS) < 0.1
    assert any("лишь" in n for n in m["quality"]["notes"])


def test_analyst_uses_tools_and_keeps_history():
    """Аналитик: модель запрашивает инструмент, получает эпизоды, отвечает; второй вопрос — в той же истории."""
    import json

    from tracker.analyst import Analyst, PlayerContext, build_context

    records, inputs = build(90, lambda i: (300 + 10 * i, 300, 330 + 10 * i, 400))
    metrics = compute(records, inputs, scale=1.0, fps=FPS, width=W)

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, tools):
            self.calls.append(messages)
            if len(self.calls) == 1:
                assert "Данные анализа" in messages[1]["content"] and "timeline" not in messages[1]["content"]
                return {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_episodes", "arguments": json.dumps({"kind": "sprints"})}}]}
            if messages[-1]["role"] == "tool":
                eps = json.loads(messages[-1]["content"])
                return {"content": f"## Коротко\nСпринтов: {len(eps)}, первый в [{eps[0]['start']}]"}
            return {"content": "Ответ на уточнение"}

    job = {"events": [{"time": 0.0, "event": "locked"}], "corrections": [], "init": {}, "fps": FPS}
    fake = FakeClient()
    analyst = Analyst(fake, metrics, job)
    answer, history = analyst.ask([], "Разбор", PlayerContext.from_dict({"age": 12, "position": "forward"}))
    assert "Спринтов: 1" in answer and "первый в [00:0" in answer
    assert [m["role"] for m in history] == ["user", "user", "assistant", "tool", "assistant"]
    answer2, history2 = analyst.ask(history, "А что со скоростью?", PlayerContext())
    assert answer2 == "Ответ на уточнение" and len(history2) == len(history) + 2
    ctx = build_context(metrics, job, PlayerContext.from_dict({"age": 12, "position": "full_back", "game_format": "6x6"}))
    assert "возраст: 12" in ctx and "крайний защитник" in ctx and "40–50 × 25–35 м" in ctx
    assert "в команде 7 игроков вместе с вратарём, на поле 14" in ctx
    assert "спринт 4.5–∞" in ctx and "типичный для 12 лет" in ctx


def test_identity_change_is_not_a_sprint():
    """Слежение перескочило на другого игрока в 200 px (повторный захват через 3 кадра): это не бег."""
    def box(i):
        return None if 30 <= i < 33 else ((300, 300, 330, 400) if i < 30 else (500, 300, 530, 400))
    records, inputs = build(60, box)
    for i in range(33, 60):
        records[i]["track_id"] = 7
    for i in range(30):
        records[i]["track_id"] = 3
    records[33]["event"] = "reacquired"
    m = compute(records, inputs, scale=1.0, fps=FPS, width=W)
    assert m["movement"]["max_speed_ms"] < 0.5 and m["presence"]["appearances"] == 2


def test_profile_drives_zones_and_height():
    from tracker.profile import PlayerProfile, speed_zones, typical_height

    assert speed_zones(8)[-1][1] == 4.0 and speed_zones(11)[-1][1] == 4.5 and speed_zones(14)[-1][1] == 5.0
    assert speed_zones(17)[-1][1] == 5.5 and typical_height(12) == 1.50
    assert PlayerProfile.from_dict({"age": "9"}).effective_height() == (1.33, "типичный для 9 лет")
    assert PlayerProfile.from_dict({"age": 9, "height_m": 1.41}).effective_height() == (1.41, "указан")
    # один и тот же бег 4.2 м/с: для 9-летнего — спринт, для 15-летнего — бег
    records, inputs = build(90, lambda i: (300 + 9 * i, 300, 330 + 9 * i, 400))   # ~4.05 м/с при росте 1.5 м
    young = compute(records, inputs, 1.0, FPS, W, MetricsConfig(player_height_m=1.5, zones=speed_zones(8)))
    older = compute(records, inputs, 1.0, FPS, W, MetricsConfig(player_height_m=1.5, zones=speed_zones(15)))
    assert young["movement"]["sprints"] and not older["movement"]["sprints"]


def test_format_counts_goalkeeper():
    from tracker.profile import GAME_FORMATS, players_on_field

    assert players_on_field("6x6") == (7, 14) and players_on_field("4x4") == (5, 10)
    assert players_on_field("11x11") == (11, 22) and players_on_field("training") is None
    assert "+ вратарь, 7 в команде" in GAME_FORMATS["6x6"]["label"]


def test_pitch_orientation_thirds_and_run_direction():
    """Камера за ближней бровкой напротив центра, смотрит в центр; свои ворота слева. Игрок бежит вправо в кадре —
    к чужим воротам, из середины в атакующую треть."""
    from tracker.pitch import PitchSetup

    setup = PitchSetup((0.5, 1.4), (0.5, 0.5), "left", 60.0, 40.0)
    # центр кадра (u = 640) — центр поля; игрок с рамкой 100 px (1.5 см/px) уходит вправо на 20 м
    records, inputs = build(60, lambda i: (640 + 22 * i, 300, 670 + 22 * i, 400))
    m = compute(records, inputs, 1.0, FPS, W, MetricsConfig(player_height_m=1.5), pitch=setup)
    pt = m["pitch"]
    assert pt["towards_opponent_goal_m"] > 15 and pt["towards_own_goal_m"] < 1
    assert m["movement"]["sprints"][0]["direction"] == "к чужим воротам"
    assert pt["thirds"]["чужая треть (атака)"] > 0.2 and pt["thirds"]["своя треть (оборона)"] == 0
    assert m["timeline"][-1]["third"] == "чужая треть (атака)"
    # те же данные, но свои ворота справа — тот же рывок становится возвратом к своим воротам
    flipped = compute(records, inputs, 1.0, FPS, W, MetricsConfig(player_height_m=1.5),
                      pitch=PitchSetup((0.5, 1.4), (0.5, 0.5), "right", 60.0, 40.0))
    assert flipped["movement"]["sprints"][0]["direction"] == "к своим воротам"
    # камера за левыми воротами, смотрит вдоль поля: «вправо в кадре» — поперёк поля
    behind = compute(records, inputs, 1.0, FPS, W, MetricsConfig(player_height_m=1.5),
                     pitch=PitchSetup((-0.2, 0.5), (0.5, 0.5), "left", 60.0, 40.0))
    assert behind["movement"]["sprints"][0]["direction"] == "поперёк поля"
    assert "за левыми воротами" in behind["pitch"]["description"]


def test_ignored_neighbour_is_not_a_contact():
    """Сосед в метре от цели весь ролик: без исключения — единоборство, с ignore_boxes — нет."""
    other = lambda i: (300 + 10 * i + 40, 300, 330 + 10 * i + 40, 400)   # noqa: E731 — 40 px ≈ 0.6 м при 1.5 см/px
    records, inputs = build(90, lambda i: (300 + 10 * i, 300, 330 + 10 * i, 400), others=[other])
    with_neighbour = compute(records, inputs, scale=1.0, fps=FPS, width=W)
    assert with_neighbour["involvement"]["close_contacts"] and with_neighbour["involvement"]["avg_nearest_m"] < 1.5
    ignore = [np.array([other(i)], float) for i in range(90)]
    without = compute(records, inputs, scale=1.0, fps=FPS, width=W, ignore_boxes=ignore)
    assert without["involvement"]["close_contacts"] == [] and without["involvement"]["avg_nearest_m"] is None


def test_ground_plane_position_ignores_partial_box():
    """Рамка цели 1.3 с подряд на 30 % ниже (частично закрыта): по росту он «улетает» вглубь,
    по плоскости поля (строка ног та же) положение на поле не прыгает."""
    from tracker.ground import GroundConfig, GroundModel
    from tracker.pitch import PitchSetup

    def target(i):
        top = 330 if 30 <= i < 70 else 300
        return (600, top, 630, 400)

    records, inputs = build(90, target, others=[lambda i: (200, 250, 230, 350), lambda i: (900, 350, 930, 450),
                                                lambda i: (400, 300, 430, 400), lambda i: (1000, 200, 1030, 300),
                                                lambda i: (700, 320, 730, 420), lambda i: (100, 380, 130, 480)])
    people = [np.array([[*d.box, d.score] for d in x.detections], np.float32).reshape(-1, 5) for x in inputs]
    ground = GroundModel.fit(people, [None] * 90, W, GroundConfig(player_height_m=1.5))
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    by_height = compute(records, inputs, scale=1.0, fps=FPS, width=W, pitch=pitch)
    by_ground = compute(records, inputs, scale=1.0, fps=FPS, width=W, pitch=pitch, ground=ground)
    z_h = [r["z_m"] for r in by_height["timeline"] if "z_m" in r]
    z_g = [r["z_m"] for r in by_ground["timeline"] if "z_m" in r]
    assert max(z_h) - min(z_h) > 2.0          # по росту глубина прыгает
    assert max(z_g) - min(z_g) < 1.0          # по плоскости — нет
