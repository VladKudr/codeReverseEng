"""Сквозной прогон на рисованных кадрах: цель уходит за край и возвращается,
рядом бегает одноклубник с другими волосами/гетрами."""
import numpy as np

from _synth import draw_player, grass_frame
from tracker.appearance import PartColorEncoder
from tracker.detection import ScriptedDetector
from tracker.export import observation_to_dict, summarize
from tracker.jersey import NumberRead, ScriptedNumberReader
from tracker.pipeline import InitSpec, Pipeline, PipelineConfig
from tracker.render import draw
from tracker.target import TargetState

W, H = 640, 360
RED, WHITE, BLOND, DARK, BLUE = (0, 0, 255), (255, 255, 255), (120, 200, 230), (30, 30, 30), (255, 60, 0)


def scenario(n=220):
    """Возвращает (frames, script, truth): truth[i] = рамка цели или None."""
    frames, script, truth = [], {}, {}
    for i in range(n):
        f = grass_frame(W, H)
        dets = []
        # цель: идёт вправо, на 60..140 за кадром, возвращается слева
        if i < 60:
            tb = (60 + 4 * i, 100, 100 + 4 * i, 220)
        elif i >= 140:
            tb = (-40 + 4 * (i - 140), 180, 0 + 4 * (i - 140), 300)
        else:
            tb = None
        if tb is not None and tb[2] > 8:
            cb = (max(tb[0], 0), tb[1], min(tb[2], W), tb[3])
            draw_player(f, cb, RED, WHITE, RED, BLOND)
            if cb[2] - cb[0] > 16:
                dets.append((*cb, 0.9))
                truth[i] = cb
        # одноклубник: бегает по кругу в правой части
        ang = i / 25
        mx, my = int(420 + 120 * np.cos(ang)), int(120 + 60 * np.sin(ang))
        mb = (mx, my, mx + 40, my + 120)
        draw_player(f, mb, RED, WHITE, WHITE, DARK)
        dets.append((*mb, 0.9))
        # соперник
        ob = (300, 200, 340, 320)
        draw_player(f, ob, BLUE, BLUE, BLUE, DARK)
        dets.append((*ob, 0.85))
        frames.append(f)
        script[i] = dets
        truth.setdefault(i, None)
    return frames, script, truth


def test_end_to_end_reacquire():
    frames, script, truth = scenario()
    cfg = PipelineConfig(init=InitSpec(frame=2, point=(80, 160)))
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), None, cfg)
    results = pipe.run(frames)
    states = [r.observation.state for r in results]
    assert states[2] == TargetState.ACTIVE
    # пока цели нет, захвата не происходит
    for r in results[62:140]:
        assert r.observation.track_id is None, f"кадр {r.frame_idx}: захвачен {r.observation.track_id}"
    # после возвращения цель захватывается и рамка совпадает с истиной
    reacq = [r for r in results if r.observation.event == "reacquired"]
    assert reacq, "цель не была захвачена повторно"
    first = reacq[0].frame_idx
    assert 140 < first < 175, first
    for r in results[first + 5:]:
        assert r.observation.track_id == reacq[0].observation.track_id
        tb = truth[r.frame_idx]
        cx = (r.observation.box[0] + r.observation.box[2]) / 2
        assert abs(cx - (tb[0] + tb[2]) / 2) < 25
    # экспорт и сводка не падают, рамки в исходных координатах
    recs = [observation_to_dict(r.observation, scale=0.5, fps=30.0) for r in results]
    assert recs[2]["box"][2] > truth[2][2]
    s = summarize(recs)
    assert s["reacquired"] >= 1 and s["lost_events"] >= 1 and s["tracked_share"] > 0.5
    img = draw(frames[first], reacq[0].observation, reacq[0].tracks)
    assert img.shape == frames[first].shape


def test_number_reader_is_consulted():
    frames, script, truth = scenario(80)
    reads = {i: {1: NumberRead("10", 0.9)} for i in range(3, 30)}
    cfg = PipelineConfig(init=InitSpec(frame=2, point=(80, 160)))
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), ScriptedNumberReader(reads), cfg)
    pipe.run(frames[:40])
    assert pipe.follower.model.numbers.best()[0] == "10"


def test_init_by_box_and_team_learned():
    frames, script, truth = scenario(120)
    cfg = PipelineConfig(init=InitSpec(frame=0, box=truth[0]))
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), None, cfg)
    results = pipe.run(frames[:100])
    assert results[3].observation.state == TargetState.ACTIVE
    assert pipe.teams.ready
    # соперник и цель — разные кластеры
    from tracker.team import torso_color

    a = pipe.teams.predict(torso_color(frames[0], truth[0]))
    b = pipe.teams.predict(torso_color(frames[0], (300, 200, 340, 320)))
    assert a != b
    assert pipe.follower.model.team == a


def test_init_timeout_logged_or_raised():
    import pytest

    frames, script, _ = scenario(30)
    cfg = PipelineConfig(init=InitSpec(frame=0, point=(5, 5), max_wait_frames=5))
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), None, cfg)
    results = pipe.run(frames)
    assert all(r.observation.state == TargetState.IDLE for r in results)
    assert pipe.errors.counts() == {"init_failed": 1}
    cfg = PipelineConfig(init=InitSpec(frame=0, point=(5, 5), max_wait_frames=5), tolerate_errors=False)
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), None, cfg)
    with pytest.raises(RuntimeError):
        pipe.run(frames)
