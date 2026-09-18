import json

import numpy as np
import pytest

from _synth import draw_player, grass_frame, make_track
from tracker.appearance import PartColorEncoder
from tracker.detection import Detection, ScriptedDetector
from tracker.export import observation_to_dict
from tracker.metrics import ErrorLog, RunMetrics, evaluate, load_truth
from tracker.pipeline import InitSpec, Pipeline, PipelineConfig
from tracker.target import Candidate, TargetObservation, TargetState
from test_pipeline import H, W, scenario


def obs(frame, state, track_id=None, box=None, conf=1.0, event=None, ambiguous=False, cands=()):
    return TargetObservation(frame, state, track_id, None if box is None else np.asarray(box, float), conf,
                             list(cands), ambiguous, event)


def test_error_log_dedupes_consecutive_and_renders():
    log = ErrorLog(fps=30)
    for f in range(5):
        log.add(f, "no_detections", "пусто")
    log.add(7, "no_detections", "пусто")
    log.add(8, "released_swap", "подмена", track_id=3)
    log.add(9, "released_swap", "подмена", track_id=4)   # события не схлопываются
    assert log.counts() == {"no_detections": 2, "released_swap": 2}
    assert log.records[0].details["run_length"] == 5 and log.records[0].details["last_frame"] == 4
    assert log.by_severity() == {"warning": 4}
    md = log.render_markdown()
    assert "серия 5 кадров" in md and "released_swap" in md
    try:
        raise ValueError("boom")
    except ValueError as exc:
        rec = log.exception(10, "detector", exc)
    assert rec.severity == "error" and "boom" in rec.message and "traceback" in rec.details


def test_run_metrics_summary_and_loss_episodes():
    m = RunMetrics((640, 360), fps=30, long_loss_sec=0.5)
    det = [Detection(np.array([10, 10, 50, 150.]), 0.9), Detection(np.array([100, 10, 140, 150.]), 0.3)]
    t1 = make_track(1, [10, 10, 50, 150])
    m.observe(0, det, [t1], obs(0, TargetState.ACTIVE, 1, [10, 10, 50, 150], event="locked"), 5.0)
    for f in range(1, 4):
        m.observe(f, det, [t1], obs(f, TargetState.ACTIVE, 1, [10, 10, 50, 150], conf=0.4), 5.0)
    m.observe(4, [], [], obs(4, TargetState.LOST, event="lost"), 5.0)
    for f in range(5, 30):
        c = Candidate(2, 0.7, np.array([0, 0, 10, 10.]))
        m.observe(f, [], [], obs(f, TargetState.LOST, ambiguous=True, cands=[c]), 5.0)
    m.observe(30, det, [make_track(2, [10, 10, 50, 150])], obs(30, TargetState.ACTIVE, 2, [10, 10, 50, 150], event="reacquired"), 5.0)
    s = m.summary()
    assert s["frames"] == 31
    assert s["detection"]["per_frame_high_mean"] == pytest.approx(5 / 31, abs=0.01)
    assert s["detection"]["frames_without_detections"] == 26
    assert s["target"]["loss_episodes"] == 1 and s["target"]["loss_frames_max"] == 26
    assert s["target"]["reacquired"] == 1 and s["target"]["id_changes"] == 1
    kinds = m.errors.counts()
    assert kinds["target_lost"] == 1 and kinds["long_loss"] == 1 and kinds["ambiguous"] == 1
    assert kinds["low_confidence"] == 1 and kinds["no_detections"] == 1
    assert s["speed"]["fps"] == 200.0


def test_metrics_write(tmp_path):
    m = RunMetrics((640, 360), fps=30)
    m.observe(0, [], [], obs(0, TargetState.LOST, event="lost"), 3.0)
    s = m.write(tmp_path)
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "frame_stats.csv").exists()
    assert (tmp_path / "errors.jsonl").read_text().count("\n") == 2   # target_lost + no_detections
    assert "Ошибки прогона" in (tmp_path / "errors.md").read_text()
    assert json.loads((tmp_path / "metrics.json").read_text())["frames"] == s["frames"] == 1


def test_evaluate_against_truth(tmp_path):
    records = [
        {"frame": 0, "state": "active", "track_id": 1, "box": [0, 0, 10, 10], "candidates": []},   # TP
        {"frame": 1, "state": "active", "track_id": 1, "box": [50, 50, 60, 60], "candidates": []}, # wrong_target
        {"frame": 2, "state": "lost", "track_id": None, "box": None, "candidates": [{"track_id": 3, "score": 0.5}]},  # missed
        {"frame": 3, "state": "lost", "track_id": None, "box": None, "candidates": []},           # true absent
        {"frame": 4, "state": "active", "track_id": 2, "box": [0, 0, 10, 10], "candidates": []},   # false_track
        {"frame": 5, "state": "active", "track_id": 2, "box": [0, 0, 10, 10], "candidates": []},   # TP, id switch
    ]
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"frames": {"0": [0, 0, 10, 10], "1": [0, 0, 10, 10], "2": [0, 0, 10, 10],
                                                 "3": None, "4": None, "5": [1, 1, 10, 10], "99": [0, 0, 1, 1]}}))
    log = ErrorLog()
    ev = evaluate(records, load_truth(truth_path), log)
    assert (ev["tp"], ev["wrong_target"], ev["missed_target"], ev["true_absent"], ev["false_track"]) == (2, 1, 1, 1, 1)
    assert ev["id_switches"] == 1 and ev["frames_evaluated"] == 6
    assert ev["recall"] == 0.5 and ev["mota"] == pytest.approx(1 - 4 / 4)
    assert log.counts() == {"wrong_target": 2, "missed_target": 1}
    assert [r.frame for r in log.sorted_records()] == [1, 2, 4]
    csv_path = tmp_path / "truth.csv"
    csv_path.write_text("frame,x1,y1,x2,y2\n0,0,0,10,10\n3,,,,\n")
    t = load_truth(csv_path)
    assert t == {0: [0, 0, 10, 10], 3: None}


class FlakyDetector(ScriptedDetector):
    def detect(self, frame):
        if self.frame_idx == 9:
            self.frame_idx += 1
            raise RuntimeError("cuda out of memory")
        return super().detect(frame)


def test_pipeline_records_component_exception_and_continues():
    frames, script, truth = scenario(40)
    cfg = PipelineConfig(init=InitSpec(frame=2, point=(80, 160)))
    pipe = Pipeline((W, H), FlakyDetector(script), PartColorEncoder(), None, cfg)
    results = pipe.run(frames)
    assert len(results) == 40
    kinds = pipe.errors.counts()
    assert kinds["exception"] == 1
    assert pipe.errors.records[0].details["where"] == "detector" if pipe.errors.records[0].kind == "exception" else True
    # после сбойного кадра слежение продолжилось
    assert results[-1].observation.state == TargetState.ACTIVE
    cfg = PipelineConfig(init=InitSpec(frame=2, point=(80, 160)), tolerate_errors=False)
    pipe = Pipeline((W, H), FlakyDetector(script), PartColorEncoder(), None, cfg)
    with pytest.raises(RuntimeError):
        pipe.run(frames)


def test_pipeline_metrics_and_truth_end_to_end(tmp_path):
    frames, script, truth = scenario()
    cfg = PipelineConfig(init=InitSpec(frame=2, point=(80, 160)))
    pipe = Pipeline((W, H), ScriptedDetector(script), PartColorEncoder(), None, cfg)
    results = pipe.run(frames)
    s = pipe.metrics.write(tmp_path)
    assert s["target"]["loss_episodes"] == 1 and s["target"]["reacquired"] == 1
    assert s["detection"]["per_frame_mean"] > 2
    records = [observation_to_dict(r.observation, 1.0, 30.0) for r in results]
    ev = evaluate(records, {f: (list(b) if b else None) for f, b in truth.items()}, pipe.errors)
    assert ev["recall"] > 0.8 and ev["wrong_target"] == 0 and ev["false_track"] == 0
    assert ev["missed_target"] < 40   # кадры между возвращением и подтверждением захвата


def test_evaluate_partial_and_hidden_frames(tmp_path):
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({
        "frames": {"0": [0, 0, 10, 20], "1": [0, 0, 10, 20], "2": None},
        "partial": {"3": [0, 0, 10, 20], "4": [0, 0, 10, 20]},
        "hidden": [5, 6],
    }))
    truth = load_truth(truth_path)
    assert truth[3] == {"state": "partial", "box": [0, 0, 10, 20]} and truth[5] == {"state": "hidden"}

    def rec(frame, box, state="active"):
        return {"frame": frame, "state": state, "track_id": 1 if box else None, "box": box, "candidates": []}

    records = [
        rec(0, [0, 0, 10, 20]),            # TP
        rec(1, None, "lost"),              # пропуск
        rec(2, None, "lost"),              # цели нет — верно
        rec(3, None, "lost"),              # частично закрыта, честно потеряна — верно
        rec(4, [30, 0, 40, 20]),           # частично закрыта, рамка на другом — ошибка
        rec(5, None, "lost"),              # закрыта — верно
        rec(6, [0, 0, 10, 20]),            # закрыта, а рамка есть — ошибка
    ]
    log = ErrorLog(30.0)
    ev = evaluate(records, truth, log)
    assert ev["frames_evaluated"] == 7 and ev["errors"] == 3
    assert ev["accuracy"] == round(4 / 7, 4)
    assert (ev["partial_ok"], ev["partial_wrong"], ev["hidden_ok"], ev["hidden_wrong"]) == (1, 1, 1, 1)
    assert log.counts()["wrong_target"] == 2 and log.counts()["missed_target"] == 1


def test_load_truth_csv_with_state(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("frame,x1,y1,x2,y2,state\n0,0,0,10,10,\n1,,,,,hidden\n2,1,1,5,5,partial\n")
    t = load_truth(p)
    assert t == {0: [0, 0, 10, 10], 1: {"state": "hidden"}, 2: {"state": "partial", "box": [1, 1, 5, 5]}}
