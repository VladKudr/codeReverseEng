"""Уточнение результата по всему ролику: задержка захвата, поздняя подмена id, короткие разрывы."""
from types import SimpleNamespace

import numpy as np

from tracker.offline import CameraPath, TrackLog, refine_online
from tracker.target import TargetObservation, TargetState
from _synth import unit

DIM = 8
TARGET, OTHER = unit(np.eye(DIM)[0] + 0.1), unit(np.eye(DIM)[1] + 0.1)


def trk(tid, box, feat):
    return SimpleNamespace(track_id=tid, detected_now=True, last_box=np.asarray(box, float),
                           last_feature=feat, score=0.9)


def box_at(x):
    return [x, 100, x + 30, 180]


def on(frame, tid, box, event=None):
    return TargetObservation(frame, TargetState.ACTIVE, tid, np.asarray(box, float), 1.0, event=event)


def off(frame, event=None):
    return TargetObservation(frame, TargetState.LOST, None, None, 0.0, event=event)


def test_backfill_removes_reacquisition_delay():
    """Трек 5 (цель) виден с кадра 20, автомат захватил его только на кадре 26 — кадры 20..25 дописываются.
    Трек 7 рядом — чужой: на него продление не распространяется."""
    log, obs = TrackLog(), []
    for f in range(40):
        tracks = [trk(7, box_at(400), OTHER)]
        if f < 5:
            tracks.append(trk(1, box_at(100 + 2 * f), TARGET))
        if f >= 20:
            tracks.append(trk(5, box_at(150 + 2 * f), TARGET))
        log.add(f, tracks, None)
        if f < 5:
            obs.append(on(f, 1, box_at(100 + 2 * f), "locked" if f == 0 else None))
        elif f < 26:
            obs.append(off(f, "lost" if f == 5 else None))
        else:
            obs.append(on(f, 5, box_at(150 + 2 * f), "reacquired" if f == 26 else None))
    out = refine_online(obs, log)
    assert all(o.state == TargetState.LOST for o in out[5:20])
    assert all(o.state == TargetState.ACTIVE and o.track_id == 5 for o in out[20:])
    assert out[20].event == "reacquired" and out[26].event is None


def test_late_swap_is_trimmed_to_overlap():
    """Трек 3 шёл за целью, в перекрытии (кадры 10..13) id ушёл к соседу, автомат отпустил трек на кадре 16:
    кадры, где трек уже не похож на цель, помечаются потерей, до перекрытия — остаются."""
    log, obs = TrackLog(), []
    for f in range(20):
        x = 100 + 5 * f
        feat = TARGET if f < 12 else OTHER
        tracks = [trk(3, box_at(x), feat)]
        if 10 <= f <= 13:
            tracks.append(trk(4, box_at(x + 3), OTHER))   # сосед вплотную
        elif f < 10:
            tracks.append(trk(4, box_at(400), OTHER))
        log.add(f, tracks, None)
        if f < 16:
            obs.append(on(f, 3, box_at(x), "locked" if f == 0 else None))
        else:
            obs.append(off(f, "released_swap" if f == 16 else None))
    out = refine_online(obs, log)
    assert all(o.state == TargetState.ACTIVE for o in out[:10])
    assert all(o.state == TargetState.LOST for o in out[12:])
    lost_at = next(o for o in out if o.state == TargetState.LOST)
    assert lost_at.event == "released_swap"


def test_short_gap_is_interpolated_with_camera_motion():
    """Детекция цели пропала на 4 кадра, камера при этом сдвигалась на 10 px за кадр: рамка разрыва — по
    координатам сцены, а не по экрану."""
    log, obs = TrackLog(), []
    pan = np.array([[1.0, 0, -10.0], [0, 1.0, 0]])
    for f in range(20):
        x = 300 - 10 * f                                  # игрок стоит, кадр уезжает
        visible = not (8 <= f <= 11)
        log.add(f, [trk(2, box_at(x), TARGET)] if visible else [], pan if f else None)
        obs.append(on(f, 2, box_at(x), "locked" if f == 0 else None) if visible else off(f, "lost" if f == 8 else None))
    out = refine_online(obs, log)
    for f in range(8, 12):
        assert out[f].state == TargetState.CONTESTED
        assert abs(out[f].box[0] - (300 - 10 * f)) < 1.0
    assert [o.event for o in out if o.event] == ["locked"]


def test_camera_path_roundtrip():
    log = TrackLog()
    for f in range(5):
        log.add(f, [], np.array([[1.0, 0, 5.0], [0, 1.0, -2.0]]) if f else None)
    path = CameraPath(log)
    c, h = path.to_scene(4, [100, 50, 120, 90])
    assert np.allclose(c, [110 - 20, 70 + 8]) and abs(h - 40) < 1e-6
    assert np.allclose(path.to_frame(4, c, h, 0.5), [100, 50, 120, 90])
