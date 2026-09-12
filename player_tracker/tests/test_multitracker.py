import numpy as np

from tracker.detection import Detection
from tracker.multitracker import MultiTracker, MultiTrackerConfig, TrackState
from _synth import unit


def dets(*boxes, score=0.9):
    return [Detection(np.asarray(b, dtype=float), score) for b in boxes]


def test_confirmation_and_ids_persist():
    mt = MultiTracker(MultiTrackerConfig(min_hits=3))
    for i in range(10):
        out = mt.update(dets([100 + 3 * i, 100, 140 + 3 * i, 200]))
        if i < 2:
            assert out == []
        else:
            assert [t.track_id for t in out] == [1]


def test_two_players_crossing_keep_ids_with_appearance():
    """Два игрока идут навстречу и пересекаются; дескрипторы различны — идентификаторы сохраняются."""
    mt = MultiTracker(MultiTrackerConfig(min_hits=2, appearance_weight=0.5))
    fa, fb = unit([1, 0, 0, 0]), unit([0, 1, 0, 0])
    ids_a, ids_b = set(), set()
    for i in range(40):
        a = [100 + 6 * i, 100, 140 + 6 * i, 200]
        b = [340 - 6 * i, 100, 380 - 6 * i, 200]
        feats = np.stack([fa, fb])
        out = mt.update(dets(a, b), feats)
        for t in out:
            # принадлежность трека определяем по сглаженному дескриптору
            if float(t.feature @ fa) > float(t.feature @ fb):
                ids_a.add(t.track_id)
            else:
                ids_b.add(t.track_id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_track_survives_short_gap():
    mt = MultiTracker(MultiTrackerConfig(min_hits=2, lost_max=20))
    for i in range(6):
        mt.update(dets([100 + 4 * i, 100, 140 + 4 * i, 200]))
    tid = mt.tracks[0].track_id
    for _ in range(5):
        out = mt.update([])
        assert out == []
        assert mt.get(tid).state == TrackState.LOST
    out = mt.update(dets([100 + 4 * 11, 100, 140 + 4 * 11, 200]))
    assert [t.track_id for t in out] == [tid]
    assert mt.get(tid).detected_now


def test_lost_track_removed_after_timeout():
    mt = MultiTracker(MultiTrackerConfig(min_hits=1, lost_max=3))
    mt.update(dets([100, 100, 140, 200]))
    for _ in range(5):
        mt.update([])
    assert mt.tracks == []


def test_low_score_detection_keeps_track_alive():
    mt = MultiTracker(MultiTrackerConfig(min_hits=2, det_high=0.5, det_low=0.1))
    for i in range(4):
        mt.update(dets([100 + 2 * i, 100, 140 + 2 * i, 200]))
    out = mt.update(dets([108, 100, 148, 200], score=0.2))
    assert len(out) == 1 and out[0].detected_now
