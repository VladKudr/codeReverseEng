import numpy as np

from tracker.geometry import (box_center, clip_box, iou_matrix, iou_pair, point_in_box, touches_border,
                              xyah_to_xyxy, xyxy_to_xyah)


def test_iou_matrix_values():
    a = [[0, 0, 10, 10], [20, 20, 30, 30]]
    b = [[0, 0, 10, 10], [5, 5, 15, 15]]
    m = iou_matrix(a, b)
    assert m.shape == (2, 2)
    assert np.isclose(m[0, 0], 1.0)
    assert np.isclose(m[0, 1], 25 / 175)
    assert m[1, 0] == 0.0


def test_iou_empty():
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_pair([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_xyah_roundtrip():
    box = np.array([10, 20, 50, 120], dtype=float)
    assert np.allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box)
    assert np.allclose(box_center(box)[0], [30, 70])


def test_clip_and_border():
    assert np.allclose(clip_box([-5, -5, 20, 400], 100, 100), [0, 0, 20, 100])
    assert touches_border([0, 40, 20, 60], 100, 100)
    assert not touches_border([40, 40, 60, 60], 100, 100)
    assert point_in_box((50, 50), [40, 40, 60, 60])
    assert not point_in_box((70, 50), [40, 40, 60, 60])
