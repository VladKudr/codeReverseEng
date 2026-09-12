import numpy as np
import pytest

from tracker.video import FrameSource, ffmpeg_exe, probe, tonemap_filter


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    import cv2

    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    w, h, n = 320, 180, 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (60, 140, 60)
        cv2.rectangle(frame, (10 * i, 40), (10 * i + 30, 140), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return path, w, h, n


def test_probe(synthetic_video):
    path, w, h, n = synthetic_video
    info = probe(path)
    assert (info.width, info.height) == (w, h)
    assert abs(info.fps - 24.0) < 0.5
    assert abs(info.nb_frames - n) <= 1
    assert not info.is_hdr


@pytest.mark.skipif(ffmpeg_exe() is None, reason="нет ffmpeg")
def test_frames_via_ffmpeg_with_downscale(synthetic_video):
    path, w, h, n = synthetic_video
    src = FrameSource(path, max_width=160, backend="ffmpeg")
    frames = list(src)
    assert len(frames) == n
    assert frames[0].shape == (90, 160, 3)
    assert abs(src.scale - 0.5) < 1e-6
    # красный прямоугольник виден в BGR
    assert frames[5][45, 30, 2] > 150 and frames[5][45, 30, 0] < 80


@pytest.mark.skipif(ffmpeg_exe() is None, reason="нет ffmpeg")
def test_frames_fragment(synthetic_video):
    path, *_ = synthetic_video
    src = FrameSource(path, start_sec=0.5, max_frames=5, backend="ffmpeg")
    frames = list(src)
    assert len(frames) == 5


def test_frames_via_cv2(synthetic_video):
    path, w, h, n = synthetic_video
    src = FrameSource(path, max_width=160, backend="cv2")
    with pytest.warns(UserWarning):
        frames = list(src)
    assert len(frames) == n and frames[0].shape == (90, 160, 3)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        probe("/nonexistent/clip.MOV")


def test_tonemap_filter_mentions_bt709():
    assert "bt709" in tonemap_filter() and "tonemap" in tonemap_filter()
