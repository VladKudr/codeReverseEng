"""Чтение видео с iPhone 15 Pro Max.

Особенности роликов iPhone, которые ломают наивный `cv2.VideoCapture`:
  * HEVC (H.265) в .MOV — OpenCV из pip нередко собран без HEVC-декодера;
  * HDR (Dolby Vision, 10 бит, HLG `arib-std-b67` или PQ `smpte2084`) —
    без тонмаппинга кадры блёклые и цвета формы «плывут»;
  * поворот хранится в метаданных (display matrix) — кадры приходят лежащими;
  * переменная частота кадров (VFR) в темноте и в «Кино»-режиме;
  * 4K 60 fps — декодировать и анализировать каждый кадр в полном размере
    бессмысленно: детектор работает на уменьшенной копии.

Поэтому кадры читаются через ffmpeg (бинарник из `imageio-ffmpeg`, свой
ставить не нужно): он сам поворачивает, тонмаппит HDR и выравнивает частоту
кадров. Если ffmpeg недоступен — откат на OpenCV с предупреждением.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

HDR_TRANSFERS = {"arib-std-b67", "smpte2084", "smpte428", "bt2020-10", "bt2020-12"}


@dataclass
class VideoInfo:
    path: str
    width: int              # размер кадра ПОСЛЕ поворота
    height: int
    fps: float
    nb_frames: int          # 0, если неизвестно
    duration: float
    codec: str = ""
    pix_fmt: str = ""
    color_transfer: str = ""
    rotation: int = 0
    is_vfr: bool = False

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS or "10" in self.pix_fmt

    @property
    def is_10bit(self) -> bool:
        return "10" in self.pix_fmt or "12" in self.pix_fmt


def ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - зависит от окружения
        return shutil.which("ffmpeg")


def ffprobe_exe() -> Optional[str]:
    return shutil.which("ffprobe")


def _parse_fraction(s: str) -> float:
    if not s:
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b) if float(b) else 0.0
    return float(s)


def _probe_with_ffprobe(path: Path, exe: str) -> Optional[VideoInfo]:
    cmd = [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,pix_fmt,color_transfer,duration"
           ":stream_tags=rotate:stream_side_data=rotation:format=duration", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        data = json.loads(out)
    except Exception:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    s = streams[0]
    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and (s.get("tags") or {}).get("rotate"):
        rotation = int(float(s["tags"]["rotate"]))
    w, h = int(s.get("width", 0)), int(s.get("height", 0))
    if rotation % 180 != 0:
        w, h = h, w
    r_rate = _parse_fraction(s.get("r_frame_rate", "0"))
    avg_rate = _parse_fraction(s.get("avg_frame_rate", "0"))
    fps = avg_rate or r_rate
    duration = float(s.get("duration") or (data.get("format") or {}).get("duration") or 0.0)
    nb = int(s.get("nb_frames") or 0)
    if not nb and duration and fps:
        nb = int(round(duration * fps))
    is_vfr = bool(r_rate and avg_rate and abs(r_rate - avg_rate) / max(r_rate, 1e-6) > 0.02)
    return VideoInfo(str(path), w, h, fps, nb, duration, s.get("codec_name", ""), s.get("pix_fmt", ""),
                     s.get("color_transfer", "") or "", rotation, is_vfr)


_STREAM_RE = re.compile(r"Video:\s*(?P<codec>\w+).*?,\s*(?P<pix>[\w\d]+)(?:\((?P<meta>[^)]*)\))?.*?,\s*(?P<w>\d+)x(?P<h>\d+)")
_FPS_RE = re.compile(r"(?P<fps>[\d.]+)\s*fps")
_TBR_RE = re.compile(r"(?P<tbr>[\d.]+k?)\s*tbr")
_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_ROT_RE = re.compile(r"rotation of (-?[\d.]+) degrees|rotate\s*:\s*(-?\d+)")


def _probe_with_ffmpeg(path: Path, exe: str) -> Optional[VideoInfo]:
    """Разбор `ffmpeg -i` (в imageio-ffmpeg нет ffprobe)."""
    try:
        err = subprocess.run([exe, "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=60).stderr
    except Exception:
        return None
    m = _STREAM_RE.search(err)
    if not m:
        return None
    w, h = int(m.group("w")), int(m.group("h"))
    meta = m.group("meta") or ""
    transfer = ""
    for part in meta.split("/"):
        part = part.strip()
        if part in HDR_TRANSFERS or part in ("bt709", "bt470bg", "smpte170m"):
            transfer = part
    fps_m = _FPS_RE.search(err)
    fps = float(fps_m.group("fps")) if fps_m else 0.0
    tbr_m = _TBR_RE.search(err)
    tbr = 0.0
    if tbr_m:
        v = tbr_m.group("tbr")
        tbr = float(v[:-1]) * 1000 if v.endswith("k") else float(v)
    dur = 0.0
    dm = _DUR_RE.search(err)
    if dm:
        dur = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))
    rotation = 0
    rm = _ROT_RE.search(err)
    if rm:
        rotation = int(round(float(rm.group(1) or rm.group(2))))
    if rotation % 180 != 0:
        w, h = h, w
    nb = int(round(dur * fps)) if dur and fps else 0
    is_vfr = bool(fps and tbr and abs(fps - tbr) / fps > 0.02 and tbr < 1000)
    return VideoInfo(str(path), w, h, fps, nb, dur, m.group("codec"), m.group("pix"), transfer, rotation, is_vfr)


def _probe_with_cv2(path: Path) -> Optional[VideoInfo]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return VideoInfo(str(path), w, h, fps, nb, nb / fps if fps else 0.0)


def probe(path: str | Path) -> VideoInfo:
    """Метаданные ролика: ffprobe -> ffmpeg -i -> OpenCV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    info = None
    if (fp := ffprobe_exe()):
        info = _probe_with_ffprobe(path, fp)
    if info is None and (fe := ffmpeg_exe()):
        info = _probe_with_ffmpeg(path, fe)
    if info is None:
        info = _probe_with_cv2(path)
    if info is None:
        raise RuntimeError(f"Не удалось прочитать метаданные видео: {path}")
    return info


def tonemap_filter() -> str:
    """Тонмаппинг HDR (HLG/PQ, BT.2020) в SDR BT.709 для детектора и цветовых признаков."""
    return ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def _target_size(info: VideoInfo, max_width: Optional[int]) -> tuple[int, int, float]:
    """Размер декодирования и коэффициент масштаба относительно исходного кадра."""
    if not max_width or max_width >= info.width:
        return info.width, info.height, 1.0
    scale = max_width / info.width
    w = int(round(info.width * scale)) // 2 * 2
    h = int(round(info.height * scale)) // 2 * 2
    return w, h, w / info.width


class FrameSource:
    """Итератор кадров BGR (numpy) с опциональным уменьшением, тонмаппингом и выравниванием fps.

    Параметры:
      max_width  — ширина декодируемого кадра (None — исходная). Детектору
                   хватает 1280–1920; рамки потом масштабируются `scale`.
      tonemap    — None: авто (если ролик HDR), True/False — принудительно.
      cfr        — привести к постоянной частоте кадров (VFR-ролики).
      start_sec / max_frames — фрагмент ролика.
    """

    def __init__(self, path: str | Path, max_width: Optional[int] = None, tonemap: Optional[bool] = None,
                 cfr: bool = True, start_sec: float = 0.0, max_frames: Optional[int] = None,
                 backend: str = "auto"):
        self.path = Path(path)
        self.info = probe(self.path)
        self.width, self.height, self.scale = _target_size(self.info, max_width)
        self.tonemap = self.info.is_hdr if tonemap is None else tonemap
        self.cfr = cfr
        self.start_sec = start_sec
        self.max_frames = max_frames
        self.backend = backend
        self.frames_read = 0

    @property
    def fps(self) -> float:
        return self.info.fps or 30.0

    def _ffmpeg_cmd(self, exe: str, use_tonemap: bool) -> list[str]:
        vf = []
        if use_tonemap:
            vf.append(tonemap_filter())
        if (self.width, self.height) != (self.info.width, self.info.height):
            vf.append(f"scale={self.width}:{self.height}:flags=area")
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if self.start_sec > 0:
            cmd += ["-ss", f"{self.start_sec:.3f}"]
        cmd += ["-i", str(self.path)]
        if self.cfr:
            cmd += ["-fps_mode", "cfr", "-r", f"{self.fps:.6f}"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if self.max_frames:
            cmd += ["-frames:v", str(self.max_frames)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "pipe:1"]
        return cmd

    def _iter_ffmpeg(self, exe: str) -> Iterator[np.ndarray]:
        frame_bytes = self.width * self.height * 3
        attempts = [self.tonemap, False] if self.tonemap else [False]
        for use_tm in attempts:
            proc = subprocess.Popen(self._ffmpeg_cmd(exe, use_tm), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    bufsize=frame_bytes * 4)
            got_any = False
            try:
                while True:
                    buf = proc.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break
                    got_any = True
                    self.frames_read += 1
                    yield np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            finally:
                proc.stdout.close()
                err = proc.stderr.read().decode(errors="replace")
                proc.wait()
            if got_any or proc.returncode == 0:
                return
            if use_tm:
                warnings.warn(f"Тонмаппинг HDR не удался ({err.strip()[:200]}); читаем без него — цвета будут блёклыми")
                continue
            raise RuntimeError(f"ffmpeg не смог декодировать {self.path}: {err.strip()[:500]}")

    def _iter_cv2(self) -> Iterator[np.ndarray]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV не открыл {self.path}")
        if self.start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_sec * 1000)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                self.frames_read += 1
                yield frame
                if self.max_frames and self.frames_read >= self.max_frames:
                    break
        finally:
            cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        exe = ffmpeg_exe() if self.backend in ("auto", "ffmpeg") else None
        if exe:
            yield from self._iter_ffmpeg(exe)
            return
        if self.backend == "ffmpeg":
            raise RuntimeError("ffmpeg не найден: pip install imageio-ffmpeg")
        warnings.warn("ffmpeg не найден — читаем через OpenCV (без поворота/тонмаппинга HDR)")
        yield from self._iter_cv2()
