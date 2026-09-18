"""Кто такой трек N: вырезки кадров с рамкой трека — проверять глазами, что доска и роли не врут.

    python tools/whois.py data/jobs/<job_id> 35 5 202     -> whois.jpg в каталоге задачи
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.board import load_tracks  # noqa: E402
from tracker.video import read_frame_at  # noqa: E402

job = Path(sys.argv[1])
want = [int(x) for x in sys.argv[2:]]
tr = load_tracks(job / "tracks.npz")
tiles = []
for tid in want:
    frames = [i for i in range(len(tr)) if tid in list(tr.ids[i])]
    if not frames:
        print("нет трека", tid)
        continue
    f = frames[len(frames) // 2]
    b = tr.boxes[f][list(tr.ids[f]).index(tid)]
    im = read_frame_at(job / "frames.mp4", f, 30.0, 1280, 720)
    H, W = im.shape[:2]
    cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
    Wc, Hc = 260, 200
    x0, y0 = int(np.clip(cx - Wc // 2, 0, W - Wc)), int(np.clip(cy - Hc // 2, 0, H - Hc))
    crop = im[y0:y0 + Hc, x0:x0 + Wc].copy()
    cv2.rectangle(crop, (int(b[0]) - x0, int(b[1]) - y0), (int(b[2]) - x0, int(b[3]) - y0), (0, 255, 255), 2)
    crop = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop, f"id {tid} · кадр {f} · {len(frames)} кадров", (6, 22), 0, 0.6, (0, 255, 255), 2)
    tiles.append(crop)
rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
w = max(r.shape[1] for r in rows)
out = job / "whois.jpg"
cv2.imwrite(str(out), np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out)
