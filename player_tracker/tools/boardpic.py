"""Доска рядом с кадром видео на заданные секунды — сверять расстановку с тем, что видно в кадре.

    python tools/boardpic.py data/jobs/<job_id> 12 45 70      -> board_check.jpg в каталоге задачи
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.video import read_frame_at  # noqa: E402

job = Path(sys.argv[1])
b = json.load(open(job / "board.json"))
L, W = b["field"]["length_m"], b["field"]["width_m"]


def hx(h):
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def draw(fr):
    H, Wd, M = 460, 640, 60
    img = np.full((H, Wd, 3), (60, 120, 60), np.uint8)
    x0, y0, x1, y1 = M, M, Wd - M, H - M
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.line(img, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (255, 255, 255), 1)
    P = lambda x, y: (int(x0 + (x1 - x0) * x / L), int(y0 + (y1 - y0) * y / W))  # noqa: E731
    cam = P(*b["camera"])
    cv2.circle(img, cam, 5, (0, 220, 250), -1)
    for tid, x, y in fr["p"]:
        t = b["tracks"][str(tid)]
        col = b["teams"].get(str(t["team"]), {}).get("color", "#cccccc")
        cv2.circle(img, P(x, y), 7, hx(col), -1)
        cv2.circle(img, P(x, y), 7, (20, 20, 20), 1)
        cv2.putText(img, str(tid), (P(x, y)[0] + 8, P(x, y)[1] - 6), 0, 0.35, (255, 255, 255), 1)
    if fr.get("focus"):
        cv2.circle(img, P(fr["focus"]["x"], fr["focus"]["y"]), 12, (0, 230, 255), 3)
    if fr.get("ball"):
        cv2.circle(img, P(fr["ball"][0], fr["ball"][1]), 6, (255, 255, 255), -1 if fr["ball"][2] == "d" else 2)
    cv2.putText(img, f"t={fr['t']}s", (6, 20), 0, 0.6, (255, 255, 255), 1)
    return img


out = []
for t in [float(x) for x in sys.argv[2:]]:
    fr = min(b["frames"], key=lambda q: abs(q["t"] - t))
    im = cv2.resize(read_frame_at(job / "frames.mp4", fr["f"], 30.0, 1280, 720), (820, 460))
    out.append(np.hstack([im, draw(fr)]))
path = job / "board_check.jpg"
cv2.imwrite(str(path), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(path)
