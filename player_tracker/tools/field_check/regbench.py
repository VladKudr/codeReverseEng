"""Привязка кадров гомографией (tracker.camreg) на задаче: считает, сохраняет G и время.
Запуск: PYTHONPATH=. .venv/bin/python tools/field_check/regbench.py <job> <out.npy> [step]"""
import sys, time
import cv2, numpy as np
from tracker import board as B
from tracker.camreg import RegConfig, register

job, out = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
d = f"data/jobs/{job}"
with np.load(f"{d}/inputs.npz") as z: A = z["camera"]
tr = B.load_tracks(f"{d}/tracks.npz")
boxes = [np.asarray(b, float).reshape(-1, 4) for b in tr.boxes]
def frames():
    cap = cv2.VideoCapture(f"{d}/frames.mp4")
    while True:
        ok, f = cap.read()
        if not ok: return
        yield f
t = time.time()
G, st = register(frames(), list(A), boxes, None, RegConfig(step=step))
q = st.pop("quality"); np.save(out.replace(".npy", "_q.npy"), q)
print(job, f"{time.time()-t:.0f} c", st)
np.save(out, G)
