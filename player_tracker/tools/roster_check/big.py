import sys; sys.path.insert(0, "tools/roster_check")
import numpy as np
from common import load
from tracker.roster import track_tracks, link_tracks, LinkConfig, _is_static
tr, G, st = load(sys.argv[1]); fps = st["fps"]
info = track_tracks(tr, None, list(G)); person, _ = link_tracks(info, fps)
H = 720
rows = []
for i, (ids, boxes) in enumerate(zip(tr.ids, tr.boxes)):
    for t, b in zip(ids, np.asarray(boxes, float).reshape(-1, 4)):
        if b[3] - b[1] > 200:
            P = G[i] @ [(b[0] + b[2]) / 2, b[1], 1]; rows.append((int(t), i, P[0] / P[2], P[1] / P[2], b[3] >= H - 3))
R = np.array(rows, float)
for t in np.unique(R[:, 0]).astype(int):
    r = R[R[:, 0] == t]; v = info[t]
    print(f"трек {t:4d} человек {person[t]:4d} кадры {v.start:5d}-{v.end:5d} n={v.n:4d} стоит={_is_static(v, fps, LinkConfig())!s:5} "
          f"ноги=({np.median(v.x):6.0f},{np.median(v.y):5.0f}) размах={np.hypot(np.ptp(v.x), np.ptp(v.y)):5.0f} "
          f"голова=({np.median(r[:, 2]):6.0f},{np.median(r[:, 3]):5.0f}) размах={np.hypot(np.ptp(r[:, 2]), np.ptp(r[:, 3])):5.0f} "
          f"обрезан={r[:, 4].mean():.0%} h={v.height:.0f}")
