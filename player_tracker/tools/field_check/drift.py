import cv2, numpy as np, sys
job, ax, ay, step = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
d = f"data/jobs/{job}"
with np.load(f"{d}/inputs.npz") as z: A = z["camera"]
M = np.eye(3); inv = []
for a in A:
    if not np.isnan(a).any(): M = np.vstack([a, [0, 0, 1]]) @ M
    inv.append(np.linalg.inv(M))            # кадр i -> сцена (кадр 0)
if len(sys.argv) > 6:                       # привязка гомографией (regbench.py) вместо цепочки подобий
    inv = list(np.load(sys.argv[6]))
cap = cv2.VideoCapture(f"{d}/frames.mp4")
ok, f0 = cap.read(); g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
h = 22; tpl = g0[int(ay)-h:int(ay)+h, int(ax)-2*h:int(ax)+2*h]
rows = []
i = 0
while True:
    ok, f = cap.read(); i += 1
    if not ok: break
    if i % step: continue
    T = np.linalg.inv(inv[i]) @ inv[0]
    p = T @ [ax, ay, 1]; px, py = p[0] / p[2], p[1] / p[2]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    x0, y0 = int(max(px - 250, 0)), int(max(py - 150, 0))
    roi = g[y0:int(py + 150), x0:int(px + 250)]
    if roi.shape[0] <= tpl.shape[0] or roi.shape[1] <= tpl.shape[1]: rows.append((i, px, py, np.nan, np.nan, 0)); continue
    r = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED); _, s, _, loc = cv2.minMaxLoc(r)
    mx, my = x0 + loc[0] + 2 * h, y0 + loc[1] + h
    rows.append((i, px, py, mx, my, s))
    tpl = g[int(my)-h:int(my)+h, int(mx)-2*h:int(mx)+2*h] if s > 0.8 and h <= my < g.shape[0]-h and 2*h <= mx < g.shape[1]-2*h else tpl   # обновляем шаблон при уверенном совпадении
R = np.array(rows)
good = R[:, 5] > 0.7
err = np.hypot(R[:, 1] - R[:, 3], R[:, 2] - R[:, 4])
print(f"{job}: кадров {len(R)}, уверенных {good.sum()}")
for q in (0.25, 0.5, 0.75, 1.0):
    k = int(len(R) * q) - 1
    seg = good[:k+1]
    print(f"  до кадра {int(R[k,0])}: медиана ошибки {np.nanmedian(err[:k+1][seg]):.0f}px, макс {np.nanmax(err[:k+1][seg]):.0f}px")
np.save(sys.argv[5], R)
