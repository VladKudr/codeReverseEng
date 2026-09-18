"""Загрузка задачи для стендов состава — без JobManager (он возобновляет фоновые этапы чужого сервера)."""
import json
import numpy as np
from tracker import board as B

# Эталон «кто играет» — размечено по кадрам глазами (17.09.2026). Треки с подменой человека внутри трека исключены.
TRUTH = {
    "1032b8d28d03": {
        300: {"play": [9, 31, 26, 63, 7, 20, 13, 59, 70, 60, 17, 67], "off": [64, 14, 35, 2, 5]},
        1200: {"play": [9, 162, 178, 131, 39, 180, 176, 77, 173, 202, 164], "off": [197, 14, 167, 35, 175, 192]},
        2100: {"play": [157, 215, 258, 294, 287, 276, 278, 305, 299], "off": [296, 14, 306, 234, 233, 275, 302, 309]},
    },
}


def load(job):
    d = f"data/jobs/{job}"
    tr = B.load_tracks(f"{d}/tracks.npz")
    with np.load(f"{d}/camreg.npz") as z:
        G = z["G"]
    st = json.load(open(f"{d}/state.json"))
    z = np.load(f"{d}/inputs.npz")
    off = np.concatenate([[0], np.cumsum(z["counts"])])
    colors = [z["colors"][off[i]:off[i + 1]] for i in range(len(z["counts"]))]
    feats = z["features"]
    return tr, G, st, colors, (feats, off)


def track_features(tr, feats, off):
    """Средний нормированный признак внешности по треку."""
    acc, cnt = {}, {}
    for i, (ids, det) in enumerate(zip(tr.ids, tr.det_index)):
        for t, d in zip(ids, det):
            if d < 0 or off[i] + d >= off[i + 1]:
                continue
            f = feats[off[i] + int(d)].astype(np.float32)
            n = np.linalg.norm(f)
            if n > 0 and np.isfinite(n):
                acc[int(t)] = acc.get(int(t), 0) + f / n; cnt[int(t)] = cnt.get(int(t), 0) + 1
    return {t: v / np.linalg.norm(v) for t, v in acc.items()}
