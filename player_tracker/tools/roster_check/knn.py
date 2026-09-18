"""Разведка: отделяются ли игроки от лишних по внешности (признаки трекера), росту и положению."""
import sys; sys.path.insert(0, "tools/roster_check")
import numpy as np
from common import TRUTH, load, track_features
from tracker.roster import track_tracks

job = sys.argv[1]
tr, G, st, colors, (feats, offs) = load(job)
info = track_tracks(tr, colors, list(G))
F = track_features(tr, feats, offs)
truth = TRUTH[job]
train_f = int(sys.argv[2]) if len(sys.argv) > 2 else 300
train = [(t, 1) for t in truth[train_f]["play"]] + [(t, 0) for t in truth[train_f]["off"]]
test = [(t, 1, f) for f in truth if f != train_f for t in truth[f]["play"]] + [(t, 0, f) for f in truth if f != train_f for t in truth[f]["off"]]
X = np.stack([F[t] for t, _ in train]); y = np.array([c for _, c in train])
print("обучение:", len(train), "проверка:", len(test))
for name, fn in (("1-NN cos", lambda s: y[np.argmax(s)]),
                 ("max play − max off > 0", lambda s: int(s[y == 1].max() > s[y == 0].max())),
                 ("mean top3", lambda s: int(np.sort(s[y == 1])[-3:].mean() > np.sort(s[y == 0])[-3:].mean()))):
    ok = [(fn(X @ F[t]) == c, c) for t, c, _ in test if t in F]
    print(f"  {name:28s} игроки {sum(a for a, c in ok if c == 1)}/{sum(c == 1 for _, c in ok)}  лишние {sum(a for a, c in ok if c == 0)}/{sum(c == 0 for _, c in ok)}")
# цвет формы и рост рамки
for t, c, f in test:
    s = X @ F[t]
    v = info[t]
    print(f"   кадр {f} трек {t:4d} {'игрок ' if c else 'лишний'} cos_play {s[y == 1].max():.3f} cos_off {s[y == 0].max():.3f} цвет {np.round(v.color) if v.color is not None else None} h={v.height:.0f}")
