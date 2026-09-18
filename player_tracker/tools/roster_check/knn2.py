import sys; sys.path.insert(0, "tools/roster_check")
import json
import numpy as np
from common import TRUTH, load, track_features
job = sys.argv[1]; train_fs = [int(x) for x in sys.argv[2].split(",")]
tr, G, st, colors, (feats, offs) = load(job)
F = track_features(tr, feats, offs)
T = json.load(open(f"data/jobs/{job}/board.json"))["tracks"]
hr = {int(t): (m.get("height_ratio") or 1.0) for t, m in T.items()}
truth = TRUTH[job]
train = [(t, 1) for f in train_fs for t in truth[f]["play"]] + [(t, 0) for f in train_fs for t in truth[f]["off"]]
test = [(t, c, f) for f in truth if f not in train_fs for c, k in ((1, "play"), (0, "off")) for t in truth[f][k] if t not in dict(train)]
X = np.stack([F[t] for t, _ in train]); y = np.array([c for _, c in train]); H = np.array([hr.get(t, 1.0) for t, _ in train])
for w in (0.0, 0.5, 1.0, 2.0):
    res = []
    for t, c, f in test:
        s = X @ F[t] - w * np.abs(np.log(hr.get(t, 1.0) / H))
        res.append((int(s[y == 1].max() > s[y == 0].max()) == c, c))
    print(f"вес роста {w}: игроки {sum(a for a, c in res if c)}/{sum(c for _, c in res)}  лишние {sum(a for a, c in res if not c)}/{sum(not c for _, c in res)}")
