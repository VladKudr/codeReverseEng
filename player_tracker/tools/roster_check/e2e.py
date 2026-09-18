"""Сквозная проверка состава через сервер: пометки на одном кадре эталона -> сверка на остальных."""
import json, sys, time, urllib.request
sys.path.insert(0, "tools/roster_check")
from collections import Counter
from common import TRUTH
job, mode, at = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = f"http://127.0.0.1:8010/api/jobs/{job}"
def post(path, body=None):
    r = urllib.request.Request(base + path, json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))
T = TRUTH[job]
post("/board/roster/clear"); post("/board/roster/mode", {"mode": mode})
role, key = ("sideline", "off") if mode == "all" else ("play", "play")
t = time.time()
for tid in T[at][key]: post("/board/tracks", {"track_id": tid, "role": role, "frame": at, "defer": True})
t1 = time.time() - t; t = time.time()
b = post("/board/roster/apply")
print(f"режим {mode}: {len(T[at][key])} пометок на кадре {at} за {t1:.2f} с, пересчёт {time.time() - t:.1f} с")
off = lambda tid, f: any(a <= f <= c for a, c in b["tracks"].get(str(tid), {}).get("off", []))
for f in T:
    if f == at: continue
    print(f"  кадр {f}: игроки на доске {sum(not off(x, f) for x in T[f]['play'])}/{len(T[f]['play'])}, лишние убраны {sum(off(x, f) for x in T[f]['off'])}/{len(T[f]['off'])}")
print("  вне игры по причинам:", dict(Counter(m["why"] for m in b["tracks"].values() if m["role"] == "sideline")), "из", len(b["tracks"]))
