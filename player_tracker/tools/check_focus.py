"""Согласованность положения цели: лента метрик (player_metrics.json) против доски (board.json).

    python tools/check_focus.py data/jobs/<job_id>    -> медиана и 90-й процентиль расхождения, м
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.pitch import PitchSetup  # noqa: E402

job = Path(sys.argv[1])
st = json.load(open(job / "state.json"))
board = json.load(open(job / "board.json"))
m = json.load(open(job / "player_metrics.json"))
pitch = PitchSetup.from_dict(st["player"].get("pitch"), st["player"].get("game_format"))
if pitch is None:
    sys.exit("схема поля не отмечена")
tl = {r["t"]: r for r in m["timeline"]}
diffs = []
for fr in board["frames"]:
    row = tl.get(int(fr["t"]))
    if "focus" in fr and row and "x_m" in row:
        px, py = pitch.to_pitch(row["x_m"], row["z_m"])
        diffs.append(float(np.hypot(px - fr["focus"]["x"], py - fr["focus"]["y"])))
print(f"кадров сравнено {len(diffs)}: медиана {np.median(diffs):.1f} м, 90-й процентиль {np.percentile(diffs, 90):.1f} м")
