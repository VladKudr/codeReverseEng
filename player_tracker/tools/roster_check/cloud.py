"""Облако положений игроков на плоскости поля (метры от камеры) до всякой подгонки схемы — что видит fit_to_occupancy."""
import json, sys
import numpy as np
from tracker import board as B
from tracker.ground import GroundConfig, GroundModel
from tracker.pipeline import load_inputs

job = sys.argv[1]
d = f"data/jobs/{job}"
st = json.load(open(f"{d}/state.json")); bd = json.load(open(f"{d}/board.json"))
inputs = load_inputs(f"{d}/inputs.npz"); tr = B.load_tracks(f"{d}/tracks.npz")
people = [np.array([[*x.box, x.score] for x in i.detections], np.float32).reshape(-1, 5) for i in inputs]
H = float(st["player"].get("height_m") or 1.4)
g = GroundModel.fit(people, [i.camera for i in inputs], st["process_size"][0], GroundConfig(player_height_m=H))
feet = B.play_feet(tr, g, st.get("track_roles"))
meta = bd["tracks"]
on = np.array([not B.track_off(meta.get(str(t), {}), f) for t, f in zip(feet["t"], feet["f"])])
for name, m in (("все (как видит подгонка)", np.ones(len(on), bool)), ("в игре по составу", on), ("в игре и подвижные (размах ≥ 6 м)", on & (feet["spread"] >= 6))):
    X, Z = feet["X"][m], feet["Z"][m]
    print(f"{name:36s} n={m.sum():6d}  X {np.percentile(X, [2, 25, 50, 75, 98]).round(0)}  Z {np.percentile(Z, [2, 25, 50, 75, 98]).round(0)}")
print("горизонт (строка):", np.percentile([g.horizon(i) for i in range(0, len(g), 50)], [10, 50, 90]).round(0), " a:", np.percentile(g.a[::50], [10, 50, 90]).round(2))

from tracker.pitch import PitchSetup, fit_to_occupancy
pitch = PitchSetup.from_dict(st["player"].get("pitch"), st["player"].get("game_format")) or PitchSetup.default(st["player"].get("game_format"))
for name, m in (("все", np.ones(len(on), bool)), ("в игре и подвижные", on & (feet["spread"] >= 6))):
    X, Z = feet["X"][m], feet["Z"][m]
    px, py = pitch.to_pitch(X, Z)
    print(f"\n{name}: без подгонки x {np.percentile(px, [2, 50, 98]).round(0)} y {np.percentile(py, [2, 50, 98]).round(0)} вне поля {pitch.outside_share(X, Z, 1.0):.2f}")
    fitted, info = fit_to_occupancy(pitch, X, Z)
    px, py = fitted.to_pitch(X, Z)
    print(f"   подгонка: {info}\n   x {np.percentile(px, [2, 50, 98]).round(0)} y {np.percentile(py, [2, 50, 98]).round(0)}")
