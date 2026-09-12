"""Экспорт результата слежения: JSON по кадрам, CSV, журнал событий."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .target import TargetObservation


def observation_to_dict(obs: TargetObservation, scale: float = 1.0, fps: float = 30.0) -> dict:
    """Словарь для JSON; рамки переводятся в координаты исходного кадра делением на scale."""
    k = 1.0 / scale if scale else 1.0
    box = None if obs.box is None else [round(float(v) * k, 1) for v in obs.box]
    return {
        "frame": obs.frame_idx,
        "time": round(obs.frame_idx / fps, 3) if fps else None,
        "state": obs.state.value,
        "track_id": obs.track_id,
        "box": box,
        "confidence": round(float(obs.confidence), 3),
        "ambiguous": obs.ambiguous,
        "event": obs.event,
        "candidates": [
            {"track_id": c.track_id, "score": round(float(c.score), 3), "rejected": c.rejected,
             "box": [round(float(v) * k, 1) for v in c.box], "cues": c.cues}
            for c in obs.candidates if c.rejected is None
        ],
    }


def write_json(path: str | Path, records: list[dict], meta: dict) -> None:
    Path(path).write_text(json.dumps({"meta": meta, "frames": records}, ensure_ascii=False, indent=1), encoding="utf-8")


def write_csv(path: str | Path, records: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time", "state", "track_id", "x1", "y1", "x2", "y2", "confidence", "event"])
        for r in records:
            box = r["box"] or ["", "", "", ""]
            w.writerow([r["frame"], r["time"], r["state"], r["track_id"] if r["track_id"] is not None else "",
                        *box, r["confidence"], r["event"] or ""])


def write_events(path: str | Path, records: list[dict], fps: float) -> None:
    lines = []
    for r in records:
        if r["event"]:
            lines.append(f"{r['frame']:>7}  {r['frame'] / fps if fps else 0:8.2f}s  {r['event']}  track={r['track_id']}")
        elif r["ambiguous"]:
            lines.append(f"{r['frame']:>7}  {r['frame'] / fps if fps else 0:8.2f}s  ambiguous  "
                         + ", ".join(f"{c['track_id']}:{c['score']}" for c in r["candidates"][:3]))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    total = len(records)
    tracked = sum(1 for r in records if r["state"] in ("active", "contested"))
    lost = sum(1 for r in records if r["state"] == "lost")
    events = [r["event"] for r in records if r["event"]]
    return {
        "frames": total,
        "tracked_frames": tracked,
        "lost_frames": lost,
        "tracked_share": round(tracked / total, 3) if total else 0.0,
        "reacquired": events.count("reacquired"),
        "lost_events": events.count("lost"),
        "released_swap": events.count("released_swap"),
        "released_number": events.count("released_number"),
        "released_team": events.count("released_team"),
        "ambiguous_frames": sum(1 for r in records if r["ambiguous"]),
    }
