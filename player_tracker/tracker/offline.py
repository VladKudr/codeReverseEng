"""Уточнение результата по всему ролику («взгляд вперёд»).

Онлайн-автомат (`target.TargetFollower`) решает на каждом кадре только по
прошлому, поэтому неизбежно:
  * отмечает цель с задержкой после её возвращения — пока кандидат наберёт
    чистые наблюдения и подтверждение;
  * замечает подмену id в перекрытии лишь после того, как игроки разошлись;
  * рвёт слежение на короткие пропуски детекции.

Веб-приложение и CLI обрабатывают готовый ролик, поэтому после прогона
результат уточняется по журналу мультитрекера (`TrackLog`: id, рамка,
дескриптор каждой детекции и движение камеры на каждом кадре). Новых гипотез
о личности здесь нет — только то, что следует из решений автомата и журнала
(см. `refine_online`). Попытка глобально переназначать личность по всему
ролику оказалась хрупкой: двойники из других групп выигрывают по внешности.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .appearance import l2_normalize
from .camera import IDENTITY, scale_of
from .geometry import iou_matrix
from .target import TargetObservation, TargetState


@dataclass
class FrameLog:
    frame_idx: int
    track_ids: list[int]
    boxes: np.ndarray            # (N, 4)
    scores: np.ndarray           # (N,)
    features: Optional[np.ndarray]   # (N, d)
    camera: np.ndarray           # 2x3 предыдущий кадр -> текущий
    det_index: Optional[list] = None   # номер детекции кадра у каждого трека (цвет торса для доски)


class TrackLog:
    """Журнал детекций мультитрекера за прогон: то, что нужно для уточнения после прогона."""

    def __init__(self):
        self.frames: list[FrameLog] = []

    def add(self, frame_idx: int, tracks, camera: Optional[np.ndarray]) -> None:
        seen = [t for t in tracks if t.detected_now and t.last_box is not None]
        feats = [t.last_feature for t in seen]
        has_feats = bool(seen) and all(f is not None for f in feats)
        self.frames.append(FrameLog(
            frame_idx,
            [t.track_id for t in seen],
            np.array([t.last_box for t in seen], dtype=np.float64).reshape(-1, 4),
            np.array([t.score for t in seen], dtype=np.float64),
            np.stack(feats).astype(np.float32) if has_feats else None,
            IDENTITY.copy() if camera is None else np.asarray(camera, dtype=np.float64),
            [getattr(t, "extra", {}).get("det_index") for t in seen],
        ))


class CameraPath:
    """Накопленное движение камеры: переход между координатами кадра и «координатами сцены» (первого кадра)."""

    def __init__(self, log: TrackLog):
        self.by_frame = {fl.frame_idx: fl for fl in log.frames}
        self._M: dict[int, np.ndarray] = {}
        M = np.eye(3)
        for fl in log.frames:
            M = np.vstack([fl.camera, [0.0, 0.0, 1.0]]) @ M
            self._M[fl.frame_idx] = M

    def __contains__(self, frame_idx: int) -> bool:
        return frame_idx in self._M

    def to_scene(self, frame_idx: int, box) -> tuple[np.ndarray, float]:
        """Центр рамки и её высота в координатах сцены."""
        M = np.linalg.inv(self._M[frame_idx])
        b = np.asarray(box, dtype=np.float64)
        c = M @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
        return c[:2], float(b[3] - b[1]) * scale_of(M[:2])

    def to_frame(self, frame_idx: int, center: np.ndarray, height: float, aspect: float) -> np.ndarray:
        M = self._M[frame_idx]
        c = M @ np.array([center[0], center[1], 1.0])
        h = height * scale_of(M[:2])
        w = aspect * h
        return np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])


# --- консервативное уточнение онлайн-результата ------------------------------------------------------

@dataclass
class RefineConfig:
    backfill_max: int = 90           # насколько кадров назад продлевать захват по тому же треку
    overlap_iou: float = 0.3         # кадр с таким перекрытием трека — граница продления (там мог смениться id)
    swap_lookback: int = 30          # при released_swap искать начало перекрытия не дальше N кадров назад
    interp_max_gap: int = 8
    max_speed: float = 0.12
    speed_slack: float = 1.0
    snap_iou: float = 0.4
    swap_keep_contrast: float = -0.03      # до подмены: кадры перекрытия, где трек ещё похож на цель, остаются
    backfill_min_contrast: float = -0.08   # продление назад останавливается, когда кадр трека не похож на цель
    contrast_top_k: int = 3


def refine_online(observations: list[TargetObservation], log: TrackLog,
                  cfg: RefineConfig | None = None, protected: Optional[dict[int, bool]] = None) -> list[TargetObservation]:
    """Уточнение онлайн-результата взглядом «вперёд», без новых гипотез о личности.

    Личность цели решает онлайн-автомат (со своими предохранителями). Здесь исправляется то, что
    ему недоступно из-за причинности:
      1. задержка захвата: трек, на котором цель захвачена, был виден и раньше — эти кадры
         (пока трек не перекрывался с другими) отмечаются целью;
      2. поздно замеченная подмена id: кадры от начала перекрытия до `released_swap` —
         рамка там уже на соседе, отмечаются как потеря;
      3. короткие разрывы между отрезками цели заполняются интерполяцией в координатах без
         движения камеры, с привязкой к детекции кадра.

    protected — кадры поправок оператора (кадр -> «цели нет»): их решения не меняются, продление назад
    и обрезка через них не проходят, разрыв с кадром «цели нет» не интерполируется.
    """
    cfg = cfg or RefineConfig()
    protected = protected or {}
    obs = [TargetObservation(o.frame_idx, o.state, o.track_id, None if o.box is None else o.box.copy(),
                             o.confidence, list(o.candidates), o.ambiguous, o.event, o.lost_frames)
           for o in observations]
    path = CameraPath(log)
    reasons: dict[int, str] = {}   # кадр -> причина потери, перенесённая на новое место
    on = lambda o: o.state in (TargetState.ACTIVE, TargetState.CONTESTED) and o.box is not None

    def det(frame: int, track_id: int):
        fl = path.by_frame.get(frame)
        if fl is None or track_id not in fl.track_ids:
            return None, 1.0
        i = fl.track_ids.index(track_id)
        others = [j for j in range(len(fl.track_ids)) if j != i]
        worst = float(iou_matrix([fl.boxes[i]], fl.boxes[others])[0].max()) if others else 0.0
        return fl.boxes[i], worst

    # банки внешности по итоговым кадрам цели: позитивы — сама цель, негативы — одновременные с ней треки
    pos, neg = [], []
    for o in obs:
        fl = path.by_frame.get(o.frame_idx)
        if not on(o) or fl is None or fl.features is None or o.track_id not in fl.track_ids:
            continue
        i_t = fl.track_ids.index(o.track_id)
        ious = iou_matrix(fl.boxes, fl.boxes)
        np.fill_diagonal(ious, 0.0)
        for j in range(len(fl.track_ids)):
            if ious[j].max() > 0.15:
                continue
            (pos if j == i_t else neg).append(l2_normalize(fl.features[j]))
    P = np.stack(pos[:: max(1, len(pos) // 300)]) if pos else None
    N = np.stack(neg[:: max(1, len(neg) // 1500)]) if neg else None

    def contrast(frame: int, track_id: int) -> Optional[float]:
        fl = path.by_frame.get(frame)
        if P is None or N is None or fl is None or fl.features is None or track_id not in fl.track_ids:
            return None
        x = l2_normalize(fl.features[fl.track_ids.index(track_id)])
        k = cfg.contrast_top_k
        return float(np.sort(P @ x)[-k:].mean() - np.sort(N @ x)[-k:].mean())

    # 2. подмена id, замеченная поздно
    for i, o in enumerate(obs):
        if o.event != "released_swap" or i == 0 or not on(obs[i - 1]):
            continue
        tid = obs[i - 1].track_id
        start = None
        for k in range(i - 1, max(i - 1 - cfg.swap_lookback, -1), -1):
            if not on(obs[k]) or obs[k].track_id != tid:
                break
            _, worst = det(obs[k].frame_idx, tid)
            if worst > cfg.overlap_iou:
                start = k
        if start is not None:
            # внутри перекрытия рамка ещё какое-то время на цели: режем с последнего кадра, где трек похож на неё
            cut = start
            for k in range(start, i):
                c = contrast(obs[k].frame_idx, tid)
                if c is not None and c >= cfg.swap_keep_contrast:
                    cut = k + 1
                if obs[k].frame_idx in protected:
                    cut = max(cut, k + 1)      # с кадра поправки трек подтверждён оператором
            for k in range(cut, i):
                obs[k].state, obs[k].box, obs[k].track_id, obs[k].confidence = TargetState.LOST, None, None, 0.0
            if cut < i:
                reasons[obs[cut].frame_idx] = "released_swap"

    # 1. задержка захвата
    for i, o in enumerate(obs):
        if not on(o) or i == 0 or on(obs[i - 1]):
            continue
        tid = o.track_id
        k = i - 1
        while (k >= 0 and i - k <= cfg.backfill_max and not on(obs[k]) and obs[k].state != TargetState.IDLE
               and obs[k].frame_idx not in protected):
            box, worst = det(obs[k].frame_idx, tid)
            if box is None or worst > cfg.overlap_iou:
                break
            c = contrast(obs[k].frame_idx, tid)
            if c is not None and c < cfg.backfill_min_contrast and worst <= 0.15:
                break
            obs[k].state, obs[k].box, obs[k].track_id, obs[k].confidence = TargetState.ACTIVE, box.copy(), tid, 0.7
            k -= 1

    # 3. короткие разрывы
    i = 0
    while i < len(obs):
        if on(obs[i]) and i + 1 < len(obs) and not on(obs[i + 1]):
            j = i + 1
            while j < len(obs) and not on(obs[j]):
                j += 1
            gap = j - i - 1
            absent_inside = any(protected.get(obs[k].frame_idx) for k in range(i + 1, j))
            if j < len(obs) and gap <= cfg.interp_max_gap and not absent_inside:
                a, b = obs[i], obs[j]
                ca, ha = path.to_scene(a.frame_idx, a.box)
                cb, hb = path.to_scene(b.frame_idx, b.box)
                h = max((ha + hb) / 2, 1.0)
                if float(np.linalg.norm(cb - ca)) / h <= cfg.max_speed * (gap + 1) + cfg.speed_slack:
                    asp = ((a.box[2] - a.box[0]) / max(a.box[3] - a.box[1], 1) + (b.box[2] - b.box[0]) / max(b.box[3] - b.box[1], 1)) / 2
                    for k in range(i + 1, j):
                        f = obs[k].frame_idx
                        if f not in path:
                            continue
                        t = (f - a.frame_idx) / (b.frame_idx - a.frame_idx)
                        pred = path.to_frame(f, (1 - t) * ca + t * cb, (1 - t) * ha + t * hb, asp)
                        fl = path.by_frame.get(f)
                        if fl is not None and len(fl.boxes):
                            ious = iou_matrix([pred], fl.boxes)[0]
                            m = int(np.argmax(ious))
                            if ious[m] >= cfg.snap_iou:
                                pred = fl.boxes[m].copy()
                        obs[k].state, obs[k].box, obs[k].confidence = TargetState.CONTESTED, pred, 0.5
            i = j
        else:
            i += 1

    # события заново по итоговой разметке; причина отказа от трека сохраняется, если потеря осталась
    was, first = False, True
    for o, orig in zip(obs, observations):
        now = on(o)
        o.event = None
        if orig.event in ("corrected", "corrected_absent"):
            o.event = orig.event                 # поправка оператора видна в событиях всегда
            first = first and not now
        elif now and not was:
            o.event = "locked" if first else "reacquired"
            first = False
        elif was and not now:
            released = orig.event if orig.event and orig.event.startswith("released_") else None
            o.event = reasons.get(o.frame_idx) or released or "lost"
        elif now and orig.event == "switched":
            o.event = "switched"
        was = now
    return obs
