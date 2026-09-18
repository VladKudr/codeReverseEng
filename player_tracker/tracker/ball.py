"""Игровой мяч по всему ролику: кандидаты детектора -> траектория одного мяча по времени.

Мяч с дальнего плана — 4–8 px, поэтому одиночная детекция его почти не видит (`detection.YoloDetector`
ищет его ещё и на увеличенных плитках кадра). Но даже при хорошей детекции «мяч в кадре» — не «игровой
мяч»: у забора лежат запасные мячи, за мяч принимаются головы зрителей и бутсы, а в борьбе у ног мяч
закрыт игроками. Поэтому мяч собирается как траектория по всему ролику:

  1. правдоподобие кандидата: размер против ожидаемого диаметра в этой точке земли
     (`ground.GroundModel`), ниже горизонта, не в верхней части рамки человека (головы, руки), а при
     отмеченной схеме поля — в пределах поля с запасом (за полем — запасные мячи, мячи у зрителей);
  2. кандидаты связываются в треклеты в координатах сцены (без движения камеры) с гейтом по скорости
     в метрах;
  3. неподвижные мячи — запасные у забора (игровой мяч в детском матче так долго не лежит): кандидат,
     рядом с которым в координатах сцены мяч находится и за секунды до, и через секунды после, — не
     игровой мяч. Проверка по месту в сцене, а не по треклету: запасной мяч детектируется с пропусками,
     а глубина дальнего мяча по строке кадра шумит на метры;
  4. из остальных динамическим программированием выбирается одна непротиворечивая цепочка:
     больше уверенных детекций, движение, близость к игрокам; переходы между треклетами — только с
     физически возможной скоростью;
  5. в разрывах цепочки до `interp_max_sec` положение интерполируется в метрах (состояние
     «оценка» — мяч закрыт или не найден, но был до и после рядом), дальше — «неизвестно».

Результат — положение мяча на каждом кадре в пикселях кадра и в метрах (X поперёк, Z вглубь) с
состоянием `detected` / `estimated` / None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .ground import GroundModel

DETECTED, ESTIMATED = "detected", "estimated"


@dataclass
class BallConfig:
    diameter_m: float = 0.21            # мяч №4 (дети 8–12 лет) ≈ 0.20–0.21 м, №5 ≈ 0.22 м
    min_score: float = 0.05
    size_max_factor: float = 2.6        # рамка YOLO вокруг крошечного мяча больше самого мяча
    size_max_pad: float = 5.0
    size_min_factor: float = 0.35
    size_min_factor_big: float = 0.7    # крупный (близкий) мяч рамка обводит точно: меньше — бутылка, бутса, кулак
    big_px: float = 10.0
    above_horizon_px: float = 4.0
    head_zone: float = 0.6              # верхняя доля рамки человека: мяч там — почти всегда голова/руки
    pitch_margin_m: float = 3.0         # за пределами поля дальше — не игровой мяч (плюс доля глубины: она шумит)
    pitch_margin_depth: float = 0.08
    # треклеты
    link_speed_ms: float = 22.0         # сильный удар ребёнка — до ~20 м/с
    link_slack_m: float = 0.6
    max_gap_frames: int = 6
    # неподвижные мячи
    static_radius_px: float = 5.0       # в координатах сцены; плюс доля размера мяча
    static_sec: float = 2.5             # мяч на том же месте в пределах стольких секунд — лежит
    static_side_sec: float = 0.7        # ...причём и раньше, и позже кадра хотя бы на столько
    static_window_sec: float = 4.0
    static_min_share: float = 0.04      # доля кадров окна с мячом на том же месте (запасной мяч находится с большими пропусками)
    near_player_m: float = 2.0
    # цепочка
    score_bias: float = 0.06            # слабее — кандидат вреднее, чем полезен (рядом с игроками)
    score_bias_far: float = 0.2         # вдали от игроков слабая детекция — чаще запасной мяч, конус, бутса зрителя
    move_bonus: float = 0.6
    move_min_m: float = 1.0
    near_bonus: float = 0.04
    chain_start_value: float = 0.5      # столько должна набрать цепочка сама по себе; слабые треклеты — только в цепочке
    single_min_score: float = 0.2       # короткий треклет (< min_frames) без детекции сильнее — не часть цепочки
    min_frames: int = 3
    link_max_sec: float = 3.0
    link_speed_chain_ms: float = 14.0   # средняя скорость через разрыв цепочки
    link_cost_per_sec: float = 0.15
    interp_max_sec: float = 1.5
    interp_min_people: int = 4          # в разрыве камера смотрит на игру (оператор не опустил телефон)
    # повторная детекция в разрывах цепочки на вырезке вокруг предсказания (см. predict_gaps)
    roi_gap_min_sec: float = 0.2
    roi_gap_max_sec: float = 3.0
    roi_size: int = 256                 # сторона вырезки, px кадра; сеть смотрит её со стороной roi_imgsz (×4)
    roi_imgsz: int = 1024
    roi_conf: float = 0.15
    roi_max_dist_px: float = 40.0       # кандидат дальше от предсказания — не тот мяч


@dataclass
class Tracklet:
    frames: list[int] = field(default_factory=list)
    idx: list[int] = field(default_factory=list)      # номер кандидата в кадре
    X: list[float] = field(default_factory=list)
    Z: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    near: list[bool] = field(default_factory=list)
    static: bool = False
    static_share: float = 0.0
    value: float = 0.0

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]


@dataclass
class BallTrack:
    """Мяч по кадрам: u, v — центр в пикселях кадра обработки; X, Z — метры (NaN — неизвестно)."""

    u: np.ndarray
    v: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    state: list[Optional[str]]
    score: np.ndarray
    static_spots: int = 0
    candidates: int = 0
    rejected: int = 0

    def summary(self) -> dict:
        n = max(len(self.state), 1)
        det = sum(s == DETECTED for s in self.state)
        est = sum(s == ESTIMATED for s in self.state)
        return {"frames": len(self.state), "detected_share": round(det / n, 3), "estimated_share": round(est / n, 3),
                "known_share": round((det + est) / n, 3), "static_candidates": self.static_spots,
                "candidates": self.candidates, "rejected_candidates": self.rejected}


def plausible(balls: np.ndarray, people: np.ndarray, ground: GroundModel, i: int, cfg: BallConfig,
              pitch=None) -> np.ndarray:
    """Маска правдоподобных кандидатов кадра i (balls — (M, 5), people — (N, 5)); pitch — `pitch.PitchSetup`."""
    balls = np.asarray(balls, float).reshape(-1, 5)
    if not len(balls):
        return np.zeros(0, bool)
    ok = balls[:, 4] >= cfg.min_score
    cx, cy = (balls[:, 0] + balls[:, 2]) / 2, (balls[:, 1] + balls[:, 3]) / 2
    size = np.maximum(balls[:, 2] - balls[:, 0], balls[:, 3] - balls[:, 1])
    h = ground.height_at(i, balls[:, 3])
    ok &= balls[:, 3] >= ground.horizon(i) + cfg.above_horizon_px
    d = np.maximum(h, 0.0) * cfg.diameter_m / ground.H
    ok &= size <= cfg.size_max_factor * d + cfg.size_max_pad
    ok &= size >= np.where(d >= cfg.big_px, cfg.size_min_factor_big, cfg.size_min_factor) * d
    people = np.asarray(people, float).reshape(-1, 5)
    for p in people:
        ph = p[3] - p[1]
        inside = (cx >= p[0]) & (cx <= p[2]) & (cy >= p[1]) & (cy <= p[1] + cfg.head_zone * ph)
        ok &= ~inside
    if pitch is not None and ok.any():
        X, Z = ground.ground(i, cx, balls[:, 3])
        PX, PY = pitch.frame_to_pitch(i, cx, balls[:, 3], ground)
        # по разметке поля метры точные — запас постоянный; по росту глубина шумит — запас растёт с ней
        m = cfg.pitch_margin_m + (0.0 if pitch.project is not None else cfg.pitch_margin_depth * np.nan_to_num(Z, nan=0.0))
        ok &= ~np.isnan(PX)
        ok &= (PX >= -m) & (PX <= pitch.length_m + m) & (PY >= -m) & (PY <= pitch.width_m + m)
    return ok


def static_candidates(balls: Sequence[np.ndarray], masks: Sequence[np.ndarray], ground: GroundModel, fps: float,
                      cfg: BallConfig) -> list[np.ndarray]:
    """Для каждого правдоподобного кандидата: лежит ли мяч на этом месте сцены (запасной мяч)."""
    n = len(balls)
    rows = []
    for i in range(n):
        b = np.asarray(balls[i], float).reshape(-1, 5)
        m = np.asarray(masks[i], bool)
        if not m.any():
            continue
        sx, sy = ground.scene_xy(i, (b[m, 0] + b[m, 2]) / 2, (b[m, 1] + b[m, 3]) / 2)
        size = np.maximum(b[m, 2] - b[m, 0], b[m, 3] - b[m, 1]) * ground.scene_scale(i)
        rows.append(np.stack([np.full(len(sx), i), sx, sy, size, np.flatnonzero(m)], axis=1))
    out = [np.zeros(len(np.asarray(balls[i]).reshape(-1, 5)), bool) for i in range(n)]
    if not rows:
        return out
    P = np.concatenate(rows)                       # frame, x, y, size, index — по возрастанию кадра
    win = int(cfg.static_window_sec * fps)
    need = cfg.static_sec * fps
    for row in P:
        i, x, y, sz, k = row
        lo, hi = np.searchsorted(P[:, 0], [i - win, i + win + 1])
        Q = P[lo:hi]
        near = Q[(Q[:, 0] != i) & (np.hypot(Q[:, 1] - x, Q[:, 2] - y) <= cfg.static_radius_px + 0.5 * sz)]
        if not len(near):
            continue
        frames = np.unique(near[:, 0])
        span = max(frames.max(), i) - min(frames.min(), i)
        window = min(n, i + win + 1) - max(0, i - win)
        side = cfg.static_side_sec * fps
        both = frames.min() <= i - side and frames.max() >= i + side
        if both and span >= need and len(frames) >= cfg.static_min_share * window:
            out[int(i)][int(k)] = True
    return out


def _near_players(X: float, Z: float, px: np.ndarray, pz: np.ndarray, radius: float) -> bool:
    if not len(px) or np.isnan(X):
        return False
    return bool(np.nanmin(np.hypot(px - X, pz - Z)) <= radius)


def build_tracklets(balls: Sequence[np.ndarray], people: Sequence[np.ndarray], ground: GroundModel, fps: float,
                    cfg: BallConfig, pitch=None) -> tuple[list[Tracklet], dict]:
    dt = 1.0 / fps
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    total = rejected = 0
    masks = [plausible(balls[i], people[i], ground, i, cfg, pitch) for i in range(len(balls))]
    static = static_candidates(balls, masks, ground, fps, cfg)
    stats = {"candidates": 0, "rejected": 0, "static": int(sum(int(x.sum()) for x in static))}
    for i in range(len(balls)):
        b = np.asarray(balls[i], float).reshape(-1, 5)
        p = np.asarray(people[i], float).reshape(-1, 5)
        total += len(b)
        mask = masks[i]
        rejected += int((~mask).sum())
        keep = np.flatnonzero(mask & ~static[i])
        cu = (b[keep, 0] + b[keep, 2]) / 2
        X, Z = ground.ground(i, cu, b[keep, 3])
        pX, pZ = ground.ground(i, (p[:, 0] + p[:, 2]) / 2, p[:, 3]) if len(p) else (np.zeros(0), np.zeros(0))
        # продолжение треклетов: ближайшие пары в метрах с гейтом по скорости
        pairs = []
        for ti, t in enumerate(active):
            gap = i - t.end
            if len(t.frames) >= 2 and t.frames[-1] - t.frames[-2] <= cfg.max_gap_frames:
                g0 = t.frames[-1] - t.frames[-2]
                vx, vz = (t.X[-1] - t.X[-2]) / g0, (t.Z[-1] - t.Z[-2]) / g0
            else:
                vx = vz = 0.0
            px_, pz_ = t.X[-1] + vx * gap, t.Z[-1] + vz * gap
            gate = cfg.link_speed_ms * gap * dt + cfg.link_slack_m
            for k in range(len(keep)):
                if np.isnan(X[k]):
                    continue
                d = float(np.hypot(X[k] - px_, Z[k] - pz_))
                # глубина мяча шумит сильнее поперечного положения — гейт по глубине шире
                if d <= gate * (1.0 + 0.1 * Z[k] / 10.0):
                    pairs.append((d, ti, k))
        pairs.sort()
        used_t, used_k = set(), set()
        for d, ti, k in pairs:
            if ti in used_t or k in used_k:
                continue
            used_t.add(ti)
            used_k.add(k)
            t = active[ti]
            t.frames.append(i), t.idx.append(int(keep[k])), t.X.append(float(X[k])), t.Z.append(float(Z[k]))
            t.scores.append(float(b[keep[k], 4])), t.near.append(_near_players(X[k], Z[k], pX, pZ, cfg.near_player_m))
        for k in range(len(keep)):
            if k in used_k or np.isnan(X[k]):
                continue
            active.append(Tracklet([i], [int(keep[k])], [float(X[k])], [float(Z[k])], [float(b[keep[k], 4])],
                                   [_near_players(X[k], Z[k], pX, pZ, cfg.near_player_m)]))
        still = []
        for t in active:
            (still if i - t.end < cfg.max_gap_frames else done).append(t)
        active = still
    done.extend(active)
    stats.update(candidates=total, rejected=rejected)
    return sorted(done, key=lambda t: t.start), stats


def _classify(t: Tracklet, fps: float, cfg: BallConfig) -> None:
    """Ценность треклета для цепочки: уверенность детекций, движение, близость к игрокам."""
    X, Z = np.array(t.X), np.array(t.Z)
    moved = float(np.hypot(X.max() - X.min(), Z.max() - Z.min())) if len(X) > 1 else 0.0
    near = np.array(t.near, bool)
    scores = np.array(t.scores)
    value = float(np.sum(scores - np.where(near, cfg.score_bias, cfg.score_bias_far))) + cfg.near_bonus * float(near.sum())
    if moved >= cfg.move_min_m:
        value += cfg.move_bonus
    if len(t.frames) < cfg.min_frames and max(t.scores) < cfg.single_min_score:
        value = -1.0
    if not near.any() and max(t.scores) < cfg.single_min_score:
        value = -1.0          # слабый мяч вдали от всех игроков — запасной, у зрителей, за полем
    t.value = value


def _link_ok(x0: float, z0: float, x1: float, z1: float, gap: int, fps: float, cfg: BallConfig) -> bool:
    """Мог ли мяч за `gap` кадров переместиться так (глубина дальнего мяча шумит — допуск растёт с глубиной)."""
    dist = float(np.hypot(x1 - x0, (z1 - z0) * 0.6))
    return dist <= cfg.link_speed_chain_ms * gap / fps + 1.5 + 0.04 * max(z0, z1)


def select_chain(tracklets: list[Tracklet], fps: float, cfg: BallConfig) -> list[Tracklet]:
    """Непротиворечивая цепочка треклетов с наибольшей суммой (без перекрытия по времени)."""
    cand = [t for t in tracklets if t.value > -0.5]
    cand.sort(key=lambda t: t.end)
    n = len(cand)
    best = np.zeros(n)
    prev = np.full(n, -1)
    ends = np.array([t.end for t in cand])
    max_gap = cfg.link_max_sec * fps
    for j, t in enumerate(cand):
        best[j] = t.value if t.value >= cfg.chain_start_value else -np.inf
        lo = int(np.searchsorted(ends, t.start - max_gap))
        for i in range(lo, j):
            s = cand[i]
            gap = t.start - s.end
            if gap <= 0:
                continue
            if not _link_ok(s.X[-1], s.Z[-1], t.X[0], t.Z[0], gap, fps, cfg):
                continue
            if not np.isfinite(best[i]):
                continue
            v = best[i] + t.value - cfg.link_cost_per_sec * gap / fps
            if v > best[j]:
                best[j], prev[j] = v, i
        # цепочку можно и начать заново: разрыв «ничего не знаем» ничего не стоит
    # лучшие непересекающиеся цепочки: берём лучшую, затем лучшие среди не пересекающихся с уже выбранными
    chosen: list[Tracklet] = []
    taken = np.zeros(n, bool)
    busy: list[tuple[int, int]] = []
    for j in np.argsort(-best):
        if taken[j] or not best[j] >= cfg.chain_start_value:
            continue
        chain, k = [], j
        while k >= 0:
            chain.append(k)
            k = prev[k]
        chain = [k for k in chain if not taken[k]]
        span = (cand[chain[-1]].start, cand[chain[0]].end)
        if any(not (span[1] < a or span[0] > b) for a, b in busy):
            # пересекается с уже выбранной: берём только отдельные треклеты, которые не пересекаются
            chain = [k for k in chain if all(cand[k].end < a or cand[k].start > b for a, b in busy)]
        for k in chain:
            taken[k] = True
            busy.append((cand[k].start, cand[k].end))
            chosen.append(cand[k])
    return sorted(chosen, key=lambda t: t.start)


def track_ball(balls: Sequence[np.ndarray], people: Sequence[np.ndarray], ground: GroundModel, fps: float,
               cfg: BallConfig | None = None, pitch=None) -> BallTrack:
    """balls[i], people[i] — (M, 5) кандидаты мяча и (N, 5) рамки людей кадра i (пиксели кадра обработки);
    pitch — схема поля (`pitch.PitchSetup`), если отмечена: мяч за пределами поля отбрасывается."""
    cfg = cfg or BallConfig()
    fps = fps or 30.0
    n = len(balls)
    tracklets, stats = build_tracklets(balls, people, ground, fps, cfg, pitch)
    for t in tracklets:
        _classify(t, fps, cfg)
    chain = select_chain(tracklets, fps, cfg)
    u = np.full(n, np.nan)
    v = np.full(n, np.nan)
    X = np.full(n, np.nan)
    Z = np.full(n, np.nan)
    score = np.zeros(n)
    state: list[Optional[str]] = [None] * n
    for t in chain:
        for f, k, x, z, s in zip(t.frames, t.idx, t.X, t.Z, t.scores):
            b = np.asarray(balls[f], float).reshape(-1, 5)[k]
            u[f], v[f] = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            X[f], Z[f], score[f], state[f] = x, z, s, DETECTED
    # разрывы до interp_max_sec, если мяч мог так переместиться: положение — по прямой в метрах и в сцене
    known = np.flatnonzero(~np.isnan(X))
    max_gap = cfg.interp_max_sec * fps
    people_n = np.array([len(np.asarray(p).reshape(-1, 5)) for p in people])
    for a, b in zip(known[:-1], known[1:]):
        if not (1 < b - a <= max_gap and _link_ok(X[a], Z[a], X[b], Z[b], int(b - a), fps, cfg)):
            continue
        if (people_n[a + 1:b] < cfg.interp_min_people).any():
            continue
        w = (np.arange(a + 1, b) - a) / (b - a)
        X[a + 1:b] = X[a] + (X[b] - X[a]) * w
        Z[a + 1:b] = Z[a] + (Z[b] - Z[a]) * w
        sa, sb = ground.scene_xy(a, u[a], v[a]), ground.scene_xy(b, u[b], v[b])
        for f, wf in zip(range(a + 1, b), w):
            sx, sy = sa[0] + (sb[0] - sa[0]) * wf, sa[1] + (sb[1] - sa[1]) * wf
            M = np.linalg.inv(ground.inv[f])
            u[f], v[f] = M[0, 0] * sx + M[0, 1] * sy + M[0, 2], M[1, 0] * sx + M[1, 1] * sy + M[1, 2]
            state[f] = ESTIMATED
    return BallTrack(u, v, X, Z, state, score, stats["static"], stats["candidates"], stats["rejected"])


def predict_gaps(track: BallTrack, ground: GroundModel, fps: float, cfg: BallConfig | None = None) -> dict[int, tuple[float, float]]:
    """Где искать мяч в разрывах цепочки: кадр -> (u, v) — линейное предсказание между известными положениями
    в координатах сцены (движение камеры убрано), переведённое в кадр. Разрывы короче roi_gap_min_sec
    и так интерполируются, длиннее roi_gap_max_sec — мяч мог улететь куда угодно."""
    cfg = cfg or BallConfig()
    known = [i for i, st in enumerate(track.state) if st == DETECTED]
    out: dict[int, tuple[float, float]] = {}
    for a, b in zip(known[:-1], known[1:]):
        gap = b - a
        if not (cfg.roi_gap_min_sec * fps <= gap <= cfg.roi_gap_max_sec * fps):
            continue
        sa, sb = ground.scene_xy(a, track.u[a], track.v[a]), ground.scene_xy(b, track.u[b], track.v[b])
        for f in range(a + 1, b):
            w = (f - a) / gap
            sx, sy = sa[0] + (sb[0] - sa[0]) * w, sa[1] + (sb[1] - sa[1]) * w
            M = np.linalg.inv(ground.inv[f])
            out[f] = (float(M[0, 0] * sx + M[0, 1] * sy + M[0, 2]), float(M[1, 0] * sx + M[1, 1] * sy + M[1, 2]))
    return out


def accept_roi(cands: np.ndarray, pred: tuple[float, float], people: np.ndarray, ground: GroundModel, i: int,
               cfg: BallConfig, pitch=None) -> np.ndarray:
    """Из кандидатов повторной детекции оставить правдоподобные и не дальше roi_max_dist_px от предсказания."""
    cands = np.asarray(cands, float).reshape(-1, 5)
    if not len(cands):
        return cands
    ok = plausible(cands, people, ground, i, cfg, pitch)
    cu, cv = (cands[:, 0] + cands[:, 2]) / 2, (cands[:, 1] + cands[:, 3]) / 2
    ok &= np.hypot(cu - pred[0], cv - pred[1]) <= cfg.roi_max_dist_px
    return cands[ok]
