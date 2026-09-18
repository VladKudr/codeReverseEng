"""Состав: кто на доске участвует в игре. Пометки оператора относятся к ЧЕЛОВЕКУ, а не к треку.

Трекер рвёт человека на фрагменты: игрок зашёл за другого, камера ушла и вернулась — новый номер трека (замер
17.09.2026, игра 7×7, 83 с: 385 треков на ~19 человек в кадре, у 92 % новых треков рядом есть только что оборванный).
Пометка «вне игры» на одном фрагменте поэтому «слетала»: человек возвращался на доску под новым номером.

Здесь фрагменты склеиваются в людей (`link_tracks`):
  * продолжение — новый трек начался вскоре после конца старого, там, куда тот двигался, того же роста и цвета формы;
    связь принимается только взаимно лучшая (у А лучший преемник — Б, у Б лучший предшественник — А);
  * стоящий человек — два неподвижных трека на одном месте сцены (зритель, тренер, запасной у бровки) — один человек,
    сколько бы времени ни прошло (положения — в координатах сцены, `camreg`);
  * двое одновременно в кадре одним человеком быть не могут — такая склейка отвергается.

Пометка (`Mark`) — событие на линии времени человека: «вне игры» / «в игре», целиком или с кадра `at` (замена: игрок
ушёл с поля — «вне игры с этого момента», вышедший — «в игре с этого момента»). Состояние человека на кадре — роль
последнего события не позже кадра; до первого события — его же роль, если оно «целиком», иначе — роль по умолчанию.
Роль по умолчанию задаёт режим состава: «все, кроме выключенных» (play) или «только отмеченные» (sideline).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

PLAY, SIDELINE = "play", "sideline"


@dataclass
class LinkConfig:
    max_gap_sec: float = 3.0            # продолжение ищется не дольше
    reach: float = 1.0                  # допустимый промах прогноза, в ростах человека, при нулевом разрыве…
    reach_per_sec: float = 2.0          # …и его рост с длиной разрыва
    height_ratio: float = 1.35          # рамки продолжения и оригинала — сопоставимого роста
    max_color: float = 40.0             # расстояние цвета формы (Lab)
    color_weight: float = 0.02          # вес цвета в стоимости связи (в ростах на единицу Lab)
    static_min_sec: float = 1.0         # «стоит»: трек не короче…
    static_spread: float = 0.6          # …и размах положений меньше стольких ростов
    static_radius: float = 0.5          # два стоящих трека — один человек, если ближе стольких ростов
    static_max_gap_sec: Optional[float] = None   # без точной привязки кадров место «уплывает» — ограничить разрыв
    max_overlap_frames: int = 2         # одновременно в кадре — разные люди


@dataclass
class TrackInfo:
    tid: int
    frames: np.ndarray                  # кадры, где трек виден
    x: np.ndarray                       # точка ног в координатах сцены
    y: np.ndarray
    h: np.ndarray                       # рост рамки в масштабе сцены
    color: Optional[np.ndarray] = None  # медианный цвет формы (Lab)
    box0: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])

    @property
    def height(self) -> float:
        return float(np.median(self.h))


def track_tracks(tracks, colors: Optional[Sequence[np.ndarray]], inv: Sequence[np.ndarray]) -> dict[int, TrackInfo]:
    """Сводка по каждому треку в координатах сцены. inv[i] — «кадр i -> сцена» (camreg или цепочка подобий)."""
    rows: dict[int, list] = {}
    cols: dict[int, list] = {}
    box0: dict[int, list] = {}
    n = min(len(tracks), len(inv))
    for i in range(n):
        ids = tracks.ids[i]
        if not len(ids):
            continue
        boxes = np.asarray(tracks.boxes[i], float).reshape(-1, 4)
        M = np.asarray(inv[i], float)
        P = M @ np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], np.ones(len(boxes))])
        w = np.where(np.abs(P[2]) > 1e-9, P[2], np.nan)
        s = np.sqrt(abs(np.linalg.det(M[:2, :2] / M[2, 2])))          # местный масштаб «кадр -> сцена»
        col = colors[i] if colors is not None and i < len(colors) else None
        det = tracks.det_index[i] if hasattr(tracks, "det_index") else None
        for k, t in enumerate(ids):
            t = int(t)
            rows.setdefault(t, []).append((i, P[0, k] / w[k], P[1, k] / w[k], (boxes[k, 3] - boxes[k, 1]) * s))
            box0.setdefault(t, [round(float(v), 1) for v in boxes[k]])
            if col is not None and det is not None:
                d = int(det[k])
                if 0 <= d < len(col) and not np.isnan(col[d]).any():
                    cols.setdefault(t, []).append(np.asarray(col[d], np.float32))
    out = {}
    for t, r in rows.items():
        A = np.array(r, float)
        A = A[np.isfinite(A).all(axis=1)]
        if not len(A):
            continue
        c = np.median(np.stack(cols[t]), axis=0) if cols.get(t) else None
        out[t] = TrackInfo(t, A[:, 0].astype(int), A[:, 1], A[:, 2], np.maximum(A[:, 3], 1.0), c, box0[t])
    return out


def _is_static(v: TrackInfo, fps: float, cfg: LinkConfig) -> bool:
    if v.n < cfg.static_min_sec * fps:
        return False
    # размах без выбросов: рамка человека у камеры дёргается, когда её режет край кадра
    sx, sy = np.subtract(*np.percentile(v.x, [90, 10])), np.subtract(*np.percentile(v.y, [90, 10]))
    return float(np.hypot(sx, sy)) < cfg.static_spread * v.height


def _velocity(v: TrackInfo, at_end: bool, k: int = 8) -> np.ndarray:
    """Скорость у конца (или начала) трека, точек сцены на кадр; ограничена четвертью роста за кадр."""
    if v.n < 3:
        return np.zeros(2)
    sl = slice(-k, None) if at_end else slice(0, k)
    f, x, y = v.frames[sl], v.x[sl], v.y[sl]
    dt = max(int(f[-1] - f[0]), 1)
    vel = np.array([x[-1] - x[0], y[-1] - y[0]]) / dt
    lim = 0.25 * v.height
    sp = float(np.hypot(*vel))
    return vel * (lim / sp) if sp > lim else vel


class _Groups:
    """Объединение фрагментов в людей с запретом на одновременное присутствие."""

    def __init__(self, info: dict[int, TrackInfo], max_overlap: int):
        self.parent = {t: t for t in info}
        self.frames = {t: set(v.frames.tolist()) for t, v in info.items()}
        self.max_overlap = max_overlap

    def find(self, t: int) -> int:
        while self.parent[t] != t:
            self.parent[t] = self.parent[self.parent[t]]
            t = self.parent[t]
        return t

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        small, big = sorted((ra, rb), key=lambda r: len(self.frames[r]))
        if len(self.frames[small] & self.frames[big]) > self.max_overlap:
            return False
        self.parent[small] = big
        self.frames[big] |= self.frames.pop(small)
        return True


def link_tracks(info: dict[int, TrackInfo], fps: float, cfg: LinkConfig | None = None) -> tuple[dict[int, int], dict]:
    """Человек (номер группы) для каждого трека. Возвращает ({трек: человек}, сводка)."""
    cfg = cfg or LinkConfig()
    fps = fps or 30.0
    max_gap = int(cfg.max_gap_sec * fps)
    order = sorted(info.values(), key=lambda v: v.start)
    # --- продолжения: стоимость каждой допустимой пары «конец А -> начало Б» --------------------------------
    cand: list[tuple[float, int, int]] = []
    for b in order:
        for a in order:
            gap = b.start - a.end
            if a.tid == b.tid or gap <= 0 or gap > max_gap:
                continue
            ha, hb = float(np.median(a.h[-8:])), float(np.median(b.h[:8]))
            if max(ha, hb) / max(min(ha, hb), 1e-6) > cfg.height_ratio:
                continue
            h = (ha + hb) / 2
            # встречный прогноз: А вперёд и Б назад на половину разрыва
            pa = np.array([a.x[-1], a.y[-1]]) + _velocity(a, True) * gap / 2
            pb = np.array([b.x[0], b.y[0]]) - _velocity(b, False) * gap / 2
            d = float(np.hypot(*(pa - pb))) / h
            if d > cfg.reach + cfg.reach_per_sec * gap / fps:
                continue
            dc = float(np.linalg.norm(a.color - b.color)) if a.color is not None and b.color is not None else 0.0
            if dc > cfg.max_color:
                continue
            cand.append((d + cfg.color_weight * dc + 0.3 * gap / fps, a.tid, b.tid))
    best_next: dict[int, tuple[float, int]] = {}
    best_prev: dict[int, tuple[float, int]] = {}
    for c, a, b in cand:
        if a not in best_next or c < best_next[a][0]:
            best_next[a] = (c, b)
        if b not in best_prev or c < best_prev[b][0]:
            best_prev[b] = (c, a)
    links = sorted((c, a, b) for c, a, b in cand if best_next[a][1] == b and best_prev[b][1] == a)
    groups = _Groups(info, cfg.max_overlap_frames)
    n_succ = sum(groups.union(a, b) for _, a, b in links)
    # --- стоящие на одном месте -----------------------------------------------------------------------------
    static = [v for v in order if _is_static(v, fps, cfg)]
    pairs = []
    for i, a in enumerate(static):
        pa = np.array([np.median(a.x), np.median(a.y)])
        for b in static[i + 1:]:
            if cfg.static_max_gap_sec is not None and b.start - a.end > cfg.static_max_gap_sec * fps:
                continue
            h = (a.height + b.height) / 2
            if max(a.height, b.height) / min(a.height, b.height) > cfg.height_ratio:
                continue
            d = float(np.hypot(*(pa - [np.median(b.x), np.median(b.y)]))) / h
            if d <= cfg.static_radius:
                pairs.append((d, a.tid, b.tid))
    n_static = sum(groups.union(a, b) for _, a, b in sorted(pairs))
    person = {t: groups.find(t) for t in info}
    return person, {"tracks": len(info), "people": len(set(person.values())), "links": int(n_succ),
                    "static_links": int(n_static), "static_tracks": len(static)}


# --- пометки оператора -> состояние по кадрам -------------------------------------------------------------------
@dataclass
class Mark:
    tid: int                 # трек, по которому щёлкнули (уже сопоставленный с текущим журналом треков)
    role: str                # play | sideline
    at: Optional[int] = None  # None — человек целиком; кадр — с этого момента (замена)


def _merge(intervals: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def off_intervals(info: dict[int, TrackInfo], person: dict[int, int], marks: Sequence[Mark],
                  default: str = PLAY) -> dict[int, list[list[int]]]:
    """Для каждого трека — отрезки кадров [с, по], где человек вне игры (пусто — всё время в игре)."""
    events: dict[int, list[tuple[int, int, str]]] = {}
    for m in marks:
        if m.tid not in info:
            continue
        whole = m.at is None
        frame = info[m.tid].start if whole else int(m.at)
        events.setdefault(person[m.tid], []).append((frame, 0 if whole else 1, m.role))
    out: dict[int, list[list[int]]] = {}
    for t, v in info.items():
        ev = sorted(events.get(person[t], []))
        if not ev:
            out[t] = [[v.start, v.end]] if default == SIDELINE else []
            continue
        before = ev[0][2] if ev[0][1] == 0 else default      # «целиком» действует и назад, «с момента» — нет
        cuts = [(-1, before)] + [(f, r) for f, _, r in ev]
        spans = []
        for k, (f, r) in enumerate(cuts):
            nxt = cuts[k + 1][0] - 1 if k + 1 < len(cuts) else 10 ** 9
            if r == SIDELINE:
                a, b = max(f, v.start), min(nxt, v.end)
                if a <= b:
                    spans.append((a, b))
        out[t] = _merge(spans)
    return out


def is_off(intervals: Sequence[Sequence[int]], frame: int) -> bool:
    return any(a <= frame <= b for a, b in intervals)


def off_share(v: TrackInfo, intervals: Sequence[Sequence[int]]) -> float:
    if not intervals or not v.n:
        return 0.0
    f = v.frames
    return float(np.mean(np.any([(f >= a) & (f <= b) for a, b in intervals], axis=0)))


# --- состав целиком: пометки -> люди -> похожие по внешности ------------------------------------------------------
@dataclass
class RosterConfig:
    implicit_handicap: float = 0.02     # «не отмеченные на кадре разметки» — примеры слабее явных пометок
    margin: float = 0.0                 # насколько сходство с примерами «не по умолчанию» должно превышать обратное
    min_frames: int = 10                # короче — внешность не оценивается (на доску такие треки и не попадают)


def track_features_frames(tracks, per_frame: Sequence[np.ndarray]) -> dict[int, np.ndarray]:
    """То же, что `track_features`, но признаки заданы по кадрам: per_frame[i][d] — детекция d кадра i."""
    acc: dict[int, np.ndarray] = {}
    for i, (ids, det) in enumerate(zip(tracks.ids, tracks.det_index)):
        if i >= len(per_frame):
            break
        F = per_frame[i]
        for t, d in zip(ids, det):
            if 0 <= int(d) < len(F):
                f = np.asarray(F[int(d)], np.float32)
                n = float(np.linalg.norm(f))
                if n > 0 and np.isfinite(n):
                    acc[int(t)] = acc.get(int(t), 0) + f / n
    return {t: v / max(float(np.linalg.norm(v)), 1e-9) for t, v in acc.items()}


def track_features(tracks, features, offsets) -> dict[int, np.ndarray]:
    """Средний нормированный признак внешности каждого трека. features — признаки всех детекций ролика подряд,
    offsets[i] — начало детекций кадра i (как в inputs.npz)."""
    acc: dict[int, np.ndarray] = {}
    for i, (ids, det) in enumerate(zip(tracks.ids, tracks.det_index)):
        if i + 1 >= len(offsets):
            break
        for t, d in zip(ids, det):
            j = int(offsets[i]) + int(d)
            if d < 0 or j >= int(offsets[i + 1]):
                continue
            f = np.asarray(features[j], np.float32)
            n = float(np.linalg.norm(f))
            if n > 0 and np.isfinite(n):
                acc[int(t)] = acc.get(int(t), 0) + f / n
    return {t: v / max(float(np.linalg.norm(v)), 1e-9) for t, v in acc.items()}


def resolve(info: dict[int, TrackInfo], marks: Sequence[Mark], fps: float, default: str = PLAY,
            feats: Optional[dict[int, np.ndarray]] = None, mark_frames: Sequence[int] = (),
            link_cfg: LinkConfig | None = None, cfg: RosterConfig | None = None) -> dict:
    """Кто вне игры и почему. Возвращает {"off": {трек: [[с, по], …]}, "why": {трек: причина}, "person": {…}, "stats"}.

    Причины: manual — пометка на самом треке; person — тот же человек, что помеченный (склейка фрагментов);
    similar — похож по внешности на помеченных; default — режим состава («только отмеченные»)."""
    cfg = cfg or RosterConfig()
    person, stats = link_tracks(info, fps, link_cfg)
    marks = [m for m in marks if m.tid in info]
    marked_people = {person[m.tid] for m in marks}
    other = SIDELINE if default == PLAY else PLAY
    # --- похожие по внешности: примеры — помеченные люди (все их фрагменты) и «остальные на кадрах разметки» -----
    similar: set[int] = set()
    if feats:
        role_of = {}
        for m in marks:
            if m.at is None:                                  # «с этого момента» — человек и то и другое, не пример
                role_of.setdefault(person[m.tid], m.role)
        ex = {PLAY: [], SIDELINE: []}
        for t, v in info.items():
            r = role_of.get(person[t])
            if r is not None and t in feats and v.n >= cfg.min_frames:
                ex[r].append((feats[t], 0.0))
        for f in mark_frames:
            for t, v in info.items():
                if person[t] not in marked_people and t in feats and v.n >= cfg.min_frames and v.start <= f <= v.end:
                    ex[default].append((feats[t], cfg.implicit_handicap))
        if ex[other] and ex[default]:
            A = {r: (np.stack([e[0] for e in ex[r]]), np.array([e[1] for e in ex[r]])) for r in ex}
            groups: dict[int, list[int]] = {}
            for t in info:
                groups.setdefault(person[t], []).append(t)
            for p, members in groups.items():
                if p in marked_people:
                    continue
                vec = [feats[t] * info[t].n for t in members if t in feats and info[t].n >= cfg.min_frames]
                if not vec:
                    continue
                q = np.sum(vec, axis=0)
                q /= max(float(np.linalg.norm(q)), 1e-9)
                score = {r: float(np.max(A[r][0] @ q - A[r][1])) for r in A}
                if score[other] > score[default] + cfg.margin:
                    similar.add(p)
    # --- итог -----------------------------------------------------------------------------------------------
    hard = off_intervals(info, person, marks, default)
    own = {m.tid for m in marks}
    off, why = {}, {}
    for t, v in info.items():
        p = person[t]
        if p in marked_people:
            off[t], why[t] = hard[t], ("manual" if t in own else "person")
        elif p in similar:
            off[t], why[t] = ([[v.start, v.end]] if other == SIDELINE else []), "similar"
        else:
            off[t], why[t] = ([[v.start, v.end]] if default == SIDELINE else []), "default"
    stats.update({"marks": len(marks), "similar_people": len(similar)})
    return {"off": off, "why": why, "person": person, "stats": stats}
