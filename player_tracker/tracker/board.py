"""Тактическая доска: все игроки, игрок в фокусе и мяч на схеме поля по времени ролика.

Источники — то, что уже сохранено прогоном:
  * журнал мультитрекера (`tracks.npz`): id и рамка каждого трека на каждом кадре и номер детекции
    (по нему — цвет торса из `inputs.npz`);
  * плоскость поля (`ground.GroundModel`): ноги игрока -> метры (X поперёк, Z вглубь от камеры);
  * мяч (`ball.BallTrack`), решения о цели (`track.json`) и схема поля (`pitch.PitchSetup`).

Команды — кластеры цвета формы по игрокам ролика. Сначала отсекаются взрослые: рамка судьи, тренера
или зрителя выше ожидаемого роста ребёнка в этой точке земли (`ground.height_at`) на 17–33 %, у детей
отношение 0.84–1.06 — порог 1.12 по медиане трека. Оставшиеся дети делятся на два кластера по
median-цвету торса; если кластеры не разделяются (центры ближе двух разбросов) или в какой-то
«команде» на поле меньше двух человек, команды не назначаются и причина пишется в `teams_note`.
«Свои» — команда игрока в фокусе. Вратарь узнаётся по положению (почти всё время у одной из линий
ворот, малый разброс) и получает команду по стороне: у своих ворот — свои, у чужих — соперник; его
форма другого цвета кластеризацию не портит.

Не все люди в кадре играют: у бровки стоят запасные, тренеры, зрители у камеры и дети с соседнего поля.
Их пометка «вне игры» — ручная: на доске видны все, оператор выключает неактивных кликом по точке
(`manual_roles`), выключенные не попадают ни в игровой контекст, ни в метрики («ближайший игрок»).
Автоматическое правило по следу на поле (`classify_roles`) есть, но на реальном ролике ошибалось и по
умолчанию выключено (`BoardConfig.auto_roles`).

Ручная пометка хранится не только по id трека, но и по его началу (кадр и рамка, `track_signature`):
после поправки цели мультитрекер нумерует треки заново, и пометка находит «своего» человека
(`resolve_roles`), а не переезжает на соседа.

У каждого трека есть короткий номер (`display_numbers`: 1, 2, 3… по порядку появления) — один и тот же
над рамкой на видео и над точкой на доске, чтобы было видно, кого выключать.

Точность — как у метрик: оценка ±20–30 % в метрах плюс неточность отметок на схеме. Доска показывает
расстановку и перемещения, а не сантиметры.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .ball import DETECTED, BallTrack
from .ground import GroundModel
from .pitch import PitchSetup
from .team import OTHER, UNKNOWN

TEAM_OWN, TEAM_OPP = "own", "opp"


@dataclass
class BoardConfig:
    step: int = 3                    # кадр доски — каждый N-й кадр ролика (30 fps -> 10 fps)
    outside_margin_m: float = 3.0       # при разметке поля на кадре: дальше за бровкой — «за полем»
    outside_share: float = 0.8          # доля времени за полем, с которой человек предлагается как «вне игры»
    outside_min_frames: int = 45
    smooth_frames: int = 15          # сглаживание положения трека (кадры ролика)
    smooth_board: int = 5            # сглаживание цели и мяча (кадры доски)
    min_track_frames: int = 10       # короче — мелькнувшая ложная детекция
    auto_roles: bool = False         # автоматически помечать «вне игры» (ошибалось на реальном ролике — выключено)
    team_min_frames: int = 15        # трек короче — цвет формы ненадёжен, команда не назначается
    adult_ratio: float = 1.12        # рост рамки / ожидаемый рост ребёнка выше — взрослый (судья, тренер, зритель)
    adult_min_frames: int = 60       # ...если трек живёт хотя бы столько кадров
    team_min_separation: float = 2.0 # центры кластеров ближе (в разбросах) — формы не различимы, команды не назначать
                                     # (красная и белая формы на IMG_7463 разделяются на 2.5; 3.0 из team.py — для детекций, не треков)
    team_min_presence: float = 2.0   # в «команде» одновременно на поле меньше людей — это не команда
    gk_line: float = 0.33            # вратарь: доля длины поля от линии ворот (схема на глаз ошибается на метры)...
    gk_share: float = 0.6            # ...где он проводит не меньше этой доли времени...
    gk_spread_m: float = 8.0         # ...с разбросом положения не больше этого
    gk_min_sec: float = 4.0
    gk_overlap: float = 0.2          # у ворот один вратарь: второй трек там же — только если по времени почти не пересекается
    gk_band: float = 0.25            # вратарь стоит в створе: медиана y в пределах ±25 % ширины от середины
    gk_action_m: float = 10.0        # ...и далеко от места игры (нападающий у чужих ворот — рядом с игрой)
    # владение мячом с гистерезисом (game_context)
    poss_radius_m: float = 2.0       # ближе — претендент на владение...
    poss_depth_factor: float = 0.05  # ...радиус растёт с глубиной (положение дальних игроков шумит)
    poss_switch_m: float = 1.0       # смена владельца: новый ближе прежнего на столько...
    poss_switch_frames: int = 5      # ...столько кадров доски подряд (0.5 с)
    poss_keep_frames: int = 10       # никого в радиусе — прежний владелец держит мяч ещё столько кадров
    # «вне игры»: запасные, тренеры, зрители у камеры (см. classify_roles)
    role_min_sec: float = 4.0        # короткий трек не классифицируем: за пару секунд и игрок стоит на месте
    edge_m: float = 2.0              # «у бровки» — ближе этого к боковой линии (лицевые не считаются: там вратарь)
    outside_m: float = 1.0           # «за полем» — дальше этого за линией
    outside_share: float = 0.5
    still_rate: float = 0.25         # разброс положения на секунду жизни трека, м/с: меньше — стоит
    still_edge_share: float = 0.3
    idle_edge_share: float = 0.35    # держится у бровки...
    idle_action_m: float = 16.0      # ...и далеко от места игры (медиана по всем игрокам кадра)
    idle_rate: float = 0.8


@dataclass
class TrackFrames:
    """Журнал мультитрекера по кадрам: id, рамка (кадр обработки), номер детекции (-1 — нет)."""

    ids: list[np.ndarray]
    boxes: list[np.ndarray]
    det_index: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.ids)


def save_tracks(path, log) -> None:
    """`offline.TrackLog` -> .npz (без дескрипторов: они уже в inputs.npz)."""
    frames = log.frames
    np.savez_compressed(
        path,
        frame=np.array([fl.frame_idx for fl in frames], np.int32),
        counts=np.array([len(fl.track_ids) for fl in frames], np.int32),
        ids=np.array([i for fl in frames for i in fl.track_ids], np.int32),
        boxes=np.concatenate([fl.boxes.reshape(-1, 4) for fl in frames]).astype(np.float32) if frames else np.zeros((0, 4), np.float32),
        det_index=np.array([(-1 if d is None else d) for fl in frames for d in (fl.det_index or [None] * len(fl.track_ids))], np.int32),
    )


def load_tracks(path) -> TrackFrames:
    data = np.load(path)
    counts = data["counts"]
    pos = np.concatenate([[0], np.cumsum(counts)])
    ids, boxes, det = data["ids"], data["boxes"], data["det_index"]
    return TrackFrames([ids[a:b] for a, b in zip(pos[:-1], pos[1:])], [boxes[a:b] for a, b in zip(pos[:-1], pos[1:])],
                       [det[a:b] for a, b in zip(pos[:-1], pos[1:])])


def display_numbers(tracks: TrackFrames, min_frames: int = 10) -> dict[int, int]:
    """Короткие номера для показа: треки не короче min_frames кадров, по порядку появления -> 1, 2, 3…
    Зависят только от журнала треков, поэтому на видео и на доске совпадают."""
    first: dict[int, int] = {}
    count: dict[int, int] = defaultdict(int)
    for i, ids in enumerate(tracks.ids):
        for t in ids:
            t = int(t)
            first.setdefault(t, i)
            count[t] += 1
    order = sorted((f, t) for t, f in first.items() if count[t] >= min_frames)
    return {t: k + 1 for k, (_, t) in enumerate(order)}


def track_signatures(tracks: TrackFrames) -> dict[int, tuple[int, list[float]]]:
    """Начало каждого трека: (первый кадр, рамка на нём) — чтобы узнать того же человека после перенумерации."""
    out: dict[int, tuple[int, list[float]]] = {}
    for i, (ids, boxes) in enumerate(zip(tracks.ids, tracks.boxes)):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        for t, b in zip(ids, boxes):
            if int(t) not in out:
                out[int(t)] = (i, [round(float(v), 1) for v in b])
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def resolve_marks(stored: Optional[dict], signatures: dict[int, tuple[int, list[float]]]) -> dict[int, dict]:
    """Как `resolve_roles`, но возвращает пометку целиком ({"role", "at", "frame", …}) — для состава (`roster`)."""
    return resolve_roles(stored, signatures, full=True)


def resolve_roles(stored: Optional[dict], signatures: dict[int, tuple[int, list[float]]],
                  max_shift: int = 2, min_iou: float = 0.5, full: bool = False) -> dict:
    """Ручные пометки -> текущие id треков. stored: {id: {"role", "f", "box"}} (или старый вид {id: role}).
    Если трек с тем же id начинается там же — пометка его; иначе ищется трек, начавшийся на том же кадре
    (±max_shift) с перекрывающейся рамкой; не нашёлся — пометка пропускается (человек исчез после поправки)."""
    out: dict[int, str] = {}
    by_start: dict[int, list[int]] = defaultdict(list)
    for t, (f, _) in signatures.items():
        by_start[f].append(t)
    for key, val in (stored or {}).items():
        tid = int(key)
        if isinstance(val, str):                      # пометка до появления подписей — только по id
            if tid in signatures:
                out[tid] = {"role": val} if full else val
            continue
        role, f0, box0 = (val if full else val.get("role")), val.get("f"), val.get("box")
        sig = signatures.get(tid)
        if sig is not None and (f0 is None or (sig[0] == f0 and _iou(sig[1], box0) >= min_iou)):
            out[tid] = role
            continue
        best, best_iou = None, min_iou
        for f in range(f0 - max_shift, f0 + max_shift + 1):
            for t in by_start.get(f, []):
                v = _iou(signatures[t][1], box0)
                if v >= best_iou:
                    best, best_iou = t, v
        if best is not None:
            out[best] = role
    return out


def track_off(meta: dict, frame: int) -> bool:
    """Человек трека вне игры на этом кадре (`off` — отрезки кадров; без них — по роли трека целиком)."""
    spans = meta.get("off")
    if spans is None:
        return meta.get("role") == "sideline"
    return any(a <= frame <= b for a, b in spans)


def _lab_to_hex(lab: np.ndarray) -> str:
    import cv2

    px = np.clip(np.asarray(lab, np.float32), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _smooth_rows(rows: list[tuple], win: int) -> list[tuple]:
    """Сглаживание ряда (кадр доски, x, y, ...) с разрывом там, где кадры не подряд."""
    if not rows:
        return []
    out = []
    k = np.array([r[0] for r in rows])
    cuts = list(np.flatnonzero(np.diff(k) > 2) + 1)
    for a, b in zip([0] + cuts, cuts + [len(rows)]):
        xs = _smooth(np.array([r[1] for r in rows[a:b]]), win)
        ys = _smooth(np.array([r[2] for r in rows[a:b]]), win)
        out += [(rows[a + j][0], float(xs[j]), float(ys[j]), *rows[a + j][3:]) for j in range(b - a)]
    return out


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x
    half = min(win // 2, len(x) - 1)
    padded = np.concatenate([np.repeat(x[0], half), x, np.repeat(x[-1], half)])
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")


def _kmeans(data: np.ndarray, k: int, weights: Optional[np.ndarray] = None, iters: int = 40,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """k-means с инициализацией «самые далёкие точки» (k-means++ без sklearn)."""
    rng = np.random.default_rng(seed)
    w = np.ones(len(data)) if weights is None else np.asarray(weights, float)
    centers = [data[int(rng.integers(len(data)))]]
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(data[:, None, :] - np.stack(centers)[None], axis=2), axis=1) * w
        centers.append(data[int(np.argmax(d))])
    C = np.stack(centers)
    labels = np.zeros(len(data), int)
    for _ in range(iters):
        new = np.argmin(np.linalg.norm(data[:, None, :] - C[None], axis=2), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = np.average(data[m], axis=0, weights=w[m])
    return C, labels


def assign_teams(track_colors: dict[int, list], track_frames: dict[int, list], cfg: BoardConfig,
                 height_ratio: Optional[dict[int, float]] = None) -> tuple[dict[int, int], dict[int, np.ndarray], str]:
    """Команда каждого трека: взрослые — «прочие», дети — два кластера цвета формы.

    Возвращает (трек -> команда, центры Lab по командам, примечание — почему команды не назначены)."""
    height_ratio = height_ratio or {}
    ids = [t for t, cs in track_colors.items() if len(cs) >= cfg.team_min_frames]
    adults = {t for t in ids if height_ratio.get(t, 1.0) >= cfg.adult_ratio
              and len(track_frames.get(t, [])) >= cfg.adult_min_frames}
    kids = [t for t in ids if t not in adults]
    team_of = {t: OTHER for t in adults}
    centers_out: dict[int, np.ndarray] = {}
    if adults:
        centers_out[OTHER] = np.median(np.stack([np.median(np.stack(track_colors[t]), axis=0) for t in adults]), axis=0)
    if len(kids) < 4:
        return team_of, centers_out, "слишком мало игроков-детей, чтобы различить команды"
    med = np.stack([np.median(np.stack(track_colors[t]), axis=0) for t in kids])
    weights = np.array([len(track_frames.get(t, [])) for t in kids], float)
    centers, labels = _kmeans(med, 2, weights)
    within = np.linalg.norm(med - centers[labels], axis=1)
    spread = float(np.median(within)) + 1e-6
    separation = float(np.linalg.norm(centers[0] - centers[1])) / spread
    if separation < cfg.team_min_separation:
        return team_of, centers_out, f"формы команд не различаются по цвету (разделение {separation:.1f} < {cfg.team_min_separation})"
    presence = np.zeros(2)
    for j in range(2):
        frames = [f for t, lab in zip(kids, labels) if lab == j for f in track_frames.get(t, [])]
        if frames:
            counts = np.bincount(np.asarray(frames, int))
            presence[j] = counts[counts > 0].mean()
    if presence.min() < cfg.team_min_presence:
        return team_of, centers_out, f"в одной из групп одновременно на поле меньше {cfg.team_min_presence:g} человек"
    for t, lab in zip(kids, labels):
        team_of[t] = int(lab)
    centers_out[0], centers_out[1] = centers[0], centers[1]
    return team_of, centers_out, ""


def find_goalkeepers(per_track: dict[int, list], pitch: PitchSetup, fps: float, cfg: BoardConfig,
                     adults: set) -> dict[int, str]:
    """Вратари по положению: почти всё время у одной линии ворот, в створе (по ширине), с малым разбросом и
    далеко от места игры. -> {трек: "left"|"right"}. Нападающий, застоявшийся у чужих ворот, отличается тем,
    что игра рядом с ним и он обычно не в створе; зритель у камеры — тем, что он у боковой линии.

    У каждых ворот вратарь один: из кандидатов берётся самый долгий, остальные — только если по времени с
    уже выбранными почти не пересекаются (тот же вратарь, трек прервался). Защитник, постоявший у ворот
    одновременно с вратарём, вратарём не станет."""
    by_frame: dict[int, list] = defaultdict(list)
    for pts in per_track.values():
        for f, x, y in pts:
            by_frame[int(f)].append((x, y))
    action = {f: np.median(np.array(p), axis=0) for f, p in by_frame.items()}
    cands: dict[str, list] = {"left": [], "right": []}
    for tid, pts in per_track.items():
        if tid in adults or len(pts) < cfg.gk_min_sec * fps:
            continue
        arr = np.array(pts, float)
        x, y = arr[:, 1], arr[:, 2]
        near_left = float(np.mean(x < cfg.gk_line * pitch.length_m))
        near_right = float(np.mean(x > (1 - cfg.gk_line) * pitch.length_m))
        spread = float(np.percentile(np.hypot(x - np.median(x), y - np.median(y)), 90))
        in_band = abs(float(np.median(y)) - pitch.width_m / 2) <= cfg.gk_band * pitch.width_m
        to_action = float(np.median([np.hypot(xi - action[int(f)][0], yi - action[int(f)][1]) for f, xi, yi in pts]))
        if in_band and to_action >= cfg.gk_action_m and spread <= cfg.gk_spread_m and max(near_left, near_right) >= cfg.gk_share:
            side = "left" if near_left >= near_right else "right"
            cands[side].append((len(pts), tid, set(int(f) for f in arr[:, 0])))
    out = {}
    for side, items in cands.items():
        taken: set = set()
        for _, tid, frames in sorted(items, key=lambda r: -r[0]):
            if len(frames & taken) <= cfg.gk_overlap * len(frames):
                out[tid] = side
                taken |= frames
    return out


def classify_roles(per_track: dict[int, list], pitch: PitchSetup, fps: float, cfg: BoardConfig,
                  manual: Optional[dict] = None, protect: Optional[set] = None) -> dict[int, tuple[str, str]]:
    """Роль трека: ("play", "") или ("sideline", причина). per_track[tid] — [(кадр, x, y на поле, м)]."""
    manual = {int(k): v for k, v in (manual or {}).items()}
    protect = protect or set()
    L, W = pitch.length_m, pitch.width_m
    # «место игры» на кадре — медиана положений всех треков (устойчива к паре стоящих с краю)
    action: dict[int, np.ndarray] = {}
    by_frame: dict[int, list] = defaultdict(list)
    for tid, pts in per_track.items():
        for f, x, y in pts:
            by_frame[f].append((x, y))
    for f, pts in by_frame.items():
        action[f] = np.median(np.array(pts), axis=0)

    roles: dict[int, tuple[str, str]] = {}
    for tid, pts in per_track.items():
        if tid in manual:
            roles[tid] = (manual[tid], "выключен вручную" if manual[tid] == "sideline" else "включён вручную")
            continue
        if not cfg.auto_roles:
            roles[tid] = ("play", "")
            continue
        arr = np.array(pts, float)
        f, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
        dur = (f[-1] - f[0]) / fps
        if tid in protect or dur < cfg.role_min_sec:
            roles[tid] = ("play", "")
            continue
        m = cfg.outside_m
        outside = float(np.mean((x < -m) | (x > L + m) | (y < -m) | (y > W + m)))
        # «у бровки» — только боковые линии: у линии ворот в центре стоит вратарь, его убирать нельзя
        e = cfg.edge_m
        edge = float(np.mean((y < e) | (y > W - e)))
        spread = float(np.percentile(np.hypot(x - np.median(x), y - np.median(y)), 90))
        rate = spread / max(dur, 1e-6)
        dist = [float(np.hypot(*(np.array([xi, yi]) - action[int(fi)]))) for fi, xi, yi in pts if int(fi) in action]
        to_action = float(np.median(dist)) if dist else 0.0
        if outside >= cfg.outside_share:
            roles[tid] = ("sideline", "почти всё время за пределами поля")
        elif rate < cfg.still_rate and edge >= cfg.still_edge_share:
            roles[tid] = ("sideline", "стоит у бровки")
        elif edge >= cfg.idle_edge_share and to_action >= cfg.idle_action_m and rate < cfg.idle_rate:
            roles[tid] = ("sideline", "держится у бровки далеко от игры")
        else:
            roles[tid] = ("play", "")
    return roles


def play_feet(tracks: TrackFrames, ground: GroundModel, manual_roles: Optional[dict] = None,
              cfg: BoardConfig | None = None) -> dict[str, np.ndarray]:
    """Точки ног детей-участников по всему ролику: кадр `f`, точка кадра `u`, `v`, метры от камеры `X`, `Z`.

    Без взрослых (рамка выше ожидаемого роста ребёнка), без выключенных оператором и без мелькнувших треков:
    зрители и тренеры у камеры растягивали бы охват. Основа подгонки схемы (`pitch.fit_to_occupancy`) и
    подсказки «где играли» в разметке поля. `h` — высота рамки (NaN, если обрезана
    краем кадра), `t` — трек, `spread` — подвижность трека (размах положений, м)."""
    cfg = cfg or BoardConfig()
    n = min(len(tracks), len(ground))
    frame_h = float(max((np.asarray(b, float).reshape(-1, 4)[:, 3].max() for b in tracks.boxes if len(b)), default=1e9))
    rows: dict[int, list] = defaultdict(list)
    ratios: dict[int, list] = defaultdict(list)
    for i in range(n):
        boxes = np.asarray(tracks.boxes[i], float).reshape(-1, 4)
        if not len(boxes):
            continue
        u, v = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]
        X, Z = ground.ground(i, u, v)
        expected = ground.height_at(i, v)
        for k, tid in enumerate(tracks.ids[i]):
            t = int(tid)
            if np.isfinite(X[k]) and np.isfinite(Z[k]):
                whole = boxes[k, 3] < frame_h - 2 and boxes[k, 1] > 2          # рамка не обрезана краем кадра
                rows[t].append((i, float(u[k]), float(v[k]), float(X[k]), float(Z[k]),
                                float(boxes[k, 3] - boxes[k, 1]) if whole else np.nan))
            if expected[k] > 8 and boxes[k, 3] < frame_h - 2 and boxes[k, 1] > 2:
                ratios[t].append(float((boxes[k, 3] - boxes[k, 1]) / expected[k]))
    off = {t for t, role in resolve_roles(manual_roles, track_signatures(tracks)).items() if role == "sideline"}
    out: list = []
    for t, r in rows.items():
        if len(r) < cfg.min_track_frames or t in off:
            continue
        if ratios.get(t) and len(r) >= cfg.adult_min_frames and np.median(ratios[t]) >= cfg.adult_ratio:
            continue
        R = np.array(r, float)
        # подвижность трека: разброс положений на земле, м (игрок бегает, зритель и запасной стоят)
        spread = float(np.hypot(*(np.percentile(R[:, 3:5], 90, axis=0) - np.percentile(R[:, 3:5], 10, axis=0))))
        out += [(*row, t, spread) for row in r]
    A = np.array(out, float).reshape(-1, 8)
    return {"f": A[:, 0].astype(int), "u": A[:, 1], "v": A[:, 2], "X": A[:, 3], "Z": A[:, 4], "h": A[:, 5],
            "t": A[:, 6].astype(int), "spread": A[:, 7]}


def active_feet(feet: dict, off: dict, min_spread_m: float = 6.0, min_points: int = 300) -> np.ndarray:
    """Какие точки `play_feet` годятся для подгонки схемы поля: человек в игре на этом кадре (`off` — отрезки «вне
    игры» по трекам из `roster.resolve`) и двигается. Мало таких — все, кто в игре; совсем мало — все."""
    on = np.array([not any(a <= f <= b for a, b in off.get(int(t), [])) for t, f in zip(feet["t"], feet["f"])], bool)
    for mask in (on & (feet["spread"] >= min_spread_m), on):
        if mask.sum() >= min_points:
            return mask
    return np.ones(len(on), bool)


def occupancy_points(tracks: TrackFrames, ground: GroundModel, manual_roles: Optional[dict] = None,
                     cfg: BoardConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Положения (X, Z, метры от камеры) детей-участников по всему ролику — для `pitch.fit_to_occupancy`."""
    feet = play_feet(tracks, ground, manual_roles, cfg)
    return feet["X"], feet["Z"]


def build_board(tracks: TrackFrames, colors: Sequence[np.ndarray], ground: GroundModel, ball: Optional[BallTrack],
                records: Sequence[dict], scale: float, fps: float, pitch: Optional[PitchSetup],
                cfg: BoardConfig | None = None, manual_roles: Optional[dict] = None,
                pitch_marked: Optional[bool] = None, roster: Optional[dict] = None) -> dict:
    """colors[i] — (N, 3) Lab торса детекций кадра i; records — кадры track.json (рамки исходного кадра);
    manual_roles — ручные пометки «в игре / вне игры» в виде, который хранит задача
    ({id: {"role", "f", "box"}}, см. `resolve_roles`), или просто {id: "play" | "sideline"}."""
    cfg = cfg or BoardConfig()
    fps = fps or 30.0
    n = min(len(tracks), len(ground), len(records))
    signatures = track_signatures(tracks)
    manual_roles = resolve_roles(manual_roles, signatures)
    numbers = display_numbers(tracks, cfg.min_track_frames)
    default_pitch = pitch is None if pitch_marked is None else not pitch_marked
    if pitch is None:
        pitch = PitchSetup.default()
    L, W = pitch.length_m, pitch.width_m

    # --- положение каждого трека по кадрам ------------------------------------------------------------
    per_track: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    track_colors: dict[int, list[np.ndarray]] = defaultdict(list)
    ratios: dict[int, list[float]] = defaultdict(list)       # рост рамки / ожидаемый рост ребёнка
    frame_h = float(max((np.asarray(b, float).reshape(-1, 4)[:, 3].max() for b in tracks.boxes if len(b)), default=1e9))
    for i in range(n):
        ids, boxes, det = tracks.ids[i], np.asarray(tracks.boxes[i], float).reshape(-1, 4), tracks.det_index[i]
        if not len(ids):
            continue
        PX, PY = pitch.frame_to_pitch(i, (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3], ground)
        expected = ground.height_at(i, boxes[:, 3])
        col = colors[i] if i < len(colors) else np.zeros((0, 3))
        for k, tid in enumerate(ids):
            if np.isnan(PX[k]):
                continue
            per_track[int(tid)].append((i, float(PX[k]), float(PY[k])))
            # рамка, срезанная краем кадра, занижает рост (зритель у камеры кажется ребёнком)
            if expected[k] > 8 and boxes[k, 3] < frame_h - 2 and boxes[k, 1] > 2:
                ratios[int(tid)].append(float((boxes[k, 3] - boxes[k, 1]) / expected[k]))
            d = int(det[k])
            if 0 <= d < len(col) and not np.isnan(col[d]).any():
                track_colors[int(tid)].append(np.asarray(col[d], np.float32))

    # --- команды ----------------------------------------------------------------------------------------
    height_ratio = {t: float(np.median(r)) for t, r in ratios.items() if r}
    frames_of = {t: [f for f, _, _ in pts] for t, pts in per_track.items()}
    team_of, centers, teams_note = assign_teams(track_colors, frames_of, cfg, height_ratio)
    adults = {t for t, team in team_of.items() if team == OTHER}
    focus_ids = Counter(r["track_id"] for r in records[:n] if r.get("track_id") is not None
                        and r.get("state") in ("active", "contested"))
    own = None
    for tid, _ in focus_ids.most_common():
        if team_of.get(tid) in (0, 1):
            own = team_of[tid]
            break
    # вратари: команда по стороне ворот, а не по цвету формы — только играющие дети
    roles = classify_roles(per_track, pitch, fps, cfg, manual_roles, protect=set(focus_ids))
    # состав (`roster.resolve`): кто вне игры по кадрам — по пометкам оператора, перенесённым на человека целиком
    # (все фрагменты его трека) и на похожих по внешности. Роль трека — «вне игры», если он вне игры почти всё время
    off_of: dict[int, list] = {}
    why_of: dict[int, str] = {}
    if roster is not None:
        reasons = {"manual": "выключен вручную", "person": "тот же человек, что выключенный",
                   "similar": "похож на выключенных", "default": "не отмечен как игрок"}
        for tid, pts in per_track.items():
            spans = roster["off"].get(tid, [])
            off_of[tid], why_of[tid] = spans, roster["why"].get(tid, "default")
            fr = np.array([f for f, _, _ in pts])
            share = float(np.mean(np.any([(fr >= a) & (fr <= b) for a, b in spans], axis=0))) if spans and len(fr) else 0.0
            if tid in focus_ids and why_of[tid] != "manual":
                off_of[tid], share = [], 0.0                    # цель слежения выключается только явной пометкой
            roles[tid] = ("sideline", reasons[why_of[tid]]) if share >= 0.9 else ("play", "")
    skip = adults | {t for t, (r, _) in roles.items() if r == "sideline"}
    keepers = find_goalkeepers(per_track, pitch, fps, cfg, skip)
    for tid, side in keepers.items():
        if own is None or 0 not in centers:
            continue
        at_own_goal = (side == "left") == (pitch.own_goal == "left")
        team_of[tid] = own if at_own_goal else 1 - own
    teams = {}
    for k, center in centers.items():
        if k == OTHER:
            continue
        role = TEAM_OWN if own == k else TEAM_OPP if own is not None else f"team{k}"
        teams[str(k)] = {"role": role, "label": "свои" if role == TEAM_OWN else "соперник" if role == TEAM_OPP
                         else f"команда {'АБ'[k]}", "color": _lab_to_hex(center)}
    teams[str(OTHER)] = {"role": "other", "label": "судья, тренеры, взрослые",
                         "color": _lab_to_hex(centers[OTHER]) if OTHER in centers else "#9aa0a6"}

    # --- сглаживание и выборка кадров доски ---------------------------------------------------------------
    frames: list[dict] = [{"f": f, "t": round(f / fps, 2), "p": []} for f in range(0, n, cfg.step)]
    track_meta = {}
    sideline = []
    for tid, pts in per_track.items():
        if len(pts) < cfg.min_track_frames:
            continue
        arr = np.array(pts)
        f, x, y = arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
        # в пропусках трека не сглаживаем через разрыв: отрезки подряд
        cuts = np.flatnonzero(np.diff(f) > cfg.step * 2) + 1
        xs, ys = np.empty_like(x), np.empty_like(y)
        for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(f)]])):
            xs[a:b], ys[a:b] = _smooth(x[a:b], cfg.smooth_frames), _smooth(y[a:b], cfg.smooth_frames)
        team = team_of.get(tid, UNKNOWN)
        cs = track_colors.get(tid)
        role, why = roles.get(tid, ("play", ""))
        manual = tid in manual_roles
        sig = signatures.get(tid, (int(f[0]), []))
        track_meta[str(tid)] = {"num": numbers.get(tid), "team": team,
                                "color": _lab_to_hex(np.median(np.stack(cs), axis=0)) if cs else "#cccccc",
                                "frames": len(pts), "seconds": round(len(pts) / fps, 1), "role": role,
                                "reason": why, "manual": manual, "gk": tid in keepers,
                                "from": round(float(f[0]) / fps, 1), "to": round(float(f[-1]) / fps, 1),
                                "sig": {"f": sig[0], "box": sig[1]}, "off": off_of.get(tid, [[int(f[0]), int(f[-1])]]
                                                                                    if role == "sideline" else []),
                                "why": why_of.get(tid, "manual" if manual else "default"),
                                "person": (roster or {}).get("person", {}).get(tid),
                                "height_ratio": round(height_ratio[tid], 2) if tid in height_ratio else None}
        # поле отмечено на кадре — границы известны точно: кто почти всё время за ними, тот кандидат «вне игры»
        # (только предложение: выключает оператор); цель слежения и отмеченные вручную не предлагаются
        if pitch.project is not None and role == "play" and not manual and tid not in focus_ids \
                and why_of.get(tid, "default") == "default":
            m = cfg.outside_margin_m
            out = (x < -m) | (x > pitch.length_m + m) | (y < -m) | (y > pitch.width_m + m)
            if len(pts) >= cfg.outside_min_frames and out.mean() >= cfg.outside_share:
                track_meta[str(tid)]["suggest"] = "sideline"
                track_meta[str(tid)]["suggest_reason"] = f"{round(100 * float(out.mean()))} % времени за границей поля"
        if role == "sideline":
            sideline.append({"id": tid, "num": numbers.get(tid), "seconds": round(len(pts) / fps, 1), "reason": why,
                             "manual": manual, "why": why_of.get(tid, "manual" if manual else "default"), "color": track_meta[str(tid)]["color"],
                             "from": round(float(f[0]) / fps, 1), "to": round(float(f[-1]) / fps, 1)})
        # на доске все, включая выключенных и стоящих за полем: кого выключать, решает оператор
        for fi, xi, yi in zip(f, xs, ys):
            if fi % cfg.step == 0:
                frames[fi // cfg.step]["p"].append([tid, round(float(xi), 1), round(float(yi), 1)])

    # --- цель и мяч ----------------------------------------------------------------------------------------
    focus_rows, ball_rows = [], []
    for k, fr in enumerate(frames):
        i = fr["f"]
        r = records[i]
        if r.get("state") in ("active", "contested") and r.get("box"):
            b = np.asarray(r["box"], float) * scale
            PX, PY = pitch.frame_to_pitch(i, (b[0] + b[2]) / 2, b[3], ground)
            if not np.isnan(PX):
                focus_rows.append((k, float(PX), float(PY), r.get("track_id"), r["state"]))
        if ball is not None and i < len(ball.state) and ball.state[i] is not None:
            if pitch.project is not None:
                BX, BY = pitch.project(i, ball.u[i], ball.v[i])
            else:
                BX, BY = pitch.to_pitch(ball.X[i], ball.Z[i])
            if not np.isnan(BX):
                ball_rows.append((k, float(BX), float(BY), "d" if ball.state[i] == DETECTED else "e"))
    for k, x, y, tid, state in _smooth_rows(focus_rows, cfg.smooth_board):
        frames[k]["focus"] = {"id": tid, "x": round(x, 1), "y": round(y, 1), "state": state}
    for k, x, y, state in _smooth_rows(ball_rows, cfg.smooth_board):
        frames[k]["ball"] = [round(x, 1), round(y, 1), state]

    C, f_dir, _ = pitch._basis()
    return {
        "fps": round(fps / cfg.step, 2), "step": cfg.step, "frames_total": n,
        "field": {"length_m": L, "width_m": W, "own_goal": pitch.own_goal, "default": default_pitch},
        "camera": [round(float(C[0]), 1), round(float(C[1]), 1)],
        "teams": teams, "own_team": own, "teams_note": teams_note, "tracks": track_meta,
        "sideline": sorted(sideline, key=lambda r: r["num"] or 0),
        "auto_roles": cfg.auto_roles,
        "ball": ball.summary() if ball is not None else None,
        "frames": frames,
    }


# --- игровой контекст для метрик и ИИ-разбора ------------------------------------------------------------

def game_context(board: dict, pitch: PitchSetup, cfg: BoardConfig | None = None) -> dict:
    """По доске: где мяч, у кого он (ближайший игрок какой команды), где игрок в фокусе относительно мяча и
    своей команды. Посекундно — для ленты метрик; итог — доли времени."""
    teams = board.get("teams", {})
    own = board.get("own_team")
    tracks = board.get("tracks", {})
    fps = board["fps"]
    cfg = cfg or BoardConfig()
    rows: dict[int, dict] = {}
    poss = Counter()
    known_ball = 0
    dist_ball = []
    behind_line = []
    # владение с гистерезисом: владелец сохраняется, пока другой не окажется ближе на poss_switch_m
    # подряд poss_switch_frames кадров; никого рядом — владелец держит мяч ещё poss_keep_frames кадров
    holder: Optional[int] = None
    holder_missing = 0
    challenger: Optional[int] = None
    challenge_frames = 0
    spells: list[str] = []
    for fr in board["frames"]:
        sec = int(fr["t"])
        row = rows.setdefault(sec, {"ball": [], "poss": [], "dist": [], "depth": [], "team_prog": [], "line": []})
        # выключенные оператором (запасные, тренеры, зрители) в игре не участвуют
        pts = [q for q in fr["p"] if not track_off(tracks.get(str(q[0]), {}), fr["f"])]
        ball = fr.get("ball")
        focus = fr.get("focus")
        if ball:
            known_ball += 1
            row["ball"].append((ball[0], ball[1]))
            bz = float(np.hypot(ball[0] - board["camera"][0], ball[1] - board["camera"][1]))
            radius = cfg.poss_radius_m + cfg.poss_depth_factor * bz
            dists = {tid: float(np.hypot(x - ball[0], y - ball[1])) for tid, x, y in pts
                     if tracks.get(str(tid), {}).get("team") in (0, 1)}
            nearest = min(dists, key=dists.get) if dists else None
            d_holder = dists.get(holder) if holder is not None else None
            if nearest is not None and dists[nearest] <= radius:
                if holder is None or d_holder is None or d_holder > radius:
                    holder, challenger, challenge_frames = nearest, None, 0
                elif nearest != holder and d_holder - dists[nearest] >= cfg.poss_switch_m:
                    challenge_frames = challenge_frames + 1 if challenger == nearest else 1
                    challenger = nearest
                    if challenge_frames >= cfg.poss_switch_frames:
                        holder, challenger, challenge_frames = nearest, None, 0
                else:
                    challenger, challenge_frames = None, 0
                holder_missing = 0
            else:
                holder_missing += 1
                if holder_missing > cfg.poss_keep_frames:
                    holder = None
            team = tracks.get(str(holder), {}).get("team", UNKNOWN) if holder is not None else UNKNOWN
            if team in (0, 1):
                who = "own" if own is not None and team == own else "opp" if own is not None else f"team{team}"
            else:
                who = "loose"
            if not spells or spells[-1] != who:
                spells.append(who)
            poss[who] += 1
            row["poss"].append(who)
            if focus:
                dd = float(np.hypot(focus["x"] - ball[0], focus["y"] - ball[1]))
                dist_ball.append(dd)
                row["dist"].append(dd)
        if focus and own is not None:
            mates = [x for tid, x, y in pts if tracks.get(str(tid), {}).get("team") == own and tid != focus.get("id")
                     and not tracks.get(str(tid), {}).get("gk")]
            if len(mates) >= 2:
                prog = pitch.progress(np.array(mates))
                fprog = float(pitch.progress(focus["x"]))
                # последний полевой своей команды (вратарь узнаётся отдельно и сюда не входит)
                line = float(np.min(prog))
                row["depth"].append(fprog - float(np.mean(prog)))
                row["line"].append(fprog - line)
                row["team_prog"].append(float(np.mean(prog)))
                behind_line.append(fprog <= line + 0.03)
    L = pitch.length_m
    timeline = {}
    for sec, r in rows.items():
        out = {}
        if r["ball"]:
            bx = float(np.mean([b[0] for b in r["ball"]]))
            by = float(np.mean([b[1] for b in r["ball"]]))
            prog = float(pitch.progress(bx))
            out["ball_third"] = ("своя треть", "средняя треть", "чужая треть")[min(max(int(prog * 3), 0), 2)]
            out["ball_to_goal"] = round(prog, 2)
        if r["poss"]:
            c = Counter(r["poss"]).most_common(1)[0][0]
            out["ball_with"] = {"own": "у своих", "opp": "у соперника", "loose": "ничей/в борьбе"}.get(c, c)
        if r["dist"]:
            out["dist_to_ball_m"] = round(float(np.median(r["dist"])), 1)
        if r["depth"]:
            out["vs_team_m"] = round(float(np.mean(r["depth"])) * L, 1)
            out["vs_last_line_m"] = round(float(np.mean(r["line"])) * L, 1)
        if out:
            timeline[sec] = out
    total = max(sum(poss.values()), 1)
    frames = max(len(board["frames"]), 1)
    summary = {
        "ball_known_share": round(known_ball / frames, 3),
        "ball_detected_share": (board.get("ball") or {}).get("detected_share"),
        "possession_estimate": {k: round(v / total, 3) for k, v in poss.items()},
        "possession_spells": dict(Counter(s for s in spells if s != "loose")),
        "focus_dist_to_ball_m": {"median": round(float(np.median(dist_ball)), 1),
                                 "share_within_5m": round(float(np.mean(np.array(dist_ball) <= 5)), 3),
                                 "share_within_15m": round(float(np.mean(np.array(dist_ball) <= 15)), 3)}
        if dist_ball else None,
        "focus_last_line_share": round(float(np.mean(behind_line)), 3) if behind_line else None,
        "own_team_color": teams.get(str(own), {}).get("color") if own is not None else None,
        "note": ("владение — по ближайшему к мячу игроку (в 2 м + 5 % глубины, с гистерезисом: владелец держит мяч, "
                 "пока другой не окажется ближе на 1 м полсекунды), это оценка; possession_spells — число отрезков "
                 "владения; «последняя линия» — самый задний полевой своей команды на доске (вратарь не считается)"),
    }
    return {"summary": summary, "timeline": timeline, "fps": fps}
