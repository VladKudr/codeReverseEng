"""Разметка поля на кадре: четыре угла площадки -> перспективное преобразование «кадр -> метры поля».

Оценка метров по росту игрока (`ground.GroundModel`) зависит от неизвестного зума телефона и роста детей, а
отметка камеры на схеме — от глазомера. Если оператор отметил на кадре углы площадки (или места, где они
должны быть, если разметки нет или угол не попал в кадр), положение любой точки земли в кадре переводится в
метры поля точно — гомографией H по четырём соответствиям.

Углы отмечаются на опорном кадре `frame`, в координатах исходного кадра, и могут лежать за его краями.
Порядок точек — 1: левый ближний, 2: правый ближний, 3: правый дальний, 4: левый дальний («ближний» — ближе к
камере, «левый» — слева в кадре). Им соответствуют углы схемы поля: при камере у ближней длинной бровки это
(0, W), (L, W), (L, 0), (0, 0); `rotation` поворачивает соответствие на 90° (камера за воротами или у другой
бровки).

На другом кадре точка переносится в опорный привязкой кадров к сцене: p_ref = M_ref · inv_i · p_i, где inv_i — «кадр i ->
сцена». Точная привязка — гомографией по опорным кадрам (`camreg.register`, сползание 1–3 px); пока она не посчитана —
цепочкой подобий покадрового движения камеры (`GroundModel.inv`, сползание 15–25 px к концу ролика).

Разметок может быть несколько (`parse_marks`, `MultiProjector`): оператор перешёл на другое место — поле отмечается
заново на кадре оттуда, каждый кадр ролика считается по ближайшей по времени разметке.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pitch import PitchSetup


def pitch_corners(length_m: float, width_m: float, rotation: int = 0) -> np.ndarray:
    """Углы схемы поля для точек 1–4 (левый ближний, правый ближний, правый дальний, левый дальний)."""
    L, W = length_m, width_m
    ring = [(0.0, W), (L, W), (L, 0.0), (0.0, 0.0)]           # обход схемы: камера у ближней длинной бровки
    k = rotation % 4
    return np.array(ring[k:] + ring[:k], dtype=np.float64)


def convex(corners) -> bool:
    """Четыре точки по кругу образуют выпуклый невырожденный четырёхугольник."""
    q = np.asarray(corners, float).reshape(4, 2)
    d = np.roll(q, -1, axis=0) - q
    n = np.roll(d, -1, axis=0)
    cross = d[:, 0] * n[:, 1] - d[:, 1] * n[:, 0]
    return bool((np.abs(cross) > 1e-6).all() and (np.sign(cross) == np.sign(cross[0])).all())


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """H (3×3) по четырём соответствиям src -> dst (прямое решение 8×8, без OpenCV)."""
    A, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float)):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def apply_h(H: np.ndarray, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w, w


CORNERS, SIDES = "corners", "sides"


@dataclass
class FieldMarks:
    frame: int
    corners: np.ndarray            # (4, 2) координаты исходного кадра
    rotation: int = 0
    # corners — все четыре точки стоят в углах площадки; sides — 3 и 4 стоят в дальних углах, а 1 и 2 — в любом
    # видимом месте боковых линий (ближние углы у ног оператора уходят на тысячи пикселей за кадр)
    mode: str = CORNERS

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["FieldMarks"]:
        if not d or d.get("corners") is None or len(d["corners"]) != 4:
            return None
        return cls(int(d.get("frame") or 0), np.asarray(d["corners"], dtype=np.float64).reshape(4, 2),
                   int(d.get("rotation") or 0), SIDES if d.get("mode") == SIDES else CORNERS)


@dataclass
class CameraPrior:
    """Камера опорного кадра для режима `sides`: строка горизонта и высота камеры — по росту людей на кадрах
    (`scene_horizon`), фокус — типичный для телефона. Координаты — кадра обработки."""
    v0: float
    hc: float
    cx: float
    f0: float


def scene_horizon(f, u, v, h, inv: list, ref: int, player_height_m: float) -> Optional[tuple[float, float]]:
    """Строка горизонта на кадре `ref` и высота камеры (м) по росту людей ВСЕГО ролика.

    Рост рамки человека линейно растёт со строкой ног: h = a·(v − v0); на одном кадре людей мало и прямая шумит
    (горизонт ±15 px). С привязкой кадров к сцене (`inv`) все наблюдения ролика переносятся на кадр `ref` и
    прямая подгоняется по десяткам тысяч точек (замер: ±4 px между половинами ролика). Высота камеры — H / a."""
    from .ground import _robust_line

    f, u, v, h = np.asarray(f, int), np.asarray(u, float), np.asarray(v, float), np.asarray(h, float)
    ref = int(min(max(ref, 0), len(inv) - 1))
    back = np.linalg.inv(inv[ref])
    rows, heights = [], []
    for i in np.unique(f):
        m = (f == i) & np.isfinite(h)
        if not m.any() or i >= len(inv):
            continue
        T = back @ inv[i]
        one = np.ones(int(m.sum()))
        foot, head = T @ np.stack([u[m], v[m], one]), T @ np.stack([u[m], v[m] - h[m], one])
        ok = (foot[2] > 1e-9) & (head[2] > 1e-9)
        rows += list((foot[1] / foot[2])[ok])
        heights += list((foot[1] / foot[2] - head[1] / head[2])[ok])
    rows, heights = np.array(rows), np.array(heights)
    ok = np.isfinite(rows) & np.isfinite(heights) & (heights > 8)
    if ok.sum() < 300:
        return None
    line = _robust_line(rows[ok], heights[ok])
    if line is None:
        return None
    a, c = line
    return float(-c / a), float(player_height_m / a)


def _sides_matrix(p, cam: CameraPrior) -> np.ndarray:
    """Гомография «кадр -> метры поля» камеры без крена: u = cx + f·X/Z, v = v0 + f·hc/Z; поле повёрнуто на θ
    и сдвинуто (tx, ty); ось y поля направлена к камере (отражение)."""
    lf, theta, tx, ty, lh = p
    f, hc = cam.f0 * np.exp(lf), cam.hc * np.exp(lh)
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, s], [s, -c]])
    M = np.zeros((3, 3))
    for r, t in ((0, tx), (1, ty)):
        M[r] = [A[r, 0] * hc, t, -A[r, 0] * hc * cam.cx + A[r, 1] * f * hc - t * cam.v0]
    M[2] = [0.0, 1.0, -cam.v0]
    return M


def fit_sides(points: np.ndarray, rotation: int, length_m: float, width_m: float,
              cam: CameraPrior) -> tuple[np.ndarray, dict]:
    """Режим `sides`: H «кадр -> метры поля» по двум дальним углам (точки 3, 4) и по точке на каждой боковой
    линии (1 — на линии «угол 4 — угол 1», 2 — на линии «угол 3 — угол 2»).

    Неизвестные — фокус, поворот и сдвиг поля, высота камеры; горизонт — из `cam`. Прямой угол площадки даёт фокус,
    длина дальней стороны — масштаб. Когда дальняя линия параллельна кадру, фокус из разметки не определяется —
    остаётся априорный (слабые априорные члены на фокус и высоту камеры). Возвращает H и {"rms_m", "f", "hc"}."""
    from scipy.optimize import least_squares

    pts = np.asarray(points, float).reshape(4, 2)
    pc = pitch_corners(length_m, width_m, rotation)

    def project(p):
        x, y, w = apply_h(_sides_matrix(p, cam), pts[:, 0], pts[:, 1])
        return np.stack([x, y], axis=1), w

    def line_dist(P, a, b):
        d = (b - a) / np.linalg.norm(b - a)
        r = P - a
        return d[0] * r[1] - d[1] * r[0]

    def resid(p):
        P, w = project(p)
        if (w <= 0).any():                                   # точка выше горизонта — модель неприменима
            return np.full(8, 1e3)
        return np.array([*(P[3] - pc[3]), *(P[2] - pc[2]), line_dist(P[0], pc[3], pc[0]), line_dist(P[1], pc[2], pc[1]),
                         p[0] / 0.35 * 0.5, p[4] / 0.15 * 0.5])          # априорные: фокус ±35 %, высота камеры ±15 %

    best = None
    for k in range(4):                                       # поворот поля неизвестен заранее — четыре начала
        for lf in (0.0, 0.5):
            p0 = np.array([lf, k * np.pi / 2, length_m / 2, width_m, 0.0])
            r = least_squares(resid, p0, method="lm", max_nfev=200)
            if best is None or r.cost < best.cost:
                best = r
    H = _sides_matrix(best.x, cam)
    rms = float(np.sqrt(np.mean(best.fun[:6] ** 2)))
    return H, {"rms_m": round(rms, 2), "f": float(cam.f0 * np.exp(best.x[0])), "hc": float(cam.hc * np.exp(best.x[4]))}


def parse_marks(d: Optional[dict]) -> list[FieldMarks]:
    """Разметки из `player["field"]`: {"marks": [{frame, corners, rotation}, ...], own_goal, length_m, width_m};
    прежний вид с одной разметкой ({frame, corners, rotation, ...}) читается как список из одной."""
    if not d:
        return []
    items = d.get("marks") if d.get("marks") is not None else [d]
    out = [m for m in (FieldMarks.from_dict(x) for x in items) if m is not None]
    return sorted(out, key=lambda m: m.frame)


class FieldProjector:
    """Точка земли на кадре i (пиксели кадра обработки) -> метры поля по разметке оператора."""

    def __init__(self, marks: FieldMarks, scale: float, inv: list[np.ndarray], length_m: float, width_m: float,
                 max_out_m: float = 60.0, camera: Optional[CameraPrior] = None):
        self.marks = marks
        self.length_m, self.width_m = length_m, width_m
        self.inv = inv
        self.ref = int(min(max(marks.frame, 0), len(inv) - 1))
        self.M_ref = np.linalg.inv(inv[self.ref])
        img = marks.corners * scale                                    # в координатах кадра обработки
        self.fit: dict = {}
        if marks.mode == SIDES and camera is not None:
            self.H, self.fit = fit_sides(img, marks.rotation, length_m, width_m, camera)
        else:
            self.H = homography(img, pitch_corners(length_m, width_m, marks.rotation))
        self.max_out_m = max_out_m
        # сторона кадра, где лежит поле: у точек поля знак знаменателя гомографии одинаков
        _, _, w = apply_h(self.H, img[:, 0], img[:, 1])
        self.sign = float(np.sign(np.median(w))) or 1.0

    def to_ref(self, i: int, u, v) -> tuple[np.ndarray, np.ndarray]:
        """Точка кадра i -> координаты опорного кадра."""
        T = self.M_ref @ self.inv[min(max(int(i), 0), len(self.inv) - 1)]
        u, v = np.asarray(u, float), np.asarray(v, float)
        w = T[2, 0] * u + T[2, 1] * v + T[2, 2]
        return (T[0, 0] * u + T[0, 1] * v + T[0, 2]) / w, (T[1, 0] * u + T[1, 1] * v + T[1, 2]) / w

    def __call__(self, i: int, u, v) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.to_ref(i, u, v)
        px, py, w = apply_h(self.H, x, y)
        L, W, m = self.length_m, self.width_m, self.max_out_m
        # за горизонтом поля (другой знак знаменателя) или безумно далеко — не точка этого поля
        bad = (w * self.sign <= 0) | (px < -m) | (px > L + m) | (py < -m) | (py > W + m)
        return np.where(bad, np.nan, px), np.where(bad, np.nan, py)

    def pitch_setup(self, own_goal: str, frame_size: tuple[int, int]) -> PitchSetup:
        """Схема для «своих/чужих ворот», третей и описания для ИИ: камера — проекция низа опорного кадра,
        взгляд — проекция его центра (метры по-прежнему считаются гомографией, не этой схемой)."""
        Wf, Hf = frame_size
        L, W = self.length_m, self.width_m
        cx, cy, _ = apply_h(self.H, np.array([Wf / 2, Wf / 2]), np.array([Hf * 0.98, Hf / 2]))
        cam = (float(np.clip(cx[0] / L, -0.5, 1.5)), float(np.clip(cy[0] / W, -0.5, 1.5)))
        look = (float(np.clip(cx[1] / L, 0, 1)), float(np.clip(cy[1] / W, 0, 1)))
        if abs(cam[0] - look[0]) + abs(cam[1] - look[1]) < 1e-3:
            cam = (look[0], look[1] + 0.6)
        return PitchSetup(cam, look, own_goal, L, W)


class MultiProjector:
    """Несколько разметок поля: кадр считается по ближайшей по времени."""

    def __init__(self, projectors: list[FieldProjector]):
        self.projectors = sorted(projectors, key=lambda p: p.ref)
        self.refs = np.array([p.ref for p in self.projectors])
        self.length_m, self.width_m = self.projectors[0].length_m, self.projectors[0].width_m

    def nearest(self, i: int) -> FieldProjector:
        return self.projectors[int(np.argmin(np.abs(self.refs - int(i))))]

    def __call__(self, i: int, u, v) -> tuple[np.ndarray, np.ndarray]:
        return self.nearest(i)(i, u, v)

    def pitch_setup(self, own_goal: str, frame_size: tuple[int, int]) -> PitchSetup:
        return self.projectors[0].pitch_setup(own_goal, frame_size)


def inside_share(px, py, length_m: float, width_m: float, margin_m: float = 1.0) -> float:
    """Доля точек (метры поля) внутри площадки с запасом; NaN — не в счёт."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    ok = np.isfinite(px) & np.isfinite(py)
    if not ok.any():
        return float("nan")
    m = margin_m
    inside = (px >= -m) & (px <= length_m + m) & (py >= -m) & (py <= width_m + m)
    return float(inside[ok].mean())


def expand_rect(H: np.ndarray, length_m: float, width_m: float, u, v, lo: float = 2.0, hi: float = 98.0,
                margin_m: float = 1.0, min_shift_m: float = 1.0) -> Optional[tuple[float, float, float, float]]:
    """Границы площадки (x0, x1, y0, y1 в метрах прежней разметки H), раздвинутые до точек ног игроков (u, v).

    Сторона переносится, только если игроки выходят за неё больше чем на min_shift_m: там точно играли. Стороны, до
    которых игроки не доходят, не трогаются — оператор мог отметить всё поле, а играли на части."""
    px, py, w = apply_h(H, u, v)
    _, _, wc = apply_h(H, *apply_h(np.linalg.inv(H), np.array([length_m / 2]), np.array([width_m / 2]))[:2])
    ok = np.isfinite(px) & np.isfinite(py) & (w * np.sign(wc[0]) > 0)
    if ok.sum() < 50:
        return None
    x0, x1 = np.percentile(px[ok], [lo, hi])
    y0, y1 = np.percentile(py[ok], [lo, hi])
    return (x0 - margin_m if x0 < -min_shift_m else 0.0, x1 + margin_m if x1 > length_m + min_shift_m else length_m,
            y0 - margin_m if y0 < -min_shift_m else 0.0, y1 + margin_m if y1 > width_m + min_shift_m else width_m)


def expand_to_points(corners: np.ndarray, rotation: int, length_m: float, width_m: float, u, v,
                     H: Optional[np.ndarray] = None, mode: str = CORNERS, **kw) -> np.ndarray:
    """Раздвинуть разметку так, чтобы в неё попали точки ног игроков (u, v — на том же кадре, те же координаты).
    H — «кадр -> метры» этой разметки (для режима `sides` — из `fit_sides`); без него — гомография по углам."""
    corners = np.asarray(corners, float).reshape(4, 2)
    pc = pitch_corners(length_m, width_m, rotation)
    if H is None:
        H = homography(corners, pc)
    rect = expand_rect(H, length_m, width_m, u, v, **kw)
    if rect is None:
        return corners
    nx0, nx1, ny0, ny1 = rect
    sx = lambda x: nx0 if x == 0.0 else nx1          # noqa: E731
    sy = lambda y: ny0 if y == 0.0 else ny1          # noqa: E731
    new_m = np.array([[sx(x), sy(y)] for x, y in pc])            # новые углы в метрах прежней разметки
    if mode == SIDES:
        # точки 1, 2 остаются на своей глубине, но переезжают на новые боковые линии
        mx, my, _ = apply_h(H, corners[:2, 0], corners[:2, 1])
        for k in (0, 1):
            a, b = new_m[3 - k], new_m[k]                        # боковая линия: дальний угол -> ближний
            d = (b - a) / np.linalg.norm(b - a)
            t = float(np.clip(np.dot([mx[k], my[k]] - a, d), 0.05 * np.linalg.norm(b - a), np.linalg.norm(b - a)))
            new_m[k] = a + d * t
    Hi = np.linalg.inv(H)
    nu, nv, _ = apply_h(Hi, new_m[:, 0], new_m[:, 1])
    return np.stack([nu, nv], axis=1)


def visible_side_points(corners: np.ndarray, size: tuple[int, int], inset: float = 0.04,
                        reach: float = 0.3) -> tuple[np.ndarray, str]:
    """Если ближние углы (1, 2) дальше `reach` кадра за его краем — мышью их не поставить: разметка переводится в
    режим `sides`, точки 1, 2 съезжают по боковым линиям к дальним углам до края кадра (с отступом)."""
    corners = np.asarray(corners, float).reshape(4, 2)
    w, h = size
    inside = lambda p, m: -m * w <= p[0] <= (1 + m) * w and -m * h <= p[1] <= (1 + m) * h   # noqa: E731
    if inside(corners[0], reach) and inside(corners[1], reach):
        return corners, CORNERS
    out = corners.copy()
    for k in (0, 1):
        far = corners[3 - k]
        if inside(corners[k], -inset):
            continue
        ts = np.linspace(1.0, 0.0, 400)
        pts = far + (corners[k] - far) * ts[:, None]
        ok = [inside(p, -inset) for p in pts]
        out[k] = pts[int(np.argmax(ok))] if any(ok) else far + (corners[k] - far) * 0.3
    return out, SIDES


def corners_from_ground(X, Z, to_image, trim: float = 10.0, margin_m: float = 1.0,
                        min_depth_m: float = 3.0) -> Optional[tuple[np.ndarray, int, bool]]:
    """Начальная разметка по игрокам: прямоугольник наименьшей площади вокруг точек ног на плоскости поля
    (X поперёк, Z вглубь, метры — `GroundModel.ground`), перенесённый на кадр (`to_image(X, Z) -> (u, v)`).

    Возвращает углы 1–4 (левый ближний, правый ближний, правый дальний, левый дальний), rotation: 0 — к камере
    обращена длинная сторона (камера у бровки), 1 — короткая (камера за воротами), и признак «ближние точки сдвинуты
    по боковым линиям» (угол у ног оператора на кадр не переносится — тогда это разметка режима `sides`).
    Это заготовка: оператор её правит."""
    import cv2

    X, Z = np.asarray(X, float), np.asarray(Z, float)
    ok = np.isfinite(X) & np.isfinite(Z)
    P = np.stack([X[ok], Z[ok]], axis=1)
    if len(P) < 200:
        return None
    # плотное ядро облака: клетки 1×1 м по убыванию заполненности, пока не наберётся (100 − trim) % точек —
    # одиночные прохожие у камеры и ложные треки вдали лежат в редких клетках и прямоугольник не растягивают
    cell = np.floor(P).astype(np.int64)
    keys, inv_idx, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    keep_n = int(np.searchsorted(np.cumsum(counts[order]), (1 - trim / 100) * len(P))) + 1
    keep = np.zeros(len(keys), bool)
    keep[order[:keep_n]] = True
    P = P[keep[inv_idx.ravel()]]
    (cx, cy), (w, h), ang = cv2.minAreaRect(P.astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (w + 2 * margin_m, h + 2 * margin_m), ang)).astype(float)
    # ближняя сторона — соседняя пара углов с наименьшей глубиной; по кругу: ближний A, ближний B, дальний за B, дальний за A
    k = int(np.argmin([box[a, 1] + box[(a + 1) % 4, 1] for a in range(4)]))
    ring = box[[(k + a) % 4 for a in range(4)]]
    near_len = float(np.hypot(*(ring[0] - ring[1])))
    side_len = float(np.hypot(*(ring[1] - ring[2])))
    clipped = False
    for a, far in ((0, 3), (1, 2)):                          # угол у ног оператора или за его спиной на кадр не переносится:
        if ring[a, 1] < min_depth_m:                         # сдвигаем его по боковой линии к дальнему углу
            if ring[far, 1] <= min_depth_m:
                return None
            t = (min_depth_m - ring[far, 1]) / (ring[a, 1] - ring[far, 1])
            ring[a] = ring[far] + (ring[a] - ring[far]) * t
            clipped = True
    u, v = to_image(ring[:, 0], ring[:, 1])
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        return None
    pts = np.stack([u, v], axis=1)
    if pts[0, 0] > pts[1, 0]:                                # 1 — левый ближний
        pts = pts[[1, 0, 3, 2]]
    # глубина по росту людей бывает растянута до полутора раз (зум телефона неизвестен) — «камера за воротами»
    # только при явном перевесе боковой стороны
    return pts, (1 if side_len > 1.6 * near_len else 0), clipped
