"""Схема поля: где камера, куда смотрит, где свои ворота — и координаты игрока на поле.

Без схемы положение игрока известно только относительно камеры: X — поперёк взгляда (вправо в
кадре), Z — вглубь от камеры (`player_metrics`). Чтобы говорить «своя половина», «атакующая треть»,
«левый фланг», «рывок к чужим воротам», оператор отмечает на схеме поля (вид сверху):

  * камеру — точку, где стоит снимающий (обычно за бровкой или за воротами, может быть вне поля);
  * куда смотрит камера в начале ролика — точку на поле в центре первого кадра (дальше панорамирование
    учитывается компенсацией движения камеры);
  * свои ворота — левые или правые на схеме.

Координаты схемы: x — вдоль длины поля от левой линии ворот (0) до правой (1), y — поперёк от дальней
(верхней на схеме) бровки (0) до ближней (1); точки вне поля — < 0 или > 1. Размер поля в метрах —
из формата игры (`profile.GAME_FORMATS`) или указанный.

Переход в метры поля: C — камера, f — единичный вектор взгляда (к точке «куда смотрит»), r — вправо
от взгляда (на схеме с осью y вниз это (−f_y, f_x)); игрок = C + kx·X·r + kz·Z·f. Ошибка — как у X и Z
(±20–30 %) плюс неточность отметок: это оценка «где примерно», не трекинг по разметке поля.

kx и kz — поправки масштаба, которые подбираются по самим игрокам (`fit_to_occupancy`): глубина Z = f·H/h
зависит от угла обзора (зум телефона неизвестен) и роста детей, поперечное X — только от роста. За матч
дети покрывают почти всё поле, поэтому масштабы и положение камеры подгоняются так, чтобы облако их
положений легло в поле формата; направление взгляда и свои ворота остаются отметками оператора.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .profile import GAME_FORMATS

DEFAULT_FIELD = (40.0, 25.0)   # тренировка / иное без указанного размера


def default_field(game_format: Optional[str]) -> tuple[float, float]:
    fmt = GAME_FORMATS.get(game_format or "")
    if fmt and fmt.get("field"):
        (l1, l2), (w1, w2) = fmt["field"]
        return (l1 + l2) / 2, (w1 + w2) / 2
    return DEFAULT_FIELD


@dataclass
class PitchSetup:
    camera: tuple[float, float]
    look: tuple[float, float]
    own_goal: str                 # "left" | "right" на схеме
    length_m: float
    width_m: float
    kx: float = 1.0               # поправка масштаба поперёк взгляда (рост игроков)
    kz: float = 1.0               # поправка масштаба вглубь (угол обзора / зум и рост)
    # разметка поля на кадре (`field.FieldProjector`): если есть, точка кадра -> метры поля берётся из неё
    project: Optional[Callable] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: Optional[dict], game_format: Optional[str] = None) -> Optional["PitchSetup"]:
        if not d or d.get("camera") is None or d.get("look") is None or d.get("own_goal") not in ("left", "right"):
            return None
        L, W = default_field(game_format)
        return cls(tuple(map(float, d["camera"])), tuple(map(float, d["look"])), d["own_goal"],
                   float(d.get("length_m") or L), float(d.get("width_m") or W))

    @classmethod
    def default(cls, game_format: Optional[str] = None) -> "PitchSetup":
        """Схема, если оператор ничего не отметил: камера за серединой ближней длинной бровки, смотрит поперёк
        поля (так снимает большинство родителей). Свои ворота не известны — «left» условно."""
        L, W = default_field(game_format)
        return cls((0.5, 1.1), (0.5, 0.5), "left", L, W)

    # --- геометрия ------------------------------------------------------------------------------------
    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C = np.array([self.camera[0] * self.length_m, self.camera[1] * self.width_m])
        T = np.array([self.look[0] * self.length_m, self.look[1] * self.width_m])
        f = T - C
        n = float(np.hypot(*f))
        f = f / n if n > 1e-6 else np.array([0.0, -1.0])
        r = np.array([-f[1], f[0]])
        return C, f, r

    def to_pitch(self, X, Z) -> tuple[np.ndarray, np.ndarray]:
        """Метры поля (px вдоль длины от левой линии ворот, py поперёк от дальней бровки)."""
        C, f, r = self._basis()
        X, Z = np.asarray(X, float) * self.kx, np.asarray(Z, float) * self.kz
        return C[0] + X * r[0] + Z * f[0], C[1] + X * r[1] + Z * f[1]

    def frame_to_pitch(self, i: int, u, v, ground) -> tuple[np.ndarray, np.ndarray]:
        """Точка земли на кадре i -> метры поля: по разметке поля, если она есть, иначе по росту и схеме."""
        if self.project is not None:
            return self.project(i, u, v)
        X, Z = ground.ground(i, u, v)
        return self.to_pitch(X, Z)

    def outside_share(self, X, Z, margin_m: float = 0.0) -> float:
        px, py = self.to_pitch(X, Z)
        ok = ~np.isnan(px)
        if not ok.any():
            return 0.0
        px, py, m = px[ok], py[ok], margin_m
        return float(np.mean((px < -m) | (px > self.length_m + m) | (py < -m) | (py > self.width_m + m)))

    @property
    def attack_sign(self) -> float:
        """+1 — атака слева направо на схеме (свои ворота слева), −1 — справа налево."""
        return 1.0 if self.own_goal == "left" else -1.0

    def progress(self, px) -> np.ndarray:
        """0 — у своих ворот, 1 — у чужих."""
        p = np.asarray(px, float) / self.length_m
        return p if self.own_goal == "left" else 1.0 - p

    def side(self, py) -> np.ndarray:
        """0 — левый фланг, 1 — правый, если смотреть в сторону атаки своей команды."""
        s = np.asarray(py, float) / self.width_m      # 0 — дальняя (верхняя) бровка
        # атака вправо: левая рука — к верхней бровке (y = 0); атака влево — наоборот
        return s if self.own_goal == "left" else 1.0 - s

    def describe(self) -> str:
        C, f, _ = self._basis()
        angle = math.degrees(math.atan2(f[1], f[0]))
        where = []
        cx, cy = self.camera
        if cy > 1:
            where.append("за ближней бровкой")
        elif cy < 0:
            where.append("за дальней бровкой")
        if cx < 0:
            where.append("за левыми воротами")
        elif cx > 1:
            where.append("за правыми воротами")
        if not where:
            where.append("на поле")
        along = "у левой половины" if cx < 0.4 else "у правой половины" if cx > 0.6 else "напротив центра"
        return (f"Поле {self.length_m:.0f} × {self.width_m:.0f} м. Камера {' и '.join(where)}, {along} "
                f"(в {C[0]:.0f} м от левой линии ворот), смотрит под {angle:.0f}° к оси поля. "
                f"Свои ворота {'слева' if self.own_goal == 'left' else 'справа'} на схеме: атака своей команды — "
                f"{'слева направо' if self.own_goal == 'left' else 'справа налево'}.")


THIRDS = ("своя треть (оборона)", "средняя треть", "чужая треть (атака)")
CHANNELS = ("левый фланг", "центр", "правый фланг")


def run_direction(dx_attack: float, dy: float, min_m: float = 3.0) -> str:
    if abs(dx_attack) < min_m and abs(dy) < min_m:
        return "на месте"
    if abs(dx_attack) >= abs(dy) * 0.8:
        return "к чужим воротам" if dx_attack > 0 else "к своим воротам"
    return "поперёк поля"


# Выход за поле штрафуется с весом 1; недобор охвата — слабо: на тренировке или в отрезке матча дети занимают
# часть площадки, и растягивать их на всё поле нельзя (12-секундное упражнение растягивалось до ×2.5 по глубине).
UNDER_WEIGHT = 0.05
PRIOR_KX = 0.15            # тяга к «без поправки»: поправка нужна, только когда игроки выходят за поле
PRIOR_KZ = 0.1


def fit_to_occupancy(pitch: PitchSetup, X, Z, margin: float = 0.03, coverage: tuple[float, float] = (3, 97),
                     kx_range: tuple[float, float] = (0.7, 1.45), kz_range: tuple[float, float] = (0.35, 2.5),
                     steps: int = 40, max_points: int = 6000) -> tuple[PitchSetup, dict]:
    """Подогнать масштабы (kx, kz) и положение камеры так, чтобы положения игроков легли в поле.

    X, Z — положения детей-участников (метры в системе камеры) по всему ролику. Для каждой пары масштабов
    на сетке облако поворачивается по направлению взгляда, берётся его охват (процентили coverage) вдоль
    длины и ширины поля и сравнивается с полем без полей margin. Выход охвата за поле штрафуется сильно,
    недобор — слабо (камера могла снимать не всё поле, упражнение идёт на части площадки), отклонение
    масштабов от 1 — умеренно.
    Камера сдвигается так, чтобы центр охвата совпал с центром поля по той оси, где охват почти во всё
    поле; по оси, где охват мал, сохраняется отмеченное положение (только не выпуская облако за поле)."""
    X, Z = np.asarray(X, float), np.asarray(Z, float)
    ok = np.isfinite(X) & np.isfinite(Z)
    X, Z = X[ok], Z[ok]
    if len(X) < 50:
        return pitch, {"fitted": False, "reason": "мало положений игроков для подгонки"}
    if len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points).astype(int)
        X, Z = X[idx], Z[idx]
    L, W = pitch.length_m, pitch.width_m
    C0, f, r = pitch._basis()
    target = np.array([L, W]) * (1 - 2 * margin)
    lo_q, hi_q = coverage
    best = None
    for kx in np.geomspace(*kx_range, steps):
        for kz in np.geomspace(*kz_range, steps):
            px = kx * X * r[0] + kz * Z * f[0]
            py = kx * X * r[1] + kz * Z * f[1]
            lo = np.array([np.percentile(px, lo_q), np.percentile(py, lo_q)])
            hi = np.array([np.percentile(px, hi_q), np.percentile(py, hi_q)])
            ratio = (hi - lo) / target
            over = np.maximum(ratio - 1, 0)
            under = np.maximum(1 - ratio, 0)
            cost = float(np.sum(over ** 2) + UNDER_WEIGHT * np.sum(under ** 2)
                         + PRIOR_KX * np.log(kx) ** 2 + PRIOR_KZ * np.log(kz) ** 2)
            if best is None or cost < best[0]:
                best = (cost, kx, kz, lo, hi, ratio)
    cost, kx, kz, lo, hi, ratio = best
    size = np.array([L, W])
    C = C0.copy()
    for a in range(2):
        span_lo, span_hi = C0[a] + lo[a], C0[a] + hi[a]
        if ratio[a] >= 0.8:
            C[a] = size[a] / 2 - (lo[a] + hi[a]) / 2          # охват почти во всё поле — центрируем
        elif span_lo < margin * size[a]:
            C[a] += margin * size[a] - span_lo                 # иначе только не выпускаем за поле
        elif span_hi > (1 - margin) * size[a]:
            C[a] -= span_hi - (1 - margin) * size[a]
    T = C + f * 10.0
    fitted = PitchSetup((float(C[0] / L), float(C[1] / W)), (float(T[0] / L), float(T[1] / W)), pitch.own_goal,
                        L, W, float(kx), float(kz))
    info = {"fitted": True, "kx": round(float(kx), 3), "kz": round(float(kz), 3),
            "coverage": [round(float(v), 2) for v in ratio],
            "camera_m": [round(float(C[0]), 1), round(float(C[1]), 1)],
            "outside_before": round(pitch.outside_share(X, Z, 1.0), 3),
            "outside_after": round(fitted.outside_share(X, Z, 1.0), 3)}
    return fitted, info
