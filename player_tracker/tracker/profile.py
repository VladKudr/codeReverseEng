"""Профиль игрока из опроса перед слежением: возраст, позиция, формат игры (и рост).

Опрос нужен не для красоты, каждое поле меняет расчёт:
  * возраст — пороги зон скорости («спринт» девятилетнего и пятнадцатилетнего — разные скорости) и
    типичный рост, если его не указали: рост — масштаб для метров, от него зависят дистанция и скорости;
  * позиция — по чему судить игрока (защитник держит позицию и страхует, нападающий открывается
    и идёт в мяч) — это правила для ИИ-разбора;
  * формат игры — размер поля и плотность игроков: 100 м/мин и «единоборства» на 30×20 м и на 60×40 м
    означают разное; тепловая карта и расстояния до соседей читаются относительно поля.

Все ориентиры — справочные (рекомендации по детским форматам и медианы роста ВОЗ), не нормативы.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

POSITIONS = {
    "goalkeeper": "вратарь",
    "center_back": "центральный защитник",
    "full_back": "крайний защитник",
    "defensive_mid": "опорный полузащитник",
    "central_mid": "центральный полузащитник",
    "wide_mid": "крайний полузащитник / вингер",
    "forward": "нападающий",
    "universal": "универсал / позиция не закреплена",
    "other": "другое",
}

POSITION_FOCUS = {
    "goalkeeper": "игра на линии и выходы, расположение относительно ворот; беговые метрики второстепенны, "
                  "дистанция и спринты у вратаря естественно малы",
    "center_back": "удержание позиции и компактность, страховка, единоборства в своей половине, реакция на "
                   "прорывы (ускорения назад/в сторону); много спринтов вперёд — скорее тревожный признак",
    "full_back": "работа по флангу в обе стороны, возврат после подключений, спринты вдоль бровки",
    "defensive_mid": "положение перед защитниками, перекрытие центра, частые короткие перемещения, "
                     "борьба за мяч в центре",
    "central_mid": "объём работы и связь линий: высокий темп, участие и в обороне, и в атаке, близость к мячу",
    "wide_mid": "игра по флангу, рывки в свободную зону, спринты с мячом и без, возврат в оборону",
    "forward": "открывания и рывки в зону ворот, спринты, близость к мячу в атаке, прессинг защитников",
    "universal": "где игрок реально проводил время и чем был полезен; не оценивать строго ни по одной позиции",
    "other": "оценивать по фактическому положению и действиям, позиция уточнена в комментарии",
}

# Формат «N × N» — N полевых игроков в команде ПЛЮС вратарь (6 × 6 — 7 игроков в команде, 14 на поле).
# Исключение — 11 × 11: классический футбол, 10 полевых + вратарь.
# Поле, м (длина × ширина) — ориентировочно по рекомендациям для детского футбола.
GAME_FORMATS = {
    "4x4": {"label": "4 × 4 (+ вратарь, 5 в команде)", "outfield": 4, "field": ((25, 35), (18, 25)),
            "note": "очень плотно, постоянное участие, короткие рывки 5–15 м; единоборства — норма каждые несколько секунд"},
    "5x5": {"label": "5 × 5 (+ вратарь, 6 в команде)", "outfield": 5, "field": ((35, 40), (20, 25)),
            "note": "плотная игра, рывки 10–20 м, частая смена фаз атака/оборона"},
    "6x6": {"label": "6 × 6 (+ вратарь, 7 в команде)", "outfield": 6, "field": ((40, 50), (25, 35)),
            "note": "появляются линии (2–3 защитника), рывки 10–25 м"},
    "7x7": {"label": "7 × 7 (+ вратарь, 8 в команде)", "outfield": 7, "field": ((50, 60), (35, 45)),
            "note": "позиции выражены, рывки до 30 м"},
    "8x8": {"label": "8 × 8 (+ вратарь, 9 в команде)", "outfield": 8, "field": ((60, 70), (40, 50)),
            "note": "почти «большой» футбол по структуре, больше беговой работы без мяча"},
    "11x11": {"label": "11 × 11 (10 + вратарь, классика)", "outfield": 10, "field": ((100, 110), (64, 75)),
              "note": "полноразмерное поле: длинные рывки, мяч рядом у отдельного игрока бывает редко"},
    "training": {"label": "тренировка / упражнение", "outfield": None, "field": None,
                 "note": "не матч: темп и метрики зависят от упражнения, сравнивать с игровыми ориентирами нельзя"},
    "other": {"label": "иное", "outfield": None, "field": None, "note": "формат описан в комментарии"},
}


def players_on_field(game_format: Optional[str]) -> Optional[tuple[int, int]]:
    """(в команде, всего на поле) с вратарями; None — для тренировки и «иного»."""
    fmt = GAME_FORMATS.get(game_format or "")
    if not fmt or not fmt["outfield"]:
        return None
    per_team = fmt["outfield"] + 1
    return per_team, per_team * 2


# медиана роста мальчиков (ВОЗ), м; для девочек до 12 лет почти та же, позже ниже на 5–10 см
_HEIGHT_BY_AGE = {5: 1.10, 6: 1.16, 7: 1.22, 8: 1.28, 9: 1.33, 10: 1.38, 11: 1.44, 12: 1.50, 13: 1.56,
                  14: 1.63, 15: 1.69, 16: 1.73, 17: 1.75, 18: 1.77}


def typical_height(age: Optional[float]) -> Optional[float]:
    if age is None:
        return None
    a = int(round(min(max(age, 5), 18)))
    return _HEIGHT_BY_AGE[a]


def speed_zones(age: Optional[float]) -> tuple[tuple[str, float, float], ...]:
    """Зоны скорости, м/с, по возрастной группе (ориентировочно; у взрослых спринт — от ~7 м/с)."""
    if age is not None and age < 10:
        cuts = (1.3, 2.5, 4.0)
    elif age is None or age < 13:
        cuts = (1.5, 3.0, 4.5)
    elif age < 16:
        cuts = (1.5, 3.5, 5.0)
    else:
        cuts = (2.0, 4.0, 5.5)
    return (("шаг", 0.0, cuts[0]), ("трусца", cuts[0], cuts[1]), ("бег", cuts[1], cuts[2]), ("спринт", cuts[2], math.inf))


@dataclass
class PlayerProfile:
    age: Optional[float] = None
    position: Optional[str] = None            # ключ POSITIONS
    position_note: Optional[str] = None
    game_format: Optional[str] = None         # ключ GAME_FORMATS
    format_note: Optional[str] = None
    height_m: Optional[float] = None          # не указан — типичный по возрасту

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PlayerProfile":
        d = d or {}
        age = d.get("age")
        height = d.get("height_m")
        return cls(float(age) if age not in (None, "") else None, d.get("position") or None, d.get("position_note") or None,
                   d.get("game_format") or None, d.get("format_note") or None,
                   float(height) if height not in (None, "") else None)

    def as_dict(self) -> dict:
        return asdict(self)

    def effective_height(self) -> tuple[float, str]:
        """Рост для масштаба и откуда он: указан / типичный для возраста / по умолчанию."""
        if self.height_m:
            return self.height_m, "указан"
        h = typical_height(self.age)
        if h:
            return h, f"типичный для {int(self.age)} лет"
        return 1.5, "по умолчанию (возраст не указан)"

    def position_label(self) -> Optional[str]:
        if not self.position:
            return None
        label = POSITIONS.get(self.position, self.position)
        return f"{label} ({self.position_note})" if self.position_note else label

    def format_label(self) -> Optional[str]:
        if not self.game_format:
            return None
        label = GAME_FORMATS.get(self.game_format, {}).get("label", self.game_format)
        return f"{label} ({self.format_note})" if self.format_note else label

    def guide(self) -> str:
        """Ориентиры для ИИ-разбора: формат (поле, характер игры), роль позиции, возрастные зоны скорости."""
        lines = []
        fmt = GAME_FORMATS.get(self.game_format or "")
        if fmt:
            if fmt["field"]:
                (l1, l2), (w1, w2) = fmt["field"]
                team, total = players_on_field(self.game_format)
                lines.append(f"Формат {fmt['label']}: в команде {team} игроков вместе с вратарём, на поле {total}; "
                             f"поле примерно {l1}–{l2} × {w1}–{w2} м ({(l1 * w1) // total}–{(l2 * w2) // total} м² на игрока); "
                             f"{fmt['note']}.")
            else:
                lines.append(f"Формат: {fmt['label']} — {fmt['note']}.")
        if self.format_note:
            lines.append(f"Уточнение формата: {self.format_note}.")
        if self.position:
            lines.append(f"Позиция: {self.position_label()} — оценивать по: {POSITION_FOCUS.get(self.position, 'фактической игре')}.")
        zones = speed_zones(self.age)
        lines.append("Зоны скорости для возраста " + (f"{int(self.age)} лет" if self.age else "(возраст не указан)") + ": "
                     + ", ".join(f"{n} {lo:g}–{'∞' if math.isinf(hi) else f'{hi:g}'} м/с" for n, lo, hi in zones) + ".")
        h, src = self.effective_height()
        lines.append(f"Рост для масштаба метров: {h:.2f} м ({src}).")
        return "\n".join(lines)
