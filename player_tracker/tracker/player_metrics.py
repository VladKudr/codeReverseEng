"""Метрики игрока по результату слежения (без калибровки поля).

Камера в руках и не откалибрована, поэтому метры берутся из самого игрока: рамка ростом
h пикселей вокруг игрока ростом H метров даёт масштаб H/h метра на пиксель на его глубине.

  * поперечное смещение (вдоль линии взгляда камеры поперёк) — ΔX = Δu · H / h: от объектива не
    зависит, точность — точность рамки;
  * глубина — Z = f · H / h, где f — фокусное расстояние в пикселях из угла обзора (`hfov_deg`):
    изменение глубины оценивается по изменению роста рамки, это шумнее — рост сглаживается сильнее;
  * панорамирование камеры убирается накопленным движением кадра (`FrameInputs.camera`), поэтому
    координаты — «координаты сцены», а не экрана.

Положение на поле (раздел `pitch`, лента, тепловая карта поля) — не по росту рамки, а по плоскости поля
(`ground.GroundModel`, строка ног), если она передана: частично закрытый игрок даёт рамку ниже, и по росту
он «улетал» бы вглубь на метры, а строка ног от перекрытия не меняется. Скорость и дистанция остаются по
росту: там важнее устойчивость к шуму строки ног при беге, и они калибруются тем же ростом, что и
плоскость. Так доска и метрики показывают игрока в одном месте.

Всё — оценка (±20–30 % на дистанции и скоростях): рамка «дышит» при беге, дети разного роста,
угол обзора при зуме неизвестен. Точные метры требуют калибровки поля по разметке.

Люди вне игры (запасные, тренеры, зрители у бровки — `board.classify_roles`) в расчёте не участвуют:
иначе «ближайший игрок» и единоборства считаются по стоящему рядом зрителю.

Мяч (класс COCO «sports ball») мелкий и находится не на каждом кадре. Если передана траектория мяча
(`ball.BallTrack`: найден / оценён в разрыве), «мяч рядом» считается по ней, иначе — по сырым детекциям
кадра; «мяч рядом» — мяч в пределах `ball_near_m` от ног игрока. Это признак владения/борьбы за мяч, а не
счёт касаний. Игровой контекст с тактической доски (`board.game_context`: где мяч, у кого, где игрок
относительно своей команды) добавляется в посекундную ленту и раздел `game`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .camera import scale_of
from .geometry import iou_matrix
from .pitch import CHANNELS, THIRDS, PitchSetup, run_direction
from .profile import speed_zones

SPEED_ZONES = speed_zones(None)   # 10–12 лет; для другого возраста — MetricsConfig.zones = speed_zones(age)


@dataclass
class MetricsConfig:
    player_height_m: float = 1.5
    hfov_deg: float = 70.0            # основная камера iPhone в 16:9 ≈ 70°; ультраширокая ≈ 108°; 3x ≈ 26°; 5x ≈ 17°
    smooth_sec: float = 0.5           # сглаживание положения
    height_smooth_sec: float = 1.5    # сглаживание роста рамки (глубина) для крупной рамки...
    height_ref_px: float = 60.0       # ...рамка ниже этой высоты шумит сильнее: окно растёт как (ref/h)², до
    height_smooth_max_sec: float = 5.0  # ...этого предела (±2 px на 30-px рамке — ±3 м глубины на 50 м)
    max_gap_frames: int = 6           # разрыв слежения дольше — новый отрезок (скорость через разрыв не считается)
    min_segment_sec: float = 1.0      # отрезки короче идут только в «время в кадре»
    max_speed_ms: float = 9.0         # выше — ошибка рамки, обрезаем
    sprint_min_sec: float = 1.0
    accel_ms2: float = 2.0
    accel_min_sec: float = 0.5
    contact_m: float = 1.5            # ближе — единоборство / плотная опека
    contact_min_sec: float = 0.5
    ball_near_m: float = 1.2
    ball_min_score: float = 0.25
    ball_min_sec: float = 0.3
    heat_bins: tuple[int, int] = (12, 6)
    zones: tuple = SPEED_ZONES        # (название, от, до) м/с — по возрасту игрока (profile.speed_zones)


@dataclass
class Episode:
    start_s: float
    end_s: float
    peak: float = 0.0
    distance_m: float = 0.0

    def as_dict(self) -> dict:
        return {k: round(v, 2) for k, v in asdict(self).items()}


def _moving(x: np.ndarray, win: int, fn=np.mean) -> np.ndarray:
    """Скользящее окно; для среднего края дополняются нечётным отражением — линейное движение у краёв
    отрезка не «съедается»."""
    x = np.asarray(x, dtype=float)
    if win <= 1 or len(x) < 3:
        return x.copy()
    half = min(win // 2, len(x) - 1)
    if fn is np.mean and np.isfinite(x[[0, -1]]).all():
        left = 2 * x[0] - x[1:half + 1][::-1]
        right = 2 * x[-1] - x[-half - 1:-1][::-1]
    else:                                   # медиана и значения-заглушки (inf): края просто повторяются
        left, right = np.repeat(x[0], half), np.repeat(x[-1], half)
    padded = np.concatenate([left, x, right])
    return np.array([fn(padded[i - half:i + half + 1]) for i in range(half, half + len(x))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Отрезки подряд идущих True: [(начало, конец включительно)]."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


class _Scene:
    """Кадр -> координаты сцены (без движения камеры) и положение на земле в метрах."""

    def __init__(self, cameras: list[Optional[np.ndarray]], width: int, cfg: MetricsConfig):
        self.inv: list[np.ndarray] = []
        self.scale: list[float] = []
        M = np.eye(3)
        for A in cameras:
            if A is not None:
                M = np.vstack([A, [0.0, 0.0, 1.0]]) @ M
            inv = np.linalg.inv(M)
            self.inv.append(inv)
            self.scale.append(scale_of(inv[:2]))
        self.cx = width / 2
        self.f = (width / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)
        self.H = cfg.player_height_m

    def feet(self, i: int, box) -> tuple[float, float, float]:
        """Ноги (u, v) и рост h рамки в координатах сцены."""
        x1, y1, x2, y2 = [float(v) for v in box]
        p = self.inv[i] @ np.array([(x1 + x2) / 2, y2, 1.0])
        return float(p[0]), float(p[1]), (y2 - y1) * self.scale[i]

    def ground(self, u, h):
        """(X поперёк, Z вглубь) в метрах; u, h — числа или массивы."""
        h = np.maximum(h, 1.0)
        return (u - self.cx) * self.H / h, self.f * self.H / h


def compute(records: list[dict], inputs: list, scale: float, fps: float, width: int,
            cfg: MetricsConfig | None = None, corrections: int = 0, pitch: Optional[PitchSetup] = None,
            ball=None, game: Optional[dict] = None, ignore_boxes: Optional[list] = None, ground=None) -> dict:
    """records — кадры track.json (рамки в координатах исходного кадра), inputs — FrameInputs прогона
    (кадр обработки), scale — кадр обработки / исходный кадр."""
    cfg = cfg or MetricsConfig()
    fps = fps or 30.0
    n = min(len(records), len(inputs))
    scene = _Scene([inputs[i].camera for i in range(n)], width, cfg)
    dt = 1.0 / fps

    tracked = np.zeros(n, bool)
    uncertain = np.zeros(n, bool)
    U = np.full(n, np.nan)
    Hh = np.full(n, np.nan)
    boxes: list[Optional[np.ndarray]] = [None] * n
    for i in range(n):
        r = records[i]
        if r["state"] in ("active", "contested") and r["box"]:
            b = np.asarray(r["box"], float) * scale
            boxes[i] = b
            tracked[i] = True
            uncertain[i] = r["state"] == "contested"
            U[i], _, Hh[i] = scene.feet(i, b)

    # --- отрезки слежения ---------------------------------------------------------------------------
    # отрезок — один и тот же игрок подряд: разрыв дольше max_gap_frames или смена личности (повторный
    # захват, переход на другой трек, поправка) начинают новый — иначе скачок рамки к другому игроку
    # посчитается как бег
    identity_change = {"locked", "reacquired", "switched", "corrected"}
    segments: list[tuple[int, int]] = []
    last_tid, last_i = None, -10**9
    for i in np.flatnonzero(tracked):
        tid, ev = records[i].get("track_id"), records[i].get("event")
        new = (not segments or i - last_i - 1 > cfg.max_gap_frames or ev in identity_change
               or (tid is not None and last_tid is not None and tid != last_tid))
        if new:
            segments.append((int(i), int(i)))
        else:
            segments[-1] = (segments[-1][0], int(i))
        last_i = i
        if tid is not None:
            last_tid = tid
    X = np.full(n, np.nan)
    Z = np.full(n, np.nan)
    # положение на поле по плоскости поля (строка ног): считается по тем же отрезкам и сглаживается так же
    GX = np.full(n, np.nan)
    GZ = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    moving_segments = []
    for a, b in segments:
        idx = np.arange(a, b + 1)
        ok = ~np.isnan(U[idx])
        u = np.interp(idx, idx[ok], U[idx][ok])
        h = np.interp(idx, idx[ok], Hh[idx][ok])
        win_h = cfg.height_smooth_sec * max(1.0, (cfg.height_ref_px / max(float(np.median(h)), 1.0)) ** 2)
        win_h = min(win_h, cfg.height_smooth_max_sec)
        if win_h * fps >= len(h):
            # отрезок короче окна: изменение глубины по росту рамки здесь — шум; глубина постоянна
            h = np.full_like(h, float(np.median(h)))
        else:
            h = _moving(_moving(h, int(win_h * fps), np.median), int(win_h * fps / 2))
        u = _moving(u, int(cfg.smooth_sec * fps))
        x, z = scene.ground(u, h)
        X[idx], Z[idx] = x, z
        if ground is not None:
            gx = np.array([ground.ground(int(i), (boxes[i][0] + boxes[i][2]) / 2, boxes[i][3])[0] if boxes[i] is not None
                           else np.nan for i in idx], float).ravel()
            gz = np.array([ground.ground(int(i), (boxes[i][0] + boxes[i][2]) / 2, boxes[i][3])[1] if boxes[i] is not None
                           else np.nan for i in idx], float).ravel()
            okg = ~np.isnan(gx) & ~np.isnan(gz)
            if okg.sum() >= 2:
                gx = np.interp(idx, idx[okg], gx[okg])
                gz = np.interp(idx, idx[okg], gz[okg])
                GX[idx] = _moving(gx, int(cfg.smooth_sec * fps))
                GZ[idx] = _moving(gz, int(cfg.smooth_sec * fps))
            elif okg.any():
                GX[idx], GZ[idx] = gx[okg][0], gz[okg][0]
        if (b - a + 1) * dt >= cfg.min_segment_sec:
            step = np.hypot(np.diff(x), np.diff(z)) / dt
            v = np.concatenate([[step[0] if len(step) else 0.0], step])
            # медиана гасит одиночные всплески рамки, среднее — сглаживает
            v = np.minimum(_moving(_moving(v, int(cfg.smooth_sec * fps), np.median), int(cfg.smooth_sec * fps)),
                           cfg.max_speed_ms)
            speed[idx] = v
            moving_segments.append((a, b))

    # положение на поле: по плоскости поля, если она есть, иначе по росту
    FX, FZ = (GX, GZ) if ground is not None and (~np.isnan(GX)).any() else (X, Z)
    has_speed = ~np.isnan(speed)
    total_dist = float(np.nansum(speed[has_speed]) * dt)
    zones = []
    for name, lo, hi in cfg.zones:
        m = has_speed & (speed >= lo) & (speed < hi)
        zones.append({"zone": name, "from_ms": lo, "to_ms": None if math.isinf(hi) else hi,
                      "seconds": round(m.sum() * dt, 1), "distance_m": round(float(speed[m].sum() * dt), 1)})

    sprint_lo = cfg.zones[-1][1]
    sprints, sprint_idx = [], []
    for a, b in _runs(has_speed & (speed >= sprint_lo)):
        if (b - a + 1) * dt >= cfg.sprint_min_sec:
            sprints.append(Episode(a * dt, (b + 1) * dt, float(speed[a:b + 1].max()), float(speed[a:b + 1].sum() * dt)))
            sprint_idx.append((a, b))

    accel = np.full(n, np.nan)
    for a, b in moving_segments:
        v = speed[a:b + 1]
        accel[a:b + 1] = _moving(np.gradient(v, dt), int(cfg.accel_min_sec * fps))
    accels = [Episode(a * dt, (b + 1) * dt, float(np.nanmax(accel[a:b + 1])))
              for a, b in _runs(~np.isnan(accel) & (accel >= cfg.accel_ms2)) if (b - a + 1) * dt >= cfg.accel_min_sec]
    decels = [Episode(a * dt, (b + 1) * dt, float(-np.nanmin(accel[a:b + 1])))
              for a, b in _runs(~np.isnan(accel) & (accel <= -cfg.accel_ms2)) if (b - a + 1) * dt >= cfg.accel_min_sec]

    # --- соседи и мяч -----------------------------------------------------------------------------------
    nearest = np.full(n, np.nan)
    ball_near = np.zeros(n, bool)
    ball_seen = 0
    all_x, all_z = [], []
    for i in range(n):
        dets = inputs[i].detections
        # люди вне игры (запасные, тренеры, зрители у бровки) — не соседи и не часть расстановки
        skip = np.asarray(ignore_boxes[i], float).reshape(-1, 4) if ignore_boxes is not None and i < len(ignore_boxes) \
            else np.zeros((0, 4))
        others = []
        for d in dets:
            if len(skip) and float(iou_matrix([d.box], skip).max()) > 0.6:
                continue
            u, _, h = scene.feet(i, d.box)
            gx, gz = scene.ground(u, h)
            all_x.append(gx)
            all_z.append(gz)
            others.append((d.box, gx, gz))
        if ball is not None:
            # траектория мяча: одна точка (найден или оценён в разрыве) вместо всех кандидатов кадра
            st = ball.state[i] if i < len(ball.state) else None
            balls = (np.array([[ball.u[i] - 3, ball.v[i] - 3, ball.u[i] + 3, ball.v[i] + 3, 1.0]])
                     if st is not None else np.zeros((0, 5)))
            ball_seen += st == "detected"
        else:
            balls = inputs[i].balls if getattr(inputs[i], "balls", None) is not None else np.zeros((0, 5))
            balls = balls[balls[:, 4] >= cfg.ball_min_score] if len(balls) else balls
            ball_seen += bool(len(balls))
        if boxes[i] is None or np.isnan(X[i]):
            continue
        tb = boxes[i]
        dists = [math.hypot(gx - X[i], gz - Z[i]) for b, gx, gz in others
                 if iou_matrix([tb], [b])[0, 0] < 0.5]
        if dists:
            nearest[i] = min(dists)
        if len(balls):
            th = max(tb[3] - tb[1], 1.0)
            fx, fy = (tb[0] + tb[2]) / 2, tb[3]
            for bb in balls:
                bx, by = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                if math.hypot(bx - fx, by - fy) / th * cfg.player_height_m <= cfg.ball_near_m:
                    ball_near[i] = True
                    break
    nearest_s = _moving(np.where(np.isnan(nearest), np.inf, nearest), int(cfg.smooth_sec * fps), np.median)
    nearest_s[np.isinf(nearest_s)] = np.nan
    contact_mask = ~np.isnan(nearest_s) & (nearest_s < cfg.contact_m)
    contacts = [Episode(a * dt, (b + 1) * dt, float(np.nanmin(nearest_s[a:b + 1])))
                for a, b in _runs(contact_mask) if (b - a + 1) * dt >= cfg.contact_min_sec]
    # мяч детектируется с пропусками: разрывы до 0.3 с внутри эпизода закрываем
    gap = int(0.3 * fps)
    filled = ball_near.copy()
    for a, b in _runs(~ball_near):
        if 0 < a and b < n - 1 and b - a + 1 <= gap:
            filled[a:b + 1] = tracked[a:b + 1]
    ball_eps = [Episode(a * dt, (b + 1) * dt, 0.0, float(np.nansum(speed[a:b + 1]) * dt))
                for a, b in _runs(filled & tracked) if (b - a + 1) * dt >= cfg.ball_min_sec]

    # --- положение в сцене ---------------------------------------------------------------------------------
    heat = None
    thirds = None
    if all_x and (~np.isnan(X)).any():
        xs, zs = np.array(all_x), np.array(all_z)
        x_lo, x_hi = np.percentile(xs, 5), np.percentile(xs, 95)
        z_lo, z_hi = np.percentile(zs, 5), np.percentile(zs, 95)
        ok = ~np.isnan(X)
        gx = np.clip((X[ok] - x_lo) / max(x_hi - x_lo, 1e-6), 0, 0.999)
        gz = np.clip((Z[ok] - z_lo) / max(z_hi - z_lo, 1e-6), 0, 0.999)
        bx, bz = cfg.heat_bins
        grid = np.zeros((bz, bx))
        np.add.at(grid, ((gz * bz).astype(int), (gx * bx).astype(int)), dt)
        heat = {"bins": [bx, bz], "x_range_m": [round(float(x_lo), 1), round(float(x_hi), 1)],
                "z_range_m": [round(float(z_lo), 1), round(float(z_hi), 1)], "seconds": np.round(grid, 2).tolist(),
                "note": "поперёк — слева направо в кадре, вглубь — от камеры; границы — где вообще были игроки"}
        lat = np.bincount((gx * 3).astype(int), minlength=3) * dt
        dep = np.bincount((gz * 3).astype(int), minlength=3) * dt
        tot = max(ok.sum() * dt, 1e-6)
        thirds = {"слева": round(lat[0] / tot, 3), "в центре": round(lat[1] / tot, 3), "справа": round(lat[2] / tot, 3),
                  "ближе к камере": round(dep[0] / tot, 3), "середина": round(dep[1] / tot, 3),
                  "дальше от камеры": round(dep[2] / tot, 3)}

    # --- на поле (если отмечена схема) ---------------------------------------------------------------------
    pitch_out = None
    PX = PY = None
    sprint_dicts = [e.as_dict() for e in sprints]
    if pitch is not None and pitch.project is not None:
        # по разметке поля: ноги цели на каждом кадре -> метры поля, сглаживание по отрезкам, как у положения
        PX, PY = np.full(n, np.nan), np.full(n, np.nan)
        for a, b in segments:
            idx = np.arange(a, b + 1)
            px = np.full(len(idx), np.nan)
            py = np.full(len(idx), np.nan)
            for k, i in enumerate(idx):
                if boxes[i] is not None:
                    x, y = pitch.project(int(i), (boxes[i][0] + boxes[i][2]) / 2, boxes[i][3])
                    px[k], py[k] = float(x), float(y)
            okp = ~np.isnan(px)
            if okp.sum() >= 2:
                PX[idx] = _moving(np.interp(idx, idx[okp], px[okp]), int(cfg.smooth_sec * fps))
                PY[idx] = _moving(np.interp(idx, idx[okp], py[okp]), int(cfg.smooth_sec * fps))
        if np.isnan(PX).all():
            PX = PY = None
    elif pitch is not None and (~np.isnan(FX)).any():
        PX, PY = pitch.to_pitch(FX, FZ)
    if PX is not None:
        ok = ~np.isnan(PX)
        prog, side = pitch.progress(PX), pitch.side(PY)
        tot = max(ok.sum() * dt, 1e-6)
        third_i = np.clip((prog[ok] * 3).astype(int), 0, 2)
        chan_i = np.clip((side[ok] * 3).astype(int), 0, 2)
        L, Wd = pitch.length_m, pitch.width_m
        outside = ok & ((PX < -0.1 * L) | (PX > 1.1 * L) | (PY < -0.15 * Wd) | (PY > 1.15 * Wd))
        fwd = back = 0.0
        for a, b in moving_segments:
            d = np.diff(PX[a:b + 1]) * pitch.attack_sign
            fwd += float(d[d > 0].sum())
            back += float(-d[d < 0].sum())
        for e, (a, b) in zip(sprint_dicts, sprint_idx):
            dxa = float((PX[b] - PX[a]) * pitch.attack_sign)
            e.update(direction=run_direction(dxa, float(PY[b] - PY[a])), towards_goal_m=round(dxa, 1))
        grid = np.zeros((8, 12))
        gx = np.clip(PX[ok] / L, 0, 0.999)
        gy = np.clip(PY[ok] / Wd, 0, 0.999)
        np.add.at(grid, ((gy * 8).astype(int), (gx * 12).astype(int)), dt)
        pitch_out = {
            "setup": {"camera": list(pitch.camera), "look": list(pitch.look), "own_goal": pitch.own_goal,
                      "length_m": L, "width_m": Wd},
            "description": pitch.describe(),
            "thirds": {name: round(float((third_i == k).sum() * dt / tot), 3) for k, name in enumerate(THIRDS)},
            "channels": {name: round(float((chan_i == k).sum() * dt / tot), 3) for k, name in enumerate(CHANNELS)},
            "avg_position_m": [round(float(np.nanmean(PX)), 1), round(float(np.nanmean(PY)), 1)],
            "avg_progress": round(float(np.nanmean(prog)), 2),
            "towards_opponent_goal_m": round(fwd, 1),
            "towards_own_goal_m": round(back, 1),
            "outside_share": round(float(outside.sum() / max(ok.sum(), 1)), 3),
            "heatmap": {"bins": [12, 8], "seconds": np.round(grid, 2).tolist(),
                        "note": "x — вдоль поля слева направо на схеме, y — от дальней бровки к ближней"},
        }

    # --- посекундная лента -----------------------------------------------------------------------------------
    timeline = []
    per = max(int(round(fps)), 1)
    for s in range(0, n, per):
        sl = slice(s, min(s + per, n))
        tr = tracked[sl]
        sp = speed[sl][~np.isnan(speed[sl])]
        near = nearest_s[sl][~np.isnan(nearest_s[sl])]
        row = {"t": s // per, "in_view": round(float(tr.mean()), 2)}
        if len(sp):
            v = float(sp.mean())
            row.update(speed_ms=round(v, 2), speed_max_ms=round(float(sp.max()), 2),
                       zone=next(z for z, lo, hi in cfg.zones if lo <= v < hi))
        if tr.any() and not np.isnan(FX[sl]).all():
            row.update(x_m=round(float(np.nanmean(FX[sl])), 1), z_m=round(float(np.nanmean(FZ[sl])), 1))
        if len(near):
            row["nearest_m"] = round(float(near.min()), 1)
        if PX is not None and not np.isnan(PX[sl]).all():
            pr = float(np.nanmean(pitch.progress(PX[sl])))
            sd = float(np.nanmean(pitch.side(PY[sl])))
            row.update(third=THIRDS[min(max(int(pr * 3), 0), 2)], channel=CHANNELS[min(max(int(sd * 3), 0), 2)],
                       to_goal=round(pr, 2))
        if ball_near[sl].any():
            row["ball_near"] = round(float(filled[sl].mean()), 2)
        if game and (s // per) in game["timeline"]:
            row.update(game["timeline"][s // per])
        events = [records[i]["event"] for i in range(sl.start, sl.stop) if records[i].get("event")]
        if events:
            row["events"] = events
        timeline.append(row)

    tracked_sec = tracked.sum() * dt
    seg_lengths = [(b - a + 1) * dt for a, b in segments]
    moving_sec = has_speed.sum() * dt
    notes = ["Метры и скорости — оценка по росту игрока в кадре (±20–30 %), без калибровки поля."]
    if tracked.mean() < 0.6:
        notes.append(f"Игрок в кадре и отслежен лишь {tracked.mean():.0%} ролика — итоги только по этому времени.")
    ball_summary = ball.summary() if ball is not None else None
    if ball_summary is not None:
        if ball_summary["known_share"] < 0.5:
            notes.append(f"Игровой мяч найден на {ball_summary['detected_share']:.0%} кадров, с оценкой в коротких "
                         f"разрывах — на {ball_summary['known_share']:.0%}: в остальное время (борьба у ног, мяч закрыт, "
                         "дальний план) где мяч — неизвестно, «мяч рядом» и владение занижены.")
    elif ball_seen / max(n, 1) < 0.3:
        notes.append(f"Мяч найден лишь на {ball_seen / max(n, 1):.0%} кадров — «мяч рядом» сильно занижен.")
    if pitch_out and pitch_out["outside_share"] > 0.15:
        notes.append(f"На схеме игрок {pitch_out['outside_share']:.0%} времени оказывается за пределами поля — "
                     "проверьте отметки камеры и направления взгляда.")
    if uncertain.mean() > 0.1:
        notes.append("Заметная доля кадров — в перекрытии/интерполяции: там рамка менее точна.")
    return {
        "config": {"player_height_m": cfg.player_height_m, "hfov_deg": cfg.hfov_deg},
        "duration_s": round(n * dt, 1),
        "presence": {
            "tracked_s": round(tracked_sec, 1),
            "tracked_share": round(float(tracked.mean()), 3) if n else 0.0,
            "appearances": len(segments),
            "longest_s": round(max(seg_lengths), 1) if seg_lengths else 0.0,
            "uncertain_share": round(float(uncertain.sum() / max(tracked.sum(), 1)), 3),
        },
        "movement": {
            "measured_s": round(moving_sec, 1),
            "distance_m": round(total_dist, 1),
            "distance_per_min_m": round(total_dist / (moving_sec / 60), 1) if moving_sec > 5 else None,
            "avg_speed_ms": round(float(np.nanmean(speed)), 2) if has_speed.any() else None,
            "max_speed_ms": round(float(np.nanmax(speed)), 2) if has_speed.any() else None,
            "zones": zones,
            "sprints": sprint_dicts,
            "accelerations": len(accels),
            "decelerations": len(decels),
        },
        "involvement": {
            "avg_nearest_m": round(float(np.nanmean(nearest_s)), 1) if (~np.isnan(nearest_s)).any() else None,
            "close_contact_s": round(contact_mask.sum() * dt, 1),
            "close_contacts": [e.as_dict() for e in contacts],
            "ball_near_s": round(float((filled & tracked).sum() * dt), 1),
            "ball_episodes": [e.as_dict() for e in ball_eps],
            "ball_detected_share": round(ball_seen / max(n, 1), 3),
        },
        "position": {"thirds": thirds, "heatmap": heat},
        "pitch": pitch_out,
        "ball": ball_summary,
        "game": game["summary"] if game else None,
        "quality": {"corrections": corrections, "notes": notes},
        "timeline": timeline,
    }
