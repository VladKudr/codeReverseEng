"""Модель целевого игрока и автомат слежения с повторным захватом.

Почему не достаточно мультитрекера: идентификаторы ByteTrack живут до первого
серьёзного перекрытия или выхода из кадра. `TargetFollower` держит поверх
него долгоживущую модель ОДНОГО игрока и решает три задачи:

  * верификация — пока цель «на треке», каждый кадр проверяется, что трек
    всё ещё похож на цель (защита от подмены идентификатора при столкновении
    двух игроков в одинаковой форме);
  * повторный захват — когда цель потеряна, все видимые треки оцениваются по
    нескольким независимым признакам, и захват происходит только при
    уверенном отрыве лучшего кандидата от второго и подтверждении на
    нескольких кадрах подряд;
  * накопление модели — галерея дескрипторов внешности с разных ракурсов,
    голоса за номер, метка команды, размер фигуры.

Признаки кандидата и их роль:
  app     — сходство с галереей внешности (всегда есть; калибруется по
            статистике «цель vs. остальные», чтобы одинаковая форма не
            давала всем высокие оценки);
  number  — номер на футболке: совпадение подтверждает, противоречие — жёсткий отказ;
  team    — цвет формы: другая команда — жёсткий отказ;
  motion  — близость к экстраполированному положению; вес быстро затухает
            с временем потери (камера в руках движется, поле большое);
  size    — высота фигуры относительно последней виденной (слабый признак);
  cont    — непрерывность: тот же идентификатор мультитрекера после краткого
            пропуска детекций (бонус, затухает за десяток кадров).

Итог = взвешенное среднее доступных признаков. Порог, отрыв от второго и
подтверждение на K кадрах — три независимых предохранителя от ложного
захвата похожего одноклубника: лучше сообщить «цель потеряна» и список
кандидатов, чем молча следить не за тем игроком.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .appearance import cosine_similarity, l2_normalize
from .geometry import box_center, iou_pair, touches_border
from .jersey import NumberRead, NumberVotes
from .multitracker import Track
from .team import UNKNOWN, teams_compatible


class TargetState(Enum):
    IDLE = "idle"            # цель ещё не выбрана
    ACTIVE = "active"        # цель на треке, модель обновляется
    CONTESTED = "contested"  # цель на треке, но перекрыта другим игроком — модель заморожена
    LOST = "lost"            # трек цели утрачен, идёт поиск


@dataclass
class TargetConfig:
    # галерея внешности
    gallery_size: int = 40
    gallery_novelty: float = 0.02       # добавлять образец, если max-сходство < 1 - novelty
    gallery_update_every: int = 5       # не чаще, чем раз в N кадров
    gallery_min_mapped_sim: float = 0.35  # не добавлять образец, слишком непохожий на галерею
    # калибровка сходства внешности
    app_sim_lo: float = 0.60            # стартовые границы, пока нет статистики
    app_sim_hi: float = 0.95
    calib_min_samples: int = 30
    calib_momentum: float = 0.98
    # веса признаков
    w_app: float = 1.0
    w_num: float = 1.5
    w_motion: float = 0.8
    w_size: float = 0.3
    w_cont: float = 0.7
    # решение о захвате
    accept_thr: float = 0.55
    margin: float = 0.12                # отрыв лучшего кандидата от второго по общему баллу
    margin_identity: float = 0.08       # отрыв по признакам личности (app/number/cont) — движение
                                        # после долгой потери не должно решать спор двойников
    confirm_frames: int = 3
    quick_confirm_frames: int = 1       # для «непрерывного» кандидата после краткого пропуска
    quick_gap_frames: int = 12
    # верификация на треке
    verify_thr: float = 0.30            # калиброванное сходство ниже — подозрительный кадр
    verify_frames: int = 6              # подряд подозрительных кадров -> отпустить трек
    contest_iou: float = 0.25
    post_contest_freeze: int = 5        # кадров после перекрытия не обновлять модель
    # модель движения
    motion_sigma0: float = 0.8          # в высотах рамки
    motion_sigma_rate: float = 0.12     # рост σ за кадр потери
    motion_tau: float = 45.0            # затухание веса движения (кадры)
    motion_max_extrap: int = 8          # экстраполяция скоростью не дальше N кадров
    size_sigma: float = 0.35            # σ log-отношения высот
    size_tau: float = 150.0
    cont_tau: float = 12.0
    # отказы
    reject_cooldown: int = 20           # кадров игнорировать трек после жёсткого отказа
    contradict_min_weight: float = 2.5  # вес чужого номера на треке цели, чтобы отпустить трек
    team_votes_min: int = 5             # голосов для уверенной метки команды трека


IDENTITY_CUES = ("app", "number", "cont")


@dataclass
class Candidate:
    track_id: int
    score: float
    box: np.ndarray
    cues: dict[str, float] = field(default_factory=dict)
    rejected: Optional[str] = None
    identity: float = 0.0   # балл только по признакам личности


@dataclass
class TargetObservation:
    frame_idx: int
    state: TargetState
    track_id: Optional[int]
    box: Optional[np.ndarray]
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    ambiguous: bool = False
    event: Optional[str] = None
    lost_frames: int = 0


class TargetModel:
    """Долгоживущая модель игрока: галерея внешности, номер, команда, размер."""

    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg
        self.gallery: list[np.ndarray] = []
        self.centroid: Optional[np.ndarray] = None
        self.numbers = NumberVotes()
        self.team: int = UNKNOWN
        self._team_counts: dict[int, int] = defaultdict(int)
        self.last_box: Optional[np.ndarray] = None
        self.last_velocity: np.ndarray = np.zeros(2)
        self.last_height: float = 0.0
        self.last_seen_frame: int = -1
        self.exited_via_border: bool = False
        self._last_gallery_frame = -10**9
        # калибровка: сходство «своих» и «чужих» образцов
        self._self_mean: Optional[float] = None
        self._other_mean: Optional[float] = None
        self._other_std: Optional[float] = None
        self._self_n = 0
        self._other_n = 0

    # --- внешность --------------------------------------------------------
    def raw_similarity(self, feat: np.ndarray) -> float:
        if not self.gallery:
            return 0.0
        f = l2_normalize(feat)[None, :]
        g = np.stack(self.gallery)
        best = float(cosine_similarity(f, g).max())
        cen = float(cosine_similarity(f, self.centroid[None, :])[0, 0]) if self.centroid is not None else best
        return 0.6 * best + 0.4 * cen

    def bounds(self) -> tuple[float, float]:
        """Границы отображения косинусного сходства в [0, 1] (адаптивные)."""
        lo, hi = self.cfg.app_sim_lo, self.cfg.app_sim_hi
        if self._self_n >= self.cfg.calib_min_samples and self._other_n >= self.cfg.calib_min_samples:
            lo_c = self._other_mean + 0.5 * (self._other_std or 0.0)
            hi_c = self._self_mean
            if hi_c - lo_c > 0.02:
                lo, hi = lo_c, hi_c
        return lo, hi

    def mapped_similarity(self, feat: np.ndarray) -> float:
        lo, hi = self.bounds()
        return float(np.clip((self.raw_similarity(feat) - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def add_feature(self, feat: np.ndarray, frame_idx: int, force: bool = False) -> bool:
        f = l2_normalize(feat)
        if not self.gallery:
            self.gallery.append(f)
            self.centroid = f.copy()
            self._last_gallery_frame = frame_idx
            return True
        if not force and frame_idx - self._last_gallery_frame < self.cfg.gallery_update_every:
            return False
        sims = cosine_similarity(f[None, :], np.stack(self.gallery))[0]
        if not force and sims.max() > 1.0 - self.cfg.gallery_novelty:
            return False
        if len(self.gallery) >= self.cfg.gallery_size:
            # вытесняем образец, самый похожий на остальные (наименее информативный)
            g = np.stack(self.gallery)
            s = cosine_similarity(g, g)
            np.fill_diagonal(s, -1)
            self.gallery.pop(int(np.argmax(s.max(axis=1))))
        self.gallery.append(f)
        self.centroid = l2_normalize(np.mean(np.stack(self.gallery), axis=0))
        self._last_gallery_frame = frame_idx
        return True

    def observe_self(self, raw_sim: float) -> None:
        m = self.cfg.calib_momentum
        self._self_mean = raw_sim if self._self_mean is None else m * self._self_mean + (1 - m) * raw_sim
        self._self_n += 1

    def observe_other(self, raw_sim: float) -> None:
        m = self.cfg.calib_momentum
        if self._other_mean is None:
            self._other_mean, self._other_std = raw_sim, 0.05
        else:
            self._other_mean = m * self._other_mean + (1 - m) * raw_sim
            dev = abs(raw_sim - self._other_mean)
            self._other_std = m * self._other_std + (1 - m) * dev
        self._other_n += 1

    # --- прочие признаки ---------------------------------------------------
    def observe_team(self, label: int) -> None:
        if label == UNKNOWN:
            return
        self._team_counts[label] += 1
        self.team = max(self._team_counts.items(), key=lambda kv: kv[1])[0]

    def remember_position(self, track: Track, frame_idx: int, width: int, height: int) -> None:
        box = track.last_box if track.last_box is not None else track.box
        self.last_box = np.asarray(box, dtype=np.float64)
        self.last_velocity = track.velocity
        self.last_height = max(float(track.height), 1.0)
        self.last_seen_frame = frame_idx
        self.exited_via_border = touches_border(self.last_box, width, height, margin=0.03)


class TargetFollower:
    def __init__(self, frame_size: tuple[int, int], cfg: TargetConfig | None = None):
        self.cfg = cfg or TargetConfig()
        self.width, self.height = frame_size
        self.model = TargetModel(self.cfg)
        self.state = TargetState.IDLE
        self.track_id: Optional[int] = None
        self._pending_track_id: Optional[int] = None   # id, потерянный мультитрекером ненадолго
        self._suspicious = 0
        self._contest_until = -1
        self._confirm_id: Optional[int] = None
        self._confirm_count = 0
        self._cooldown: dict[int, int] = {}
        self.track_numbers: dict[int, NumberVotes] = defaultdict(NumberVotes)
        self.track_teams: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.lost_since: int = -1
        self.events: list[tuple[int, str]] = []
        self._reads_now: dict[int, NumberRead] = {}
        self._teams_now: dict[int, int] = {}

    # --- служебное -------------------------------------------------------
    def _event(self, frame_idx: int, name: str) -> str:
        self.events.append((frame_idx, name))
        return name

    def _track_team(self, track_id: int) -> int:
        counts = self.track_teams.get(track_id)
        if not counts:
            return UNKNOWN
        label, n = max(counts.items(), key=lambda kv: kv[1])
        return label if n >= self.cfg.team_votes_min else UNKNOWN

    def _gc(self, tracks: list[Track]) -> None:
        alive = {t.track_id for t in tracks}
        for d in (self.track_numbers, self.track_teams):
            for tid in [k for k in d if k not in alive]:
                del d[tid]
        self._cooldown = {k: v for k, v in self._cooldown.items() if k in alive}

    # --- внешнее управление ----------------------------------------------
    def lock(self, track: Track, feature: Optional[np.ndarray], frame_idx: int, number: Optional[str] = None) -> TargetObservation:
        """Ручной выбор цели (клик пользователя / стартовая рамка)."""
        self.model = TargetModel(self.cfg)
        if feature is not None:
            self.model.add_feature(feature, frame_idx, force=True)
        if number:
            self.model.numbers.set_known(number)
        self.track_id = track.track_id
        self._pending_track_id = None
        self.state = TargetState.ACTIVE
        self._suspicious = 0
        self.model.remember_position(track, frame_idx, self.width, self.height)
        ev = self._event(frame_idx, "locked")
        return TargetObservation(frame_idx, self.state, self.track_id, track.box.copy(), 1.0, event=ev)

    def release(self, frame_idx: int, reason: str = "released_manual") -> None:
        if self.state in (TargetState.ACTIVE, TargetState.CONTESTED):
            self._pending_track_id = None
        self.track_id = None
        self.state = TargetState.LOST
        self.lost_since = frame_idx
        self._confirm_id, self._confirm_count = None, 0
        self._event(frame_idx, reason)

    # --- основной шаг ------------------------------------------------------
    def step(self, frame_idx: int, tracks: list[Track], features: Optional[dict[int, np.ndarray]] = None,
             number_reads: Optional[dict[int, NumberRead]] = None,
             team_labels: Optional[dict[int, int]] = None) -> TargetObservation:
        """Один кадр. `tracks` — подтверждённые треки мультитрекера (и потерянные, если переданы).

        features     — дескриптор по track_id (только у детектированных сейчас);
        number_reads — прочтения номеров по track_id за этот кадр;
        team_labels  — метки команды по track_id за этот кадр.
        """
        features = features or {}
        number_reads = number_reads or {}
        team_labels = team_labels or {}
        self._reads_now, self._teams_now = number_reads, team_labels
        for tid, r in number_reads.items():
            self.track_numbers[tid].add(r)
        for tid, lbl in team_labels.items():
            if lbl != UNKNOWN:
                self.track_teams[tid][lbl] += 1
        self._gc(tracks)
        by_id = {t.track_id: t for t in tracks}

        if self.state == TargetState.IDLE:
            return TargetObservation(frame_idx, self.state, None, None, 0.0)

        event = None
        if self.state in (TargetState.ACTIVE, TargetState.CONTESTED):
            event = self._step_on_track(frame_idx, tracks, by_id, features)
            if self.state != TargetState.LOST:
                t = by_id[self.track_id]
                conf = self.model.mapped_similarity(features[t.track_id]) if t.track_id in features else 0.5
                return TargetObservation(frame_idx, self.state, self.track_id, t.box.copy(),
                                         float(0.5 + 0.5 * conf), event=event)

        obs = self._step_lost(frame_idx, tracks, by_id, features)
        if event and obs.event is None:
            obs.event = event
        return obs

    # --- цель на треке -------------------------------------------------------
    def _step_on_track(self, frame_idx: int, tracks: list[Track], by_id: dict[int, Track],
                       features: dict[int, np.ndarray]) -> Optional[str]:
        t = by_id.get(self.track_id)
        if t is None or not t.detected_now:
            # мультитрекер потерял цель (или трек удалён) — переходим в поиск
            self._pending_track_id = self.track_id
            self.track_id = None
            self.state = TargetState.LOST
            self.lost_since = frame_idx
            self._confirm_id, self._confirm_count = None, 0
            return self._event(frame_idx, "lost")

        # жёсткие противоречия по номеру / команде (по номеру — со строгим порогом:
        # пара ошибочных прочтений не должна сбрасывать цель)
        rel = self.model.numbers.relation(self.track_numbers[t.track_id], self.cfg.contradict_min_weight)
        if rel is False:
            self._hard_release(frame_idx, t.track_id, "released_number")
            return "released_number"
        if teams_compatible(self.model.team, self._track_team(t.track_id)) is False:
            self._hard_release(frame_idx, t.track_id, "released_team")
            return "released_team"

        contested = any(
            o.track_id != t.track_id and o.detected_now and iou_pair(o.box, t.box) > self.cfg.contest_iou
            for o in tracks
        )
        if contested:
            self._contest_until = frame_idx + self.cfg.post_contest_freeze
        frozen = frame_idx <= self._contest_until
        self.state = TargetState.CONTESTED if contested else TargetState.ACTIVE

        feat = features.get(t.track_id)
        event = None
        if feat is not None and self.model.gallery:
            raw = self.model.raw_similarity(feat)
            mapped = self.model.mapped_similarity(feat)
            # во время перекрытия дескриптор заведомо грязный — подозрения не копим
            if frozen:
                self._suspicious = 0
            elif mapped < self.cfg.verify_thr:
                self._suspicious += 1
            else:
                self._suspicious = 0
            if self._suspicious >= self.cfg.verify_frames:
                # трек больше не похож на цель: вероятна подмена идентификатора
                self._hard_release(frame_idx, t.track_id, "released_swap", cooldown=False)
                return "released_swap"
            if not frozen:
                self.model.observe_self(raw)
                for o in tracks:
                    if o.track_id != t.track_id and o.track_id in features:
                        self.model.observe_other(self.model.raw_similarity(features[o.track_id]))
                if not touches_border(t.box, self.width, self.height) and mapped >= self.cfg.gallery_min_mapped_sim:
                    self.model.add_feature(feat, frame_idx)
        if not frozen:
            # номер и команда учатся только вне перекрытий: иначе чужой номер попадёт в модель
            self._absorb_track_identity(t.track_id)
        self.model.remember_position(t, frame_idx, self.width, self.height)
        return event

    def _absorb_track_identity(self, track_id: int) -> None:
        """Прочтения номера и метка команды текущего кадра идут в модель цели.

        Голоса самого трека не стираются: именно по ним ловится противоречие
        («на треке стабильно читается чужой номер»).
        """
        read = self._reads_now.get(track_id)
        if read is not None:
            self.model.numbers.add(read)
        label = self._teams_now.get(track_id, UNKNOWN)
        if label != UNKNOWN:
            self.model.observe_team(label)

    def _hard_release(self, frame_idx: int, track_id: int, reason: str, cooldown: bool = True) -> None:
        if cooldown:
            self._cooldown[track_id] = frame_idx + self.cfg.reject_cooldown
        self._pending_track_id = None
        self.track_id = None
        self.state = TargetState.LOST
        self.lost_since = frame_idx
        self._suspicious = 0
        self._confirm_id, self._confirm_count = None, 0
        self._event(frame_idx, reason)

    # --- поиск -------------------------------------------------------------
    def _score(self, frame_idx: int, t: Track, feat: Optional[np.ndarray]) -> Candidate:
        cand = Candidate(t.track_id, 0.0, t.box.copy())
        if teams_compatible(self.model.team, self._track_team(t.track_id)) is False:
            cand.rejected = "team"
            return cand
        rel = self.model.numbers.relation(self.track_numbers.get(t.track_id, NumberVotes()))
        if rel is False:
            cand.rejected = "number"
            return cand
        if self._cooldown.get(t.track_id, -1) >= frame_idx:
            cand.rejected = "cooldown"
            return cand
        if feat is None:
            cand.rejected = "no_feature"
            return cand

        m = self.model
        dt = max(frame_idx - m.last_seen_frame, 1)
        cues: dict[str, tuple[float, float]] = {}
        cues["app"] = (m.mapped_similarity(feat), self.cfg.w_app)
        if rel is True:
            cues["number"] = (1.0, self.cfg.w_num)
        if m.last_box is not None:
            k = min(dt, self.cfg.motion_max_extrap)
            pred = box_center(m.last_box)[0] + m.last_velocity * k
            d = float(np.linalg.norm(box_center(t.box)[0] - pred)) / m.last_height
            sigma = self.cfg.motion_sigma0 + self.cfg.motion_sigma_rate * dt
            w = self.cfg.w_motion * math.exp(-dt / self.cfg.motion_tau)
            if m.exited_via_border:
                w *= 0.5
            cues["motion"] = (math.exp(-d * d / (2 * sigma * sigma)), w)
            ratio = math.log(max(float(t.height), 1.0) / m.last_height)
            cues["size"] = (math.exp(-ratio * ratio / (2 * self.cfg.size_sigma ** 2)),
                            self.cfg.w_size * math.exp(-dt / self.cfg.size_tau))
        if self._pending_track_id is not None and t.track_id == self._pending_track_id:
            cues["cont"] = (1.0, self.cfg.w_cont * math.exp(-dt / self.cfg.cont_tau))
        total_w = sum(w for _, w in cues.values())
        cand.score = sum(s * w for s, w in cues.values()) / max(total_w, 1e-9)
        ident = {k: v for k, v in cues.items() if k in IDENTITY_CUES}
        cand.identity = sum(s * w for s, w in ident.values()) / max(sum(w for _, w in ident.values()), 1e-9)
        cand.cues = {k: round(s, 3) for k, (s, _) in cues.items()}
        return cand

    def _step_lost(self, frame_idx: int, tracks: list[Track], by_id: dict[int, Track],
                   features: dict[int, np.ndarray]) -> TargetObservation:
        lost_frames = frame_idx - self.lost_since
        cands = [self._score(frame_idx, t, features.get(t.track_id)) for t in tracks if t.detected_now]
        ok = sorted((c for c in cands if c.rejected is None), key=lambda c: c.score, reverse=True)
        obs = TargetObservation(frame_idx, TargetState.LOST, None, None, 0.0, candidates=cands, lost_frames=lost_frames)
        if not ok:
            self._confirm_id, self._confirm_count = None, 0
            return obs
        best = ok[0]
        second = ok[1].score if len(ok) > 1 else 0.0
        obs.confidence = best.score
        if best.score < self.cfg.accept_thr:
            self._confirm_id, self._confirm_count = None, 0
            return obs
        quick = (best.track_id == self._pending_track_id and lost_frames <= self.cfg.quick_gap_frames)
        second_identity = max((c.identity for c in ok[1:]), default=0.0)
        ambiguous = best.score - second < self.cfg.margin
        if not quick and len(ok) > 1 and best.identity - second_identity < self.cfg.margin_identity:
            ambiguous = True   # двойники: движение после потери спор не решает
        if ambiguous:
            obs.ambiguous = True
            self._confirm_id, self._confirm_count = None, 0
            return obs
        if self._confirm_id == best.track_id:
            self._confirm_count += 1
        else:
            self._confirm_id, self._confirm_count = best.track_id, 1
        need = self.cfg.quick_confirm_frames if quick else self.cfg.confirm_frames
        if self._confirm_count < need:
            return obs
        # захват
        t = by_id[best.track_id]
        self.track_id = t.track_id
        self._pending_track_id = None
        self.state = TargetState.ACTIVE
        self._suspicious = 0
        self._confirm_id, self._confirm_count = None, 0
        self.model.remember_position(t, frame_idx, self.width, self.height)
        if t.track_id in features and best.score >= self.cfg.accept_thr + self.cfg.margin:
            self.model.add_feature(features[t.track_id], frame_idx)
        obs.state = TargetState.ACTIVE
        obs.track_id = t.track_id
        obs.box = t.box.copy()
        obs.event = self._event(frame_idx, "reacquired")
        return obs
