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

Внешность оценивается контрастно: сходство с галереей цели минус сходство
с банком «чужих» — дескрипторов игроков, которые были видны одновременно с
целью. Одинаковая форма поднимает оба сходства, а разность остаётся
информативной: игрок, похожий на цель, но стоявший рядом с ней, похож на
свои же негативные образцы сильнее. Треки, долго видимые одновременно с
целью без перекрытия, считаются «соседями» и не могут быть ею (пока трек
жив и не участвовал в перекрытии с целью, где возможна подмена id).

Итог = взвешенное среднее доступных признаков. Порог, отрыв от второго и
подтверждение на K кадрах — три независимых предохранителя от ложного
захвата похожего одноклубника: лучше сообщить «цель потеряна» и список
кандидатов, чем молча следить не за тем игроком.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .appearance import cosine_similarity, l2_normalize
from .camera import warp_box
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
    gallery_size: int = 60
    gallery_novelty: float = 0.02       # добавлять образец, если max-сходство < 1 - novelty
    gallery_update_every: int = 2       # не чаще, чем раз в N кадров
    gallery_min_mapped_sim: float = 0.35  # не добавлять образец, слишком непохожий на галерею
    # калибровка сходства внешности
    app_sim_lo: float = 0.60            # стартовые границы, пока нет статистики
    app_sim_hi: float = 0.95
    calib_min_samples: int = 30
    calib_momentum: float = 0.98
    # контраст «цель vs. соседи» (банк негативных дескрипторов)
    negatives_per_track: int = 12
    negatives_max_tracks: int = 60
    negatives_every: int = 3            # не чаще раза в N кадров на трек
    negatives_min: int = 20             # меньше образцов — контраст не используется
    positives_min: int = 3
    contrast_top_k: int = 2
    contrast_lo: float = -0.12          # разность сходств -> [0, 1]
    contrast_hi: float = 0.04
    learn_min_contrast: float = 0.0     # учиться (галерея, негативы) только когда уверены, что на цели
    bootstrap_margin: float = 0.02      # на старте: сходство трека цели с галереей выше, чем у любого соседа, на столько
    evidence_len: int = 8               # контраст трека усредняется по последним N чистым кадрам
    swap_contrast: float = -0.13        # среднее контраста трека цели ниже -> подмена id
    swap_window: int = 3                # по скольким последним чистым кадрам судить о подмене
    swap_contrast_dirty: float = -0.2   # в затяжном перекрытии чистых кадров нет: судим по грязным, строже
    swap_window_dirty: int = 4
    accept_evidence_min: int = 2        # столько чистых наблюдений нужно кандидату для захвата
    coast_frames: int = 0               # детекция цели пропала: столько кадров вести рамку по Калману
    trust_frames: int = 90              # после поправки оператора: столько кадров указанный трек — цель без проверок
    coexist_min_frames: int = 8         # кадров одновременно с целью без перекрытия -> «сосед», не цель
    coexist_reset_gap: int = 8          # трек пропадал дольше — id мог перейти к другому, «соседство» сбрасывается
    reach_base: float = 2.0             # цель, скрывшаяся за игроками (не за краем кадра), не телепортируется:
    reach_rate: float = 0.015           #   кандидат дальше base + rate*dt высот фигуры от места потери отвергается
    app_min_accept: float = 0.45        # внешность ниже — захват запрещён при любых прочих признаках
    app_min_accept_long: float = 0.65   # то же после долгой потери (движение уже ничего не говорит)
    long_loss_frames: int = 30
    app_floor: float = 0.25             # относительный захват: внешность не ниже...
    app_rel_margin: float = 0.25        # ...и выше лучшего из остальных кандидатов на столько
    switch_margin: float = 0.08         # в перекрытии соседний трек похож на цель сильнее на столько...
    switch_frames: int = 2              # ...столько кадров подряд -> мультитрекер перепутал id, переходим
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
    post_contest_freeze: int = 10       # кадров после перекрытия не обновлять модель
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
    team_share_min: float = 0.8         # доля голосов за метку (у цели и у трека)
    team_window: int = 15               # по скольким последним кадрам трека судить о его команде


IDENTITY_CUES = ("app", "number", "cont")


@dataclass
class Candidate:
    track_id: int
    score: float
    box: np.ndarray
    cues: dict[str, float] = field(default_factory=dict)
    rejected: Optional[str] = None
    identity: float = 0.0   # балл только по признакам личности
    extra_n: int = 0        # чистых наблюдений внешности у трека
    far: bool = False       # дальше, чем цель могла уйти за время потери: захватить нельзя, но как двойник учитывается


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
        self.gallery_meta: list[tuple[int, bool]] = []   # (кадр, образец указан оператором) для каждого образца
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
        # негативные образцы: дескрипторы соседей по track_id
        self.negatives: dict[int, deque] = {}
        self._neg_last: dict[int, int] = {}
        self._neg_cache: Optional[np.ndarray] = None

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
        """Сходство с целью в [0, 1]: контрастное, если банк соседей набран, иначе калиброванное."""
        c = self.contrast(feat)
        if c is not None:
            lo, hi = self.cfg.contrast_lo, self.cfg.contrast_hi
            return float(np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        lo, hi = self.bounds()
        return float(np.clip((self.raw_similarity(feat) - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    # --- контраст с соседями ------------------------------------------------
    def add_negative(self, track_id: int, feat: np.ndarray, frame_idx: int) -> bool:
        if frame_idx - self._neg_last.get(track_id, -10**9) < self.cfg.negatives_every:
            return False
        bank = self.negatives.get(track_id)
        if bank is None:
            if len(self.negatives) >= self.cfg.negatives_max_tracks:
                oldest = min(self._neg_last, key=self._neg_last.get)
                self.negatives.pop(oldest, None)
                self._neg_last.pop(oldest, None)
            bank = self.negatives[track_id] = deque(maxlen=self.cfg.negatives_per_track)
        bank.append((frame_idx, l2_normalize(feat)))
        self._neg_last[track_id] = frame_idx
        self._neg_cache = None
        return True

    def n_negatives(self) -> int:
        return sum(len(b) for b in self.negatives.values())

    def _neg_matrix(self) -> Optional[np.ndarray]:
        if self._neg_cache is None and self.negatives:
            self._neg_cache = np.stack([f for b in self.negatives.values() for _, f in b])
        return self._neg_cache

    def contrast_ready(self) -> bool:
        return len(self.gallery) >= self.cfg.positives_min and self.n_negatives() >= self.cfg.negatives_min

    def contrast(self, feat: np.ndarray) -> Optional[float]:
        """Сходство с галереей цели минус сходство с банком соседей (среднее top-k); None — данных мало."""
        if not self.contrast_ready():
            return None
        f = l2_normalize(feat)[None, :]
        k = self.cfg.contrast_top_k
        sp = np.sort(cosine_similarity(f, np.stack(self.gallery))[0])[::-1][:k].mean()
        sn = np.sort(cosine_similarity(f, self._neg_matrix())[0])[::-1][:k].mean()
        return float(sp - sn)

    def add_feature(self, feat: np.ndarray, frame_idx: int, force: bool = False, anchor: bool = False) -> bool:
        f = l2_normalize(feat)
        if not self.gallery:
            self.gallery.append(f)
            self.gallery_meta.append((frame_idx, anchor))
            self.centroid = f.copy()
            self._last_gallery_frame = frame_idx
            return True
        if not force and frame_idx - self._last_gallery_frame < self.cfg.gallery_update_every:
            return False
        sims = cosine_similarity(f[None, :], np.stack(self.gallery))[0]
        if not force and sims.max() > 1.0 - self.cfg.gallery_novelty:
            return False
        if len(self.gallery) >= self.cfg.gallery_size:
            # вытесняем образец, самый похожий на остальные (наименее информативный); указанные оператором — никогда
            g = np.stack(self.gallery)
            s = cosine_similarity(g, g)
            np.fill_diagonal(s, -1)
            redundancy = s.max(axis=1)
            redundancy[[i for i, (_, a) in enumerate(self.gallery_meta) if a]] = -np.inf
            k = int(np.argmax(redundancy))
            self.gallery.pop(k)
            self.gallery_meta.pop(k)
        self.gallery.append(f)
        self.gallery_meta.append((frame_idx, anchor))
        self.centroid = l2_normalize(np.mean(np.stack(self.gallery), axis=0))
        self._last_gallery_frame = frame_idx
        return True

    def forget_since(self, frame_idx: int, target_track: Optional[int] = None) -> None:
        """Оператор поправил цель: всё выученное с кадра `frame_idx` могло быть выучено на чужом игроке.

        С этого кадра удаляются образцы галереи (кроме указанных оператором) и образцы соседей — пока
        автомат вёл не того игрока, настоящая цель попадала в «соседи». Выученное раньше остаётся: банк
        соседей — главный признак, без него все похожие игроки неразличимы. Образцы трека, который
        оператор назвал целью, из соседей удаляются целиком. Метка команды сбрасывается: её могли выучить
        на чужом игроке, а решение оператора важнее цвета формы."""
        keep = [i for i, (f, a) in enumerate(self.gallery_meta) if a or f < frame_idx]
        self.gallery = [self.gallery[i] for i in keep]
        self.gallery_meta = [self.gallery_meta[i] for i in keep]
        self.centroid = l2_normalize(np.mean(np.stack(self.gallery), axis=0)) if self.gallery else None
        for tid in list(self.negatives):
            if tid == target_track:
                self.negatives.pop(tid)
                self._neg_last.pop(tid, None)
                continue
            kept = deque(((f, x) for f, x in self.negatives[tid] if f < frame_idx), maxlen=self.cfg.negatives_per_track)
            if kept:
                self.negatives[tid] = kept
            else:
                self.negatives.pop(tid)
                self._neg_last.pop(tid, None)
        self._neg_cache = None
        self._team_counts.clear()
        self.team = UNKNOWN

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
        best, n = max(self._team_counts.items(), key=lambda kv: kv[1])
        # метка цели — только при устойчивом большинстве: разнобой значит, что цвет торса не информативен
        self.team = best if n >= self.cfg.team_share_min * sum(self._team_counts.values()) else UNKNOWN

    def apply_camera_motion(self, A: np.ndarray) -> None:
        if self.last_box is not None:
            self.last_box = warp_box(A, self.last_box)
            self.last_velocity = np.asarray(A, dtype=np.float64)[:, :2] @ self.last_velocity

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
        # метки команды трека — только последние кадры: id мультитрекера может перейти к другому игроку,
        # и голоса прежнего владельца не должны решать за нового
        self.track_teams: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.cfg.team_window))
        self.lost_since: int = -1
        self.events: list[tuple[int, str]] = []
        self._reads_now: dict[int, NumberRead] = {}
        self._teams_now: dict[int, int] = {}
        self.coexist: dict[int, int] = defaultdict(int)   # кадров рядом с целью без перекрытия
        self.evidence: dict[int, deque] = {}               # контраст по track_id на чистых кадрах
        self._evidence_frame: dict[int, int] = {}
        self._swap_count = 0
        self._coast = 0
        self._switch: tuple[Optional[int], int] = (None, 0)
        self._dirty: deque = deque(maxlen=8)     # (кадр, контраст) трека цели в кадрах перекрытия
        self._last_detected: dict[int, int] = {}
        self._acquired_at = -1                   # кадр, с которого автомат ведёт текущий трек (захват/переход)
        self._trusted: tuple[Optional[int], int] = (None, -1)   # (трек, до кадра): поправка оператора

    # --- служебное -------------------------------------------------------
    def _event(self, frame_idx: int, name: str) -> str:
        self.events.append((frame_idx, name))
        return name

    def _track_team(self, track_id: int) -> int:
        votes = self.track_teams.get(track_id)
        if not votes:
            return UNKNOWN
        counts: dict[int, int] = defaultdict(int)
        for lbl in votes:
            counts[lbl] += 1
        label, n = max(counts.items(), key=lambda kv: kv[1])
        if n < self.cfg.team_votes_min or n < self.cfg.team_share_min * len(votes):
            return UNKNOWN
        return label

    def _gc(self, tracks: list[Track]) -> None:
        alive = {t.track_id for t in tracks}
        for d in (self.track_numbers, self.track_teams):
            for tid in [k for k in d if k not in alive]:
                del d[tid]
        self._cooldown = {k: v for k, v in self._cooldown.items() if k in alive}
        for tid in [k for k in self.coexist if k not in alive]:
            del self.coexist[tid]
        for tid in [k for k in self.evidence if k not in alive]:
            del self.evidence[tid]
            self._evidence_frame.pop(tid, None)

    def _update_evidence(self, frame_idx: int, tracks: list[Track], features: dict[int, np.ndarray]) -> None:
        """Контраст «цель vs. соседи» каждого видимого трека; кадры с перекрытием — грязные, не копим."""
        seen = [t for t in tracks if t.detected_now]
        for t in seen:
            feat = features.get(t.track_id)
            if feat is None:
                continue
            if any(o is not t and iou_pair(o.box, t.box) > 0.1 for o in seen):
                continue
            c = self.model.contrast(feat)
            if c is None:
                continue
            bank = self.evidence.get(t.track_id)
            if bank is None or self._evidence_frame.get(t.track_id, -1) < frame_idx - 2 * self.cfg.evidence_len:
                bank = self.evidence[t.track_id] = deque(maxlen=self.cfg.evidence_len)
            bank.append(c)
            self._evidence_frame[t.track_id] = frame_idx

    def track_evidence(self, track_id: int, last: Optional[int] = None) -> tuple[Optional[float], int]:
        """Средний контраст трека (по последним `last` чистым кадрам) и число наблюдений."""
        bank = self.evidence.get(track_id)
        if not bank:
            return None, 0
        vals = list(bank)[-last:] if last else list(bank)
        return float(np.mean(vals)), len(vals)

    def _evidence_fresh(self, track_id: int, frame_idx: int) -> bool:
        return self._evidence_frame.get(track_id, -10**9) == frame_idx

    def apply_camera_motion(self, A: np.ndarray) -> None:
        """Движение камеры: последняя позиция цели переносится в координаты текущего кадра."""
        self.model.apply_camera_motion(A)

    # --- внешнее управление ----------------------------------------------
    def lock(self, track: Track, feature: Optional[np.ndarray], frame_idx: int, number: Optional[str] = None) -> TargetObservation:
        """Ручной выбор цели (клик пользователя / стартовая рамка)."""
        self.model = TargetModel(self.cfg)
        if feature is not None:
            self.model.add_feature(feature, frame_idx, force=True, anchor=True)
        if number:
            self.model.numbers.set_known(number)
        self._acquired_at = frame_idx
        self.track_id = track.track_id
        self._pending_track_id = None
        self.state = TargetState.ACTIVE
        self._suspicious = 0
        self._swap_count = 0
        self._coast = 0
        self.coexist.clear()
        self.evidence.clear()
        self.model.remember_position(track, frame_idx, self.width, self.height)
        ev = self._event(frame_idx, "locked")
        return TargetObservation(frame_idx, self.state, self.track_id, track.box.copy(), 1.0, event=ev)

    def correct(self, track: Track, feature: Optional[np.ndarray], frame_idx: int) -> TargetObservation:
        """Поправка оператора: на этом кадре цель — `track`. Выученное на прежнем (неверном) треке забывается,
        выученное раньше и все прежние указания оператора остаются."""
        if self.state == TargetState.IDLE:
            obs = self.lock(track, feature, frame_idx)
            obs.event = self._retag_event(frame_idx, "corrected")
            return obs
        on_track = self.state in (TargetState.ACTIVE, TargetState.CONTESTED)
        self.model.forget_since(self._acquired_at if on_track else frame_idx, target_track=track.track_id)
        if feature is not None:
            self.model.add_feature(feature, frame_idx, force=True, anchor=True)
        neighbours = {} if on_track else dict(self.coexist)   # соседи, набранные на чужом игроке, недостоверны
        self._reset_identity_state()
        self.coexist.update({k: v for k, v in neighbours.items() if k != track.track_id})
        self.track_teams.pop(track.track_id, None)
        self.track_numbers.pop(track.track_id, None)
        self._trusted = (track.track_id, frame_idx + self.cfg.trust_frames)
        self.track_id = track.track_id
        self.state = TargetState.ACTIVE
        self._acquired_at = frame_idx
        self._cooldown.pop(track.track_id, None)
        self.model.remember_position(track, frame_idx, self.width, self.height)
        ev = self._event(frame_idx, "corrected")
        return TargetObservation(frame_idx, self.state, self.track_id, track.box.copy(), 1.0, event=ev)

    def mark_absent(self, frame_idx: int, tracks: list[Track]) -> TargetObservation:
        """Поправка оператора: на этом кадре цели нет. Трек, который автомат вёл, — не цель; и никто из видимых
        сейчас игроков не цель (пока их треки не прервутся)."""
        if self.state in (TargetState.ACTIVE, TargetState.CONTESTED):
            self.model.forget_since(self._acquired_at)
            self._cooldown[self.track_id] = frame_idx + self.cfg.reject_cooldown
        self._reset_identity_state()
        self._trusted = (None, -1)
        for t in tracks:
            if t.detected_now:
                self.coexist[t.track_id] = self.cfg.coexist_min_frames
        self.track_id = None
        if self.state != TargetState.IDLE:
            self.state = TargetState.LOST
            self.lost_since = frame_idx
        ev = self._event(frame_idx, "corrected_absent")
        return TargetObservation(frame_idx, self.state, None, None, 0.0, event=ev)

    def _reset_identity_state(self) -> None:
        self.coexist.clear()
        self.evidence.clear()
        self._evidence_frame.clear()
        self._dirty.clear()
        self._pending_track_id = None
        self._suspicious = 0
        self._swap_count = 0
        self._coast = 0
        self._switch = (None, 0)
        self._confirm_id, self._confirm_count = None, 0

    def _retag_event(self, frame_idx: int, name: str) -> str:
        if self.events and self.events[-1][0] == frame_idx:
            self.events[-1] = (frame_idx, name)
        return name

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
                self.track_teams[tid].append(lbl)
        self._gc(tracks)
        by_id = {t.track_id: t for t in tracks}
        for t in tracks:
            if t.detected_now:
                if frame_idx - self._last_detected.get(t.track_id, frame_idx - 1) > self.cfg.coexist_reset_gap:
                    # трек возобновился после разрыва: id мог перейти к другому игроку — «соседство» не в счёт
                    self.coexist.pop(t.track_id, None)
                self._last_detected[t.track_id] = frame_idx

        if self.state == TargetState.IDLE:
            return TargetObservation(frame_idx, self.state, None, None, 0.0)
        self._update_evidence(frame_idx, tracks, features)

        event = None
        if self.state in (TargetState.ACTIVE, TargetState.CONTESTED):
            event = self._step_on_track(frame_idx, tracks, by_id, features)
            if self.state != TargetState.LOST:
                t = by_id[self.track_id]
                ev, _ = self.track_evidence(t.track_id)
                if ev is not None:
                    lo, hi = self.cfg.contrast_lo, self.cfg.contrast_hi
                    conf = float(np.clip((ev - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
                else:
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
        if (t is not None and not t.detected_now and self._coast < self.cfg.coast_frames
                and not touches_border(t.box, self.width, self.height)):
            # детекция пропала на кадр-другой (слабый отклик, частичное перекрытие): ведём рамку
            # по предсказанию, модель не трогаем
            self._coast += 1
            return None
        if t is None or not t.detected_now:
            # мультитрекер потерял цель (или трек удалён) — переходим в поиск
            self._coast = 0
            self._pending_track_id = self.track_id
            self.track_id = None
            self.state = TargetState.LOST
            self.lost_since = frame_idx
            self._confirm_id, self._confirm_count = None, 0
            return self._event(frame_idx, "lost")

        # жёсткие противоречия по номеру / команде (по номеру — со строгим порогом:
        # пара ошибочных прочтений не должна сбрасывать цель)
        trusted = self._trusted[0] == t.track_id and frame_idx <= self._trusted[1]
        rel = self.model.numbers.relation(self.track_numbers[t.track_id], self.cfg.contradict_min_weight)
        if rel is False and not trusted:
            self._hard_release(frame_idx, t.track_id, "released_number")
            return "released_number"
        if not trusted and teams_compatible(self.model.team, self._track_team(t.track_id)) is False:
            self._hard_release(frame_idx, t.track_id, "released_team")
            return "released_team"

        overlaps = {o.track_id: iou_pair(o.box, t.box) for o in tracks if o.track_id != t.track_id and o.detected_now}
        contested = any(v > self.cfg.contest_iou for v in overlaps.values())
        for tid, v in overlaps.items():
            if v > 0:
                # касание/перекрытие с целью: возможна подмена id — «соседство» не в счёт
                self.coexist.pop(tid, None)
        if contested:
            self._contest_until = frame_idx + self.cfg.post_contest_freeze
            other = None if trusted else self._check_switch(frame_idx, t, by_id, features, overlaps)
            if other is not None:
                # в перекрытии мультитрекер отдал id цели соседу: цель — на соседнем треке
                self.track_id = other.track_id
                self._switch = (None, 0)
                self.evidence.pop(t.track_id, None)
                t = other
                self._acquired_at = frame_idx
                self._event(frame_idx, "switched")
        else:
            self._switch = (None, 0)
        frozen = frame_idx <= self._contest_until
        self.state = TargetState.CONTESTED if contested else TargetState.ACTIVE

        self._coast = 0
        feat = features.get(t.track_id)
        event = None
        if feat is not None and self.model.gallery:
            raw = self.model.raw_similarity(feat)
            mapped = self.model.mapped_similarity(feat)
            contrast = self.model.contrast(feat)
            if contrast is not None:
                # подмена id: в среднем по последним чистым кадрам трек похож на соседей сильнее,
                # чем на цель (один кадр в необычной позе решения не принимает)
                ev, n = self.track_evidence(t.track_id, last=self.cfg.swap_window)
                if (not trusted and self._evidence_fresh(t.track_id, frame_idx) and n >= self.cfg.swap_window
                        and ev < self.cfg.swap_contrast):
                    self.evidence.pop(t.track_id, None)
                    self.coexist.clear()   # «соседи», набранные на подменённом треке, недостоверны
                    self._hard_release(frame_idx, t.track_id, "released_swap", cooldown=False)
                    return "released_swap"
                if not self._evidence_fresh(t.track_id, frame_idx):
                    self._dirty.append((frame_idx, contrast))
                    recent = [c for f, c in self._dirty if f > frame_idx - 2 * self.cfg.swap_window_dirty]
                    last_clean = self._evidence_frame.get(t.track_id, -10**9)
                    if (not trusted and len(recent) >= self.cfg.swap_window_dirty
                            and last_clean < frame_idx - self.cfg.swap_window_dirty
                            and float(np.mean(recent[-self.cfg.swap_window_dirty:])) < self.cfg.swap_contrast_dirty):
                        self._dirty.clear()
                        self.coexist.clear()
                        self._hard_release(frame_idx, t.track_id, "released_swap", cooldown=False)
                        return "released_swap"
                self._suspicious = 0
            # во время перекрытия дескриптор заведомо грязный, а по галерее из пары образцов сходство
            # ничего не значит — подозрения не копим
            elif frozen or len(self.model.gallery) < self.cfg.positives_min:
                self._suspicious = 0
            elif mapped < self.cfg.verify_thr:
                self._suspicious += 1
            else:
                self._suspicious = 0
            if self._suspicious >= self.cfg.verify_frames and not trusted:
                # трек больше не похож на цель: вероятна подмена идентификатора
                self.coexist.clear()
                self._hard_release(frame_idx, t.track_id, "released_swap", cooldown=False)
                return "released_swap"
            if contrast is not None:
                confident = contrast >= self.cfg.learn_min_contrast
            else:
                # старт: банка соседей ещё нет, калибровать нечем — учимся, только если трек цели похож на
                # галерею сильнее любого другого видимого сейчас игрока (относительный тест без порогов)
                rivals = [self.model.raw_similarity(features[o.track_id]) for o in tracks
                          if o.track_id != t.track_id and o.track_id in features]
                confident = not rivals or raw >= max(rivals) + self.cfg.bootstrap_margin
            # учимся на чистых кадрах (цель ни с кем не соприкасается): «заморозка после перекрытия»
            # не нужна — подмену id ловит контраст, а обучение на соседе отсекает `confident`
            clean_now = max(overlaps.values(), default=0.0) <= 0.1
            trusted = (contrast >= self.cfg.swap_contrast) if contrast is not None else self._contest_until < 0
            if clean_now and trusted:
                # «соседство» — факт времени, а не внешности: считаем на чистых кадрах, если нет подозрения на
                # подмену (без контраста — только пока цель ни с кем не перекрывалась с момента выбора)
                for o in tracks:
                    if o.track_id != t.track_id and o.detected_now and overlaps.get(o.track_id, 0.0) == 0.0:
                        self.coexist[o.track_id] += 1
            if clean_now and confident:
                self.model.observe_self(raw)
                for o in tracks:
                    if o.track_id != t.track_id and o.track_id in features:
                        self.model.observe_other(self.model.raw_similarity(features[o.track_id]))
                        if overlaps.get(o.track_id, 0.0) == 0.0:
                            self.model.add_negative(o.track_id, features[o.track_id], frame_idx)
                if not touches_border(t.box, self.width, self.height) and (
                        mapped >= self.cfg.gallery_min_mapped_sim or not self.model.contrast_ready()):
                    # на старте «похожесть» в абсолютных числах не откалибрована — годится относительный тест выше
                    self.model.add_feature(feat, frame_idx)
        if not frozen:
            # номер и команда учатся только вне перекрытий: иначе чужой номер попадёт в модель
            self._absorb_track_identity(t.track_id)
        self.model.remember_position(t, frame_idx, self.width, self.height)
        return event

    def _check_switch(self, frame_idx: int, t: Track, by_id: dict[int, Track], features: dict[int, np.ndarray],
                      overlaps: dict[int, float]) -> Optional[Track]:
        """В перекрытии сравнить трек цели с перекрывающими: чей дескриптор ближе к цели.

        Дескрипторы в перекрытии грязные у обоих, но у рамки, центрированной на
        цели, пикселей цели больше. Переход — только при устойчивом перевесе
        несколько кадров подряд."""
        feat = features.get(t.track_id)
        own = self.model.contrast(feat) if feat is not None else None
        if own is None:
            return None
        best, best_c = None, -1e9
        for tid, v in overlaps.items():
            if v <= self.cfg.contest_iou or tid not in features or tid not in by_id:
                continue
            c = self.model.contrast(features[tid])
            if c is not None and c > best_c:
                best, best_c = by_id[tid], c
        if best is None or best_c - own < self.cfg.switch_margin:
            self._switch = (None, 0)
            return None
        sid, n = self._switch
        n = n + 1 if sid == best.track_id else 1
        self._switch = (best.track_id, n)
        return best if n >= self.cfg.switch_frames else None

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
        self._swap_count = 0
        self._coast = 0
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
        if self.coexist.get(t.track_id, 0) >= self.cfg.coexist_min_frames:
            cand.rejected = "neighbour"
            return cand
        m = self.model
        far = False
        if m.last_box is not None and not m.exited_via_border and rel is not True:
            dt_reach = max(frame_idx - m.last_seen_frame, 1)
            d = float(np.linalg.norm(box_center(t.box)[0] - box_center(m.last_box)[0])) / m.last_height
            far = d > self.cfg.reach_base + self.cfg.reach_rate * dt_reach
        if feat is None:
            cand.rejected = "no_feature"
            return cand

        m = self.model
        dt = max(frame_idx - m.last_seen_frame, 1)
        cues: dict[str, tuple[float, float]] = {}
        ev, n_ev = self.track_evidence(t.track_id)
        if ev is not None:
            lo, hi = self.cfg.contrast_lo, self.cfg.contrast_hi
            cues["app"] = (float(np.clip((ev - lo) / max(hi - lo, 1e-6), 0.0, 1.0)), self.cfg.w_app)
        elif self.model.contrast(feat) is not None:
            cand.rejected = "no_clean_view"   # виден только в перекрытии — личность не проверить
            return cand
        else:
            cues["app"] = (m.mapped_similarity(feat), self.cfg.w_app)
        cand.extra_n = n_ev
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
        cand.far = far
        return cand

    def _step_lost(self, frame_idx: int, tracks: list[Track], by_id: dict[int, Track],
                   features: dict[int, np.ndarray]) -> TargetObservation:
        lost_frames = frame_idx - self.lost_since
        tid, until = self._trusted
        if tid is not None and frame_idx <= until and tid in by_id and by_id[tid].detected_now:
            # игрок, указанный оператором, снова виден (пропуск детекции, перекрытие) — это цель без проверок
            t = by_id[tid]
            self.track_id, self._pending_track_id, self.state = tid, None, TargetState.ACTIVE
            self._confirm_id, self._confirm_count = None, 0
            self.model.remember_position(t, frame_idx, self.width, self.height)
            ev = self._event(frame_idx, "reacquired")
            return TargetObservation(frame_idx, self.state, tid, t.box.copy(), 1.0, event=ev, lost_frames=lost_frames)
        cands = [self._score(frame_idx, t, features.get(t.track_id)) for t in tracks if t.detected_now]
        scored = sorted((c for c in cands if c.rejected is None), key=lambda c: c.score, reverse=True)
        ok = [c for c in scored if not c.far]
        obs = TargetObservation(frame_idx, TargetState.LOST, None, None, 0.0, candidates=cands, lost_frames=lost_frames)
        if not ok:
            self._confirm_id, self._confirm_count = None, 0
            return obs
        best = ok[0]
        second = ok[1].score if len(ok) > 1 else 0.0
        obs.confidence = best.score
        app_min = self.cfg.app_min_accept_long if lost_frames > self.cfg.long_loss_frames else self.cfg.app_min_accept
        app = best.cues.get("app", 1.0)
        rival_apps = [c.cues.get("app", 0.0) for c in scored if c is not best]
        # относительный захват: внешность средняя (слабый детектор, дрожащие рамки), но все остальные
        # видимые кандидаты заметно хуже; одинокого кандидата это правило не касается
        relative_ok = bool(rival_apps) and app >= self.cfg.app_floor and app - max(rival_apps) >= self.cfg.app_rel_margin
        if best.score < self.cfg.accept_thr or (app < app_min and not relative_ok):
            self._confirm_id, self._confirm_count = None, 0
            return obs
        quick = (best.track_id == self._pending_track_id and lost_frames <= self.cfg.quick_gap_frames)
        if self.model.contrast_ready() and not quick and best.extra_n < self.cfg.accept_evidence_min:
            return obs   # мало чистых наблюдений кандидата — ждём
        # далёкий кандидат захвачен быть не может, но похожий на цель двойник вдали всё равно делает
        # выбор неоднозначным: цель могла оказаться и там (например, если оценка движения камеры сбилась)
        rivals = [c for c in scored if c is not best]
        second_identity = max((c.identity for c in rivals), default=0.0)
        ambiguous = best.score - second < self.cfg.margin
        if not quick and rivals and best.identity - second_identity < self.cfg.margin_identity:
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
        self._swap_count = 0
        self._confirm_id, self._confirm_count = None, 0
        self.model.remember_position(t, frame_idx, self.width, self.height)
        if t.track_id in features and best.score >= self.cfg.accept_thr + self.cfg.margin:
            self.model.add_feature(features[t.track_id], frame_idx)
        obs.state = TargetState.ACTIVE
        obs.track_id = t.track_id
        obs.box = t.box.copy()
        obs.event = self._event(frame_idx, "reacquired")
        self._acquired_at = frame_idx
        return obs
