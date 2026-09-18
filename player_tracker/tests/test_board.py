"""Тактическая доска: команды по цвету формы, координаты на поле, мяч, игровой контекст."""
import numpy as np

from tracker.ball import BallTrack
from tracker.board import (BoardConfig, TrackFrames, build_board, classify_roles, display_numbers, game_context,
                           load_tracks, resolve_roles, save_tracks, track_signatures)
from tracker.ground import GroundConfig, GroundModel
from tracker.pitch import PitchSetup
from tracker.team import OTHER

FPS, W = 30.0, 1280
A, V0, H = 0.8, 400.0, 1.5
RED = np.array([120.0, 180.0, 150.0], np.float32)     # Lab: «красная» форма
DARK = np.array([30.0, 128.0, 128.0], np.float32)     # тёмная форма соперника
GREY = np.array([180.0, 128.0, 128.0], np.float32)    # судья/тренер


def person(u, v, tall=1.0):
    """Рамка ребёнка ростом H у строки ног v; tall > 1 — взрослый (рамка выше ожидаемого роста)."""
    h = A * (v - V0) * tall
    return [u - h / 4, v - h, u + h / 4, v]


def scene(n=60, mates=4, opps=4, others=1, opps_every=1):
    """mates и opps — по обе стороны кадра, судьи (взрослые) — с краю; трек 1 — цель.
    opps_every — соперники видны только на каждом k-м кадре (проверка выбора команд по присутствию)."""
    ids, boxes, dets, colors = [], [], [], []
    for i in range(n):
        row_id, row_box, row_det, row_col = [], [], [], []
        k = 0
        for j in range(mates):
            row_id.append(1 + j)
            row_box.append(person(400 + 60 * j + i, 560 + 30 * j))
            row_det.append(k)
            row_col.append(RED)
            k += 1
        for j in range(opps if i % opps_every == 0 else 0):
            row_id.append(20 + j)
            row_box.append(person(700 + 60 * j - i, 480 + 30 * j))
            row_det.append(k)
            row_col.append(DARK)
            k += 1
        for j in range(others):
            row_id.append(40 + j)
            row_box.append(person(1000 + 40 * j, 570 + 20 * j, tall=1.3))
            row_det.append(k)
            row_col.append(GREY)
            k += 1
        ids.append(np.array(row_id))
        boxes.append(np.array(row_box, np.float32))
        dets.append(np.array(row_det))
        colors.append(np.stack(row_col))
    return TrackFrames(ids, boxes, dets), colors


def records_for(tracks, n):
    out = []
    for i in range(n):
        box = tracks.boxes[i][0]
        out.append({"frame": i, "state": "active", "box": [float(v) for v in box], "track_id": 1, "event": None})
    return out


def ground_for(tracks, n):
    people = [np.hstack([tracks.boxes[i], np.full((len(tracks.boxes[i]), 1), 0.9, np.float32)]) for i in range(n)]
    return GroundModel.fit(people, [None] * n, W, GroundConfig(player_height_m=H))


def build(n=60, ball=True):
    tracks, colors = scene(n)
    g = ground_for(tracks, n)
    bt = None
    if ball:
        u = np.array([430.0 + i for i in range(n)])
        v = np.full(n, 565.0)
        X, Z = g.ground(0, u, v)
        bt = BallTrack(u, v, np.array(X), np.array(Z), ["detected"] * n, np.full(n, 0.8))
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    board = build_board(tracks, colors, g, bt, records_for(tracks, n), 1.0, FPS, pitch,
                        BoardConfig(step=3, min_track_frames=5, team_min_frames=5))
    return board, pitch


def test_teams_split_and_own_team_is_focus_team():
    board, _ = build()
    teams = board["teams"]
    assert teams[str(board["own_team"])]["role"] == "own"
    assert board["tracks"]["1"]["team"] == board["own_team"]
    assert board["tracks"]["20"]["team"] not in (board["own_team"], OTHER)
    assert board["tracks"]["40"]["team"] == OTHER      # судья: взрослый по росту рамки — не команда
    assert board["tracks"]["40"]["height_ratio"] > 1.12 and board["tracks"]["1"]["height_ratio"] < 1.1
    assert teams[str(OTHER)]["role"] == "other" and board["teams_note"] == ""


def test_adults_always_present_do_not_become_a_team():
    """Четверо взрослых в тёмном стоят весь ролик (их не меньше, чем соперников) — команды всё равно
    красные и белые, взрослые — прочие (дефект D1 ревизии 2026-09-16)."""
    tracks, colors = scene(90, others=4)
    g = ground_for(tracks, 90)
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    board = build_board(tracks, colors, g, None, records_for(tracks, 90), 1.0, FPS, pitch,
                        BoardConfig(step=3, min_track_frames=5, team_min_frames=5, adult_min_frames=10))
    own = board["own_team"]
    assert board["tracks"]["1"]["team"] == own
    assert board["tracks"]["20"]["team"] == 1 - own and board["teams"][str(1 - own)]["role"] == "opp"
    for j in range(4):
        assert board["tracks"][str(40 + j)]["team"] == OTHER


def test_goalkeeper_gets_team_by_goal_side():
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    from tracker.board import find_goalkeepers
    keeper = track([(1.5 + 0.01 * (i % 20), 12.0 + 0.02 * i) for i in range(200)])   # у левых ворот, в створе
    runner = track([(5 + 0.2 * i, 12.0) for i in range(200)])
    crowd = {k: track([(25 + (k % 3), 10 + k) for _ in range(200)]) for k in range(10, 16)}   # игра в центре
    gks = find_goalkeepers({7: keeper, 8: runner, **crowd}, pitch, FPS, BoardConfig(), adults=set())
    assert gks == {7: "left"}
    # нападающий застоялся у чужих ворот, но игра рядом с ним — не вратарь; зритель у камеры — не в створе
    striker = track([(38.0 + 0.01 * (i % 20), 12.0) for i in range(200)])
    near_play = {k: track([(36 + (k % 3), 9 + (k % 5)) for _ in range(200)]) for k in range(10, 16)}
    fan = track([(1.5, 24.0) for _ in range(200)])
    assert find_goalkeepers({9: striker, 3: fan, **near_play}, pitch, FPS, BoardConfig(), adults=set()) == {}


def test_frames_have_players_ball_and_focus_on_pitch():
    board, _ = build()
    fr = board["frames"][5]
    assert len(fr["p"]) >= 8
    assert fr["focus"]["id"] == 1
    assert fr["ball"][2] == "d"
    L, Wd = board["field"]["length_m"], board["field"]["width_m"]
    for _, x, y in fr["p"]:
        assert -5 < x < L + 5 and -5 < y < Wd + 5
    assert board["fps"] == round(FPS / 3, 2)


def test_game_context_possession_and_distance_to_ball():
    board, pitch = build()
    ctx = game_context(board, pitch)
    s = ctx["summary"]
    assert s["ball_known_share"] == 1.0
    assert s["possession_estimate"].get("own", 0) > 0.5      # мяч у ног цели
    assert s["focus_dist_to_ball_m"]["median"] < 3
    assert 0 in ctx["timeline"] and "ball_third" in ctx["timeline"][0]


def test_board_without_ball_reports_unknown():
    board, pitch = build(ball=False)
    assert all("ball" not in fr for fr in board["frames"])
    assert game_context(board, pitch)["summary"]["ball_known_share"] == 0.0


def test_save_load_tracks_roundtrip(tmp_path):
    tracks, _ = scene(5)
    log = type("Log", (), {})()
    log.frames = [type("FL", (), {"frame_idx": i, "track_ids": list(tracks.ids[i]), "boxes": tracks.boxes[i],
                                  "det_index": list(tracks.det_index[i])})() for i in range(5)]
    save_tracks(tmp_path / "tracks.npz", log)
    back = load_tracks(tmp_path / "tracks.npz")
    assert len(back) == 5
    assert list(back.ids[2]) == list(tracks.ids[2])
    assert np.allclose(back.boxes[2], tracks.boxes[2])
    assert list(back.det_index[0]) == list(tracks.det_index[0])


def track(points, start=0):
    return [(start + i, x, y) for i, (x, y) in enumerate(points)]


def test_classify_roles_marks_standing_and_outside_people():
    """Игрок бегает по полю, зритель стоит за бровкой, запасной ходит вдоль линии далеко от игры."""
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    n = 150
    player = track([(10 + 0.15 * i, 12 + 0.05 * i) for i in range(n)])
    keeper = track([(0.8 + 0.01 * i, 12.5) for i in range(n)])              # вратарь: стоит на линии ворот в центре
    fan = track([(20.0, 26.5) for _ in range(n)])                           # стоит за боковой линией
    sub = track([(3 + 0.005 * i, 24.4) for i in range(n)])                  # переминается на бровке в углу
    blink = track([(20.0, 12.0) for _ in range(20)])                        # мелькнул меньше секунды
    per_track = {1: player, 2: keeper, 3: fan, 4: sub, 5: blink}
    assert all(r == ("play", "") for r in classify_roles(per_track, pitch, FPS, BoardConfig()).values())  # по умолчанию все играют
    roles = classify_roles(per_track, pitch, FPS, BoardConfig(auto_roles=True))
    assert roles[1][0] == "play" and roles[2][0] == "play"                  # игрок и вратарь остаются
    assert roles[3][0] == "sideline" and "поля" in roles[3][1]
    assert roles[4][0] == "sideline" and "бровки" in roles[4][1]
    assert roles[5][0] == "play"                                            # короткий трек не классифицируем


def test_manual_role_wins_over_rule_and_focus_is_protected():
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    fan = track([(20.0, 26.5) for _ in range(150)])
    player = track([(10 + 0.15 * i, 12) for i in range(150)])
    per_track = {3: fan, 1: player}
    roles = classify_roles(per_track, pitch, FPS, BoardConfig(auto_roles=True), manual={"3": "play", "1": "sideline"})
    assert roles[3] == ("play", "включён вручную") and roles[1] == ("sideline", "выключен вручную")
    # цель (protect) автоматика не трогает
    roles = classify_roles({3: fan}, pitch, FPS, BoardConfig(auto_roles=True), protect={3})
    assert roles[3][0] == "play"


def test_sideline_track_leaves_board_and_is_listed():
    tracks, colors = scene(60)
    g = ground_for(tracks, 60)
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    board = build_board(tracks, colors, g, None, records_for(tracks, 60), 1.0, FPS, pitch,
                        BoardConfig(step=3, min_track_frames=5, team_min_frames=5), manual_roles={"40": "sideline"})
    assert [r["id"] for r in board["sideline"]] == [40]
    assert board["sideline"][0]["manual"] is True and board["tracks"]["40"]["role"] == "sideline"
    # выключенный остаётся в кадрах доски (интерфейс показывает его бледно), но не участвует в игре
    on_board = {tid for fr in board["frames"] for tid, _, _ in fr["p"]}
    assert 40 in on_board and 1 in on_board
    pitch_ctx = game_context(board, pitch)
    assert pitch_ctx["summary"]["ball_known_share"] == 0.0


def test_everyone_is_on_board_with_short_numbers():
    tracks, colors = scene(60)
    g = ground_for(tracks, 60)
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    board = build_board(tracks, colors, g, None, records_for(tracks, 60), 1.0, FPS, pitch,
                        BoardConfig(step=3, min_track_frames=5, team_min_frames=5))
    assert board["sideline"] == [] and board["auto_roles"] is False
    nums = {tid: m["num"] for tid, m in board["tracks"].items()}
    assert sorted(nums.values()) == list(range(1, len(nums) + 1))       # 1..N без пропусков
    assert display_numbers(tracks, 5) == {int(t): n for t, n in nums.items()}


def test_manual_role_follows_person_after_renumbering():
    """Пометка хранится с началом трека: после поправки id сдвинулись — пометка на том же человеке."""
    tracks, _ = scene(30)
    sig = track_signatures(tracks)
    stored = {"40": {"role": "sideline", "f": sig[40][0], "box": sig[40][1]}}
    shifted = TrackFrames([np.array([t + 1 for t in ids]) for ids in tracks.ids], tracks.boxes, tracks.det_index)
    resolved = resolve_roles(stored, track_signatures(shifted))
    assert resolved == {41: "sideline"}
    # человек исчез (другие рамки) — пометка никуда не переезжает
    moved = TrackFrames(tracks.ids, [b + 300 for b in tracks.boxes], tracks.det_index)
    assert resolve_roles({"40": {"role": "sideline", "f": sig[40][0], "box": sig[40][1]}},
                         {k: v for k, v in track_signatures(moved).items() if k != 40}) == {}
    assert resolve_roles({"40": "sideline"}, sig) == {40: "sideline"}      # старый формат — по id


def test_possession_hysteresis_keeps_holder_through_noise():
    """Мяч у своего игрока; соперник на один кадр оказывается на 0.3 м ближе — владение не переключается;
    когда соперник ближе на 1.5 м пять кадров подряд — переключается; мяч без людей рядом — прежний
    владелец держит его ещё секунду, потом «ничей»."""
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    tracks = {"1": {"team": 0}, "2": {"team": 1}}
    frames = []
    for k in range(60):
        t = round(k / 10, 2)
        if k < 10:
            p = [[1, 10.0, 12.0], [2, 12.0, 12.0]]          # свой в 0.5 м, соперник в 1.5 м
        elif k == 10:
            p = [[1, 10.9, 12.0], [2, 10.6, 12.0]]          # соперник на 0.3 м ближе, один кадр
        elif k < 20:
            p = [[1, 10.0, 12.0], [2, 12.0, 12.0]]
        elif k < 30:
            p = [[1, 12.5, 12.0], [2, 10.4, 12.0]]          # соперник ближе на 2 м много кадров
        else:
            p = [[1, 30.0, 20.0], [2, 30.0, 5.0]]           # никого рядом с мячом
        frames.append({"f": 3 * k, "t": t, "p": p, "ball": [10.5, 12.0, "d"]})
    board = {"fps": 10, "own_team": 0, "camera": [20.0, 27.0], "teams": {"0": {"role": "own"}, "1": {"role": "opp"}},
             "tracks": tracks, "frames": frames}
    ctx = game_context(board, pitch)
    tl = ctx["timeline"]
    assert tl[0]["ball_with"] == "у своих" and tl[1]["ball_with"] == "у своих"     # шумный кадр не переключил
    assert tl[2]["ball_with"] == "у соперника"
    assert tl[3]["ball_with"] == "у соперника"          # секунда удержания без людей рядом
    assert tl[5]["ball_with"] == "ничей/в борьбе"
    assert ctx["summary"]["possession_spells"] == {"own": 1, "opp": 1}


def test_render_draws_numbers():
    from tracker.render import draw
    from tracker.target import TargetObservation, TargetState
    from types import SimpleNamespace

    frame = np.zeros((200, 300, 3), np.uint8)
    obs = TargetObservation(0, TargetState.ACTIVE, 1, np.array([50, 60, 80, 150.0]), 0.9)
    others = [SimpleNamespace(track_id=7, detected_now=True, box=np.array([150, 60, 180, 150.0]))]
    out = draw(frame, obs, others, numbers={1: 3, 7: 12})
    assert out[40:58, 150:185].any() and out[40:58, 50:120].any()      # плашки с номерами над рамками


def test_field_marks_suggest_people_outside_the_pitch():
    """Поле отмечено на кадре: кто всё время за его границей — предложение «вне игры», но не выключен;
    цель слежения и отмеченные вручную не предлагаются."""
    from tracker.field import FieldMarks, FieldProjector

    n = 60
    tracks, colors = scene(n)
    g = ground_for(tracks, n)
    # разметка накрывает игроков (u 380..960, v 470..660), взрослый у u≈1000 — далеко справа за полем
    corners = np.array([[300.0, 680.0], [900.0, 680.0], [880.0, 460.0], [330.0, 460.0]])
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    pitch.project = FieldProjector(FieldMarks(0, corners, 0), 1.0, g.inv, 40.0, 25.0)
    cfg = BoardConfig(step=3, min_track_frames=5, team_min_frames=5, outside_min_frames=20)
    board = build_board(tracks, colors, g, None, records_for(tracks, n), 1.0, FPS, pitch, cfg)
    sug = {int(t) for t, m in board["tracks"].items() if m.get("suggest") == "sideline"}
    assert sug == {40}
    assert board["tracks"]["40"]["role"] == "play" and not board["sideline"]          # только предложение
    manual = build_board(tracks, colors, g, None, records_for(tracks, n), 1.0, FPS, pitch, cfg, manual_roles={40: "play"})
    assert "suggest" not in manual["tracks"]["40"]


def test_pitch_fit_ignores_people_switched_off():
    """Выключенные оператором (зрители у камеры, люди на дальних полях) не должны сжимать схему поля:
    подгонка идёт по тем, кто в игре и двигается."""
    from tracker.board import active_feet
    from tracker.pitch import fit_to_occupancy

    rng = np.random.default_rng(0)
    base = PitchSetup((0.2, 1.07), (0.5, 0.5), "left", 45.0, 30.0)
    C, f_dir, r_dir = base._basis()
    P = np.stack([rng.uniform(3, 42, 4000), rng.uniform(2, 28, 4000)], axis=1)          # игроки по всему полю
    crowd = np.stack([rng.uniform(-30, 80, 2500), rng.uniform(-60, 45, 2500)], axis=1)   # зрители и дальние поля
    allp = np.vstack([P, crowd])
    feet = {"X": (allp - C) @ r_dir, "Z": (allp - C) @ f_dir, "f": np.tile(np.arange(100), 65),
            "t": np.r_[np.full(4000, 1), np.full(2500, 2)], "spread": np.full(6500, 20.0)}
    squeezed, _ = fit_to_occupancy(base, feet["X"], feet["Z"])
    mask = active_feet(feet, {2: [[0, 99]]})
    assert mask.sum() == 4000
    good, _ = fit_to_occupancy(base, feet["X"][mask], feet["Z"][mask])
    span = lambda pitch: np.ptp(np.percentile(pitch.to_pitch(feet["X"][:4000], feet["Z"][:4000])[0], [2, 98]))   # noqa: E731
    assert span(good) > 34 and span(good) > span(squeezed) + 5          # игроки занимают поле, а не кучку
    assert active_feet(feet, {1: [[0, 99]], 2: [[0, 99]]}).all()         # никого в игре — подгонка по всем
