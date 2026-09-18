"""Стенд состава: оператор размечает людей на одном кадре, проверяем по эталону на других кадрах.
Сценарий «выключаю лишних» (все в игре, кроме выключенных) и «отмечаю игроков» (в игре только отмеченные)."""
import sys; sys.path.insert(0, "tools/roster_check")
import numpy as np
from common import TRUTH, load, track_features
from tracker.roster import LinkConfig, Mark, PLAY, SIDELINE, is_off, link_tracks, off_intervals, resolve, track_tracks

job = sys.argv[1]
tr, G, st, colors, (feats, offs) = load(job)
fps = st["fps"]
info = track_tracks(tr, colors, list(G))
truth = TRUTH[job]
mark_frames = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [min(truth)]
test_frames = [f for f in truth if f not in mark_frames]


def score(name, off):
    ok_p = n_p = ok_o = n_o = 0
    for f in test_frames:
        for t in truth[f]["play"]:
            n_p += 1; ok_p += not is_off(off.get(t, []), f)
        for t in truth[f]["off"]:
            n_o += 1; ok_o += is_off(off.get(t, []), f)
    print(f"  {name:34s} игроки на доске {ok_p}/{n_p}   лишние убраны {ok_o}/{n_o}")


for scen, role, default in (("выключаю лишних", SIDELINE, PLAY), ("отмечаю игроков", PLAY, SIDELINE)):
    key = "off" if role == SIDELINE else "play"
    marks = [Mark(t, role) for f in mark_frames for t in truth[f][key]]
    print(f"{scen}: пометки на кадрах {mark_frames} ({len(marks)} щелчков), проверка на {test_frames}")
    score("по треку (как сейчас)", off_intervals(info, {t: t for t in info}, marks, default))
    person, stats = link_tracks(info, fps, LinkConfig())
    score(f"по человеку ({stats['people']} людей)", off_intervals(info, person, marks, default))
    F = track_features(tr, feats, offs)
    r = resolve(info, marks, fps, default, F, mark_frames)
    score(f"человек + похожие ({r['stats']['similar_people']} людей)", r["off"])
