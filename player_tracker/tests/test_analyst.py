"""Инструменты ИИ-аналитика по данным доски: расстановка на момент, officials и вратари отдельно."""
from tracker.analyst import Analyst
from tracker.pitch import PitchSetup


def small_board():
    return {"fps": 10, "own_team": 0,
            "teams": {"0": {"role": "own"}, "1": {"role": "opp"}, "2": {"role": "other"}},
            "tracks": {"1": {"team": 0}, "2": {"team": 0}, "3": {"team": 1}, "4": {"team": 2}, "5": {"team": 1, "gk": True}},
            "frames": [{"f": 0, "t": 0.0, "p": [[1, 10, 12], [2, 15, 10], [3, 25, 12], [4, 30, 1], [5, 38, 12]],
                        "focus": {"id": 1, "x": 10, "y": 12, "state": "active"}, "ball": [11, 12, "d"]},
                       {"f": 30, "t": 3.0, "p": [[2, 16, 10]], "focus": {"id": 1, "x": 12, "y": 12, "state": "active"}}]}


def test_positions_tool_separates_officials_and_goalkeepers():
    pitch = PitchSetup((0.5, 1.1), (0.5, 0.5), "left", 40.0, 25.0)
    a = Analyst(None, {"timeline": []}, {"fps": 30.0, "events": [], "corrections": []}, board=small_board(), pitch=pitch)
    r = a._positions(0.2)                                # ближайший кадр доски — t=0
    assert r["time"] == "00:00" and r["focus"]["to_goal"] == 0.25
    assert len(r["teammates"]) == 1 and len(r["opponents"]) == 1
    assert r["officials"] == 1 and r["goalkeepers"] == [{"x": 38, "y": 12, "to_goal": 0.95, "team": "соперник"}]
    assert r["ball"]["state"] == "найден" and r["focus_to_ball_m"] == 1.0
    r3 = a._positions(2.9)
    assert r3["ball"] == "неизвестно" and r3["teammates"] == [{"x": 16, "y": 10, "to_goal": 0.4}]


def test_positions_tool_without_board():
    a = Analyst(None, {"timeline": []}, {"fps": 30.0})
    assert "error" in a._positions(1.0)
