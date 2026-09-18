import numpy as np

from tracker.team import OTHER, UNKNOWN, TeamClassifier, teams_compatible, torso_color
from _synth import draw_player, grass_frame


def test_two_kits_and_referee():
    rng = np.random.default_rng(1)
    clf = TeamClassifier(min_samples=40)
    red = np.array([130, 180, 160], dtype=np.float32)
    white = np.array([240, 128, 128], dtype=np.float32)
    for _ in range(25):
        clf.observe(red + rng.normal(0, 3, 3))
        clf.observe(white + rng.normal(0, 3, 3))
    assert clf.predict(red) == UNKNOWN
    assert clf.fit()
    a, b = clf.predict(red), clf.predict(white)
    assert {a, b} == {0, 1}
    black = np.array([30, 128, 128], dtype=np.float32)
    assert clf.predict(black) == OTHER
    assert teams_compatible(a, a) is True
    assert teams_compatible(a, b) is False
    assert teams_compatible(a, UNKNOWN) is None


def test_torso_color_from_synthetic_frame():
    frame = grass_frame()
    draw_player(frame, (100, 50, 140, 170), shirt=(0, 0, 255), shorts=(255, 255, 255), socks=(0, 0, 255), hair=(0, 0, 0))
    lab = torso_color(frame, (100, 50, 140, 170))
    assert lab is not None
    # красная футболка: канал a* высокий
    assert lab[1] > 170


def test_many_bib_colours_make_classifier_silent():
    """Тренировка с манишками четырёх цветов: два кластера ничего не объясняют — метки «неизвестно»."""
    rng = np.random.default_rng(2)
    clf = TeamClassifier(min_samples=40)
    colours = [np.array(c, dtype=np.float32) for c in ([200, 160, 120], [150, 90, 170], [170, 170, 190], [120, 175, 150])]
    for _ in range(30):
        for c in colours:
            clf.observe(c + rng.normal(0, 3, 3))
    assert clf.fit()
    assert not clf.reliable
    assert clf.predict(colours[0]) == UNKNOWN
