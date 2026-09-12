import numpy as np

from tracker.appearance import CompositeEncoder, PartColorEncoder, cosine_similarity
from _synth import draw_player, grass_frame

RED, WHITE, BLACK, BLOND, DARK = (0, 0, 255), (255, 255, 255), (20, 20, 20), (120, 200, 230), (30, 30, 30)


def test_same_kit_different_hair_and_socks_is_separable():
    enc = PartColorEncoder()
    f1 = grass_frame()
    draw_player(f1, (100, 40, 140, 180), RED, WHITE, RED, BLOND)
    f2 = grass_frame()
    draw_player(f2, (300, 60, 340, 200), RED, WHITE, RED, BLOND)   # тот же игрок в другом месте
    f3 = grass_frame()
    draw_player(f3, (300, 60, 340, 200), RED, WHITE, WHITE, DARK)  # одноклубник: волосы и гетры другие
    a = enc.encode(f1, [(100, 40, 140, 180)])
    b = enc.encode(f2, [(300, 60, 340, 200)])
    c = enc.encode(f3, [(300, 60, 340, 200)])
    assert a.shape == (1, enc.dim)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    same = cosine_similarity(a, b)[0, 0]
    other = cosine_similarity(a, c)[0, 0]
    assert same > 0.98
    assert other < same - 0.15


def test_opponent_far_from_teammate():
    enc = PartColorEncoder()
    f = grass_frame()
    draw_player(f, (100, 40, 140, 180), RED, WHITE, RED, DARK)
    draw_player(f, (300, 40, 340, 180), RED, WHITE, RED, DARK)
    draw_player(f, (500, 40, 540, 180), (255, 0, 0), (255, 0, 0), (255, 0, 0), DARK)
    feats = enc.encode(f, [(100, 40, 140, 180), (300, 40, 340, 180), (500, 40, 540, 180)])
    s = cosine_similarity(feats, feats)
    assert s[0, 1] > 0.99
    assert s[0, 2] < 0.8


def test_degenerate_box_gives_zero_vector():
    enc = PartColorEncoder()
    f = grass_frame()
    feats = enc.encode(f, [(10, 10, 11, 11)])
    assert np.allclose(feats, 0)


def test_composite_concatenates():
    enc = CompositeEncoder([(PartColorEncoder(), 1.0), (PartColorEncoder(h_bins=6), 0.5)])
    f = grass_frame()
    draw_player(f, (100, 40, 140, 180), RED, WHITE, RED, DARK)
    feats = enc.encode(f, [(100, 40, 140, 180)])
    assert feats.shape == (1, enc.dim)
    assert np.isclose(np.linalg.norm(feats[0]), 1.0)
