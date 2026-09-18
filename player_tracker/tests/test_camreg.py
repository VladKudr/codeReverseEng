"""Привязка кадров к сцене гомографией по опорным кадрам: поворот камеры без накопления ошибки."""
import cv2
import numpy as np

from tracker.camreg import RegConfig, register

W, H = 640, 360


def world(seed=0):
    """Широкая «панорама» с текстурой: много контрастных фигур и линий (есть за что зацепиться)."""
    rng = np.random.default_rng(seed)
    img = np.full((700, 2400, 3), 110, np.uint8)
    for _ in range(2500):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, 2400)), int(rng.integers(0, 700)))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(img, p, (p[0] + int(rng.integers(6, 30)), p[1] + int(rng.integers(6, 30))), c, -1)
        elif kind == 1:
            cv2.circle(img, p, int(rng.integers(4, 16)), c, -1)
        else:
            cv2.line(img, p, (p[0] + int(rng.integers(-40, 40)), p[1] + int(rng.integers(-40, 40))), c, 2)
    return cv2.GaussianBlur(img, (3, 3), 0)


def view(i, n):
    """Сцена -> кадр i: камера уезжает вправо на 700 px с лёгкой перспективой (поворот), потом возвращается."""
    t = np.sin(np.pi * i / (n - 1))
    dx = 300 + 700 * t
    P = np.array([[1, 0, -dx], [0, 1, -150], [0, 0, 1.0]])
    K = np.array([[1, 0, 0], [0, 1, 0], [2.5e-4 * t, 0, 1.0]])                # перспектива растёт с поворотом
    return K @ P


def test_registration_recovers_pan_without_drift():
    n = 60
    pano = world()
    frames = [cv2.warpPerspective(pano, view(i, n), (W, H)) for i in range(n)]
    # покадровое «подобие» с систематической ошибкой 1 px/кадр — цепочка уплывает, привязка нет
    true_shift = [(view(i, n) @ np.linalg.inv(view(i - 1, n))) for i in range(1, n)]
    cams = [None] + [np.array([[1, 0, T[0, 2] / T[2, 2] + 1.0], [0, 1, 0.0]]) for T in true_shift]
    G, stats = register(iter(frames), cams, [np.zeros((0, 4))] * n, None, RegConfig(step=3, min_inliers=25))
    assert stats["matched"] >= 15 and stats["keyframes"] >= 2
    pts = np.array([[100.0, 80, 1], [500, 100, 1], [320, 300, 1]]).T
    err_reg, err_chain, M = [], [], np.eye(3)
    for i in range(n):
        truth = view(0, n) @ np.linalg.inv(view(i, n)) @ pts                  # кадр i -> сцена (кадр 0)
        got = G[i] @ pts
        err_reg.append(np.abs(got[:2] / got[2] - truth[:2] / truth[2]).max())
        if i:
            M = np.vstack([cams[i], [0, 0, 1]]) @ M
        ch = np.linalg.inv(M) @ pts
        err_chain.append(np.abs(ch[:2] / ch[2] - truth[:2] / truth[2]).max())
    assert np.median(err_reg) < 1.5 and max(err_reg[::3]) < 4.0               # обработанные кадры — в пределах пикселей
    assert err_chain[-1] > 30                                                 # цепочка подобий уплыла


def test_registration_falls_back_to_chain_without_texture():
    n = 12
    frames = [np.full((H, W, 3), 90, np.uint8) for _ in range(n)]
    cams = [None] + [np.array([[1, 0, -5.0], [0, 1, 0.0]])] * (n - 1)
    G, stats = register(iter(frames), cams, [np.zeros((0, 4))] * n, None, RegConfig(step=3))
    assert stats["matched"] == 0
    assert np.allclose(G[6] @ [0, 0, 1], [30, 0, 1])                          # кадр 6 сдвинут на 6×5 px
