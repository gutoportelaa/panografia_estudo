"""Testes com translações conhecidas (recortes sintéticos de uma imagem real)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from panografia.compositing import composite, compute_canvas
from panografia.matching import match_descriptors
from panografia.pipeline import StitchParams, stitch
from panografia.ransac import estimate_translation_ransac, required_iterations

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def scene():
    img = cv2.imread(str(DATA / "img1.png"))
    assert img is not None
    return img


def crops(scene, dx, dy, w=380, h=400):
    """Dois recortes da mesma cena; o 2º começa (dx, dy) à direita/abaixo do 1º.

    Um ponto na posição x2 do recorte 2 está em x1 = x2 + (dx, dy) no recorte 1,
    portanto a translação esperada é t = (dx, dy).
    """
    y0, x0 = 40, 20
    a = scene[y0:y0 + h, x0:x0 + w]
    b = scene[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w]
    return a.copy(), b.copy()


@pytest.mark.parametrize("detector", ["sift", "harris", "opencv-sift"])
@pytest.mark.parametrize("shift", [(180, 0), (150, 25), (200, -30)])
def test_recovers_known_translation(scene, detector, shift):
    dx, dy = shift
    if dy < 0:  # recorte 2 acima do 1: desloca a origem para caber na imagem
        a, b = crops(scene[60:], dx, 0)
        b = scene[60 + 40 + dy:60 + 40 + dy + 400, 20 + dx:20 + dx + 380]
    else:
        a, b = crops(scene, dx, dy)
    res = stitch(a, b, StitchParams(detector=detector))
    assert res.n_inliers >= 20
    assert res.t == pytest.approx((dx, dy), abs=0.5)


def test_sample_pair_matches_reference():
    a, b = cv2.imread(str(DATA / "img1.png")), cv2.imread(str(DATA / "img2.png"))
    res = stitch(a, b)
    assert res.t == pytest.approx((361.9, -0.4), abs=0.5)
    assert res.inlier_ratio > 0.9
    assert res.images["mosaico"].shape[:2] == (res.mosaic_size[1], res.mosaic_size[0])


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    p2 = rng.uniform(0, 500, (200, 2))
    t = np.array([123.4, -7.8])
    p1 = p2 + t + rng.normal(0, 0.3, p2.shape)
    out = rng.choice(200, 80, replace=False)           # 40% de outliers
    p1[out] = rng.uniform(0, 500, (80, 2))
    est = estimate_translation_ransac(p1, p2, threshold=2.0)
    assert np.allclose(est.t, t, atol=0.15)
    assert est.n_inliers >= 115
    assert not est.inliers[out].mean() > 0.05


def test_required_iterations_formula():
    # S = log(1 − P) / log(1 − p^k)   (Szeliski eq. 8.30)
    assert required_iterations(0.5, 1, 0.99) == 7
    assert required_iterations(0.5, 4, 0.99) == 72
    assert required_iterations(1.0) == 1


def test_nndr_and_mutual():
    d1 = np.array([[0, 0], [10, 0], [5, 5]], np.float32)
    d2 = np.array([[0.1, 0], [10, 0.2], [5, 5.1], [4.9, 5]], np.float32)
    m = match_descriptors(d1, d2, ratio=0.8)
    # O 3º descritor é ambíguo (dois vizinhos quase equidistantes): rejeitado.
    assert list(zip(m.idx1, m.idx2)) == [(0, 0), (1, 1)]


@pytest.mark.parametrize("t", [(100.0, 20.0), (-50.5, -10.25), (0.0, 0.0)])
def test_canvas_contains_both_images(t):
    c = compute_canvas((100, 200), (100, 200), t)
    assert c.offset1.min() >= 0 and c.offset2.min() >= 0
    assert c.width >= c.offset1[0] + 200 and c.width >= c.offset2[0] + 200 - 1e-9
    assert c.height >= c.offset2[1] + 100 - 1e-9


@pytest.mark.parametrize("mode", ["feather", "average", "overlay"])
def test_composite_identity_on_perfect_alignment(scene, mode):
    a, b = crops(scene, 120, 0)
    mosaic, canvas, overlap = composite(a, b, (120, 0), mode)
    assert mosaic.shape[:2] == (400, 500)
    # Com alinhamento exato, o mosaico reproduz a cena original.
    original = scene[40:440, 20:520]
    assert np.abs(mosaic.astype(int) - original.astype(int)).max() <= 1
    assert overlap.sum() == 400 * 260
