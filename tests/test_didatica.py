"""Testes dos dados e figuras do passo a passo."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from panografia.compositing import composite
from panografia.pipeline import stitch
from panografia.visualization import scale_space_figure, weights_figure

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def result():
    return stitch(cv2.imread(str(DATA / "img1.png")), cv2.imread(str(DATA / "img2.png")))


def test_votes_match_ransac(result):
    v = result.didatica["votos"]
    assert len(v["t"]) == result.n_matches          # menos de 1500 pares: todos os votos
    assert sum(v["inlier"]) == result.n_inliers
    inl = np.array(v["t"])[np.array(v["inlier"], bool)]
    assert inl.mean(axis=0) == pytest.approx(result.t, abs=0.01)   # t* = média dos inliers


def test_nndr_histogram_counts_every_point(result):
    n = result.didatica["nndr"]
    assert sum(n["contagens"]) == n["total"] == result.n_keypoints[0]
    assert len(n["bordas"]) == len(n["contagens"]) + 1
    assert result.n_matches <= n["aceitos"] <= n["total"]


def test_blend_alpha_is_a_partition(result):
    a = cv2.imread(str(DATA / "img1.png"))
    b = cv2.imread(str(DATA / "img2.png"))
    _, canvas, overlap, alpha = composite(a, b, result.t, "feather", return_alpha=True)
    inside = ~np.isnan(alpha)
    assert np.nanmin(alpha) >= 0 and np.nanmax(alpha) <= 1
    assert alpha[:, 5][inside[:, 5]].max() < 0.01           # borda esquerda: só a imagem 1
    assert alpha[:, -5][inside[:, -5]].min() > 0.99         # borda direita: só a imagem 2
    assert inside.sum() >= overlap.sum()


def test_scale_space_figure_layout():
    img = cv2.imread(str(DATA / "img1.png"))
    fig = scale_space_figure(img, tile_width=200)
    assert fig.shape[1] == 3 * 200 + 2 * 6 and fig.ndim == 3


def test_weights_figure_colors():
    alpha = np.array([[0.0, 1.0, np.nan]], np.float32)
    fig = weights_figure(alpha)
    assert fig.shape == (1, 3, 3)
    assert fig[0, 1, 0] > fig[0, 0, 0]        # cobalto tem mais azul (B em BGR) que o grafite
