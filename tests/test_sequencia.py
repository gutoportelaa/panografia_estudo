"""Panografia de mais de duas imagens: translações encadeadas."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.server import app
from panografia import service
from panografia.compositing import composite, composite_many
from panografia.pipeline import stitch_many

ROOT = Path(__file__).resolve().parent.parent
WIDE = cv2.imread(str(ROOT / "resultados" / "mosaico.jpg"))   # 1002 × 557, sem bordas pretas no meio
ORIGINS = [(0, 10), (230, 0), (450, 22)]                     # canto de cada recorte na imagem larga
W, H = 520, 520


def _crops():
    return [WIDE[y:y + H, x:x + W].copy() for x, y in ORIGINS]


def test_positions_are_the_sum_of_pair_translations():
    res = stitch_many(_crops(), figures=[])
    # x1 = xk + (origem_k − origem_1)
    expected = [(x - ORIGINS[0][0], y - ORIGINS[0][1]) for x, y in ORIGINS]
    assert np.allclose(res.positions, expected, atol=0.5)
    assert len(res.pairs) == 2
    assert np.allclose(np.add(res.pairs[0].t, res.pairs[1].t), res.positions[2])


def test_composite_many_matches_two_image_composite():
    a, b = _crops()[:2]
    t = (230.0, -10.0)
    m2, _, _ = composite(a, b, t)
    mn, _, size = composite_many([a, b], [(0, 0), t])
    assert size == (m2.shape[1], m2.shape[0])
    assert np.abs(m2.astype(int) - mn.astype(int)).max() <= 1


def test_service_and_api_accept_a_sequence():
    datas = [cv2.imencode(".png", c)[1].tobytes() for c in _crops()]
    summary, images = service.run_many(datas, {"detalhe": 1})
    assert summary["n_imagens"] == 3 and summary["par_detalhe"] == 1
    assert summary["t"] == pytest.approx(summary["pares"][1]["t"])
    assert {"mosaico", "justaposicao", "correspondencias"} <= set(images)

    files = [("imagens", (f"{k}.png", d, "image/png")) for k, d in enumerate(datas)]
    r = TestClient(app).post("/api/stitch", files=files)
    assert r.status_code == 200
    assert r.json()["posicoes"][2] == pytest.approx([450, 12], abs=0.5)

    with pytest.raises(service.InputError):
        service.run_many(datas[:1])
