"""Testes da camada de serviço (usada pelo servidor e pelo Pyodide) e da API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import app
from panografia import service

DATA = Path(__file__).resolve().parent.parent / "data"
ROOT = DATA.parent


def _bytes(name):
    return (DATA / name).read_bytes()


def test_service_run_returns_jpegs():
    summary, images = service.run(_bytes("img1.png"), _bytes("img2.png"), {"detector": "harris"})
    assert summary["t"] == pytest.approx((361.9, -0.4), abs=0.5)
    assert summary["scale"] == 1.0
    assert set(images) >= {"mosaico", "justaposicao", "correspondencias"}
    assert all(v[:2] == b"\xff\xd8" for v in images.values())  # cabeçalho JPEG


def test_service_renders_only_requested_figures():
    """A comparação entre detectores pede só os pontos, para não pagar pelas outras figuras."""
    summary, images = service.run(_bytes("img1.png"), _bytes("img2.png"),
                                  {"detector": "opencv-orb", "figuras": "pontos1,inexistente"})
    assert set(images) == {"pontos1"}
    assert summary["n_inliers"] > 50


def test_service_downscales_large_inputs_consistently():
    summary, _ = service.run(_bytes("img1.png"), _bytes("img2.png"), {"max_side": 320})
    assert summary["scale"] == pytest.approx(0.5)
    assert summary["t_original"] == pytest.approx((361.9, -0.4), abs=1.5)


def test_service_rejects_bad_input():
    with pytest.raises(service.InputError):
        service.run(b"nao-e-imagem", _bytes("img2.png"))
    with pytest.raises(service.InputError):
        service.run(_bytes("img1.png"), _bytes("img2.png"), {"detector": "xyz"})


def test_api_and_static_site():
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/samples/samples.json").status_code == 200
    assert client.get("/api/config").json()["engine"] == "server"
    r = client.post("/api/stitch", files={"img1": _bytes("img1.png"), "img2": _bytes("img2.png")})
    assert r.status_code == 200 and r.json()["n_inliers"] > 200
    r = client.post("/api/stitch", files={"img1": b"x", "img2": _bytes("img2.png")})
    assert r.status_code == 400


def test_samples_catalog_points_to_existing_files():
    import json
    samples = json.loads((ROOT / "web" / "samples" / "samples.json").read_text())
    for s in samples:
        imgs = s.get("imgs") or [s["img1"], s["img2"]]   # sequência ou par
        assert len(imgs) >= 2
        assert all((ROOT / "web" / f).is_file() for f in imgs)
