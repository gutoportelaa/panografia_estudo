"""Ponto de entrada comum ao servidor FastAPI e ao Pyodide (navegador).

Recebe as imagens como bytes codificados (PNG/JPEG), executa o pipeline e
devolve as métricas (dict serializável) e as imagens de saída em JPEG.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from .compositing import BLEND_MODES
from .features import DETECTORS
from .pipeline import StitchError, StitchParams, stitch_many

DEFAULT_MAX_SIDE = 1600
MULTI_MAX_SIDE = 1000   # por imagem, quando há mais de duas
MAX_IMAGES = 8


class InputError(ValueError):
    """Entrada inválida (imagem ilegível ou parâmetro fora do intervalo)."""


def config() -> dict[str, Any]:
    """Opções disponíveis e valores padrão, usados para montar o formulário."""
    return {
        "detectors": [{"id": k, "label": v[1]} for k, v in DETECTORS.items()],
        "blend_modes": [{"id": k, "label": v} for k, v in BLEND_MODES.items()],
        "defaults": {**StitchParams().__dict__, "max_side": DEFAULT_MAX_SIDE},
        "max_images": MAX_IMAGES,
    }


def decode(data: bytes, label: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"{label}: formato de imagem não reconhecido")
    return img


def run(data1: bytes, data2: bytes, options: dict[str, Any] | None = None, quality: int = 92):
    """Executa a panografia de duas imagens. Retorna (resumo, {nome: bytes JPEG})."""
    return run_many([data1, data2], options, quality)


def run_many(datas: list[bytes], options: dict[str, Any] | None = None, quality: int = 92):
    """Panografia de uma sequência de imagens, cada uma sobrepondo a seguinte.

    O resumo traz, no nível de cima, os números do par escolhido em
    ``options["detalhe"]`` (o par mostrado no passo a passo), e em ``pares`` e
    ``posicoes`` os dados de toda a sequência.
    """
    o = {**config()["defaults"], **(options or {})}
    if not 2 <= len(datas) <= MAX_IMAGES:
        raise InputError(f"envie de 2 a {MAX_IMAGES} imagens")
    if o["detector"] not in DETECTORS:
        raise InputError(f"detector inválido: {o['detector']}")
    if o["blend"] not in BLEND_MODES:
        raise InputError(f"modo de blending inválido: {o['blend']}")
    ratio, eps = float(o["ratio"]), float(o["ransac_threshold"])
    if not (0.1 <= ratio <= 1.0) or not (0.1 <= eps <= 50):
        raise InputError("parâmetros fora do intervalo permitido")

    imgs = [decode(d, f"imagem {k + 1}") for k, d in enumerate(datas)]
    # Todas são reduzidas pelo MESMO fator, preservando as translações relativas.
    # Com mais de duas o mosaico cresce a cada foto, então o limite é menor.
    max_side = max(256, int(o["max_side"]))
    if len(imgs) > 2:
        max_side = min(max_side, MULTI_MAX_SIDE)
    largest = max(max(im.shape[:2]) for im in imgs)
    scale = min(1.0, max_side / largest)
    if scale < 1.0:
        imgs = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in imgs]

    params = StitchParams(
        detector=o["detector"], blend=o["blend"], ratio=ratio, mutual=bool(o["mutual"]),
        ransac_threshold=eps, max_points=max(100, min(int(o["max_points"]), 10000)), seed=int(o["seed"]),
    )
    figures = o.get("figuras")
    if isinstance(figures, str):
        figures = [f for f in figures.split(",") if f] or None
    t0 = time.perf_counter()
    result = stitch_many(imgs, params, int(o.get("detalhe", 0) or 0), figures)  # pode lançar StitchError
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    detail = result.pairs[result.detail]
    summary = detail.summary()
    summary["timings_ms"]["total"] = total_ms
    summary["mosaic_size"] = result.mosaic_size
    summary["scale"] = scale
    summary["t_original"] = (detail.t[0] / scale, detail.t[1] / scale)
    summary["n_imagens"] = len(imgs)
    summary["par_detalhe"] = result.detail
    summary["posicoes"] = [list(p) for p in result.positions]
    summary["pares"] = [
        {"t": list(r.t), "n_inliers": r.n_inliers, "n_matches": r.n_matches,
         "inlier_ratio": r.inlier_ratio, "rmse": r.rmse, "n_keypoints": list(r.n_keypoints)}
        for r in result.pairs
    ]
    images = {}
    for name, img in result.images.items():
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        images[name] = buf.tobytes()
    return summary, images


__all__ = ["DEFAULT_MAX_SIDE", "MAX_IMAGES", "InputError", "StitchError", "config", "decode", "run", "run_many"]
