"""Ponto de entrada comum ao servidor FastAPI e ao Pyodide (navegador).

Recebe as imagens como bytes codificados (PNG/JPEG), executa o pipeline e
devolve as métricas (dict serializável) e as imagens de saída em JPEG.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .compositing import BLEND_MODES
from .features import DETECTORS
from .pipeline import StitchError, StitchParams, stitch

DEFAULT_MAX_SIDE = 1600


class InputError(ValueError):
    """Entrada inválida (imagem ilegível ou parâmetro fora do intervalo)."""


def config() -> dict[str, Any]:
    """Opções disponíveis e valores padrão, usados para montar o formulário."""
    return {
        "detectors": [{"id": k, "label": v[1]} for k, v in DETECTORS.items()],
        "blend_modes": [{"id": k, "label": v} for k, v in BLEND_MODES.items()],
        "defaults": {**StitchParams().__dict__, "max_side": DEFAULT_MAX_SIDE},
    }


def decode(data: bytes, label: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"{label}: formato de imagem não reconhecido")
    return img


def run(data1: bytes, data2: bytes, options: dict[str, Any] | None = None, quality: int = 92):
    """Executa a panografia. Retorna (resumo, {nome: bytes JPEG})."""
    o = {**config()["defaults"], **(options or {})}
    if o["detector"] not in DETECTORS:
        raise InputError(f"detector inválido: {o['detector']}")
    if o["blend"] not in BLEND_MODES:
        raise InputError(f"modo de blending inválido: {o['blend']}")
    ratio, eps = float(o["ratio"]), float(o["ransac_threshold"])
    if not (0.1 <= ratio <= 1.0) or not (0.1 <= eps <= 50):
        raise InputError("parâmetros fora do intervalo permitido")

    a, b = decode(data1, "imagem 1"), decode(data2, "imagem 2")
    # Imagens grandes são reduzidas pelo MESMO fator, preservando a translação relativa.
    largest = max(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    scale = min(1.0, max(256, int(o["max_side"])) / largest)
    if scale < 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    params = StitchParams(
        detector=o["detector"], blend=o["blend"], ratio=ratio, mutual=bool(o["mutual"]),
        ransac_threshold=eps, max_points=max(100, min(int(o["max_points"]), 10000)), seed=int(o["seed"]),
    )
    result = stitch(a, b, params)  # pode lançar StitchError

    summary = result.summary()
    summary["scale"] = scale
    summary["t_original"] = (result.t[0] / scale, result.t[1] / scale)
    images = {}
    for name, img in result.images.items():
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        images[name] = buf.tobytes()
    return summary, images


__all__ = ["DEFAULT_MAX_SIDE", "InputError", "StitchError", "config", "decode", "run"]
