"""Detectores e descritores de características disponíveis."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .base import Features
from .harris import detect_harris
from .opencv_ref import detect_opencv_orb, detect_opencv_sift
from .sift import detect_sift

#: Detectores registrados: nome -> (função, descrição curta).
DETECTORS: dict[str, tuple[Callable[..., Features], str]] = {
    "sift": (detect_sift, "SIFT (implementação própria, DoG + descritor 128-D)"),
    "harris": (detect_harris, "Harris + ANMS + descritor MOPS (implementação própria)"),
    "opencv-sift": (detect_opencv_sift, "SIFT do OpenCV (referência)"),
    "opencv-orb": (detect_opencv_orb, "ORB do OpenCV (referência, descritor binário)"),
}


def detect(image: np.ndarray, method: str = "sift", max_points: int = 4000) -> Features:
    """Detecta e descreve características com o método ``method``."""
    try:
        fn, _ = DETECTORS[method]
    except KeyError as exc:
        raise ValueError(f"detector desconhecido: {method!r}; opções: {sorted(DETECTORS)}") from exc
    if method == "harris":
        return fn(image, max_points=min(max_points, 2000))
    return fn(image, max_points=max_points)


__all__ = ["DETECTORS", "Features", "detect", "detect_harris", "detect_sift",
           "detect_opencv_sift", "detect_opencv_orb"]
