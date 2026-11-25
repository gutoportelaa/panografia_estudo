"""Estruturas comuns aos detectores/descritores de características."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    """Conjunto de pontos de interesse e seus descritores.

    Attributes:
        points: array (N, 2) com coordenadas (x, y) em pixels da imagem original.
        descriptors: array (N, D). ``float32`` para descritores reais (SIFT, MOPS)
            ou ``uint8`` para descritores binários (ORB).
        scales: array (N,) com a escala (sigma) de cada ponto, usada só para
            visualização.
        binary: ``True`` se os descritores devem ser comparados por Hamming.
    """

    points: np.ndarray
    descriptors: np.ndarray
    scales: np.ndarray
    binary: bool = False

    def __len__(self) -> int:
        return len(self.points)


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Converte uma imagem BGR/cinza ``uint8`` para cinza ``float32`` em [0, 1]."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0
