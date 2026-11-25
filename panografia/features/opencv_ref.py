"""Detectores de referência do OpenCV (SIFT e ORB).

Servem para comparar os resultados das implementações próprias
(:mod:`panografia.features.sift` e :mod:`panografia.features.harris`) com
implementações consolidadas.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Features


def _to_features(keypoints, descriptors, binary: bool, dim: int) -> Features:
    if descriptors is None or len(keypoints) == 0:
        dtype = np.uint8 if binary else np.float32
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, dim), dtype), np.zeros(0), binary)
    pts = np.array([kp.pt for kp in keypoints], np.float32)
    scales = np.array([kp.size / 2 for kp in keypoints], np.float32)
    return Features(pts, descriptors, scales, binary)


def detect_opencv_sift(image: np.ndarray, max_points: int = 4000) -> Features:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kps, desc = cv2.SIFT_create(nfeatures=max_points).detectAndCompute(gray, None)
    return _to_features(kps, desc.astype(np.float32) if desc is not None else None, False, 128)


def detect_opencv_orb(image: np.ndarray, max_points: int = 4000) -> Features:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kps, desc = cv2.ORB_create(nfeatures=max_points).detectAndCompute(gray, None)
    feats = _to_features(kps, desc, True, 32)
    # O "size" do ORB é a janela de 31 px do BRIEF, não a escala do ponto. Para as
    # figuras ficarem comparáveis às do SIFT, usa-se a escala do nível da pirâmide.
    if len(kps):
        feats.scales = np.array([1.6 * 1.2 ** kp.octave for kp in kps], np.float32)
    return feats
