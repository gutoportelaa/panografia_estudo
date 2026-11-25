"""Visualizações auxiliares: pontos detectados e correspondências."""

from __future__ import annotations

import cv2
import numpy as np

INLIER_COLOR = (80, 200, 60)    # BGR verde
OUTLIER_COLOR = (60, 60, 230)   # BGR vermelho
POINT_COLOR = (0, 215, 255)     # BGR amarelo


def draw_keypoints(image: np.ndarray, points: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    out = image.copy()
    for i, (x, y) in enumerate(points):
        r = int(np.clip(round(float(scales[i]) * 2) if scales is not None else 3, 2, 25))
        cv2.circle(out, (int(round(x)), int(round(y))), r, POINT_COLOR, 1, cv2.LINE_AA)
    return out


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    inliers: np.ndarray,
    max_lines: int = 250,
    gap: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Imagens lado a lado com linhas ligando correspondências (verde = inlier, vermelho = outlier)."""
    h = max(img1.shape[0], img2.shape[0])
    dx = img1.shape[1] + gap
    out = np.full((h, dx + img2.shape[1], 3), 255, np.uint8)
    out[: img1.shape[0], : img1.shape[1]] = img1
    out[: img2.shape[0], dx:] = img2
    out = (out.astype(np.float32) * 0.75).astype(np.uint8)  # escurece p/ realçar as linhas

    idx = np.arange(len(pts1))
    if len(idx) > max_lines:
        idx = np.random.default_rng(seed).choice(idx, max_lines, replace=False)
    # Desenha outliers primeiro para os inliers ficarem por cima.
    for i in sorted(idx, key=lambda k: bool(inliers[k])):
        color = INLIER_COLOR if inliers[i] else OUTLIER_COLOR
        p1 = (int(round(pts1[i, 0])), int(round(pts1[i, 1])))
        p2 = (int(round(pts2[i, 0])) + dx, int(round(pts2[i, 1])))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, p1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(out, p2, 3, color, -1, cv2.LINE_AA)
    return out


def overlap_outline(mosaic: np.ndarray, canvas, shape1, shape2) -> np.ndarray:
    """Desenha os contornos das duas imagens no mosaico (ajuda a ver o alinhamento)."""
    out = mosaic.copy()
    for (ox, oy), (h, w), color in (
        (canvas.offset1, shape1[:2], (255, 160, 0)),
        (canvas.offset2, shape2[:2], (0, 140, 255)),
    ):
        x0, y0 = int(round(ox)), int(round(oy))
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), color, 2, cv2.LINE_AA)
    return out
