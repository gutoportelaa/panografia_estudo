"""Figuras didáticas geradas a partir de uma execução do pipeline.

Cores (BGR) alinhadas à interface: verde e vermelho só têm significado
(inlier e outlier), o cobalto marca a imagem 2 e o papel é o fundo.
"""

from __future__ import annotations

import cv2
import numpy as np

INLIER_COLOR = (74, 163, 22)     # #16a34a
OUTLIER_COLOR = (38, 38, 220)    # #dc2626
POINT_COLOR = (36, 165, 245)     # #f5a524, âmbar: visível sobre céu, água e neve
FRAME1_COLOR = (247, 251, 253)   # #fdfbf7, papel
FRAME2_COLOR = (216, 78, 29)     # #1d4ed8, cobalto
PAPER = np.array([238, 244, 247], np.float32)   # #f7f4ee
INK1 = np.array([47, 53, 58], np.float32)       # #3a352f, grafite quente
INK2 = np.array([216, 78, 29], np.float32)      # #1d4ed8, cobalto


def draw_keypoints(image: np.ndarray, points: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
    """Círculos nos pontos de interesse; o raio é proporcional à escala de detecção."""
    out = (image.astype(np.float32) * 0.8).astype(np.uint8)
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
    out = (out.astype(np.float32) * 0.7).astype(np.uint8)  # escurece p/ realçar as linhas

    idx = np.arange(len(pts1))
    if len(idx) > max_lines:
        idx = np.random.default_rng(seed).choice(idx, max_lines, replace=False)
    # Outliers primeiro, para os inliers ficarem por cima.
    for i in sorted(idx, key=lambda k: bool(inliers[k])):
        color = INLIER_COLOR if inliers[i] else OUTLIER_COLOR
        p1 = (int(round(pts1[i, 0])), int(round(pts1[i, 1])))
        p2 = (int(round(pts2[i, 0])) + dx, int(round(pts2[i, 1])))
        cv2.line(out, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(out, p1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(out, p2, 3, color, -1, cv2.LINE_AA)
    return out


def overlap_outline(mosaic: np.ndarray, canvas, shape1, shape2) -> np.ndarray:
    """Contornos das duas imagens no quadro global do mosaico."""
    out = mosaic.copy()
    for (ox, oy), (h, w), color in (
        (canvas.offset1, shape1[:2], FRAME1_COLOR),
        (canvas.offset2, shape2[:2], FRAME2_COLOR),
    ):
        x0, y0 = int(round(ox)), int(round(oy))
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), color, 2, cv2.LINE_AA)
    return out


def outline_many(mosaic: np.ndarray, offsets, shapes) -> np.ndarray:
    """Contornos numerados de N imagens no quadro do mosaico."""
    out = mosaic.copy()
    for k, ((ox, oy), shape) in enumerate(zip(offsets, shapes)):
        h, w = shape[:2]
        color = FRAME1_COLOR if k % 2 == 0 else FRAME2_COLOR
        x0, y0 = int(round(ox)), int(round(oy))
        cv2.rectangle(out, (x0, y0), (x0 + w - 1, y0 + h - 1), color, 2, cv2.LINE_AA)
        cv2.putText(out, str(k + 1), (x0 + 10, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, str(k + 1), (x0 + 10, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
    return out


def scale_space_figure(image: np.ndarray, sigma: float = 1.6, s: int = 3, tile_width: int = 320) -> np.ndarray:
    """Primeira oitava do espaço-escala: L(x, y, σ·kⁱ) em cima e D = L(kσ) − L(σ) embaixo.

    A DoG é mostrada com cinza médio = 0, claro = positivo e escuro = negativo.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = tile_width / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    k = 2 ** (1 / s)
    levels = [cv2.GaussianBlur(gray, (0, 0), sigma * k ** i) for i in range(4)]
    dogs = [levels[i + 1] - levels[i] for i in range(3)]
    lim = max(float(np.abs(d).max()) for d in dogs) * 0.6 + 1e-6
    top = [np.clip(l * 255, 0, 255) for l in levels[:3]]
    bottom = [np.clip(128 + d / lim * 127, 0, 255) for d in dogs]
    gap = 6
    h, w = gray.shape
    out = np.full((2 * h + gap, 3 * w + 2 * gap), 255, np.float32)
    for row, tiles in enumerate((top, bottom)):
        for col, tile in enumerate(tiles):
            y, x = row * (h + gap), col * (w + gap)
            out[y:y + h, x:x + w] = tile
    out = out.astype(np.uint8)
    paper = tuple(int(v) for v in PAPER)
    color = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # Os vãos entre os quadros ficam na cor do papel da interface.
    for col in (1, 2):
        color[:, col * (w + gap) - gap: col * (w + gap)] = paper
    color[h:h + gap, :] = paper
    return color


def weights_figure(alpha: np.ndarray) -> np.ndarray:
    """Mapa de α = w2/(w1+w2): grafite onde só a imagem 1 conta, cobalto onde só a 2 conta."""
    a = np.nan_to_num(alpha, nan=-1.0)
    inside = a >= 0
    t = np.clip(a, 0, 1)[..., None]
    img = INK1 * (1 - t) + INK2 * t
    img = np.where(inside[..., None], img, PAPER)
    return img.astype(np.uint8)
