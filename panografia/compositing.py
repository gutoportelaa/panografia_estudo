"""Aplicação da translação e composição do mosaico.

Szeliski, seção 8.4 (*Compositing*):
* **média simples** — na sobreposição cada pixel é a média das imagens;
* **feathering / blending linear** — média ponderada por um mapa de
  distância à borda de cada imagem (seção 8.4.2): o peso cai linearmente
  até zero na borda, eliminando a costura visível;
* **sobreposição** — a imagem 2 por cima da 1, para evidenciar a costura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

BLEND_MODES = {
    "feather": "Blending linear (feathering por mapa de distância)",
    "average": "Média simples na sobreposição",
    "overlay": "Sobreposição sem mistura (imagem 2 por cima)",
}


@dataclass
class Canvas:
    """Geometria do quadro global do mosaico."""

    width: int
    height: int
    offset1: np.ndarray   # posição da origem da imagem 1 no mosaico
    offset2: np.ndarray   # posição da origem da imagem 2 no mosaico (= offset1 + t)


def compute_canvas(shape1, shape2, t) -> Canvas:
    """Calcula o tamanho do mosaico que contém as duas imagens (img2 transladada por t)."""
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    tx, ty = float(t[0]), float(t[1])
    min_x, min_y = min(0.0, tx), min(0.0, ty)
    max_x, max_y = max(float(w1), tx + w2), max(float(h1), ty + h2)
    off1 = np.array([-math.floor(min_x), -math.floor(min_y)], np.float64)
    width = int(math.ceil(max_x + off1[0]))
    height = int(math.ceil(max_y + off1[1]))
    return Canvas(width, height, off1, off1 + np.array([tx, ty]))


def warp_translation(image: np.ndarray, offset, size) -> np.ndarray:
    """Translada ``image`` por ``offset`` (subpixel, bilinear) num quadro ``size`` = (w, h)."""
    m = np.array([[1, 0, offset[0]], [0, 1, offset[1]]], np.float64)
    return cv2.warpAffine(image, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _feather_weight(shape) -> np.ndarray:
    """Peso que cresce linearmente da borda (≈0) até o interior da imagem."""
    h, w = shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return dist.astype(np.float32)


def composite(img1: np.ndarray, img2: np.ndarray, t, mode: str = "feather", return_alpha: bool = False):
    """Gera o mosaico. Retorna (mosaico uint8, Canvas, máscara de sobreposição).

    Com ``return_alpha=True`` devolve também α = w2 / (w1 + w2), a fração da
    imagem 2 em cada pixel (NaN fora das imagens), usada nas figuras didáticas.

    Cada imagem k contribui com peso w_k (1 para média, mapa de distância para
    feathering). O mosaico é Σ warp(I_k·w_k) / Σ warp(w_k), isto é, uma média
    ponderada com alfa pré-multiplicado — o que trata corretamente também a
    interpolação bilinear subpixel nas bordas.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"modo de blending desconhecido: {mode!r}; opções: {sorted(BLEND_MODES)}")
    canvas = compute_canvas(img1.shape, img2.shape, t)
    size = (canvas.width, canvas.height)

    layers = []
    for img, offset in ((img1, canvas.offset1), (img2, canvas.offset2)):
        weight = _feather_weight(img.shape) if mode == "feather" else np.ones(img.shape[:2], np.float32)
        color = img.astype(np.float32) * (weight[..., None] if img.ndim == 3 else weight)
        layers.append((warp_translation(color, offset, size), warp_translation(weight, offset, size)))
    (c1, w1), (c2, w2) = layers

    cover1, cover2 = w1 > 1e-6, w2 > 1e-6
    overlap = cover1 & cover2
    if mode == "overlay":
        # A imagem 2 cobre a 1: o peso da imagem 1 é multiplicado por (1 − cobertura de 2).
        cov2 = warp_translation(np.ones(img2.shape[:2], np.float32), canvas.offset2, size)
        w1 = w1 * (1.0 - cov2)
        c1 = c1 * ((1.0 - cov2)[..., None] if c1.ndim == 3 else (1.0 - cov2))

    total = w1 + w2
    safe = np.where(total > 1e-6, total, 1.0)
    if c1.ndim == 3:
        safe = safe[..., None]
    mosaic = np.clip(np.round((c1 + c2) / safe), 0, 255).astype(np.uint8)
    if return_alpha:
        alpha = np.where(total > 1e-6, w2 / np.where(total > 1e-6, total, 1.0), np.nan).astype(np.float32)
        return mosaic, canvas, overlap, alpha
    return mosaic, canvas, overlap


def side_by_side(img1: np.ndarray, img2: np.ndarray, gap: int = 8, color=(255, 255, 255)) -> np.ndarray:
    """Justaposição direta das imagens, sem alinhamento (referência visual)."""
    h = max(img1.shape[0], img2.shape[0])
    out = np.full((h, img1.shape[1] + gap + img2.shape[1], 3), color, np.uint8)
    out[: img1.shape[0], : img1.shape[1]] = img1
    out[: img2.shape[0], img1.shape[1] + gap:] = img2
    return out
