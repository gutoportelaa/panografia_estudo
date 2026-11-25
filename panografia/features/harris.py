"""Detector de Harris + supressão não-máxima adaptativa + descritor MOPS.

Implementação própria (apenas NumPy e filtros básicos do OpenCV), seguindo
Szeliski, *Computer Vision: Algorithms and Applications*, 2ª ed., seção 7.1:

* matriz de autocorrelação (eq. 7.8)  A = w * [[Ix², IxIy], [IxIy, Iy²]];
* resposta de Harris (eq. 7.9)        R = det(A) − α·trace(A)²;
* ANMS — *adaptive non-maximal suppression* (Brown, Szeliski e Winder 2005),
  que distribui os pontos de forma uniforme pela imagem (Fig. 7.10);
* descritor MOPS simplificado (seção 7.1.2): patch 8×8 amostrado com
  espaçamento de 5 pixels sobre a imagem suavizada e normalizado em
  viés/ganho. Como o modelo desta atividade é de translação pura, o patch é
  mantido alinhado aos eixos (sem normalização de orientação).
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Features, to_gray_float


def harris_response(
    gray: np.ndarray, sigma_d: float = 1.0, sigma_i: float = 2.0, alpha: float = 0.06
) -> np.ndarray:
    """Mapa de resposta de Harris R = det(A) − α·trace(A)² (eq. 7.9)."""
    # Derivadas gaussianas com escala de diferenciação sigma_d.
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma_d)
    ix = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    iy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    # Entradas da matriz A ponderadas pela janela gaussiana w (escala sigma_i).
    a = cv2.GaussianBlur(ix * ix, (0, 0), sigma_i)
    b = cv2.GaussianBlur(ix * iy, (0, 0), sigma_i)
    c = cv2.GaussianBlur(iy * iy, (0, 0), sigma_i)

    det = a * c - b * b
    trace = a + c
    return det - alpha * trace * trace


def _local_maxima(response: np.ndarray, threshold: float, border: int) -> np.ndarray:
    """Máximos locais 3×3 acima de ``threshold``, longe da borda. Retorna (N, 2) em (x, y)."""
    dilated = cv2.dilate(response, np.ones((3, 3), np.uint8))
    mask = (response == dilated) & (response > threshold)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1)


def anms(points: np.ndarray, strengths: np.ndarray, n_keep: int, c_robust: float = 0.9) -> np.ndarray:
    """Supressão não-máxima adaptativa (Brown, Szeliski e Winder 2005).

    Para cada ponto calcula o raio de supressão r_i = min ||x_i − x_j|| sobre os
    pontos j suficientemente mais fortes (f_i < c·f_j) e mantém os ``n_keep``
    pontos de maior raio. Retorna os índices escolhidos.
    """
    n = len(points)
    if n <= n_keep:
        return np.arange(n)
    order = np.argsort(-strengths)
    pts = points[order].astype(np.float32)
    s = strengths[order]
    radii = np.full(n, np.inf, dtype=np.float32)
    # Como os pontos estão ordenados por força, só os anteriores podem suprimir.
    chunk = 512
    for start in range(1, n, chunk):
        stop = min(n, start + chunk)
        stronger = pts[:stop]
        d2 = ((pts[start:stop, None, :] - stronger[None, :, :]) ** 2).sum(-1)
        dominated = s[start:stop, None] < c_robust * s[None, :stop]
        d2 = np.where(dominated, d2, np.inf)
        radii[start:stop] = d2.min(axis=1)
    keep = np.argsort(-radii, kind="stable")[:n_keep]
    return order[keep]


def mops_descriptors(
    gray: np.ndarray, points: np.ndarray, patch: int = 8, spacing: float = 5.0
) -> np.ndarray:
    """Descritor MOPS: patch ``patch``×``patch`` amostrado com espaçamento
    ``spacing`` (janela de 40×40 px) e normalizado por média e desvio padrão."""
    # Suaviza antes da subamostragem para evitar aliasing.
    blurred = cv2.GaussianBlur(gray, (0, 0), spacing / 2.0)
    offs = (np.arange(patch, dtype=np.float32) - (patch - 1) / 2.0) * spacing
    gx, gy = np.meshgrid(offs, offs)
    map_x = (points[:, 0:1].astype(np.float32) + gx.reshape(1, -1)).astype(np.float32)
    map_y = (points[:, 1:2].astype(np.float32) + gy.reshape(1, -1)).astype(np.float32)
    samples = cv2.remap(blurred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    samples = samples.reshape(len(points), patch * patch)
    mean = samples.mean(axis=1, keepdims=True)
    std = samples.std(axis=1, keepdims=True) + 1e-6
    return ((samples - mean) / std).astype(np.float32)


def detect_harris(
    image: np.ndarray,
    max_points: int = 1500,
    rel_threshold: float = 1e-3,
    alpha: float = 0.06,
) -> Features:
    """Detecta cantos de Harris, aplica ANMS e calcula descritores MOPS."""
    gray = to_gray_float(image)
    response = harris_response(gray, alpha=alpha)
    border = 20  # metade da janela de 40 px do descritor
    pts = _local_maxima(response, rel_threshold * float(response.max()), border)
    if len(pts) == 0:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 64), np.float32), np.zeros(0))
    strengths = response[pts[:, 1], pts[:, 0]]
    # Limita o custo O(n²) do ANMS aos candidatos mais fortes.
    if len(pts) > 6000:
        top = np.argsort(-strengths)[:6000]
        pts, strengths = pts[top], strengths[top]
    keep = anms(pts, strengths, max_points)
    pts = pts[keep].astype(np.float32)
    desc = mops_descriptors(gray, pts)
    return Features(pts, desc, np.full(len(pts), 5.0, np.float32))
