"""Estimativa robusta de translação com RANSAC.

Modelo (Szeliski, seção 8.1): x1 = x2 + t, com t = (tx, ty). Cada
correspondência i fornece diretamente uma hipótese t_i = x1_i − x2_i, logo a
amostra mínima tem k = 1 correspondência.

RANSAC (seção 8.1.4, Fischler e Bolles 1981):
1. sorteia k correspondências e estima t;
2. conta os *inliers* cujo resíduo ||r_i|| = ||x2_i + t − x1_i|| ≤ ε;
3. repete S vezes e guarda a hipótese com mais inliers. O número de
   iterações é adaptativo (eq. 8.30): S = log(1 − P) / log(1 − p^k), onde p
   é a fração de inliers da melhor hipótese até o momento.

Refinamento final por mínimos quadrados nos inliers (eq. 8.12 para duas
imagens): o t que minimiza Σ||x2_i + t − x1_i||² é a média dos
deslocamentos dos inliers. Repete-se o reajuste até o conjunto estabilizar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TranslationEstimate:
    t: np.ndarray            # (2,) translação (tx, ty) que leva a imagem 2 ao quadro da imagem 1
    inliers: np.ndarray      # (M,) máscara booleana sobre as correspondências
    iterations: int          # iterações RANSAC efetivamente executadas
    rmse: float              # erro RMS dos inliers após o refinamento (px)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def required_iterations(inlier_ratio: float, sample_size: int = 1, confidence: float = 0.999) -> int:
    """Número de amostras S para obter ao menos uma amostra limpa com probabilidade P (eq. 8.30)."""
    p_k = inlier_ratio ** sample_size
    if p_k >= 1.0:
        return 1
    if p_k <= 0.0:
        return np.iinfo(np.int32).max
    return int(math.ceil(math.log(1 - confidence) / math.log(1 - p_k)))


def estimate_translation_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 3.0,
    confidence: float = 0.999,
    max_iterations: int = 2000,
    seed: int | None = 0,
) -> TranslationEstimate:
    """Estima t tal que pts1 ≈ pts2 + t, robusto a outliers.

    Args:
        pts1, pts2: arrays (M, 2) de pontos correspondentes.
        threshold: ε, distância máxima (px) para um inlier.
        confidence: P, probabilidade desejada de sortear uma amostra só de inliers.
        max_iterations: limite superior de iterações.
        seed: semente do gerador aleatório (``None`` = não determinístico).
    """
    m = len(pts1)
    if m == 0:
        return TranslationEstimate(np.zeros(2), np.zeros(0, bool), 0, float("nan"))
    rng = np.random.default_rng(seed)
    disp = (pts1 - pts2).astype(np.float64)   # hipótese t_i de cada correspondência

    best_count, best_t = -1, disp[0]
    n_iter, needed = 0, max_iterations
    while n_iter < min(needed, max_iterations):
        t = disp[rng.integers(m)]                         # amostra mínima (k = 1)
        residuals = np.linalg.norm(disp - t, axis=1)      # ||x2 + t − x1||
        count = int((residuals <= threshold).sum())
        n_iter += 1
        if count > best_count:
            best_count, best_t = count, t
            needed = required_iterations(count / m, 1, confidence)

    # Refinamento por mínimos quadrados nos inliers (média dos deslocamentos).
    t = best_t
    inliers = np.linalg.norm(disp - t, axis=1) <= threshold
    for _ in range(10):
        t = disp[inliers].mean(axis=0)
        new = np.linalg.norm(disp - t, axis=1) <= threshold
        if not new.any() or np.array_equal(new, inliers):
            inliers = new if new.any() else inliers
            break
        inliers = new
    rmse = float(np.sqrt(np.mean(np.sum((disp[inliers] - t) ** 2, axis=1))))
    return TranslationEstimate(t, inliers, n_iter, rmse)
