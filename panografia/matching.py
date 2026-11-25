"""Correspondência de descritores por vizinho mais próximo.

Szeliski, seção 7.1.3: usa-se a razão entre as distâncias ao primeiro e ao
segundo vizinho mais próximos — *nearest neighbor distance ratio* (eq. 7.18)

    NNDR = d1 / d2,

aceitando a correspondência quando NNDR < limiar (0,8 em Lowe 2004).
Opcionalmente exige consistência mútua (o vizinho de B também é A), que o
livro aponta como mais confiável que o NNDR unidirecional.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Matches:
    """Correspondências entre as características das imagens 1 e 2."""

    idx1: np.ndarray       # (M,) índices em features1
    idx2: np.ndarray       # (M,) índices em features2
    distance: np.ndarray   # (M,) distância ao vizinho mais próximo
    ratio: np.ndarray      # (M,) NNDR d1/d2

    def __len__(self) -> int:
        return len(self.idx1)


def _hamming_table() -> np.ndarray:
    return np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)


def pairwise_distances(d1: np.ndarray, d2: np.ndarray, binary: bool = False, chunk: int = 1024) -> np.ndarray:
    """Matriz (N1, N2) de distâncias euclidianas (ou de Hamming se ``binary``)."""
    if binary:
        table = _hamming_table()
        out = np.empty((len(d1), len(d2)), np.float32)
        for s in range(0, len(d1), 256):
            x = np.bitwise_xor(d1[s:s + 256, None, :], d2[None, :, :])
            out[s:s + 256] = table[x].sum(-1)
        return out
    a = d1.astype(np.float32)
    b = d2.astype(np.float32)
    bb = (b * b).sum(1)
    out = np.empty((len(a), len(b)), np.float32)
    for s in range(0, len(a), chunk):
        blk = a[s:s + chunk]
        d2sq = (blk * blk).sum(1)[:, None] + bb[None, :] - 2.0 * blk @ b.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2sq, 0))
    return out


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.8,
    mutual: bool = True,
    binary: bool = False,
) -> Matches:
    """Casa descritores com o teste de razão NNDR (e checagem mútua opcional)."""
    empty = Matches(*(np.zeros(0, t) for t in (int, int, np.float32, np.float32)))
    if len(desc1) == 0 or len(desc2) < 2:
        return empty
    dist = pairwise_distances(desc1, desc2, binary)
    nn = np.argpartition(dist, 1, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    d_a, d_b = dist[rows, nn[:, 0]], dist[rows, nn[:, 1]]
    swap = d_b < d_a
    best = np.where(swap, nn[:, 1], nn[:, 0])
    d1 = np.minimum(d_a, d_b)
    d2 = np.maximum(d_a, d_b)
    nndr = d1 / np.maximum(d2, 1e-12)
    keep = nndr < ratio
    if mutual:
        back = np.argmin(dist, axis=0)
        keep &= back[best] == rows
    idx = np.nonzero(keep)[0]
    return Matches(idx, best[idx], d1[idx], nndr[idx])
