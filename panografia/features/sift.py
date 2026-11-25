"""SIFT (Lowe 2004) implementado do zero com NumPy.

Referência: Szeliski, 2ª ed., seções 7.1.1 (detecção em espaço-escala por
diferença de gaussianas, Fig. 7.11) e 7.1.2 (descritor SIFT, Fig. 7.16).
O OpenCV é usado só para operações elementares (suavização gaussiana,
dilatação/erosão e redimensionamento).

Etapas:
1. pirâmide gaussiana com ``s`` intervalos por oitava e pilha DoG;
2. extremos locais 3×3×3 na DoG;
3. refinamento subpixel por ajuste quadrático (expansão de Taylor), rejeição
   de baixo contraste e de respostas de borda (razão de curvaturas r = 10);
4. orientação dominante via histograma de 36 bins dos gradientes;
5. descritor 4×4×8 = 128 dimensões com interpolação trilinear,
   normalização, saturação em 0,2 e renormalização.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .base import Features, to_gray_float

_BORDER = 5          # pixels ignorados na borda de cada oitava
_MAX_INTERP = 5      # iterações máximas do refinamento subpixel
_ORI_BINS = 36
_ORI_PEAK_RATIO = 0.8
_ORI_SIG_FCTR = 1.5
_ORI_RADIUS = 3 * _ORI_SIG_FCTR
_DESC_WIDTH = 4      # d: grade d×d de histogramas
_DESC_BINS = 8       # n: bins de orientação por histograma
_DESC_SCL_FCTR = 3.0
_DESC_MAG_THR = 0.2


# --------------------------------------------------------------------------- #
# 1. Espaço-escala
# --------------------------------------------------------------------------- #
def _build_pyramids(base: np.ndarray, n_octaves: int, s: int, sigma: float):
    """Retorna (gaussianas, dogs): listas por oitava de arrays (s+3, h, w) e (s+2, h, w)."""
    k = 2.0 ** (1.0 / s)
    # Desvios incrementais entre níveis consecutivos de uma oitava.
    sig = [sigma]
    for i in range(1, s + 3):
        prev = sigma * k ** (i - 1)
        total = prev * k
        sig.append(math.sqrt(total * total - prev * prev))

    gaussians, dogs = [], []
    img = base
    for o in range(n_octaves):
        levels = [img]
        for i in range(1, s + 3):
            levels.append(cv2.GaussianBlur(levels[-1], (0, 0), sig[i]))
        stack = np.stack(levels)
        gaussians.append(stack)
        dogs.append(stack[1:] - stack[:-1])
        # O nível s tem o dobro do sigma da base: subamostra por 2.
        img = levels[s][::2, ::2]
        if min(img.shape) < 2 * _BORDER + 3:
            break
    return gaussians, dogs


# --------------------------------------------------------------------------- #
# 2–3. Extremos e refinamento
# --------------------------------------------------------------------------- #
def _find_extrema(dog: np.ndarray, threshold: float) -> np.ndarray:
    """Extremos 3×3×3 nas camadas internas. Retorna (N, 3) com (camada, y, x)."""
    kernel = np.ones((3, 3), np.uint8)
    dil = np.stack([cv2.dilate(d, kernel) for d in dog])
    ero = np.stack([cv2.erode(d, kernel) for d in dog])
    found = []
    for l in range(1, dog.shape[0] - 1):
        d = dog[l]
        mx = np.maximum(np.maximum(dil[l - 1], dil[l]), dil[l + 1])
        mn = np.minimum(np.minimum(ero[l - 1], ero[l]), ero[l + 1])
        mask = ((d >= mx) & (d > threshold)) | ((d <= mn) & (d < -threshold))
        mask[:_BORDER, :] = mask[-_BORDER:, :] = False
        mask[:, :_BORDER] = mask[:, -_BORDER:] = False
        ys, xs = np.nonzero(mask)
        found.append(np.stack([np.full_like(ys, l), ys, xs], axis=1))
    return np.concatenate(found) if found else np.zeros((0, 3), int)


def _derivatives(dog: np.ndarray, l, y, x):
    """Gradiente (n, 3) e Hessiana (n, 3, 3) da DoG em (x, y, σ) por diferenças finitas."""
    v = dog[l, y, x]
    dx = (dog[l, y, x + 1] - dog[l, y, x - 1]) * 0.5
    dy = (dog[l, y + 1, x] - dog[l, y - 1, x]) * 0.5
    ds = (dog[l + 1, y, x] - dog[l - 1, y, x]) * 0.5
    dxx = dog[l, y, x + 1] + dog[l, y, x - 1] - 2 * v
    dyy = dog[l, y + 1, x] + dog[l, y - 1, x] - 2 * v
    dss = dog[l + 1, y, x] + dog[l - 1, y, x] - 2 * v
    dxy = (dog[l, y + 1, x + 1] - dog[l, y + 1, x - 1] - dog[l, y - 1, x + 1] + dog[l, y - 1, x - 1]) * 0.25
    dxs = (dog[l + 1, y, x + 1] - dog[l + 1, y, x - 1] - dog[l - 1, y, x + 1] + dog[l - 1, y, x - 1]) * 0.25
    dys = (dog[l + 1, y + 1, x] - dog[l + 1, y - 1, x] - dog[l - 1, y + 1, x] + dog[l - 1, y - 1, x]) * 0.25
    grad = np.stack([dx, dy, ds], axis=1).astype(np.float64)
    hess = np.stack(
        [np.stack([dxx, dxy, dxs], 1), np.stack([dxy, dyy, dys], 1), np.stack([dxs, dys, dss], 1)],
        axis=1,
    ).astype(np.float64)
    return v.astype(np.float64), grad, hess


def _solve_offsets(grad: np.ndarray, hess: np.ndarray):
    """x̂ = −H⁻¹∇D para cada candidato; marca Hessianas singulares como inválidas."""
    det = np.linalg.det(hess)
    ok = np.abs(det) > 1e-12
    safe = np.where(ok[:, None, None], hess, np.eye(3))
    off = -np.linalg.solve(safe, grad[..., None])[..., 0]
    return off, ok


def _refine(dog: np.ndarray, cand: np.ndarray, s: int, contrast_threshold: float, edge_threshold: float):
    """Refinamento subpixel vetorizado. Retorna (l, y, x, offset (n,3), resposta)."""
    n_layers, h, w = dog.shape
    l, y, x = (cand[:, i].copy() for i in range(3))
    alive = np.ones(len(cand), bool)
    converged = np.zeros(len(cand), bool)
    for _ in range(_MAX_INTERP):
        idx = np.nonzero(alive & ~converged)[0]
        if len(idx) == 0:
            break
        _, grad, hess = _derivatives(dog, l[idx], y[idx], x[idx])
        off, ok = _solve_offsets(grad, hess)
        alive[idx[~ok]] = False
        small = ok & np.all(np.abs(off) < 0.5, axis=1)
        converged[idx[small]] = True
        move = idx[ok & ~small]
        step = np.round(off[ok & ~small]).astype(int)
        # Evita saltos gigantes por Hessianas mal condicionadas.
        big = np.any(np.abs(off[ok & ~small]) > 1e3, axis=1)
        alive[move[big]] = False
        x[move] += step[:, 0]
        y[move] += step[:, 1]
        l[move] += step[:, 2]
        inside = (l[move] >= 1) & (l[move] <= s) & (x[move] >= _BORDER) & (x[move] < w - _BORDER) \
            & (y[move] >= _BORDER) & (y[move] < h - _BORDER)
        alive[move[~inside]] = False
    keep = np.nonzero(alive & converged)[0]
    l, y, x = l[keep], y[keep], x[keep]
    v, grad, hess = _derivatives(dog, l, y, x)
    off, ok = _solve_offsets(grad, hess)

    # Rejeição de baixo contraste: |D(x̂)| = |D + ½∇Dᵀx̂|.
    contrast = v + 0.5 * np.einsum("ij,ij->i", grad, off)
    ok &= np.abs(contrast) * s >= contrast_threshold
    # Rejeição de bordas: tr(H)²/det(H) < (r+1)²/r (Hessiana 2×2 espacial).
    tr = hess[:, 0, 0] + hess[:, 1, 1]
    det = hess[:, 0, 0] * hess[:, 1, 1] - hess[:, 0, 1] ** 2
    r = edge_threshold
    ok &= (det > 0) & (tr * tr * r < (r + 1) ** 2 * det)
    return l[ok], y[ok], x[ok], off[ok], np.abs(contrast[ok])


# --------------------------------------------------------------------------- #
# 4–5. Orientação e descritor
# --------------------------------------------------------------------------- #
def _gradients(img: np.ndarray):
    """Magnitude e ângulo (rad, em [0, 2π)) do gradiente por diferenças centrais."""
    dx = np.zeros_like(img)
    dy = np.zeros_like(img)
    dx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    dy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    return mag, ang


def _orientations(mag, ang, x: int, y: int, scale: float) -> list[float]:
    """Orientações dominantes (rad) do ponto em (x, y) com escala ``scale`` na oitava."""
    h, w = mag.shape
    radius = int(round(_ORI_RADIUS * scale))
    sig = _ORI_SIG_FCTR * scale
    y0, y1 = max(1, y - radius), min(h - 1, y + radius + 1)
    x0, x1 = max(1, x - radius), min(w - 1, x + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return []
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weight = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig * sig))
    bins = np.floor(ang[y0:y1, x0:x1] * _ORI_BINS / (2 * np.pi)).astype(int) % _ORI_BINS
    hist = np.bincount(bins.ravel(), (weight * mag[y0:y1, x0:x1]).ravel(), _ORI_BINS)
    # Suavização circular [1 4 6 4 1]/16.
    hist = (np.roll(hist, 2) + np.roll(hist, -2) + 4 * (np.roll(hist, 1) + np.roll(hist, -1)) + 6 * hist) / 16
    peak = hist.max()
    if peak <= 0:
        return []
    left, right = np.roll(hist, 1), np.roll(hist, -1)
    result = []
    for b in np.nonzero((hist > left) & (hist > right) & (hist >= _ORI_PEAK_RATIO * peak))[0]:
        # Interpolação parabólica do pico.
        denom = left[b] - 2 * hist[b] + right[b]
        delta = 0.5 * (left[b] - right[b]) / denom if denom != 0 else 0.0
        result.append(((b + 0.5 + delta) % _ORI_BINS) * 2 * np.pi / _ORI_BINS)
    return result


def _descriptor(mag, ang, x: float, y: float, scale: float, theta: float) -> np.ndarray:
    """Descritor SIFT 128-D com interpolação trilinear (Szeliski Fig. 7.16)."""
    d, n = _DESC_WIDTH, _DESC_BINS
    h, w = mag.shape
    hist_width = _DESC_SCL_FCTR * scale
    radius = int(round(hist_width * math.sqrt(2) * (d + 1) * 0.5))
    radius = min(radius, int(math.hypot(w, h)))
    xi, yi = int(round(x)), int(round(y))
    oy, ox = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    px, py = xi + ox, yi + oy
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Coordenadas relativas ao ponto (subpixel), giradas por −θ e em unidades de célula.
    dxf, dyf = px - x, py - y
    rx = (dxf * cos_t + dyf * sin_t) / hist_width
    ry = (-dxf * sin_t + dyf * cos_t) / hist_width
    rbin = ry + d / 2 - 0.5
    cbin = rx + d / 2 - 0.5
    valid = (rbin > -1) & (rbin < d) & (cbin > -1) & (cbin < d) \
        & (px > 0) & (px < w - 1) & (py > 0) & (py < h - 1)
    if not valid.any():
        return np.zeros(d * d * n, np.float32)
    rbin, cbin = rbin[valid], cbin[valid]
    m = mag[py[valid], px[valid]]
    a = ang[py[valid], px[valid]]
    obin = np.mod(a - theta, 2 * np.pi) * n / (2 * np.pi)
    weight = m * np.exp(-(rx[valid] ** 2 + ry[valid] ** 2) / (2 * (0.5 * d) ** 2))

    r0, c0, o0 = np.floor(rbin).astype(int), np.floor(cbin).astype(int), np.floor(obin).astype(int)
    fr, fc, fo = rbin - r0, cbin - c0, obin - o0
    hist = np.zeros((d + 2, d + 2, n))
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            for do, wo in ((0, 1 - fo), (1, fo)):
                np.add.at(hist, (r0 + 1 + dr, c0 + 1 + dc, (o0 + do) % n), weight * wr * wc * wo)
    vec = hist[1:d + 1, 1:d + 1, :].ravel()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = np.minimum(vec / norm, _DESC_MAG_THR)
        vec /= np.linalg.norm(vec) + 1e-12
    return vec.astype(np.float32)


# --------------------------------------------------------------------------- #
# API pública
# --------------------------------------------------------------------------- #
def detect_sift(
    image: np.ndarray,
    n_octave_layers: int = 3,
    sigma: float = 1.6,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    upsample: bool = False,
    max_points: int | None = 4000,
) -> Features:
    """Detecta pontos SIFT e calcula seus descritores de 128 dimensões.

    Args:
        image: imagem BGR ou cinza ``uint8``.
        n_octave_layers: intervalos ``s`` por oitava.
        sigma: suavização da base de cada oitava (1,6 em Lowe 2004).
        contrast_threshold: limiar de contraste de |D(x̂)| (imagem em [0, 1]).
        edge_threshold: razão máxima r entre curvaturas principais.
        upsample: duplica a imagem antes de montar a pirâmide (mais pontos, mais lento).
        max_points: mantém os pontos de maior contraste (``None`` = todos).
    """
    s = n_octave_layers
    gray = to_gray_float(image)
    assumed_blur = 0.5
    if upsample:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        assumed_blur *= 2
    base = cv2.GaussianBlur(gray, (0, 0), math.sqrt(max(sigma ** 2 - assumed_blur ** 2, 0.01)))
    n_octaves = max(1, int(round(math.log2(min(base.shape)))) - 3)
    gaussians, dogs = _build_pyramids(base, n_octaves, s, sigma)

    points, scales, descs, strengths = [], [], [], []
    pre_threshold = 0.5 * contrast_threshold / s
    for o, (gauss, dog) in enumerate(zip(gaussians, dogs)):
        cand = _find_extrema(dog, pre_threshold)
        if len(cand) == 0:
            continue
        l, y, x, off, resp = _refine(dog, cand, s, contrast_threshold, edge_threshold)
        grads = {}
        for li, yi, xi, of, rs in zip(l, y, x, off, resp):
            if li not in grads:
                grads[li] = _gradients(gauss[li])
            mag, ang = grads[li]
            layer = li + of[2]
            scale_oct = sigma * 2.0 ** (layer / s)        # escala na oitava
            fx, fy = xi + of[0], yi + of[1]
            for theta in _orientations(mag, ang, int(xi), int(yi), scale_oct):
                descs.append(_descriptor(mag, ang, fx, fy, scale_oct, theta))
                factor = 2.0 ** o
                px, py = fx * factor, fy * factor
                sc = scale_oct * factor
                if upsample:
                    # cv2.resize(×2): pixel u da imagem ampliada ↔ u/2 − 0,25 na original.
                    px, py, sc = px * 0.5 - 0.25, py * 0.5 - 0.25, sc * 0.5
                points.append((px, py))
                scales.append(sc)
                strengths.append(rs)

    if not points:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), np.zeros(0))
    pts = np.asarray(points, np.float32)
    dsc = np.asarray(descs, np.float32)
    scl = np.asarray(scales, np.float32)
    if max_points is not None and len(pts) > max_points:
        top = np.argsort(-np.asarray(strengths))[:max_points]
        pts, dsc, scl = pts[top], dsc[top], scl[top]
    return Features(pts, dsc, scl)
