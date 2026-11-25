"""Pipeline completo da panografia por translação.

detecção → correspondência (NNDR) → RANSAC (translação) → composição.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .compositing import composite, side_by_side
from .features import detect
from .matching import match_descriptors
from .ransac import estimate_translation_ransac
from .visualization import draw_keypoints, draw_matches, overlap_outline


@dataclass
class StitchParams:
    detector: str = "sift"           # sift | harris | opencv-sift | opencv-orb
    max_points: int = 4000
    ratio: float = 0.8               # limiar NNDR
    mutual: bool = True              # checagem cruzada das correspondências
    ransac_threshold: float = 3.0    # ε em pixels
    ransac_confidence: float = 0.999
    ransac_max_iterations: int = 2000
    blend: str = "feather"           # feather | average | overlay
    seed: int = 0


@dataclass
class StitchResult:
    params: StitchParams
    t: tuple[float, float]
    n_keypoints: tuple[int, int]
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    ransac_iterations: int
    rmse: float
    mosaic_size: tuple[int, int]
    timings_ms: dict[str, float]
    images: dict[str, np.ndarray] = field(repr=False)

    def summary(self) -> dict:
        """Resultados numéricos serializáveis (sem as imagens)."""
        d = asdict(self)
        d.pop("images")
        return d

    def report(self) -> str:
        tx, ty = self.t
        lines = [
            f"Detector ................ {self.params.detector}",
            f"Pontos detectados ....... img1 = {self.n_keypoints[0]}, img2 = {self.n_keypoints[1]}",
            f"Correspondências (NNDR) . {self.n_matches}",
            f"Inliers (RANSAC) ........ {self.n_inliers} ({100 * self.inlier_ratio:.1f}%)",
            f"Iterações RANSAC ........ {self.ransac_iterations}",
            f"Translação t = (tx, ty) . ({tx:.2f}, {ty:.2f}) px",
            f"Erro RMS dos inliers .... {self.rmse:.3f} px",
            f"Mosaico (L × A) ......... {self.mosaic_size[0]} × {self.mosaic_size[1]} px",
        ]
        return "\n".join(lines)


class StitchError(RuntimeError):
    """Falha ao alinhar as imagens (poucas correspondências, por exemplo)."""


def stitch(img1: np.ndarray, img2: np.ndarray, params: StitchParams | None = None) -> StitchResult:
    """Alinha ``img2`` à ``img1`` por translação e gera o mosaico.

    As imagens são arrays BGR ``uint8`` (como lidas por ``cv2.imread``).
    A translação t retornada leva coordenadas da imagem 2 para o quadro da
    imagem 1: x1 = x2 + t.
    """
    p = params or StitchParams()
    timings: dict[str, float] = {}

    def tick(name, t0):
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    f1 = detect(img1, p.detector, p.max_points)
    f2 = detect(img2, p.detector, p.max_points)
    tick("deteccao", t0)

    t0 = time.perf_counter()
    m = match_descriptors(f1.descriptors, f2.descriptors, p.ratio, p.mutual, f1.binary)
    tick("correspondencia", t0)
    if len(m) == 0:
        raise StitchError(
            f"nenhuma correspondência encontrada (pontos: {len(f1)} e {len(f2)}). "
            "Tente aumentar o limiar NNDR ou trocar o detector."
        )
    pts1, pts2 = f1.points[m.idx1], f2.points[m.idx2]

    t0 = time.perf_counter()
    est = estimate_translation_ransac(
        pts1, pts2, p.ransac_threshold, p.ransac_confidence, p.ransac_max_iterations, p.seed
    )
    tick("ransac", t0)
    if est.n_inliers < 3:
        raise StitchError(f"apenas {est.n_inliers} inlier(s): as imagens não parecem se sobrepor.")

    t0 = time.perf_counter()
    mosaic, canvas, _ = composite(img1, img2, est.t, p.blend)
    tick("composicao", t0)

    images = {
        "mosaico": mosaic,
        "justaposicao": side_by_side(img1, img2),
        "correspondencias": draw_matches(img1, img2, pts1, pts2, est.inliers),
        "contornos": overlap_outline(mosaic, canvas, img1.shape, img2.shape),
        "pontos1": draw_keypoints(img1, f1.points, f1.scales),
        "pontos2": draw_keypoints(img2, f2.points, f2.scales),
    }
    timings["total"] = round(sum(timings.values()), 1)
    return StitchResult(
        params=p,
        t=(float(est.t[0]), float(est.t[1])),
        n_keypoints=(len(f1), len(f2)),
        n_matches=len(m),
        n_inliers=est.n_inliers,
        inlier_ratio=est.n_inliers / len(m),
        ransac_iterations=est.iterations,
        rmse=est.rmse,
        mosaic_size=(canvas.width, canvas.height),
        timings_ms=timings,
        images=images,
    )
