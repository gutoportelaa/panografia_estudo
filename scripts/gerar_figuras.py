"""Gera as figuras do README (docs/figs/) e a imagem de abertura do site.

Todas vêm de uma execução real do pipeline sobre o par da atividade.

    python scripts/gerar_figuras.py          # requer matplotlib (requirements-dev.txt)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from panografia.pipeline import StitchParams, stitch  # noqa: E402

FIGS = ROOT / "docs" / "figs"
ASSETS = ROOT / "web" / "assets"

# Paleta da interface: papel, tinta, cobalto, verde/vermelho só com significado.
PAPER, INK, MUTED, LINE = "#f7f4ee", "#1f1b16", "#736a5e", "#d9d3c8"
COBALT, POS, NEG = "#1d4ed8", "#16a34a", "#dc2626"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def fit(img: np.ndarray, width: int) -> np.ndarray:
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)


def save(name: str, img: np.ndarray, width: int = 1000) -> None:
    cv2.imwrite(str(FIGS / name), fit(img, width), [cv2.IMWRITE_JPEG_QUALITY, 88])


def nndr_figure(n: dict, path: Path) -> None:
    edges, counts, thr = np.array(n["bordas"]), np.array(n["contagens"]), n["limiar"]
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=150)
    colors = [COBALT if edges[i + 1] <= thr + 1e-9 else LINE for i in range(len(counts))]
    ax.bar(edges[:-1], counts, width=np.diff(edges) * 0.9, align="edge", color=colors)
    ax.axvline(thr, color=COBALT, ls=(0, (4, 4)), lw=1)
    ax.text(thr - 0.01, counts.max() * 0.95, f"limiar {thr:.2f}".replace(".", ","), color=COBALT, ha="right")
    ax.set_xlabel("NNDR = d₁ / d₂")
    ax.set_ylabel("pontos")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def votes_figure(v: dict, t, eps: float, path: Path) -> None:
    pts, inl = np.array(v["t"]), np.array(v["inlier"], bool)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=150)
    for ax, zoom in zip(axes, (False, True)):
        ax.scatter(*pts[~inl].T, s=8, color=NEG, alpha=0.7, lw=0, label="outliers")
        ax.plot(*t, marker="o", ms=12, mfc="none", mec=INK, mew=1.2, zorder=1)
        ax.plot(*t, marker="+", ms=24, color=INK, mew=1.2, zorder=1)
        ax.scatter(*pts[inl].T, s=8, color=POS, lw=0, label="inliers", zorder=2)
        if zoom:
            r = max(3 * eps, 6)
            ax.set_xlim(t[0] - r, t[0] + r)
            ax.set_ylim(t[1] + r, t[1] - r)
            ax.add_patch(plt.Circle(t, eps, fill=False, ls=(0, (3, 3)), color=COBALT))
            ax.set_title(f"aproximação: círculo de raio ε = {eps:g} px", color=MUTED, fontsize=9)
        else:
            ax.invert_yaxis()
            ax.set_title(r"todos os votos $t_i = x_1 - x_2$", color=MUTED, fontsize=9)
            ax.annotate(f"{int(inl.sum())} inliers", xy=t, xytext=(-70, 40), textcoords="offset points", color=POS, fontsize=9, arrowprops=dict(arrowstyle="-", color=POS, lw=0.8))
            ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim" if not zoom else "box")
        ax.set_xlabel("tx (px)")
        ax.set_ylabel("ty (px)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    a = cv2.imread(str(ROOT / "data" / "img1.png"))
    b = cv2.imread(str(ROOT / "data" / "img2.png"))
    res = stitch(a, b, StitchParams())
    im, d = res.images, res.didatica

    cv2.imwrite(str(ASSETS / "capa.jpg"), im["contornos"], [cv2.IMWRITE_JPEG_QUALITY, 86])
    save("01_espaco_escala.jpg", im["escala"])
    pair = np.hstack([im["pontos1"], np.full((im["pontos1"].shape[0], 8, 3), 247, np.uint8), im["pontos2"]])
    save("02_pontos.jpg", pair)
    nndr_figure(d["nndr"], FIGS / "03_nndr.png")
    save("04_correspondencias.jpg", im["correspondencias"])
    votes_figure(d["votos"], res.t, res.params.ransac_threshold, FIGS / "05_votos.png")
    save("06_pesos.jpg", im["pesos"], 800)
    save("07_justaposicao.jpg", im["justaposicao"])
    save("08_mosaico.jpg", im["mosaico"])
    print(res.report())
    print(f"figuras em {FIGS.relative_to(ROOT)}/ e {ASSETS.relative_to(ROOT)}/capa.jpg")


if __name__ == "__main__":
    main()
