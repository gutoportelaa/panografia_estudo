"""Interface de linha de comando.

Exemplo::

    python -m panografia data/img1.png data/img2.png -o saida/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .compositing import BLEND_MODES
from .features import DETECTORS
from .pipeline import StitchError, StitchParams, stitch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="panografia", description="Panografia: costura de imagens transladadas.")
    ap.add_argument("img1", type=Path)
    ap.add_argument("img2", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("saida"), help="pasta de saída (padrão: saida/)")
    ap.add_argument("--detector", choices=sorted(DETECTORS), default="sift")
    ap.add_argument("--blend", choices=sorted(BLEND_MODES), default="feather")
    ap.add_argument("--ratio", type=float, default=0.8, help="limiar NNDR (padrão 0.8)")
    ap.add_argument("--no-mutual", action="store_true", help="desativa a checagem cruzada")
    ap.add_argument("--threshold", type=float, default=3.0, help="ε do RANSAC em px (padrão 3)")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    img1, img2 = cv2.imread(str(args.img1)), cv2.imread(str(args.img2))
    for path, img in ((args.img1, img1), (args.img2, img2)):
        if img is None:
            print(f"erro: não foi possível ler {path}", file=sys.stderr)
            return 2

    params = StitchParams(
        detector=args.detector, max_points=args.max_points, ratio=args.ratio, mutual=not args.no_mutual,
        ransac_threshold=args.threshold, blend=args.blend, seed=args.seed,
    )
    try:
        result = stitch(img1, img2, params)
    except StitchError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    print(result.report())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, image in result.images.items():
        cv2.imwrite(str(args.output / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (args.output / "resultado.json").write_text(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    print(f"\nArquivos salvos em {args.output}/ (mosaico.jpg, justaposicao.jpg, correspondencias.jpg, ...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
