"""Escolhe pares dos datasets externos e os adiciona aos exemplos do site.

Rode depois de `scripts/baixar_datasets.py`. Para cada cena, testa os pares de
imagens consecutivas com o próprio pipeline e fica com o par de mais inliers
entre os que têm deslocamento visível. As imagens são reduzidas para 960 px e
gravadas em web/samples/, e o catálogo samples.json é atualizado.

    python scripts/preparar_exemplos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from panografia.pipeline import StitchError, StitchParams, stitch  # noqa: E402

EXT = ROOT / "datasets_externos"
SAMPLES = ROOT / "web" / "samples"
CATALOG = SAMPLES / "samples.json"
WIDTH = 960

CREDIT_AERIAL = "Aerial234, por RussRobin (Hugging Face), CC BY 4.0"
CREDIT_BENCH = "StitchBench, por RussRobin (Hugging Face), CC BY 4.0"

# id, pasta, título, etiqueta, tipo, crédito
SCENES = [
    ("aerial234", "Aerial234", "Sequência aérea", "drone, câmera para baixo", "ok", CREDIT_AERIAL),
    ("sb_kindergarten", "StitchBench/Aerial/Kindergarten", "Escola vista do drone", "drone, câmera para baixo", "ok", CREDIT_BENCH),
    ("sb_gateyiplaza", "StitchBench/Aerial/GateYiPlaza", "Praça vista do drone", "drone, câmera para baixo", "ok", CREDIT_BENCH),
    ("sb_garden", "StitchBench/General/APAP-01_garden", "Jardim", "limite: paralaxe", "warn", CREDIT_BENCH),
    ("sb_carpark", "StitchBench/General/DHW-03_carpark", "Estacionamento", "limite: paralaxe", "warn", CREDIT_BENCH),
    ("sb_parksquare", "StitchBench/General/REW_parksquare", "Praça", "limite: paralaxe", "warn", CREDIT_BENCH),
    ("sb_street", "StitchBench/General/SPHP-04_street", "Rua", "limite: paralaxe", "warn", CREDIT_BENCH),
    ("sb_times_square", "StitchBench/General/CAVE-04_times_square", "Times Square", "limite: paralaxe", "warn", CREDIT_BENCH),
]


def fit(img: np.ndarray, width: int = WIDTH) -> np.ndarray:
    s = width / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def best_pair(files: list[Path]):
    best = None
    imgs = [fit(cv2.imread(str(f))) for f in files]
    for i in range(len(imgs) - 1):
        a, b = imgs[i], imgs[i + 1]
        if a is None or b is None or a.shape != b.shape:
            continue
        try:
            r = stitch(a, b, StitchParams(detector="sift"))
        except StitchError:
            continue
        shift = float(np.hypot(*r.t))
        if shift < 0.12 * a.shape[1]:   # quase o mesmo quadro: não ensina nada
            continue
        score = r.n_inliers * min(1.0, shift / (0.3 * a.shape[1]))
        if best is None or score > best[0]:
            best = (score, a, b, r, files[i].name, files[i + 1].name)
    return best


def note_for(kind: str, r, n1: str, n2: str) -> str:
    t = f"({r.t[0]:.0f}; {r.t[1]:.0f})".replace("-", "−")
    pct = f"{100 * r.inlier_ratio:.0f}%"
    if kind == "ok":
        return (f"Fotos {n1} e {n2}, de drone com a câmera apontada para baixo, voando a altura constante. "
                f"Visto de cima, o chão é quase plano, então a mudança entre as fotos é quase só translação: "
                f"t ≈ {t} px, com {pct} dos pares de acordo.")
    return (f"Fotos {n1} e {n2}, tiradas do chão. Objetos perto e longe da câmera se deslocam de forma "
            f"diferente (paralaxe), então só {pct} dos pares concordam com um único t ≈ {t} px, e as "
            f"bordas aparecem duplicadas no mosaico.")


def main() -> int:
    if not EXT.exists():
        print("rode antes: python scripts/baixar_datasets.py", file=sys.stderr)
        return 1
    catalog = json.loads(CATALOG.read_text())
    # Notas revisadas à mão sobrevivem a uma nova rodada; só as cenas novas recebem texto gerado.
    revised = {c["id"]: c for c in catalog}
    catalog = [c for c in catalog if not c["id"].startswith(("aerial234", "sb_"))]
    added = []
    for sid, folder, title, tag, kind, credit in SCENES:
        files = sorted(p for p in (EXT / folder).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                       and "ortho" not in p.name)
        if len(files) < 2:
            print(f"  {sid}: sem imagens em {folder}, pulando")
            continue
        best = best_pair(files)
        if best is None:
            print(f"  {sid}: nenhum par alinhável")
            continue
        _, a, b, r, n1, n2 = best
        for k, img in ((1, a), (2, b)):
            cv2.imwrite(str(SAMPLES / f"{sid}_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        old = revised.get(sid, {})
        added.append({"id": sid, "title": old.get("title", title), "tag": old.get("tag", tag), "kind": kind,
                      "img1": f"samples/{sid}_1.jpg", "img2": f"samples/{sid}_2.jpg",
                      "credit": credit, "note": old.get("note") or note_for(kind, r, n1, n2),
                      "_stats": {"t": [round(v, 1) for v in r.t], "inliers": r.n_inliers,
                                 "ratio": round(r.inlier_ratio, 3), "rmse": round(r.rmse, 2)}})
        print(f"  {sid}: {n1} + {n2}  t={np.round(r.t, 1)}  inliers={r.n_inliers} ({100 * r.inlier_ratio:.0f}%)")
    # Exemplos de drone logo após a paisagem; os de paralaxe antes das falhas do Rover.
    ok = [c for c in added if c["kind"] == "ok"]
    warn = [c for c in added if c["kind"] != "ok"]
    ordered = catalog[:2] + ok + catalog[2:3] + warn + catalog[3:]
    for c in ordered:
        c.pop("_stats", None)
    CATALOG.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(added)} exemplos adicionados a {CATALOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
