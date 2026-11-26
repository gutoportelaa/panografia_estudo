"""Monta o .zip de entrega da Atividade 03.

Roda a panografia nas duas imagens do enunciado (data/img1.png e data/img2.png),
grava as saídas pedidas em resultados/ e junta código, resultados e um LEIA-ME
em entrega/atividade03_panografia.zip.

    python scripts/empacotar_entrega.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "entrega" / "atividade03_panografia.zip"
SITE = "https://gutoportelaa.github.io/panografia_estudo/"
REPO = "https://github.com/gutoportelaa/panografia_estudo"

# O que vai no zip: o necessário para rodar, ler e conferir. Ficam de fora os
# exemplos extras do site, as figuras do README e a automação do GitHub.
INCLUDE = ["panografia", "app", "web", "tests", "data", "resultados", "scripts",
           "README.md", "LICENSE", "enunciado.md", "requirements.txt", "requirements-dev.txt",
           "pyproject.toml", "panografia.spec"]
SKIP_SAMPLES = ("aerial234", "sb_", "joao")   # imagens de terceiros: ficam só no site


def leia_me(r: dict) -> str:
    tx, ty = r["t"]
    return f"""Atividade 03: Panografia (costura de imagens transladadas)
==========================================================

Saídas pedidas no enunciado, para img1.png e img2.png (pasta data/):

  Mosaico ..................... resultados/mosaico.jpg
  Translação t = (tx, ty) ..... ({tx:.2f}, {ty:.2f}) px
  Correspondências válidas .... {r["n_inliers"]} inliers de {r["n_matches"]} pares
  Comparação visual ........... resultados/justaposicao.jpg (lado a lado, sem alinhamento)
                                contra resultados/mosaico.jpg

  Detector SIFT (implementação própria), teste NNDR 0,8 com checagem mútua,
  RANSAC de uma amostra com ε = 3 px, mistura por feathering.
  Erro RMS dos inliers: {r["rmse"]:.3f} px. Todos os números: resultados/resultado.json.

Relatório interativo (roda no navegador, sem instalar nada):
  {SITE}

Código-fonte: {REPO}

Como rodar
----------
  python -m venv .venv
  .venv/bin/pip install -r requirements.txt        (Windows: .venv\\Scripts\\pip)

  Linha de comando, gera as saídas acima:
    .venv/bin/python -m panografia data/img1.png data/img2.png -o resultados/

  Relatório no navegador, servido localmente:
    .venv/bin/python -m uvicorn app.server:app      e abra http://127.0.0.1:8000

  Testes:
    .venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest

Onde está cada parte
--------------------
  panografia/features/sift.py     SIFT: espaço-escala, extremos da DoG, descritor 128-D
  panografia/features/harris.py   Harris + ANMS + descritor MOPS
  panografia/matching.py          teste de razão NNDR e checagem mútua
  panografia/ransac.py            RANSAC da translação e refinamento por mínimos quadrados
  panografia/compositing.py       translação da imagem 2 e mistura (feathering ou média)
  panografia/pipeline.py          as etapas na ordem
"""


def main() -> int:
    # 1. Saídas a partir das imagens originais do enunciado.
    subprocess.run([sys.executable, "-m", "panografia", "data/img1.png", "data/img2.png", "-o", "resultados/"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    r = json.loads((ROOT / "resultados" / "resultado.json").read_text())

    # 2. Só arquivos versionados: nada de .venv, caches ou datasets baixados.
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.split()
    files = [f for f in tracked
             if f.split("/")[0] in INCLUDE and not f.startswith("resultados/")
             and f != "web/samples/samples.json"
             and not (f.startswith("web/samples/") and Path(f).name.startswith(SKIP_SAMPLES))]
    # As saídas acabaram de ser geradas: entram todas, versionadas ou não.
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "resultados").iterdir() if p.is_file())

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("atividade03_panografia/LEIA-ME.txt", leia_me(r))
        for f in files:
            zf.write(ROOT / f, f"atividade03_panografia/{f}")
        # O catálogo de exemplos do zip lista só as imagens que foram incluídas.
        cat = json.loads((ROOT / "web/samples/samples.json").read_text())
        cat = [c for c in cat if not c["id"].startswith(SKIP_SAMPLES)]
        zf.writestr("atividade03_panografia/web/samples/samples.json",
                    json.dumps(cat, ensure_ascii=False, indent=2) + "\n")

    print(f"{OUT.relative_to(ROOT)}: {len(files) + 1} arquivos, {OUT.stat().st_size / 2**20:.1f} MB")
    print(f"t = ({r['t'][0]:.2f}, {r['t'][1]:.2f}) px, {r['n_inliers']} inliers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
