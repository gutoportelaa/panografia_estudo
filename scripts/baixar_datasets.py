"""Baixa do Hugging Face as cenas usadas nos exemplos (datasets CC BY 4.0).

Os datasets são "gated": antes, aceite os termos nas páginas abaixo, logado
na sua conta, e autentique-se com `hf auth login` (ou `huggingface-cli login`).

  https://huggingface.co/datasets/RussRobin/Aerial234
  https://huggingface.co/datasets/RussRobin/StitchBench

    python scripts/baixar_datasets.py            # salva em datasets_externos/

Créditos: Aerial234 e StitchBench, por RussRobin, licença CC BY 4.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path(__file__).resolve().parent.parent / "datasets_externos"

SELECAO = {
    # Sequência de drone com câmera para baixo: quase translação pura.
    "RussRobin/Aerial234": [f"{i:03d}.JPG" for i in range(1, 9)],
    "RussRobin/StitchBench": [
        "Aerial/Kindergarten/*",
        "Aerial/GateYiPlaza/*",
        # Cenas do chão, com paralaxe: mostram o limite do modelo de translação.
        "General/APAP-01_garden/*",
        "General/DHW-03_carpark/*",
        "General/REW_parksquare/*",
        "General/SPHP-04_street/*",
        "General/CAVE-04_times_square/*",
    ],
}


def main() -> int:
    for repo, padroes in SELECAO.items():
        destino = DEST / repo.split("/")[1]
        print(f"baixando {repo} -> {destino}")
        try:
            snapshot_download(repo, repo_type="dataset", allow_patterns=padroes, local_dir=destino)
        except Exception as exc:  # acesso negado, termos não aceitos, rede
            print(f"  falhou: {exc}\n  Aceite os termos em https://huggingface.co/datasets/{repo} "
                  "e rode `hf auth login`.", file=sys.stderr)
            return 1
    print("pronto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
