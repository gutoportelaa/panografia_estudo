"""Monta o site estático para o GitHub Pages em ``_site/``.

Copia a interface (``web/``) e empacota o código Python em ``panografia.zip``,
que o Pyodide baixa e importa no navegador.

    python scripts/build_site.py            # gera _site/
    python -m http.server -d _site 8001     # testa localmente
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    with zipfile.ZipFile(out / "panografia.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((ROOT / "panografia").rglob("*.py")):
            zf.write(f, f.relative_to(ROOT).as_posix())
    (out / ".nojekyll").write_text("")
    return out


if __name__ == "__main__":
    target = build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site")
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"site gerado em {target} ({size / 1024:.0f} KB)")
