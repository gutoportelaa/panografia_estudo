"""Servidor web local da Panografia (FastAPI).

Serve a mesma interface publicada no GitHub Pages (pasta ``web/``) e uma API
que executa o pipeline em Python nativo, mais rápido que o Pyodide.

    uvicorn app.server:app --reload      # http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import sys
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from panografia import service

# Em um executável PyInstaller os arquivos de dados ficam em sys._MEIPASS.
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
WEB = BASE / "web"
PACKAGE = BASE / "panografia"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = FastAPI(title="Panografia", description="Costura de imagens transladadas", version="0.2.0")


@app.get("/api/config")
def config():
    return {**service.config(), "engine": "server"}


@app.post("/api/stitch")
def stitch_endpoint(
    imagens: list[UploadFile] | None = File(None),
    img1: UploadFile | None = File(None),
    img2: UploadFile | None = File(None),
    detector: str = Form("sift"),
    blend: str = Form("feather"),
    ratio: float = Form(0.8),
    mutual: bool = Form(True),
    ransac_threshold: float = Form(3.0),
    max_points: int = Form(4000),
    max_side: int = Form(service.DEFAULT_MAX_SIDE),
    seed: int = Form(0),
    figuras: str = Form(""),
    detalhe: int = Form(0),
):
    # Uma sequência em "imagens" (em ordem) ou o par clássico img1/img2.
    uploads = list(imagens or []) or [f for f in (img1, img2) if f is not None]
    if not 2 <= len(uploads) <= service.MAX_IMAGES:
        raise HTTPException(400, f"envie de 2 a {service.MAX_IMAGES} imagens")
    datas = [f.file.read() for f in uploads]
    if max(len(d) for d in datas) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"arquivo maior que {MAX_UPLOAD_BYTES // 2**20} MB")
    options = dict(detector=detector, blend=blend, ratio=ratio, mutual=mutual,
                   ransac_threshold=ransac_threshold, max_points=max_points, max_side=max_side, seed=seed,
                   figuras=figuras, detalhe=detalhe)
    try:
        summary, images = service.run_many(datas, options)
    except service.InputError as exc:
        raise HTTPException(400, str(exc)) from exc
    except service.StitchError as exc:
        raise HTTPException(422, str(exc)) from exc
    summary["images"] = {
        k: "data:image/jpeg;base64," + base64.b64encode(v).decode("ascii") for k, v in images.items()
    }
    return summary


@app.get("/panografia.zip")
def package_zip():
    """Pacote Python para o modo Pyodide (?engine=pyodide), montado na hora."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(PACKAGE.rglob("*.py")):
            zf.write(f, f.relative_to(PACKAGE.parent).as_posix())
    return Response(buf.getvalue(), media_type="application/zip")


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
