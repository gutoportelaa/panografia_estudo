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
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
    detector: str = Form("sift"),
    blend: str = Form("feather"),
    ratio: float = Form(0.8),
    mutual: bool = Form(True),
    ransac_threshold: float = Form(3.0),
    max_points: int = Form(4000),
    max_side: int = Form(service.DEFAULT_MAX_SIDE),
    seed: int = Form(0),
):
    data1, data2 = img1.file.read(), img2.file.read()
    if max(len(data1), len(data2)) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"arquivo maior que {MAX_UPLOAD_BYTES // 2**20} MB")
    options = dict(detector=detector, blend=blend, ratio=ratio, mutual=mutual,
                   ransac_threshold=ransac_threshold, max_points=max_points, max_side=max_side, seed=seed)
    try:
        summary, images = service.run(data1, data2, options)
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
