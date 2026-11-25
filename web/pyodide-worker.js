/* Executa o pacote Python `panografia` no navegador via Pyodide (WebAssembly).
 *
 * Mensagens recebidas:  {id, type: "init"} | {id, type: "stitch", img1, img2, options}
 * Mensagens enviadas:   {type: "progress", message}
 *                       {id, ok: true, result} | {id, ok: false, error}
 */
"use strict";

const PYODIDE_VERSION = "0.28.3";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

importScripts(PYODIDE_URL + "pyodide.js");

let ready = null;

function progress(message) {
  self.postMessage({ type: "progress", message });
}

async function boot() {
  progress("Carregando o interpretador Python (Pyodide)…");
  const py = await loadPyodide({ indexURL: PYODIDE_URL });
  progress("Carregando NumPy e OpenCV…");
  await py.loadPackage(["numpy", "opencv-python"]);
  progress("Carregando o pacote panografia…");
  const resp = await fetch(new URL("panografia.zip", self.location.href));
  if (!resp.ok) throw new Error("não foi possível baixar panografia.zip");
  py.unpackArchive(await resp.arrayBuffer(), "zip", { extractDir: "/home/pyodide" });
  py.runPython(`
import json, sys
sys.path.insert(0, "/home/pyodide")
from panografia import service

def _config():
    return json.dumps({**service.config(), "engine": "pyodide"})

def _run(b1, b2, options_json):
    try:
        summary, images = service.run(bytes(b1.to_py()), bytes(b2.to_py()), json.loads(options_json))
    except (service.InputError, service.StitchError) as exc:
        return json.dumps({"error": str(exc)}), {}
    return json.dumps(summary), images
`);
  return py;
}

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    ready = ready || boot();
    const py = await ready;
    if (type === "init") {
      self.postMessage({ id, ok: true, result: JSON.parse(py.globals.get("_config")()) });
      return;
    }
    if (type === "stitch") {
      const { img1, img2, options } = event.data;
      const out = py.globals.get("_run")(new Uint8Array(img1), new Uint8Array(img2), JSON.stringify(options));
      const summary = JSON.parse(out.get(0));
      if (summary.error) throw new Error(summary.error);
      const pyImages = out.get(1);
      const images = {};
      const transfer = [];
      for (const name of pyImages.keys()) {
        const proxy = pyImages.get(name);
        const bytes = proxy.toJs();   // Uint8Array (cópia)
        proxy.destroy();
        images[name] = bytes;
        transfer.push(bytes.buffer);
      }
      pyImages.destroy();
      out.destroy();
      self.postMessage({ id, ok: true, result: { ...summary, images } }, transfer);
    }
  } catch (err) {
    if (type === "init") ready = null; // permite tentar de novo
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
