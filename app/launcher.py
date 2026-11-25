"""Ponto de entrada do executável (PyInstaller).

* Sem argumentos: inicia o servidor local e abre a interface no navegador.
* Com argumentos: funciona como a linha de comando, por exemplo
  ``panografia img1.png img2.png -o saida/``.
"""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser


def _free_port(preferred: int = 8000) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("nenhuma porta livre")


def main() -> int:
    if len(sys.argv) > 1:
        from panografia.__main__ import main as cli
        return cli(sys.argv[1:])

    import uvicorn

    from app.server import app

    port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"Panografia rodando em {url}  (feche esta janela ou pressione Ctrl+C para sair)", flush=True)
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
