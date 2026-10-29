#!/usr/bin/env python3
"""Execute the class notebooks in a real Jupyter kernel, for a local end-to-end check.

    SLM_EPOCAS=0.05 .venv/bin/python scripts/executar_notebook.py            # check only
    SLM_EPOCAS=0.05 .venv/bin/python scripts/executar_notebook.py --salvar   # also write the outputs
    SLM_EPOCAS=0.05 .venv/bin/python scripts/executar_notebook.py --modo difusao
    .venv/bin/python scripts/executar_notebook.py --modo tokenizador --salvar
    .venv/bin/python scripts/executar_notebook.py --modo autoregressivo_en difusao_en tokenizador_en --salvar

SLM_LOCAL is set to the repository root, so nothing is cloned or installed. Prints the time
of each code cell and stops at the first error. With --salvar each executed notebook, outputs
included, replaces its file in notebooks/.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVOS = {"autoregressivo": "slm_autorregressivo.ipynb", "difusao": "slm_difusao.ipynb",
            "tokenizador": "slm_tokenizador.ipynb", "autoregressivo_en": "en/slm_autoregressive.ipynb",
            "difusao_en": "en/slm_diffusion.ipynb", "tokenizador_en": "en/slm_tokenizer.ipynb"}


def duracao(c) -> float:
    t = c.metadata.get("execution", {})
    ini, fim = t.get("iopub.execute_input"), t.get("shell.execute_reply")
    return (datetime.fromisoformat(fim.rstrip("Z")) - datetime.fromisoformat(ini.rstrip("Z"))).total_seconds() if ini and fim else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salvar", action="store_true", help="write the executed notebook in place")
    ap.add_argument("--timeout-celula", type=int, default=3600)
    ap.add_argument("--modo", choices=list(ARQUIVOS), nargs="+", default=list(ARQUIVOS))
    a = ap.parse_args()
    os.environ.setdefault("SLM_LOCAL", str(RAIZ))
    for modo in a.modo:
        executar(RAIZ / "notebooks" / ARQUIVOS[modo], a.timeout_celula, a.salvar)


def executar(caminho: Path, timeout: int, salvar: bool):
    print(f"=== {caminho.name}")
    nb = nbformat.read(caminho, as_version=4)
    cliente = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                             resources={"metadata": {"path": str(RAIZ)}})
    try:
        cliente.execute()
    except CellExecutionError as e:
        sys.exit(f"falha na execução:\n{str(e)[-3000:]}")
    total = 0.0
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            total += (dt := duracao(c))
            print(f"--- célula {i}: {dt:.1f} s")
    print(f"--- total: {total:.1f} s")
    if salvar:
        nbformat.write(nb, caminho)
        print("saídas gravadas em", caminho.relative_to(RAIZ))


if __name__ == "__main__":
    main()
