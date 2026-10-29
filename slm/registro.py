"""Pre-registration with hash and result records (CONTRATO_DADO sections 3 to 5).

A pre-registration is written once: the JSON and its sha256 are created together and never
overwritten, so any later change produces a new file instead of editing the old one.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def preregistrar(pasta: Path, nome: str, dados: dict) -> str:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arq, sha = pasta / f"{nome}.json", pasta / f"{nome}.sha256"
    if arq.exists() or sha.exists():
        raise FileExistsError(f"pre-registration {nome} already exists, create a new one")
    arq.write_text(json.dumps(dados, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    h = sha256_arquivo(arq)
    sha.write_text(f"{h}  {arq.name}\n", encoding="utf-8")
    return h


def conferir_preregistro(pasta: Path, nome: str) -> bool:
    pasta = Path(pasta)
    esperado = (pasta / f"{nome}.sha256").read_text(encoding="utf-8").split()[0]
    return sha256_arquivo(pasta / f"{nome}.json") == esperado


def conferir_tarefas(pasta: Path, tarefas: Path, corpus: str) -> None:
    """The task pre-registration and its amendments are intact, and the item files are the ones
    the latest of them froze."""
    nome = f"m14_tarefas_{corpus}"
    assert conferir_preregistro(pasta, nome), "pre-registration missing or altered"
    d = json.loads((Path(pasta) / f"{nome}.json").read_text(encoding="utf-8"))
    # amendments in order: _emenda, _emenda2, _emenda3, ...; the last one fixes the items
    k = 1
    while (Path(pasta) / f"{nome}_emenda{'' if k == 1 else k}.json").exists():
        e = f"{nome}_emenda{'' if k == 1 else k}"
        assert conferir_preregistro(pasta, e), f"amendment {e} altered"
        d = json.loads((Path(pasta) / f"{e}.json").read_text(encoding="utf-8"))
        k += 1
    for t, h in d["itens_sha256"].items():
        assert sha256_arquivo(Path(tarefas) / f"{t}.jsonl") == h, f"items of {t} differ from the frozen ones"


def maquina() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,power.limit", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            driver, pot = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
            info.update(driver=driver, teto_potencia=pot)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


class RegistroCorrida:
    """Writes the files of one run under runs/<run_id>/."""

    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)

    def config(self, cfg: dict) -> None:
        dados = dict(cfg, maquina=maquina())
        (self.pasta / "config.json").write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _linha(self, nome: str, linha: dict) -> None:
        with open(self.pasta / nome, "a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    def passo(self, linha: dict) -> None:
        self._linha("passos.jsonl", linha)

    def sentinela(self, linhas: list[dict]) -> None:
        for l in linhas:
            self._linha("sentinela.jsonl", l)

    def amostras(self, linhas: list[dict]) -> None:
        for l in linhas:
            self._linha("amostras.jsonl", l)

    def avaliacao(self, dados: dict, nome: str = "avaliacao.json") -> None:
        (self.pasta / nome).write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
