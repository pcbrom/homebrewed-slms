"""Run configuration (box 0).

Single place where a run is described. Validation enforces the accepted decisions: two modes,
fp16 or fp32 training precision, a fixed attention kernel and positive budgets.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

MODOS = ("difusao", "autoregressivo")
PRECISOES = ("fp16", "fp32")
ATENCAO = ("efficient", "math", "auto")
PAPEIS = ("calibracao", "hp", "piloto", "descritivo", "confirmatoria", "tost", "aula", "teste", "tokenizador", "preferencia", "curva", "sensibilidade")


@dataclass
class Config:
    modo: str
    vocab_size: int
    L: int = 256
    n_layer: int = 6
    d: int = 384
    n_head: int = 6
    dropout: float = 0.0
    lote: int = 64
    epocas: float = 16.0
    lr: float = 1e-3
    lr_min_frac: float = 0.1
    aquecimento_frac: float = 0.02
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    clip: float = 1.0
    precisao: str = "fp16"
    atencao: str = "efficient"
    semente: int = 0
    papel: str = "teste"
    corpus: str = "A"
    corpus_hash: str = ""
    tokenizador_hash: str = ""
    registro_cada: int = 50
    sentinela_cada: int = 500
    run_id: str = ""
    extra: dict = field(default_factory=dict)

    def validar(self) -> "Config":
        erros = []
        if self.modo not in MODOS:
            erros.append(f"modo {self.modo!r} not in {MODOS}")
        if self.precisao not in PRECISOES:
            erros.append(f"precisao {self.precisao!r} not in {PRECISOES}")
        if self.atencao not in ATENCAO:
            erros.append(f"atencao {self.atencao!r} not in {ATENCAO}")
        if self.papel not in PAPEIS:
            erros.append(f"papel {self.papel!r} not in {PAPEIS}")
        if self.d % self.n_head:
            erros.append("d must be divisible by n_head")
        if min(self.L, self.lote, self.vocab_size, self.n_layer) <= 0 or self.epocas <= 0:
            erros.append("sizes and budget must be positive")
        if erros:
            raise ValueError("; ".join(erros))
        return self

    def dicionario(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    def json(self) -> str:
        return json.dumps(self.dicionario(), sort_keys=True, ensure_ascii=False)

    def hash(self) -> str:
        return hashlib.sha256(self.json().encode("utf-8")).hexdigest()

    @classmethod
    def de_dicionario(cls, d: dict) -> "Config":
        d = dict(d)
        d["betas"] = tuple(d.get("betas", (0.9, 0.95)))
        return cls(**d).validar()
