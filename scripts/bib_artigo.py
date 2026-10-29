#!/usr/bin/env python3
"""BibTeX of the article from the verified sources (referencias/fontes.json and trava.json).

Only keys passed on the command line are written, so the bibliography holds nothing that the
article does not cite and nothing that was not downloaded and checked by scripts/referencias.py.
arXiv metadata lives in trava.json, the metadata of other sources in fontes.json.

    python scripts/bib_artigo.py ou2024radd sahoo2024mdlm ... > references.bib
"""
import ast
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
fontes = {f["chave"]: f for f in json.loads((RAIZ / "referencias" / "fontes.json").read_text(encoding="utf-8"))}
trava = json.loads((RAIZ / "referencias" / "trava.json").read_text(encoding="utf-8"))


PARTICULAS = {"Di", "De", "Da", "Do", "Dos", "Das", "Van", "Von", "Der", "Del"}


def lista(v):
    return ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else (v or [])


def autor(nome):
    # a capitalised particle before the last word belongs to the surname, which BibTeX would otherwise split
    p = nome.split()
    return " ".join(p[:-2] + ["{" + " ".join(p[-2:]) + "}"]) if len(p) > 2 and p[-2] in PARTICULAS else nome


saida = []
for chave in sys.argv[1:]:
    if chave not in trava:
        sys.exit(f"{chave}: not a verified source")
    f = fontes.get(chave, {})
    t = {**f, **trava[chave]}
    campos = {"title": "{" + t["titulo"] + "}", "author": " and ".join(autor(a) for a in lista(t.get("autores"))),
              "year": str(t["ano"])}
    if f.get("veiculo"):
        tipo = "misc"
        campos.update(howpublished=f["veiculo"].replace("&", r"\&"), url=f.get("url_pdf", ""))
    else:
        tipo = "article"
        campos.update(journal=f"arXiv preprint arXiv:{f['arxiv']}", eprint=f["arxiv"], archivePrefix="arXiv",
                      url=f"https://arxiv.org/abs/{f['arxiv']}")
    corpo = ",\n".join(f"  {k} = {{{v}}}" for k, v in campos.items() if v)
    saida.append(f"@{tipo}{{{chave},\n{corpo}\n}}")
print("\n\n".join(saida))
