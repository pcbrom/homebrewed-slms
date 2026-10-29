#!/usr/bin/env python3
"""Download, pin and document the open access sources listed in referencias/fontes.json.

Standard library only. For arXiv entries, metadata, pinned version and license come
from the arXiv OAI-PMH endpoint (arXivRaw format). A PDF is stored inside the
repository (referencias/pdf/) only when its license allows third-party
redistribution; otherwise it goes to referencias/pdf_local/, which is gitignored,
and the repository keeps the URL and the sha256 so anyone can fetch and verify it.

Usage:
    python3 scripts/referencias.py            # download missing files and rebuild the docs
    python3 scripts/referencias.py --verificar  # only check sha256 of files already present
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "referencias"
FONTES = ROOT / "fontes.json"
TRAVA = ROOT / "trava.json"
OAI = "https://oaipmh.arxiv.org/oai?verb=GetRecord&identifier=oai:arXiv.org:{}&metadataPrefix=arXivRaw"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "r": "http://arxiv.org/OAI/arXivRaw/"}
UA = "homebrewed-slms-referencias/1.0 (research; stdlib urllib)"
PAUSA_ARXIV = 3.0  # arXiv asks for at most one request every 3 seconds
REDISTRIBUIVEL = ("creativecommons.org/licenses/", "creativecommons.org/publicdomain/")


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def meta_arxiv(aid):
    root = ET.fromstring(get(OAI.format(aid)))
    m = root.find(".//r:arXivRaw", NS)
    if m is None:
        raise RuntimeError(f"arXiv {aid}: registro OAI ausente")
    versoes = m.findall("r:version", NS)
    return {
        "titulo": " ".join(m.findtext("r:title", "", NS).split()),
        "autores": [a.strip() for a in re.split(r",\s*|\s+and\s+", " ".join(m.findtext("r:authors", "", NS).split())) if a.strip()],
        # year of the first deposit: stable across later revisions, the pinned version goes in the reference
        "ano": int(re.search(r"\d{4}", versoes[0].findtext("r:date", "", NS)).group()),
        "versao": versoes[-1].get("version"),
        "licenca": m.findtext("r:license", "", NS) or "http://arxiv.org/licenses/assumed-1991-2003/",
    }


def abnt_autor(nome):
    partes = nome.split()
    if len(partes) == 1:
        return partes[0].upper()
    iniciais = " ".join(p[0] + "." for p in partes[:-1])
    return f"{partes[-1].upper()}, {iniciais}"


def abnt(f):
    a = f["autores"]
    autores = abnt_autor(a[0]) + " et al." if len(a) > 3 else "; ".join(abnt_autor(x) for x in a)
    autores += "" if autores.endswith(".") else "."
    ano = f.get("ano_rotulo", f["ano"])
    if f.get("arxiv") and not f.get("url_pdf"):
        onde = f"*arXiv*, {f['arxiv']}{f['versao']}, {ano}"
    else:
        onde = f"{f.get('veiculo', '')}, {ano}".strip(", ")
    return f"{autores} {f['titulo']}. {onde}. Disponível em: {f['url']}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", action="store_true")
    a = ap.parse_args()

    fontes = json.loads(FONTES.read_text(encoding="utf-8"))
    trava = json.loads(TRAVA.read_text(encoding="utf-8")) if TRAVA.exists() else {}
    for d in ("pdf", "pdf_local"):
        (ROOT / d).mkdir(exist_ok=True)

    falhas = []
    for f in fontes:
        k = f["chave"]
        t = trava.get(k, {})
        if f.get("arxiv") and not a.verificar and "versao" not in t:
            try:
                t.update(meta_arxiv(f["arxiv"]))
            except Exception as e:
                falhas.append(f"{k}: metadados arXiv indisponiveis ({e})")
                continue
            finally:
                time.sleep(PAUSA_ARXIV)
        f.update({kk: t[kk] for kk in ("titulo", "autores", "ano", "versao", "licenca") if kk in t and kk not in f})
        f["url"] = f.get("url_pdf") or f"https://arxiv.org/pdf/{f['arxiv']}{f['versao']}"
        lic = f.get("licenca", "")
        f["redistribuivel"] = any(s in lic for s in REDISTRIBUIVEL)
        destino = ROOT / ("pdf" if f["redistribuivel"] else "pdf_local") / f"{k}.pdf"
        # software is pinned by its commit in versao, there is no PDF to fetch
        if not destino.exists() and not a.verificar and not f.get("software"):
            try:
                dados = get(f["url"], timeout=120)
            except Exception as e:
                falhas.append(f"{k}: download falhou ({e})")
                continue
            if not dados.startswith(b"%PDF"):
                falhas.append(f"{k}: resposta nao e PDF")
                continue
            destino.write_bytes(dados)
            t["baixado_em"] = dt.date.today().isoformat()
            if f.get("arxiv"):
                time.sleep(PAUSA_ARXIV)
        if destino.exists():
            h = hashlib.sha256(destino.read_bytes()).hexdigest()
            if "sha256" in t and t["sha256"] != h:
                falhas.append(f"{k}: sha256 diverge da trava")
            t.setdefault("sha256", h)
        trava[k] = {kk: v for kk, v in {**t, "licenca": lic}.items() if kk not in ("redistribuivel",)}
        f.update(sha256=t.get("sha256", ""), baixado_em=t.get("baixado_em", ""))

    if not a.verificar:
        TRAVA.write_text(json.dumps(trava, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        linhas = [
            "# Referências",
            "",
            "Arquivo gerado por `scripts/referencias.py` a partir de `fontes.json`. Não editar à mão.",
            "",
            "Todas as fontes são de acesso aberto. O PDF fica neste repositório, em `pdf/`, apenas quando a licença da obra permite redistribuição por terceiros. Nos demais casos, o script baixa o arquivo da fonte original para `pdf_local/`, fora do versionamento, e confere o sha256 registrado abaixo.",
            "",
            "| Chave | Versão | Licença | No repositório | sha256 (12 primeiros) | Baixado em | Citado em |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in fontes:
            lic = f.get("licenca", "")
            rotulo = f"[{'/'.join(lic.rstrip('/').split('/')[-2:])}]({lic})" if lic else "não declarada"
            linhas.append(
                f"| `{f['chave']}` | {f.get('versao', '')} | {rotulo} | "
                f"{'sim' if f['redistribuivel'] else 'não'} | `{f['sha256'][:12]}` | {f['baixado_em']} | {', '.join(f.get('citado_em', []))} |"
            )
        lista = "\n\n".join(abnt(f) for f in sorted(fontes, key=lambda x: (x["autores"][0].split()[-1].upper(), str(x.get("ano_rotulo", x["ano"])))))
        (ROOT / "REFERENCIAS.md").write_text("\n".join(linhas) + "\n\n## Lista em ABNT\n\n" + lista + "\n", encoding="utf-8")

    print(f"{len(fontes)} fontes, {sum(f['redistribuivel'] for f in fontes)} redistribuíveis")
    for m in falhas:
        print("FALHA", m)
    raise SystemExit(1 if falhas else 0)


if __name__ == "__main__":
    main()
