#!/usr/bin/env python3
"""Build corpus A (Machado de Assis, Project Gutenberg) in the format of CONTRATO_DADO section 1.

Standard library only. Raw files are downloaded once to dados_brutos/A/ (gitignored)
and reused on later runs; the clean corpus goes to corpus/A/.

Usage:
    python3 scripts/construir_corpus_a.py            # download missing files and build
    python3 scripts/construir_corpus_a.py --forcar   # download every file again
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from slm import dados as D  # noqa: E402

URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
UA = "Mozilla/5.0 (compatible; corpus-builder/1.0; stdlib urllib)"
TIMEOUT = 60
PAUSA = 1.0
RETENTATIVAS = 3
ESTATUTO = "dominio_publico"


def baixar(url: str) -> tuple[bytes, str]:
    """GET with 60 s timeout, up to 3 retries on HTTP 503 or network error. Returns body and final URL."""
    for tentativa in range(RETENTATIVAS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code != 503 or tentativa == RETENTATIVAS:
                raise
        except (urllib.error.URLError, TimeoutError):
            if tentativa == RETENTATIVAS:
                raise
        time.sleep(5 * 2 ** tentativa)
    raise RuntimeError("inalcancavel")


def obter_brutos(brutos: Path, forcar: bool) -> dict[str, dict]:
    brutos.mkdir(parents=True, exist_ok=True)
    reg_path = brutos / "coleta.json"
    registro = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    ebooks = [o.ebook for o in D.OBRAS_A] + list(D.EXCLUIDAS_A)
    for n in ebooks:
        arq = brutos / f"pg{n}.txt"
        if arq.exists() and n in registro and not forcar:
            continue
        corpo, final = baixar(URL.format(n=n))
        arq.write_bytes(corpo)
        registro[n] = {"url": final, "data_coleta": dt.date.today().isoformat()}
        reg_path.write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        time.sleep(PAUSA)
    return registro


def escrever_aviso(saida: Path, data: str) -> None:
    texto = f"""# Corpus A: estatuto e atribuição

Este diretório contém obras em prosa de Machado de Assis (1839 a 1908), em português, na grafia das edições de origem. As obras estão em domínio público no Brasil desde 1979 (Lei 9.610/1998, art. 41) e também nos Estados Unidos. A autoria é atribuída a Machado de Assis em cada obra, em respeito à integridade e à autoria da obra caída em domínio público, que a mesma lei protege no art. 24, §2º.

Os arquivos de origem foram obtidos no Project Gutenberg (https://www.gutenberg.org), a quem se dá crédito pela digitação e pela revisão das edições. O texto aqui versionado é uma versão limpa, da qual foram removidas todas as referências ao projeto, conforme o §1.C da licença dele. Este corpus não é distribuído pelo Project Gutenberg nem tem o endosso dele, e o nome aparece apenas como crédito de origem.

A licença CC BY-NC-SA 4.0 deste repositório não recai sobre estes textos, que seguem em domínio público. Ela cobre só o código, a documentação e os metadados produzidos no projeto.

Data de coleta: {data}. A URL, o sha256 do arquivo baixado e o sha256 do texto limpo de cada obra estão em `manifesto.json`, e a regra de divisão está em `divisao.json`.
"""
    (saida / "AVISO.md").write_text(texto, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brutos", type=Path, default=RAIZ / "dados_brutos" / "A")
    ap.add_argument("--saida", type=Path, default=RAIZ / "corpus" / "A")
    ap.add_argument("--forcar", action="store_true")
    args = ap.parse_args()

    registro = obter_brutos(args.brutos, args.forcar)
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "nomes").mkdir(exist_ok=True)

    manifesto, documentos = [], []
    por_obra: dict[str, dict] = {}
    for n, motivo in D.EXCLUIDAS_A.items():
        bruto = (args.brutos / f"pg{n}.txt").read_bytes()
        manifesto.append({
            "fonte_id": n, "url": registro[n]["url"], "data_coleta": registro[n]["data_coleta"],
            "sha256_bruto": D.sha256_bytes(bruto), "sha256_limpo": None, "estatuto": ESTATUTO, "excluida": motivo,
        })
    for obra in D.OBRAS_A:
        bruto = (args.brutos / f"pg{obra.ebook}.txt").read_bytes()
        unidades = D.limpar_obra(bruto.decode("utf-8"), obra)
        problemas = D.verificar_numeracao([c for c, _ in unidades], obra.regra)
        if problemas:
            raise SystemExit(f"{obra.slug}: numeracao inconsistente: {problemas[:3]}")
        textos = [t for _, t in unidades]
        tamanhos = [D.n_bytes(t) for t in textos]
        rotulos = D.dividir_blocos(tamanhos)
        for i, (t, b, r) in enumerate(zip(textos, tamanhos, rotulos), start=1):
            documentos.append({
                "doc_id": f"A-{obra.ebook}-c{i:03d}", "grupo": obra.slug, "ordem": i,
                "particao": r, "texto": t, "bytes": b,
            })
        nomes = D.extrair_nomes([t for t, r in zip(textos, rotulos) if r == "treino"])
        (args.saida / "nomes" / f"{obra.slug}.txt").write_text("".join(f"{x}\n" for x in nomes), encoding="utf-8")
        manifesto.append({
            "fonte_id": obra.ebook, "url": registro[obra.ebook]["url"],
            "data_coleta": registro[obra.ebook]["data_coleta"], "sha256_bruto": D.sha256_bytes(bruto),
            "sha256_limpo": D.sha256_texto("\n\n".join(textos)), "estatuto": ESTATUTO, "excluida": None,
        })
        por_particao = {}
        for p in ("treino", "margem", "validacao", "teste"):
            sel = [b for b, r in zip(tamanhos, rotulos) if r == p]
            por_particao[p] = {"documentos": len(sel), "bytes": sum(sel)}
        por_obra[obra.slug] = {
            "ebook": obra.ebook, "titulo": obra.titulo, "regra_corte": obra.regra,
            "documentos": len(textos), "bytes": sum(tamanhos), "particoes": por_particao,
            "no_teste": por_particao["teste"]["documentos"] > 0, "nomes": len(nomes),
        }

    linhas = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documentos)
    (args.saida / "documentos.jsonl").write_text(linhas, encoding="utf-8")
    (args.saida / "manifesto.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tot = Counter()
    totb = Counter()
    for d in documentos:
        tot[d["particao"]] += 1
        totb[d["particao"]] += d["bytes"]
    divisao = {
        "corpus": "A",
        "regra": (
            "Por obra, blocos contiguos de capitulos inteiros, na ordem de leitura: treino, margem, validacao, "
            "margem, teste, com o teste no fim da obra. Cada bloco de validacao e de teste e a menor sequencia "
            "de capitulos com pelo menos min_capitulos capitulos e min_bytes bytes, e o treino precisa atingir "
            "o mesmo minimo. Obra sem espaco para os dois blocos sai do teste e fica so com validacao."
        ),
        "parametros": {"min_capitulos": D.MIN_CAPITULOS, "min_bytes": D.MIN_BYTES, "margem_capitulos": D.MARGEM},
        "unidade_bytes": "bytes UTF-8 do texto NFC",
        "sha256_limpo": "sha256 dos textos da obra unidos por duas quebras de linha",
        "particoes": {p: {"documentos": tot[p], "bytes": totb[p]} for p in ("treino", "margem", "validacao", "teste")},
        "grupos": por_obra,
        "obras_fora_do_teste": sorted(s for s, v in por_obra.items() if not v["no_teste"]),
        "sha256_documentos": D.sha256_bytes(linhas.encode("utf-8")),
    }
    (args.saida / "divisao.json").write_text(json.dumps(divisao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    escrever_aviso(args.saida, max(v["data_coleta"] for v in registro.values()))

    print(f"{'obra':22s} {'docs':>5s} {'treino':>14s} {'validacao':>14s} {'teste':>14s}")
    for s, v in por_obra.items():
        cel = [f"{v['particoes'][p]['documentos']:4d} {v['particoes'][p]['bytes']:9d}" for p in ("treino", "validacao", "teste")]
        print(f"{s:22s} {v['documentos']:5d} " + " ".join(cel))
    print(f"total: {len(documentos)} documentos, sha256 {divisao['sha256_documentos']}")


if __name__ == "__main__":
    main()
