#!/usr/bin/env python3
"""Build corpus B (federal ordinary laws 2019 to 2022) in the format of CONTRATO_DADO section 1.

With --tipos the same pipeline also takes complementary laws (LCP) and decrees (DEC), in one
grouping, so that a decree that regulates a law of the set falls in the group of that law. The
default, ordinary laws only, reproduces corpus B byte for byte.

Standard library only. The law lists come from the open data service of the
Senado and the texts from the compiled pages of the Planalto. Raw files are kept
in dados_brutos/B/ (gitignored) and reused on later runs; the clean corpus goes
to corpus/B/.

Usage:
    python3 scripts/construir_corpus_b.py            # download missing files and build
    python3 scripts/construir_corpus_b.py --forcar   # download every file again
    python3 scripts/construir_corpus_b.py --anos 2025 2026 --particao externo --saida corpus/B_externo \
        --tipos LEI LCP DEC                          # acts no model has seen, all in one partition
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from slm import dados as D  # noqa: E402

ANOS = (2019, 2020, 2021, 2022)
URL_LISTA = "https://legis.senado.leg.br/dadosabertos/legislacao/lista?tipo={tipo}&ano={ano}"
URL_LEI = "https://www.planalto.gov.br/ccivil_03/_ato{inicio}-{fim}/{ano}/lei/l{n}.htm"
TIPO_ORDINARIA = "LEI-n"
# Senado type, identifier prefix, Planalto URL and raw file of each act type
TIPOS = {
    "LEI": {"senado": TIPO_ORDINARIA, "prefixo": "lei", "url": URL_LEI, "arquivo": "planalto/l{n}.htm", "lista": "lista_{ano}.xml"},
    "LCP": {"senado": "LCP", "prefixo": "lcp", "url": "https://www.planalto.gov.br/ccivil_03/leis/lcp/lcp{n}.htm",
            "arquivo": "planalto/lcp{n}.htm", "lista": "lista_LCP_{ano}.xml"},
    "DEC": {"senado": "DEC-n", "prefixo": "dec", "url": "https://www.planalto.gov.br/ccivil_03/_ato{inicio}-{fim}/{ano}/decreto/D{n}.htm",
            "arquivo": "planalto/D{n}.htm", "lista": "lista_DEC_{ano}.xml"},
}
NOMES = {"LEI": "leis ordinárias", "LCP": "leis complementares", "DEC": "decretos"}
UA = "Mozilla/5.0 (compatible; corpus-builder/1.0; stdlib urllib)"
TIMEOUT = 60
PAUSA = 1.0
RETENTATIVAS = 3
ESTATUTO = "ato_oficial_art8_IV"
LIMIAR_MINHASH = 0.8


class Falha(Exception):
    pass


def baixar(url: str) -> tuple[bytes, str]:
    """GET with 60 s timeout, up to 3 retries on HTTP 503 or network error. Returns body and final URL."""
    for tentativa in range(RETENTATIVAS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read(), r.geturl()
        except urllib.error.HTTPError as e:
            if e.code != 503 or tentativa == RETENTATIVAS:
                raise Falha(f"HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if tentativa == RETENTATIVAS:
                raise Falha(f"rede: {e}") from e
        time.sleep(5 * 2 ** tentativa)
    raise Falha("inalcancavel")


class Coletor:
    """Download cache with a registry of final URL and collection date, and a pause between requests."""

    def __init__(self, pasta: Path, forcar: bool) -> None:
        self.pasta = pasta
        self.forcar = forcar
        pasta.mkdir(parents=True, exist_ok=True)
        self.reg_path = pasta / "coleta.json"
        self.registro = json.loads(self.reg_path.read_text(encoding="utf-8")) if self.reg_path.exists() else {}
        self.ultima = 0.0

    def obter(self, chave: str, url: str, arquivo: str) -> tuple[bytes, dict]:
        arq = self.pasta / arquivo
        if arq.exists() and chave in self.registro and not self.forcar:
            return arq.read_bytes(), self.registro[chave]
        espera = PAUSA - (time.monotonic() - self.ultima)
        if espera > 0:
            time.sleep(espera)
        try:
            corpo, final = baixar(url)
        finally:
            self.ultima = time.monotonic()
        arq.parent.mkdir(parents=True, exist_ok=True)
        arq.write_bytes(corpo)
        self.registro[chave] = {"url": final, "data_coleta": dt.date.today().isoformat()}
        self.reg_path.write_text(json.dumps(self.registro, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return corpo, self.registro[chave]


def url_lei(ano: int, n: int, tipo: str = "LEI") -> str:
    """The Planalto groups laws and decrees by presidential term, four years from 2019 (_ato2019-2022,
    _ato2023-2026); complementary laws sit in one folder."""
    inicio = 2019 + 4 * ((ano - 2019) // 4)
    return TIPOS[tipo]["url"].format(inicio=inicio, fim=inicio + 3, ano=ano, n=n)


def chave_ato(l: dict) -> tuple[int, int]:
    """Order of the acts: by type, laws first, then by number; with laws only, by number."""
    return list(TIPOS).index(l["tipo"]), l["numero"]


def ler_listas(col: Coletor, anos: tuple[int, ...], tipos: tuple[str, ...] = ("LEI",)) -> tuple[list[dict], dict]:
    leis, estado = [], {}
    for tipo, ano in ((t, a) for t in tipos for a in anos):
        chave = f"lista-{ano}" if tipo == "LEI" else f"lista-{tipo}-{ano}"
        rot = str(ano) if tipo == "LEI" else f"{tipo}-{ano}"
        try:
            corpo, _ = col.obter(chave, URL_LISTA.format(tipo=tipo, ano=ano), TIPOS[tipo]["lista"].format(ano=ano))
        except Falha as e:
            estado[rot] = f"falha: {e}"
            continue
        raiz = ET.fromstring(corpo)
        docs = raiz.findall(".//documento")
        contagem = Counter(d.findtext("tipo") for d in docs)
        estado[rot] = {"documentos": len(docs), "tipos": dict(contagem)}
        for d in docs:
            if d.findtext("tipo") != TIPOS[tipo]["senado"]:
                continue
            leis.append({
                "tipo": tipo, "numero": int(d.findtext("numero")), "ano": int(d.findtext("anoassinatura")),
                "norma": d.findtext("norma"), "ementa": " ".join((d.findtext("ementa") or "").split()),
            })
    vistos, unicas = set(), []
    for l in sorted(leis, key=chave_ato):
        if (l["tipo"], l["numero"]) not in vistos:
            vistos.add((l["tipo"], l["numero"]))
            unicas.append(l)
    return unicas, estado


def escrever_aviso(saida: Path, datas: list[str], anos: tuple[int, ...], tipos: tuple[str, ...] = ("LEI",)) -> None:
    periodo = datas[0] if datas[0] == datas[-1] else f"{datas[0]} a {datas[-1]}"
    if tipos == ("LEI",):
        descricao, verbo = "leis ordinárias federais", "sancionadas"
        atos, ato, alterado = "das leis", "cada lei", "uma lei é alterada"
    else:
        nomes = [NOMES[t] for t in tipos]
        descricao, verbo = f"atos normativos federais ({', '.join(nomes[:-1])} e {nomes[-1]})", "assinados"
        atos, ato, alterado = "dos atos", "cada ato", "um ato é alterado"
    texto = f"""# Corpus B: estatuto e atribuição

Este diretório contém o texto de {descricao} do Brasil {verbo} de {anos[0]} a {anos[-1]}, das classes substantiva e alteradora. Leis, decretos e demais atos oficiais não são objeto de proteção pelo direito autoral, conforme a Lei 9.610/1998, art. 8º, IV.

A lista {atos} e as ementas vêm do serviço de dados abertos do Senado Federal (https://legis.senado.leg.br/dadosabertos). O texto vem das páginas de legislação compilada do Portal da Legislação do Planalto (https://www.planalto.gov.br/ccivil_03), mantido pela Presidência da República. A limpeza removeu o texto revogado, marcado como riscado na página, as anotações editoriais de redação e vigência, o cabeçalho e o rodapé da página. O texto aqui versionado não substitui o publicado no Diário Oficial da União.

O Planalto reescreve as páginas compiladas quando {alterado}, de modo que o texto reflete a data de coleta: {periodo}. A URL, a data de coleta, o sha256 da página baixada e o sha256 do texto limpo de {ato} estão em `manifesto.json`.

A licença CC BY-NC-SA 4.0 deste repositório não recai sobre estes textos. Ela cobre só o código, a documentação e os metadados produzidos no projeto.
"""
    (saida / "AVISO.md").write_text(texto, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brutos", type=Path, default=RAIZ / "dados_brutos" / "B")
    ap.add_argument("--saida", type=Path, default=RAIZ / "corpus" / "B")
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--anos", type=int, nargs="+", default=list(ANOS))
    ap.add_argument("--particao", help="put every group in this partition instead of the hash split")
    ap.add_argument("--tipos", nargs="+", default=["LEI"], choices=list(TIPOS))
    args = ap.parse_args()
    anos = tuple(sorted(args.anos))
    tipos = tuple(t for t in TIPOS if t in args.tipos)

    col = Coletor(args.brutos, args.forcar)
    leis, estado_listas = ler_listas(col, anos, tipos)
    for l in leis:
        l["classe"] = D.classificar_ementa(l["ementa"])
    classes_lista = Counter(l["classe"] for l in leis)
    alvo = [l for l in leis if l["classe"] in D.CLASSES_B_MANTIDAS]

    manifesto, validas = [], []
    for l in alvo:
        fid = f"{TIPOS[l['tipo']]['prefixo']}-{l['numero']}"
        url = url_lei(l["ano"], l["numero"], l["tipo"])
        item = {
            "fonte_id": fid, "url": url, "data_coleta": None, "sha256_bruto": None, "sha256_limpo": None,
            "estatuto": ESTATUTO, "classe": l["classe"], "excluida": None,
        }
        try:
            corpo, reg = col.obter(fid, url, TIPOS[l["tipo"]]["arquivo"].format(n=l["numero"]))
        except Falha as e:
            item["excluida"] = f"falha no download: {e}"
            manifesto.append(item)
            continue
        item.update(url=reg["url"], data_coleta=reg["data_coleta"], sha256_bruto=D.sha256_bytes(corpo))
        texto, citado = D.limpar_planalto(corpo, D.TITULOS[l["tipo"]])
        if not D.titulo_confere(texto, l["numero"], D.TITULOS[l["tipo"]]):
            item["excluida"] = "pagina sem o titulo da lei esperada"
            manifesto.append(item)
            continue
        if "\n\nArt. " not in texto:
            item["excluida"] = "sem texto vigente: lei integralmente revogada na pagina compilada"
            manifesto.append(item)
            continue
        item["sha256_limpo"] = D.sha256_texto(texto)
        manifesto.append(item)
        validas.append(dict(l, fonte_id=fid, texto=texto, bytes=D.n_bytes(texto), citado=citado))

    # Groups: first act cited in the summary, then near-copies by MinHash.
    uf = D.UniaoBusca()
    citada = {}
    for l in validas:
        uf.achar(l["fonte_id"])
        c = D.primeira_lei_citada(l["ementa"])
        citada[l["fonte_id"]] = c
        if c and c != l["fonte_id"]:
            uf.unir(l["fonte_id"], c)
    conjuntos = {l["fonte_id"]: D.shingles(l["texto"]) for l in validas}
    pares = D.pares_quase_copia(conjuntos, limiar=LIMIAR_MINHASH)
    for a, b, _ in pares:
        uf.unir(a, b)

    por_grupo: dict[str, list[dict]] = {}
    for l in validas:
        por_grupo.setdefault(uf.achar(l["fonte_id"]), []).append(l)
    documentos = []
    def particao(g):
        return args.particao or D.particao_por_hash(g)

    for g in sorted(por_grupo):
        part = particao(g)
        for i, l in enumerate(sorted(por_grupo[g], key=chave_ato), start=1):
            documentos.append({
                "doc_id": f"B-{l['fonte_id']}", "grupo": g, "ordem": i, "particao": part,
                "texto": l["texto"], "bytes": l["bytes"],
            })

    args.saida.mkdir(parents=True, exist_ok=True)
    linhas = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in documentos)
    (args.saida / "documentos.jsonl").write_text(linhas, encoding="utf-8")
    (args.saida / "manifesto.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_bytes = sum(d["bytes"] for d in documentos)
    grupos = {}
    for g, membros in por_grupo.items():
        b = sum(m["bytes"] for m in membros)
        grupos[g] = {
            "particao": particao(g), "balde": D.balde_grupo(g), "documentos": len(membros),
            "bytes": b, "leis": [m["fonte_id"] for m in sorted(membros, key=chave_ato)],
        }
    maior = max(grupos, key=lambda g: grupos[g]["bytes"]) if grupos else None
    part_stats = {}
    for p in ((args.particao,) if args.particao else ("treino", "validacao", "teste")):
        sel = [d for d in documentos if d["particao"] == p]
        part_stats[p] = {
            "documentos": len(sel), "bytes": sum(d["bytes"] for d in sel),
            "grupos": len({d["grupo"] for d in sel}),
        }
    falhas = [m for m in manifesto if m["excluida"]]
    divisao = {
        "corpus": args.saida.name,
        "regra": (
            "Cada lei entra no grupo da primeira lei citada na ementa, e leis com Jaccard de 5-gramas de palavras "
            "(minusculas, digitos trocados por 0) de pelo menos 0,8 entram no mesmo grupo, por uniao-busca. "
            "O representante do grupo e o menor identificador entre os membros e as leis citadas. A particao do "
            "grupo e sha256(grupo) modulo 100: 0 a 89 treino, 90 a 94 validacao, 95 a 99 teste. Sem margem."
            if not args.particao else
            "Cada lei entra no grupo da primeira lei citada na ementa, e leis com Jaccard de 5-gramas de palavras "
            f"de pelo menos 0,8 entram no mesmo grupo. Todos os grupos vao para a particao {args.particao}."
        ),
        "parametros": {
            "anos": list(anos), "tipo_senado": TIPO_ORDINARIA if tipos == ("LEI",) else [TIPOS[t]["senado"] for t in tipos], "classes_mantidas": list(D.CLASSES_B_MANTIDAS),
            "minhash_permutacoes": 128, "minhash_bandas": 32, "shingle_palavras": 5, "limiar_jaccard": LIMIAR_MINHASH,
        },
        "listas_senado": estado_listas,
        "classes_na_lista": {c: classes_lista.get(c, 0) for c in D.CLASSES_B},
        "leis_na_lista": len(leis),
        "paginas": {
            "tentadas": len(alvo),
            "baixadas": sum(1 for m in manifesto if m["sha256_bruto"]),
            "fracao_baixadas": sum(1 for m in manifesto if m["sha256_bruto"]) / len(alvo) if alvo else 0.0,
            "validas": len(validas), "excluidas": len(falhas),
            "fracao_validas": len(validas) / len(alvo) if alvo else 0.0,
            "motivos_exclusao": dict(Counter(m["excluida"] for m in falhas)),
        },
        "classes_no_corpus": dict(Counter(l["classe"] for l in validas)),
        "bytes_total": total_bytes,
        "bytes_citados_blockquote": sum(l["citado"] for l in validas),
        "fracao_citada": sum(l["citado"] for l in validas) / total_bytes if total_bytes else 0.0,
        "pares_quase_copia": [{"a": a, "b": b, "jaccard": round(j, 4)} for a, b, j in pares],
        "maior_grupo": {
            "grupo": maior, "documentos": grupos[maior]["documentos"], "bytes": grupos[maior]["bytes"],
            "fracao_bytes": grupos[maior]["bytes"] / total_bytes,
        } if maior else None,
        "particoes": part_stats,
        "grupos": dict(sorted(grupos.items())),
        "sha256_documentos": D.sha256_bytes(linhas.encode("utf-8")),
    }
    (args.saida / "divisao.json").write_text(json.dumps(divisao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    datas = sorted({m["data_coleta"] for m in manifesto if m["data_coleta"]})
    escrever_aviso(args.saida, datas, anos, tipos)

    print(f"leis na lista: {len(leis)}; classes: {dict(classes_lista)}")
    print(f"paginas validas: {len(validas)}/{len(alvo)}; excluidas: {len(falhas)}")
    print(f"bytes: {total_bytes}; citados: {divisao['fracao_citada']:.4f}; pares quase-copia: {len(pares)}")
    if maior:
        print(f"maior grupo: {maior} com {divisao['maior_grupo']['fracao_bytes']:.4f} dos bytes")
    print(f"particoes: {part_stats}")


if __name__ == "__main__":
    main()
