#!/usr/bin/env python3
"""Milestone M9: blind reading (docs/decisao_orcamento.md, SPEC section 10, CONTRATO_DADO section 6).

Subcommands, run in order:
    amostras    generate samples from every run of the grid spec, cut at EOS and at --palavras words,
                with lexicon rate and longest copied span; writes leitura/amostras.jsonl
    controles   draw human test excerpts of the same length, compute the copy threshold k (percentile
                99 of their longest copied span) and the control pairs; writes leitura/controles.json
    lotes       apply the copy filter, pick 12 items per mode and grid point spread evenly over the
                seeds, write lote_<n>.csv and chave_<n>.json per point and leitura/n1_copia.json
    analisar    read respostas_<avaliador>_<n>.csv, filter raters by the control pairs, compute
                Krippendorff's alpha with and without the author and levels N2 to N4 per point;
                writes leitura/resultado.json

The grid spec is a JSON list of {"run_id", "modo", "ponto", "semente"}, written by the M7 script.

    python scripts/m9.py amostras --grade preregistro/m7_grade.json --dispositivo cuda
    python scripts/m9.py controles
    python scripts/m9.py lotes
    python scripts/m9.py analisar --autor pedro
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm import leitura as lt  # noqa: E402

PALAVRA = re.compile(r"\S+")


def cortar(texto: str, n: int, rng: random.Random) -> str | None:
    """n consecutive whitespace tokens from a random start, or None when the text is shorter.

    Model samples start at BOS, i.e. at a document start, while human excerpts start inside a
    document; a random start in both keeps the opening of an item from revealing its source.
    """
    toks = PALAVRA.findall(texto)
    if len(toks) < n:
        return None
    i = rng.randrange(len(toks) - n + 1)
    return " ".join(toks[i:i + n])


def cmd_amostras(a):
    import torch
    from slm.config import Config
    from slm.corpus import ler_documentos
    from slm.geracao import ate_eos, gerar_ar, gerar_difusao
    from slm.modelo import ConfigRede, Transformer
    from slm.corpus import pasta_tokenizador
    from slm.tokenizador import carregar_tokenizador
    campos = set(Config.__dataclass_fields__)
    grade = json.loads(Path(a.grade).read_text(encoding="utf-8"))
    cache, saida = {}, []
    for g in grade:
        pasta = RAIZ / "runs" / g["run_id"]
        bruto = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        cfg = Config.de_dicionario({k: v for k, v in bruto.items() if k in campos})
        chave = (cfg.corpus, str(pasta_tokenizador(cfg)))
        if chave not in cache:
            treino = [d["texto"] for d in ler_documentos(cfg.corpus, "treino")]
            cache[chave] = (carregar_tokenizador(chave[1]), lt.lexico(treino), lt.IndiceTreino(treino))
        tok, lex, indice = cache[chave]
        modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
        modelo.load_state_dict(torch.load(pasta / "pesos.pt", map_location="cpu"))
        modelo.to(a.dispositivo).eval()
        semente_geracao = 1000 * int(g["semente"]) + int(g["ponto"])
        if g["modo"] == "autoregressivo":
            ids = gerar_ar(modelo, cfg.L, n=a.n, semente=semente_geracao, dispositivo=a.dispositivo)
        else:
            ids = gerar_difusao(modelo, cfg.L, T_gen=cfg.L, n=a.n, semente=semente_geracao, dispositivo=a.dispositivo)
        validas = 0
        for j, s in enumerate(ids.tolist()):
            texto = cortar(tok.decode(ate_eos([i for i in s if i > 3 or i == 2])).decode("utf-8", "replace"), a.palavras,
                           random.Random(semente_geracao * 1000 + j))
            if texto is None:
                continue
            validas += 1
            saida.append({**g, "amostra_id": f"{g['run_id']}#{j}", "texto": texto,
                          "c": indice.maior_trecho(texto), "taxa_lexico": lt.taxa_lexico(texto, lex)})
        print(f"{g['run_id']} | {validas} de {a.n} amostras com {a.palavras} palavras", flush=True)
    pasta = RAIZ / "leitura"
    pasta.mkdir(exist_ok=True)
    with open(pasta / "amostras.jsonl", "w", encoding="utf-8") as f:
        for r in saida:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(saida)} amostras em leitura/amostras.jsonl")


def excertos_humanos(textos: list[str], n: int, palavras: int, semente: int) -> list[str]:
    """n excerpts of `palavras` whitespace tokens, with start positions drawn uniformly over every possible start."""
    rng = random.Random(semente)
    toks = [PALAVRA.findall(t) for t in textos]
    inicios = [(i, j) for i, tk in enumerate(toks) for j in range(len(tk) - palavras + 1)]
    escolhidos = rng.sample(inicios, min(n, len(inicios)))
    return [" ".join(toks[i][j:j + palavras]) for i, j in escolhidos]


def cmd_controles(a):
    from slm.corpus import ler_documentos
    treino = [d["texto"] for d in ler_documentos(a.corpus, "treino")]
    teste = [d["texto"] for d in ler_documentos(a.corpus, "teste")]
    indice, lex = lt.IndiceTreino(treino), lt.lexico(treino)
    hum = excertos_humanos(teste, a.n_referencia + a.pares * a.lotes, a.palavras, a.semente)
    ref, ctrl = hum[:a.n_referencia], hum[a.n_referencia:]
    c = indice.maiores_trechos(ref)
    k = lt.limiar_copia(c)
    pares = [(h, lt.embaralhar(h, a.semente + j)) for j, h in enumerate(ctrl)]
    res = {"corpus": a.corpus, "palavras": a.palavras, "k": k, "c_humanos": [int(x) for x in c],
           "taxas_humanos": [lt.taxa_lexico(t, lex) for t in ref], "pares": pares, "pares_por_lote": a.pares}
    (RAIZ / "leitura").mkdir(exist_ok=True)
    (RAIZ / "leitura" / "controles.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"k": k, "excertos_referencia": len(ref), "pares": len(pares),
                      "taxa_lexico_humana": round(float(np.mean(res["taxas_humanos"])), 4)}, indent=2))


def escolher(amostras: list[dict], itens: int, semente: int) -> list[dict]:
    """`itens` samples spread as evenly as possible over the seeds present, in a seeded order."""
    por_semente = defaultdict(list)
    for r in amostras:
        por_semente[r["semente"]].append(r)
    sementes = sorted(por_semente)
    rng = random.Random(semente)
    for s in sementes:
        rng.shuffle(por_semente[s])
    base, resto = divmod(itens, len(sementes))
    escolha = []
    for i, s in enumerate(sementes):
        q = base + (i < resto)
        if len(por_semente[s]) < q:
            sys.exit(f"semente {s}: {len(por_semente[s])} amostras aprovadas no filtro de cópia, {q} necessárias")
        escolha += por_semente[s][:q]
    return escolha


def cmd_lotes(a):
    pasta = RAIZ / "leitura"
    ctrl = json.loads((pasta / "controles.json").read_text(encoding="utf-8"))
    amostras = [json.loads(l) for l in open(pasta / "amostras.jsonl", encoding="utf-8")]
    pontos = sorted({r["ponto"] for r in amostras})
    modos = sorted({r["modo"] for r in amostras})
    if len(ctrl["pares"]) < ctrl["pares_por_lote"] * len(pontos):
        sys.exit("pares de controle insuficientes para um lote por ponto")
    relato = {"k": ctrl["k"], "pontos": {}}
    for n, ponto in enumerate(pontos, 1):
        itens, relato["pontos"][str(ponto)] = [], {}
        for modo in modos:
            grupo = [r for r in amostras if r["ponto"] == ponto and r["modo"] == modo]
            filtro = lt.filtro_copia([r["c"] for r in grupo], ctrl["k"])
            aprovadas = [r for r, d in zip(grupo, filtro.descartadas) if not d]
            n1 = lt.nivel_n1([r["taxa_lexico"] for r in grupo], ctrl["taxas_humanos"])
            relato["pontos"][str(ponto)][modo] = {
                "amostras": filtro.n, "descartadas_copia": filtro.n_descartadas, "taxa_descarte": filtro.taxa,
                "ic95_descarte": list(filtro.intervalo), "n1": n1.__dict__}
            itens += [{"texto": r["texto"], "modo": modo, "ponto": ponto, "semente": r["semente"],
                       "amostra_id": r["amostra_id"]} for r in escolher(aprovadas, a.itens, a.semente + n)]
        p = ctrl["pares_por_lote"]
        pares = [tuple(x) for x in ctrl["pares"][(n - 1) * p:n * p]]
        linhas, chave = lt.montar_lote(itens, pares, a.semente + 100 + n)
        lt.gravar_lote(pasta, n, linhas, chave)
        print(f"lote {n}: ponto {ponto}, {len(linhas)} itens")
    (pasta / "n1_copia.json").write_text(json.dumps(relato, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_analisar(a):
    pasta = RAIZ / "leitura"
    resultado = {"lotes": {}}
    for p_chave in sorted(pasta.glob("chave_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        n = p_chave.stem.split("_")[1]
        chave = lt.ler_chave(p_chave)
        respostas = {p.stem[len("respostas_"):-len(f"_{n}")]: lt.ler_respostas(p) for p in pasta.glob(f"respostas_*_{n}.csv")}
        if not respostas:
            continue
        filtros = {av: lt.filtro_avaliador(r, chave) for av, r in respostas.items()}
        aprovados = sorted(av for av, f in filtros.items() if f.aprovado)
        sem_autor = [av for av in aprovados if av != a.autor]
        conc = {nv: {"com_autor": lt.concordancia(respostas, chave, nv, aprovados).__dict__,
                     "sem_autor": lt.concordancia(respostas, chave, nv, sem_autor).__dict__ if len(sem_autor) >= 2 else None}
                for nv in lt.NIVEIS}
        niveis = {}
        for modo in sorted({r["modo"] for r in chave.values() if r.get("tipo") == "modelo"}):
            ids = lt.itens_modelo(chave, modo=modo)
            niveis[modo] = {nv: lt.nivel_atingido(lt.matriz_notas(respostas, ids, nv, aprovados)).__dict__ for nv in lt.NIVEIS}
        ponto = next(r["ponto"] for r in chave.values() if r.get("tipo") == "modelo")
        resultado["lotes"][n] = {"ponto": ponto, "avaliadores": {av: f.__dict__ for av, f in filtros.items()},
                                 "aprovados": aprovados, "concordancia": conc, "niveis": niveis}
    (pasta / "resultado.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    for n, r in resultado["lotes"].items():
        resumo = {m: {nv: r["niveis"][m][nv]["atingido"] for nv in lt.NIVEIS} for m in r["niveis"]}
        print(f"lote {n} (ponto {r['ponto']}) | aprovados {r['aprovados']} | níveis {resumo}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="etapa", required=True)
    s = sub.add_parser("amostras")
    s.add_argument("--grade", default=str(RAIZ / "preregistro" / "m7_grade.json"))
    s.add_argument("--n", type=int, default=8, help="samples per run; 3 are used per seed, the rest absorb the copy filter")
    s.add_argument("--palavras", type=int, default=60)
    s.add_argument("--dispositivo", default="cpu")
    s = sub.add_parser("controles")
    s.add_argument("--corpus", default="A")
    s.add_argument("--palavras", type=int, default=60)
    s.add_argument("--n-referencia", type=int, default=500, help="human excerpts for k and the N1 reference")
    s.add_argument("--pares", type=int, default=20)
    s.add_argument("--lotes", type=int, default=5)
    s.add_argument("--semente", type=int, default=0)
    s = sub.add_parser("lotes")
    s.add_argument("--itens", type=int, default=12)
    s.add_argument("--semente", type=int, default=0)
    s = sub.add_parser("analisar")
    s.add_argument("--autor", required=True, help="rater name of the author, left out in the second alpha")
    a = ap.parse_args()
    {"amostras": cmd_amostras, "controles": cmd_controles, "lotes": cmd_lotes, "analisar": cmd_analisar}[a.etapa](a)


if __name__ == "__main__":
    main()
