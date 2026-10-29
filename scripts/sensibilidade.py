#!/usr/bin/env python3
"""Milestone M7: descriptive sensitivity of Delta at the confirmatory point (SPEC section 14).

One factor at a time around the corpus A confirmatory configuration (256 epochs, L = 256,
V = 4096, fp16, K = 1, T_gen = L), on the first 3 confirmatory seeds, so every value is paired by
seed with the M7 runs already trained. No test, no change to the confirmatory reading.

Subcommands, run in order:
    preregistrar  freeze values, seeds, metrics and the reading in preregistro/sensibilidade.json
    tokenizadores train the BPE of every V of the grid on the training partition, tokenizador/A_V<V>
    rodar         train the pending runs of L, V and precision; stops after --max-min and resumes
    avaliar       K (NELBO with K draws, 3 evaluation seeds) and T_gen (diffusion samples) on the
                  M7 checkpoints, without training
    analisar      per value: bpb of each mode, Delta per seed, mean and t interval; writes
                  resultados/sensibilidade.json

    python scripts/sensibilidade.py preregistrar
    python scripts/sensibilidade.py tokenizadores
    python scripts/sensibilidade.py rodar --max-min 200
    python scripts/sensibilidade.py avaliar --dispositivo cuda
    python scripts/sensibilidade.py analisar
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from slm.registro import conferir_preregistro, preregistrar  # noqa: E402

PRE, RUNS, RES = RAIZ / "preregistro", RAIZ / "runs", RAIZ / "resultados"
NOME = "sensibilidade"
MODOS = ("autoregressivo", "difusao")
EPOCAS = 256
SEMENTES = (300, 301, 302)
TREINO = {"L": (128, 512), "V": (2048, 8192), "precisao": ("fp32",)}
PADRAO = {"L": 256, "V": 4096, "precisao": "fp16", "K": 1, "T_gen": 256}
AVALIACAO = {"K": (1, 4, 16), "T_gen": (16, 64, 256)}
SEMENTES_AVALIACAO = (0, 1, 2)
N_AMOSTRAS, PALAVRAS = 32, 60


def base_id(modo, s):
    return f"m7-{modo}-e{EPOCAS}-s{s}"


def run_id(param, valor, modo, s):
    return f"sens-{param}{valor}-{modo}-e{EPOCAS}-s{s}"


def ler(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cmd_preregistrar(_):
    assert conferir_preregistro(PRE, "m7"), "M7 not frozen"
    m7 = ler(PRE / "m7.json")
    faltam = [base_id(m, s) for m in MODOS for s in SEMENTES if not (RUNS / base_id(m, s) / "pesos.pt").exists()]
    assert not faltam, f"baseline checkpoints missing: {faltam}"
    h = preregistrar(PRE, NOME, {
        "corpus": "A", "epocas": EPOCAS, "sementes": SEMENTES, "escolha": m7["escolha"], "padrao": PADRAO,
        "fatores_com_treino": TREINO, "fatores_so_avaliacao": AVALIACAO,
        "sementes_avaliacao_K": SEMENTES_AVALIACAO, "amostras_T_gen": {"n": N_AMOSTRAS, "palavras": PALAVRAS},
        "linha_de_base": "corridas m7-<modo>-e256-s300 a s302, pareadas por semente",
        "um_fator_por_vez": True,
        "V": "BPE byte-level da biblioteca treinado no treino com V fixo, sem a regra de escolha de V",
        "L": "orcamento em epocas, iguais tokens de treino; o lote de 64 janelas muda os tokens por passo e o numero de passos",
        "criterio_de_orcamento": "sem corrida: zero passos pulados nos 104 runs do M7, contagem por passo aplicado e por passo nominal coincidem",
        "metricas": {"L_V_precisao": "bpb de validacao por modo e Delta = bpb_dif - bpb_AR por semente, media e IC de 95% por t com 2 graus de liberdade",
                     "K": "bpb da difusao e Delta por K, media e desvio entre 3 sementes de avaliacao",
                     "T_gen": "por T_gen: fracao de amostras com 60 palavras, taxa de lexico contra a humana, trigramas atestados, descarte por copia com k do M9"},
        "leitura": "descritiva; nenhum teste e nenhuma alteracao do criterio confirmatorio da decisao de orcamento",
        "precisao": "fp16 com GradScaler, atencao efficient, exceto no fator precisao"})
    print(NOME, h)


def cmd_tokenizadores(_):
    for V in TREINO["V"]:
        saida = RAIZ / "tokenizador" / f"A_V{V}"
        if (saida / "tokenizer.json").exists():
            continue
        cmd = [sys.executable, str(RAIZ / "scripts" / "treinar_tokenizador.py"), "--corpus", "A", "--grade", str(V),
               "--limiar", "1.0", "--saida", str(saida)]
        subprocess.run(cmd, check=True, timeout=3600)


def treinar(rid, modo, s, hp, param, valor):
    extra = {"L": [], "V": ["--tokenizador", str(RAIZ / "tokenizador" / f"A_V{valor}")], "precisao": []}[param]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"L": valor} if param == "L" else {}, f)
    cmd = [sys.executable, str(RAIZ / "scripts" / "treinar.py"), "--config", f.name, "--modo", modo, "--epocas", str(EPOCAS),
           "--semente", str(s), "--lr", str(hp["lr"]), "--dropout", str(hp["dropout"]), "--papel", "sensibilidade",
           "--precisao", valor if param == "precisao" else "fp16", "--atencao", "efficient", "--dispositivo", "cuda",
           "--corpus", "A", "--run-id", rid] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    Path(f.name).unlink()
    if r.returncode != 0:
        sys.exit(f"{rid} falhou:\n{r.stderr[-2000:]}")
    print(rid, round(json.loads(r.stdout)["bpb_validacao"], 4), flush=True)


def cmd_rodar(a):
    assert conferir_preregistro(PRE, NOME), "pre-registration missing or altered"
    pre = ler(PRE / f"{NOME}.json")
    t0 = time.time()
    for param, valores in TREINO.items():
        for valor in valores:
            for s in SEMENTES:
                for modo in MODOS:
                    rid = run_id(param, valor, modo, s)
                    if (RUNS / rid / "avaliacao.json").exists():
                        continue
                    if (time.time() - t0) / 60 > a.max_min:
                        print("time budget reached, rerun to resume")
                        return
                    treinar(rid, modo, s, pre["escolha"][modo], param, valor)
    print("all sensitivity runs done")


def carregar(rid, dispositivo):
    import torch
    from slm.config import Config
    from slm.corpus import pasta_tokenizador
    from slm.modelo import ConfigRede, Transformer
    from slm.tokenizador import carregar_tokenizador
    bruto = ler(RUNS / rid / "config.json")
    cfg = Config.de_dicionario({k: v for k, v in bruto.items() if k in Config.__dataclass_fields__})
    modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head))
    modelo.load_state_dict(torch.load(RUNS / rid / "pesos.pt", map_location="cpu"))
    return cfg, carregar_tokenizador(pasta_tokenizador(cfg)), modelo.to(dispositivo).eval()


def cmd_avaliar(a):
    from m9 import cortar, excertos_humanos
    from slm import leitura as lt
    from slm.avaliacao import avaliar
    from slm.corpus import ler_documentos, tokenizar
    from slm.estatistica import clopper_pearson
    from slm.geracao import ate_eos, gerar_ar, gerar_difusao
    assert conferir_preregistro(PRE, NOME), "pre-registration missing or altered"
    pasta = RES / NOME
    pasta.mkdir(parents=True, exist_ok=True)
    treino = [d["texto"] for d in ler_documentos("A", "treino")]
    lex, tri, indice = lt.lexico(treino), lt.trigramas(treino), lt.IndiceTreino(treino)
    # human reference built as in the M9 controls: 500 test excerpts, seed 0, same length
    humanos = excertos_humanos([d["texto"] for d in ler_documentos("A", "teste")], 500, PALAVRAS, 0)
    k = lt.limiar_copia(indice.maiores_trechos(humanos))
    ref = {"k": k, "taxa_lexico": float(np.mean([lt.taxa_lexico(t, lex) for t in humanos])),
           "taxa_trigramas": float(np.mean([lt.taxa_trigramas(t, tri) for t in humanos]))}
    saida_K, saida_T = {}, {"humanos": ref}
    for s in SEMENTES:
        cfg, tok, modelo = carregar(base_id("difusao", s), a.dispositivo)
        val = tokenizar(ler_documentos("A", "validacao"), tok)
        saida_K[str(s)] = {str(K): [avaliar(modelo, val, "difusao", cfg.L, K=K, semente=e, dispositivo=a.dispositivo)["bpb"]
                                    for e in SEMENTES_AVALIACAO] for K in AVALIACAO["K"]}
        assert abs(saida_K[str(s)]["1"][0] - bpb(base_id("difusao", s))) < 1e-5, "K = 1 does not reproduce the M7 evaluation"
        print("K", s, {K: round(float(np.mean(v)), 4) for K, v in saida_K[str(s)].items()}, flush=True)
        for modo, valores in (("difusao", AVALIACAO["T_gen"]), ("autoregressivo", (None,))):
            if modo == "autoregressivo":
                cfg, tok, modelo = carregar(base_id(modo, s), a.dispositivo)
            semente_geracao = 1000 * s + EPOCAS
            for T in valores:
                t0 = time.time()
                ids = (gerar_ar(modelo, cfg.L, n=N_AMOSTRAS, semente=semente_geracao, dispositivo=a.dispositivo) if T is None else
                       gerar_difusao(modelo, cfg.L, T_gen=T, n=N_AMOSTRAS, semente=semente_geracao, dispositivo=a.dispositivo))
                seg = time.time() - t0
                textos = [cortar(tok.decode(ate_eos([i for i in w if i > 3 or i == 2])).decode("utf-8", "replace"), PALAVRAS,
                                 random.Random(semente_geracao * 1000 + j)) for j, w in enumerate(ids.tolist())]
                textos = [t for t in textos if t is not None]
                c = indice.maiores_trechos(textos) if textos else []
                desc = int(sum(x > k for x in c))
                saida_T.setdefault(modo if T is None else str(T), {})[str(s)] = {
                    "segundos": seg, "validas": len(textos), "n": N_AMOSTRAS,
                    "taxa_lexico": float(np.mean([lt.taxa_lexico(t, lex) for t in textos])) if textos else None,
                    "taxa_trigramas": float(np.mean([lt.taxa_trigramas(t, tri) for t in textos])) if textos else None,
                    "descarte_copia": desc, "descarte_ic95": list(clopper_pearson(desc, len(textos))) if textos else None}
                print("T_gen", modo, T, s, {kk: v for kk, v in saida_T[modo if T is None else str(T)][str(s)].items() if kk != "descarte_ic95"}, flush=True)
    (pasta / "K.json").write_text(json.dumps(saida_K, indent=1), encoding="utf-8")
    (pasta / "T_gen.json").write_text(json.dumps(saida_T, indent=1, ensure_ascii=False), encoding="utf-8")


def intervalo_t(x):
    from scipy import stats
    x = np.asarray(x, dtype=float)
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    q = float(stats.t.ppf(0.975, len(x) - 1))
    return {"media": m, "ic95": [m - q * se, m + q * se], "por_semente": x.tolist()}


def bpb(rid):
    d = ler(RUNS / rid / "avaliacao.json")
    return None if d is None else d["bpb"]


def cmd_analisar(_):
    from slm.estatistica import clopper_pearson
    pre = ler(PRE / f"{NOME}.json")
    saida = {"padrao": pre["padrao"], "fatores": {}}
    valores = {p: [PADRAO[p], *v] for p, v in TREINO.items()}
    for param, vals in valores.items():
        linhas = {}
        for valor in vals:
            ids = {m: [base_id(m, s) if valor == PADRAO[param] else run_id(param, valor, m, s) for s in SEMENTES] for m in MODOS}
            b = {m: [bpb(r) for r in ids[m]] for m in MODOS}
            if any(x is None for m in MODOS for x in b[m]):
                continue
            linhas[str(valor)] = {"bpb_autoregressivo": intervalo_t(b["autoregressivo"]), "bpb_difusao": intervalo_t(b["difusao"]),
                                  "delta": intervalo_t(np.subtract(b["difusao"], b["autoregressivo"]))}
        saida["fatores"][param] = linhas
    K = ler(RES / NOME / "K.json")
    if K:
        ar = [bpb(base_id("autoregressivo", s)) for s in SEMENTES]
        saida["fatores"]["K"] = {
            k: {"bpb_difusao": intervalo_t([np.mean(K[str(s)][k]) for s in SEMENTES]),
                "delta": intervalo_t([np.mean(K[str(s)][k]) - a for s, a in zip(SEMENTES, ar)]),
                "desvio_entre_sementes_de_avaliacao": float(np.mean([np.std(K[str(s)][k], ddof=1) for s in SEMENTES]))}
            for k in map(str, AVALIACAO["K"])}
    T = ler(RES / NOME / "T_gen.json")
    if T:
        saida["fatores"]["T_gen"] = {"humanos": T["humanos"]}
        for chave in [*map(str, AVALIACAO["T_gen"]), "autoregressivo"]:
            r = [T[chave][str(s)] for s in SEMENTES]
            validas, desc = sum(x["validas"] for x in r), sum(x["descarte_copia"] for x in r)
            saida["fatores"]["T_gen"][chave] = {
                "segundos_por_lote": float(np.mean([x["segundos"] for x in r])), "validas": validas, "n": sum(x["n"] for x in r),
                "taxa_lexico": intervalo_t([x["taxa_lexico"] for x in r]),
                "distancia_lexico_humano": float(np.mean([x["taxa_lexico"] for x in r]) - T["humanos"]["taxa_lexico"]),
                "taxa_trigramas": intervalo_t([x["taxa_trigramas"] for x in r]),
                "descarte_copia": desc, "descarte_ic95": list(clopper_pearson(desc, validas))}
    RES.mkdir(exist_ok=True)
    (RES / f"{NOME}.json").write_text(json.dumps(saida, indent=1, ensure_ascii=False), encoding="utf-8")
    for param, linhas in saida["fatores"].items():
        for valor, r in linhas.items():
            if "delta" in r:
                print(param, valor, round(r["delta"]["media"], 4), [round(x, 4) for x in r["delta"]["ic95"]])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="etapa", required=True)
    sub.add_parser("preregistrar")
    sub.add_parser("tokenizadores")
    s = sub.add_parser("rodar")
    s.add_argument("--max-min", type=float, default=60)
    s = sub.add_parser("avaliar")
    s.add_argument("--dispositivo", default="cuda")
    sub.add_parser("analisar")
    a = ap.parse_args()
    {"preregistrar": cmd_preregistrar, "tokenizadores": cmd_tokenizadores, "rodar": cmd_rodar,
     "avaliar": cmd_avaliar, "analisar": cmd_analisar}[a.etapa](a)


if __name__ == "__main__":
    main()
