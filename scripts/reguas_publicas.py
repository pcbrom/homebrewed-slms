#!/usr/bin/env python3
"""Public yardsticks of the generative track (preregistro/m14_tarefas_B.json): TeenyTinyLlama
160M, Tucano 160M and Tucano 630M, scored on the four tasks by the rules of the n-gram and of the
autoregressive mode (slm.medida_tarefas.pontuar), with [BOS] before each sequence and the
log-probability of their [EOS] at the end.

Pre-registered checks before any number: the logits in fp16 against fp32 on a sample of items,
and the tokenizer by round trip. The measure itself runs in fp32. Revisions are fixed and written
with the results. Runs in the environment of the Laya runtime, which has transformers:
    <modelos_decisao>/.venv/bin/python scripts/reguas_publicas.py --corpus B
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from slm.medida_tarefas import pontuar  # noqa: E402
from slm.registro import conferir_tarefas  # noqa: E402

REGUAS = {"ttl-160m": ("nicholasKluge/TeenyTinyLlama-160m", "da8648b20fa9ee37980fbfdd39bda8030834e58f"),
          "tucano-160m": ("TucanoBR/Tucano-160m", "6331cbef9933c05d8d0c8ae5ee23ed2570f06571"),
          "tucano-630m": ("TucanoBR/Tucano-630m", "37baa750655d7ee54a542a41a0dc7ffb372135e7")}
AMOSTRA = 64


def itens(corpus, t, particoes):
    with open(RAIZ / "tarefas" / corpus / f"{t}.jsonl", encoding="utf-8") as f:
        return [i for i in map(json.loads, f) if i["particao"] in particoes]


@torch.no_grad()
def logp_tokens(modelo, tok, seqs, lote=16):
    """Log-probability of every token and of [EOS], with [BOS] as the first input."""
    saida = []
    for k in range(0, len(seqs), lote):
        bloco = [[tok.bos_token_id] + s + [tok.eos_token_id] for s in seqs[k:k + lote]]
        n = max(map(len, bloco))
        ids = torch.full((len(bloco), n), tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        att = torch.zeros((len(bloco), n), dtype=torch.long)
        for j, b in enumerate(bloco):
            ids[j, :len(b)], att[j, :len(b)] = torch.tensor(b), 1
        lp = torch.log_softmax(modelo(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.double(), -1)
        alvo = lp[:, :-1].gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1).cpu().numpy()
        saida += [alvo[j, :len(b) - 1] for j, b in enumerate(bloco)]
    return saida


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="B")
    a = ap.parse_args()
    conferir_tarefas(RAIZ / "preregistro", RAIZ / "tarefas" / a.corpus, a.corpus)
    pasta = RAIZ / "resultados" / f"m14_{a.corpus}"
    resumo = {}
    amostra = [i["texto"] for i in itens(a.corpus, "t2", {"externo"})[:AMOSTRA]]
    for nome, (repo, rev) in REGUAS.items():
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        cod = lambda t: tok(t, add_special_tokens=False)["input_ids"]
        ida_volta = float(np.mean([tok.decode(cod(t)) == t for t in amostra]))
        m32 = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32).cuda().eval()
        m16 = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float16).cuda().eval()
        seqs = [cod(t) for t in amostra]
        d = np.concatenate([x - y for x, y in zip(logp_tokens(m32, tok, seqs), logp_tokens(m16, tok, seqs))])
        del m16
        torch.cuda.empty_cache()
        resumo[nome] = {"repositorio": repo, "revisao": rev, "ida_e_volta": ida_volta,
                        "fp16_contra_fp32": {"media_abs_nats": float(np.abs(d).mean()), "max_abs_nats": float(np.abs(d).max()),
                                             "finitos": bool(np.isfinite(d).all())}}
        print(nome, json.dumps(resumo[nome]), flush=True)
        logp = lambda s: logp_tokens(m32, tok, s)
        for t in ("t1", "t2", "t3", "t4"):
            res = pontuar(t, itens(a.corpus, t, {"externo", "treino"} if t == "t2" else {"externo"}), logp, cod)
            with open(pasta / f"{nome}_{t}.jsonl", "w", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(nome, t, len(res), flush=True)
        del m32
        torch.cuda.empty_cache()
    (pasta / "reguas_resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
