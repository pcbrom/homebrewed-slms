#!/usr/bin/env python3
"""Build the English versions of the two class notebooks, in notebooks/en/.

English counterpart of notebooks/gerar_notebook.py: the same boxes, cells and code, with the prose
written in English, the building blocks from notebooks/teoria_en.py and the technique notes from
notebooks/tecnica_en.py. Package identifiers stay in Portuguese, because they are the code that
trained the reported runs, and each notebook opens with a glossary of them.

    python notebooks/gerar_notebook_en.py
"""
import json
import re
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from gerar_notebook import REPO, anexo, code, md  # noqa: E402
from tecnica_en import TECNICA  # noqa: E402
from teoria_en import TEORIA, mapa  # noqa: E402

SAIDA = PASTA / "en"
ARQUIVOS = {"autoregressivo": "slm_autoregressive.ipynb", "difusao": "slm_diffusion.ipynb"}
GLOSSARIO = r"""
### A note on the code

The package `slm/` is written with Portuguese identifiers, and this notebook keeps them, because they are the exact code that produced the results reported in the project. The ones you will meet most often are below. By programming convention identifiers carry no accents, so the two modes appear in the code as `autoregressivo` and `difusao`.

| Identifier | Meaning | Identifier | Meaning |
|---|---|---|---|
| `treino`, `validacao`, `teste` | training, validation, test | `treinar` | train |
| `corpus`, `documentos` | corpus, documents | `ler_documentos` | read documents |
| `tokenizador` | tokenizer | `janelas`, `janelar` | windows, cut into windows |
| `mascaramento`, `mascarar` | masking, apply masks | `modelo`, `rede` | model, network |
| `perda` | loss | `epocas`, `lote` | epochs, batch |
| `geracao`, `gerar_ar`, `gerar_difusao` | generation, generate | `avaliacao`, `avaliar` | evaluation, evaluate |
| `calibracao` | calibration | `delecao` | deletion |
| `leitura` | reading | `amostras` | samples |
| `passos_aplicados`, `passos_pulados` | applied steps, skipped steps | `tempo_s`, `perda_final` | time in seconds, final loss |
| `sem_embedding`, `com_embedding` | without, with the embedding | `particao`, `grupo` | partition, group |
"""


def celulas(modo: str) -> list[dict]:
    ar = modo == "autoregressivo"
    outro = ARQUIVOS["difusao" if ar else "autoregressivo"]
    c = []
    if ar:
        c.append(md(r"""
# A small language model from scratch, autoregressive mode

This notebook builds a small language model from nothing that generates text from left to right, one token at a time, like today's large language models. The sibling notebook, `{outro}`, trains the same architecture, with weights of its own, in the masked diffusion mode, and the two deliberately repeat the boxes they share, so that each can be read on its own.
"""))
    else:
        c.append(md(r"""
# A small language model from scratch, masked diffusion mode

This notebook builds a small language model from nothing that generates text by starting from a fully masked window and revealing positions in random order until the text is complete. The sibling notebook, `{outro}`, trains the same architecture, with weights of its own, in the autoregressive mode, and the two deliberately repeat the boxes they share, so that each can be read on its own.
"""))
    c[-1]["source"] = c[-1]["source"].replace("{outro}", outro) + "\n\n" + r"""
Each section corresponds to a box of the project's architecture diagram and is told in three parts. The theory comes broken down into mathematical building blocks, the application runs the code and shows what goes in and what comes out, and the technique covers the problems the project met and how they were solved. Nothing here uses pretrained weights or a pretrained tokenizer.

Project, decisions and references at [{REPO}]({REPO}). The corpora are Brazilian Portuguese, so the texts and samples below are in Portuguese. A Portuguese version of this notebook sits in the `notebooks/` folder. To run it in Colab, choose **Runtime, Change runtime type, T4 GPU** and run the cells in order.

If you find this material useful, please star the repository and cite it through `CITATION.cff`.
""".replace("{REPO}", REPO).strip("\n") + "\n" + GLOSSARIO.rstrip("\n")

    c.append(code(r"""
import os, subprocess, sys
LOCAL = os.environ.get("SLM_LOCAL")
if LOCAL:
    RAIZ = LOCAL
else:
    RAIZ = "/content/homebrewed-slms"
    if not os.path.exists(RAIZ):
        subprocess.run(["git", "clone", "--depth", "1", "{REPO}", RAIZ], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers==0.23.3", "regex"], check=True)
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import torch
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DISPOSITIVO, torch.cuda.get_device_name(0) if DISPOSITIVO == "cuda" else "")
""".replace("{REPO}", REPO)))

    ramo = ("This notebook follows the autoregressive branch of box 4, with a causal attention mask."
            if ar else "This notebook follows the diffusion branch of box 4, where tokens are replaced by `[MASK]`.")
    c.append(md(r"""
## The whole pipeline

![pipeline](attachment:pipeline_en.png)

Preparation (boxes 0 to 2) runs once and leaves the clean corpus and the tokenizer in the repository. Training (boxes 3 to 6) is the same in both modes up to box 4, where diffusion replaces tokens by `[MASK]` and the autoregressive mode uses causal attention. Evaluation and generation (boxes 7 to 10) measure what the model has learned. """ + ramo, anexo("pipeline_en.png")))
    c.append(md(mapa(modo)))

    c.append(md(r"""
## Box 0. Configuration

A run is described by a single object. Changing the domain, the size of the network or the mode means changing fields here, and nothing else. `CORPUS` chooses the training text, `"A"` for the prose of Machado de Assis and `"B"` for Brazilian federal laws, and `EPOCAS_AULA` sets how many times the live training goes over the text.

Before training, it is worth estimating how much text your GPU processes per hour and how much text the model needs to see. The full derivation is in `docs/DIMENSIONAMENTO.md`, and the cell below runs the same calculation for a T4. The script prints its table in Portuguese, where *pico* is peak, *tempo* is time, *orcamento* is budget, *otimo* is optimal and *custo de atencao* is the attention cost above $6N$.
"""))
    c.append(code(r"""
from slm.config import Config
MODO = "{MODO}"
CORPUS = os.environ.get("SLM_CORPUS", "A")
EPOCAS_AULA = float(os.environ.get("SLM_EPOCAS", "16"))
cfg_base = dict(L=256, n_layer=6, d=384, n_head=6, lote=64, epocas=EPOCAS_AULA, lr=1e-3,
                precisao="fp16" if DISPOSITIVO == "cuda" else "fp32",
                atencao="efficient" if DISPOSITIVO == "cuda" else "math", papel="aula", corpus=CORPUS)
print("mode:", MODO, "|", cfg_base)
print(subprocess.run([sys.executable, "scripts/dimensionar.py", "--pico-tflops", "65", "--vram-gb", "16",
                      "--arq", "6x384", "--vocab", "4096", "--horas", "0.25"],
                     check=True, capture_output=True, text=True).stdout)
""".replace("{MODO}", modo)))

    c.append(md(r"""
## Box 1. Domain adapter

Text from a source goes in, and clean NFC documents come out, each with a group, a partition and a byte count.

Corpus A gathers the prose works of Machado de Assis from Project Gutenberg, in the public domain and in the original spelling of the period editions ("elle", "annos"). Each chapter is a document. The split reserves for validation and test contiguous blocks of whole chapters from each work, with one margin chapter before each held-out block, and the tokenizer is trained only after the split, so that no statistic of the evaluation text leaks into the vocabulary. The script output below keeps the partition names of the package, `treino` (training), `margem` (margin), `validacao` and `teste`.
"""))
    c.append(code(r"""
import collections
from slm.corpus import ler_documentos
docs = ler_documentos(CORPUS)
por_particao = collections.Counter(d["particao"] for d in docs)
bytes_particao = collections.Counter()
for d in docs:
    bytes_particao[d["particao"]] += d["bytes"]
for p in ("treino", "margem", "validacao", "teste"):
    print(f"{p:10s} {por_particao[p]:5d} documents {bytes_particao[p] / 1e6:7.3f} MB")
exemplo = next(d for d in docs if d["particao"] == "treino")
print("\nexample:", exemplo["doc_id"], exemplo["grupo"], exemplo["bytes"], "bytes")
print(exemplo["texto"][:600])
"""))

    c.append(md(r"""
## Box 2. Tokenizer

Text goes in and a sequence of integer ids comes out, with the way back.

Byte-level BPE starts from the 256 bytes as its vocabulary and, at each iteration, merges the most frequent adjacent pair into a new symbol. Since any text is a sequence of bytes, there is no unknown token, and $\mathrm{decode}(\mathrm{encode}(s)) = s$ for every string. Ids 0 to 3 are reserved for `[PAD]`, `[BOS]`, `[EOS]` and `[MASK]` and are never produced from text.

Below, the didactic implementation trains on a small sample and shows the first merges, as text when the bytes form whole characters and as raw bytes when they do not. Then the experiment's tokenizer, trained by the library on the whole corpus, encodes a sentence. The two implementations produce the same merges and the same ids, which the project's tests check. The tokenizer notebook, `slm_tokenizer.ipynb`, takes this box much further.
"""))
    c.append(code(r"""
from slm.tokenizador import BPEDidatico, BPEBiblioteca
amostra = [d["texto"] for d in docs if d["particao"] == "treino"][:20]
didatico = BPEDidatico().treinar(amostra, 256 + 4 + 40)
def mostrar(x):
    try:
        return repr(x.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(x)
for a, b in didatico.merges[:15]:
    print(f"{mostrar(a):>12} + {mostrar(b):<12} -> {mostrar(a + b)}")
tok = BPEBiblioteca.carregar(f"tokenizador/{CORPUS}")
frase = "Capitú tinha olhos de ressaca, disse-lhe elle n'aquella hora."
ids = tok.encode(frase)
print("\nV =", tok.vocab_size, "| ids:", ids)
print("pieces:", [tok.decode([i]).decode("utf-8", "replace") for i in ids])
print("exact round trip:", tok.decode(ids) == frase.encode("utf-8"))
"""))

    c.append(md(r"""
## Box 3. Windowing

The ids of a document go in, and windows of $L$ tokens come out.

Each document gets an `[EOS]` at the end and is cut into $\lceil (n+1)/L \rceil$ windows of $L$ tokens, the last one padded with `[EOS]`. The padding is predicted and charged in both modes. If diffusion were given the length for free, it would gain up to $\log_2 L$ bits per document over the autoregressive mode, which pays for its `[EOS]`.
"""))
    c.append(code(r"""
from slm.corpus import tokenizar
from slm.janelas import janelar, janelas_documento, fracao_preenchimento
treino = tokenizar(ler_documentos(CORPUS, "treino"), tok)
validacao = tokenizar(ler_documentos(CORPUS, "validacao"), tok)
L = cfg_base["L"]
J = janelas_documento(treino[0]["ids"], L)
print("a document with", len(treino[0]["ids"]), "tokens becomes", tuple(J.shape), "windows")
print("end of the last window:", J[-1, -12:].tolist())
janelas_treino, _ = janelar([d["ids"] for d in treino], L)
print("training windows:", tuple(janelas_treino.shape), "| padding fraction:",
      round(fracao_preenchimento([d["ids"] for d in treino], L), 4))
"""))

    if ar:
        c.append(md(r"""
## Box 4. Shifted input and causal mask

The text stays intact. The input is the window shifted one position to the right, with a `[BOS]` at the start, and the causal attention mask stops each position from seeing the ones after it. The output at position $i$ thus predicts token $w_i$ looking only at the tokens before it.
"""))
        c.append(code(r"""
from slm.perda import entrada_ar
def texto_de(ids):
    return "".join("⟨BOS⟩" if i == 1 else tok.decode([i]).decode("utf-8", "replace") for i in ids)
w = janelas_treino[:1, :16]
print("target |", texto_de(w[0].tolist()))
print("input  |", texto_de(entrada_ar(w)[0].tolist()))
causal = torch.tril(torch.ones(6, 6, dtype=torch.int))
print("\n6x6 causal mask (1 = may attend):\n", causal)
"""))
    else:
        c.append(md(r"""
## Box 4. Masking

A number $l \sim U\{1, \dots, L\}$ is drawn, stratified within the batch, and exactly $l$ positions chosen uniformly become `[MASK]`. The network sees the text with gaps and tries to fill them in. Below, each line is the start of a window with its own $l$, and the symbol ▒ marks the masked positions.
"""))
        c.append(code(r"""
from slm.mascaramento import sortear_l, mascara_de_l, mascarar
g = torch.Generator().manual_seed(0)
w = janelas_treino[:4, :24]
l = sortear_l(4, 24, g)
z = mascarar(w, mascara_de_l(l, 24, g))
def texto_de(ids):
    return "".join("▒" if i == 3 else tok.decode([i]).decode("utf-8", "replace") for i in ids)
for i in range(4):
    print(f"l = {int(l[i]):2d} |", texto_de(z[i].tolist()))
"""))

    c.append(md(r"""
## Box 5. Transformer network

Ids go in, $(B, L)$, and logits come out, $(B, L, V)$.

The same architecture serves both modes and takes no time input. For diffusion, Ou et al. (2024) showed that the distribution the network has to learn, that of the clean tokens given the visible context, does not depend on time, as the technique of this box explains. The blocks apply LayerNorm before attention and before the MLP, positions are learned, and the embedding is shared between input and output. The `[MASK]` logit is fixed at $-\infty$, so the network never predicts the mask. """
        + ("In this notebook the network always runs with the causal mask." if ar else "In this notebook the network runs without a causal mask, and each position sees the whole window.")))
    c.append(code(r"""
from slm.modelo import ConfigRede, Transformer
rede = Transformer(ConfigRede(vocab_size=tok.vocab_size, L=L, n_layer=cfg_base["n_layer"], d=cfg_base["d"],
                              n_head=cfg_base["n_head"]))
print(rede.n_parametros())
with torch.no_grad():
    saida = rede(janelas_treino[:2], causal=MODO == "autoregressivo")
print("input", tuple(janelas_treino[:2].shape), "-> logits", tuple(saida.shape))
print("[MASK] logit:", float(saida[0, 0, 3]))
"""))

    if ar:
        perda = r"""
$$\ell_{AR}(w) = \sum_{i=1}^{L} -\log p_\theta(w_i \mid \text{BOS}, w_{<i}),$$

the exact negative log-likelihood of the window. This is what the evaluation in box 8 turns into bits per byte."""
    else:
        perda = r"""
With $l$ masks in the set $S$,

$$\hat\ell_{dif}(w; l, S) = \frac{L}{l} \sum_{i \in S} -\log p_\theta(w_i \mid z), \qquad \mathbb{E}_{l, S}\big[\hat\ell_{dif}\big] = \mathrm{NELBO}(w) \ge -\log p_\theta(w).$$

The expectation is the exact NELBO of diffusion with a linear schedule, with no cutoff at small $t$, because $\int_0^1 \binom{L}{l} t^{l-1}(1-t)^{L-l}\,dt = 1/l$."""
    c.append(md(r"""
## Box 6. Loss and optimisation
""" + perda + r"""

The loss is computed in fp32 from the logits, training runs in fp16 with gradient scaling on the T4, and only the steps actually applied count towards the budget. The training below is short, sized for a class. The sibling notebook trains the other mode with the same architecture and the same initialisation, the same data, the same order and the same number of FLOPs, which makes the two runs comparable.
"""))
    c.append(code(r"""
from slm.treino import treinar
cfg = Config(modo=MODO, vocab_size=tok.vocab_size, **cfg_base).validar()
torch.manual_seed(cfg.semente)
modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=cfg.L, n_layer=cfg.n_layer, d=cfg.d,
                                n_head=cfg.n_head)).to(DISPOSITIVO)
resumo = treinar(cfg, modelo, janelas_treino, DISPOSITIVO)
modelo.eval()
print({k: round(v, 3) if isinstance(v, float) else v for k, v in resumo.items()})
"""))

    if ar:
        c.append(md(r"""
## Box 7. Generation

One token at a time, from left to right. Starting from `[BOS]`, the network gives the distribution of the next token, one is drawn, it joins the input and the step repeats until `[EOS]` or the end of the window. The draw is made in float64, because truncating the sampling noise already reduces the diversity of samples in fp32 (Zheng et al., 2024), and in fp16 the effect would be larger.
"""))
        c.append(code(r"""
from slm.geracao import gerar_ar, ate_eos
def como_texto(ids):
    return tok.decode(ate_eos([i for i in ids if i > 3 or i == 2])).decode("utf-8", "replace")
amostras = gerar_ar(modelo, L, n=3, semente=1, dispositivo=DISPOSITIVO)
for a in amostras.tolist():
    print("AR |", como_texto(a), "\n")
"""))
    else:
        c.append(md(r"""
## Box 7. Generation

The window starts fully masked and, at each step from $t = k/T$ to $s = (k-1)/T$, each position still masked is revealed with probability $(t-s)/t$. With $T$ steps, the whole text appears in parallel, not from left to right. The window has the same $L$ positions as in training, because the bidirectional network has only seen windows of that size, and the text is cut at the first `[EOS]`. The draw is made in float64, because truncating the sampling noise already reduces the diversity of samples in fp32 (Zheng et al., 2024), and in fp16 the effect would be larger.
"""))
        c.append(code(r"""
from slm.geracao import gerar_difusao, ate_eos
def como_texto(ids):
    return tok.decode(ate_eos([i for i in ids if i > 3 or i == 2])).decode("utf-8", "replace")
amostras = gerar_difusao(modelo, L, T_gen=L, n=3, semente=1, dispositivo=DISPOSITIVO)
for a in amostras.tolist():
    print("DIF |", como_texto(a), "\n")
"""))

    leitura = ("In the autoregressive mode the NLL is exact, and the number obtained here is the model's true log-likelihood. It is the yardstick when the two modes are compared."
               if ar else "In diffusion the NLL is the NELBO, an upper bound. That is why only a diffusion win over the autoregressive mode is conclusive, since if it loses, the difference may be slack in the bound.")
    c.append(md(r"""
## Box 8. Evaluation in bits per byte

$$\mathrm{bpb} = \frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d B_d}$$

Measuring per byte rather than per token puts models with different tokenizers on the same scale, which is why bits per byte serve to compare different configurations. """ + leitura))
    c.append(code(r"""
from slm.avaliacao import avaliar
res = avaliar(modelo, validacao, MODO, L, K=1, dispositivo=DISPOSITIVO)
print(f"{MODO} | validation bpb = {res['bpb']:.4f}")
"""))
    c.append(md(r"""
### Stated confidence against observed accuracy

In a well-calibrated model, the predictions made with a stated confidence of 0.7 are right about 70% of the time. The temperature $T$ divides the logits before the softmax and is fitted by Newton's method on validation. Points below the diagonal indicate overconfidence, and points above, underconfidence.

Two precautions come before the measurement. The validation windows go in a random order, because the cap on collected predictions, in the original order, would keep only the first works. And the padding after `[EOS]` is left out. It is predicted almost with certainty and, in the project, came to occupy most of the high-confidence predictions, which made the calibration measure the padding rather than the text.
"""))
    c.append(code(r"""
import torch
from slm import calibracao as cal, graficos
janelas_val, _ = janelar([d["ids"] for d in validacao], L)
janelas_val = janelas_val[torch.randperm(janelas_val.shape[0], generator=torch.Generator().manual_seed(0))]
Z, y = cal.coletar_previsoes(modelo, janelas_val, MODO, n_max=20000, dispositivo=DISPOSITIVO, sem_preenchimento=True)
T = cal.temperatura_newton(Z, y)
sistemas = {}
for rotulo, t in ((f"{MODO} T=1", 1.0), (f"{MODO} calibrated", T)):
    cf, ac = cal.confianca_acerto(Z, y, t)
    sistemas[rotulo] = dict(cal.confiabilidade(cf, ac), ece=cal.ece(cf, ac))
print(MODO, "| T =", round(T, 3), "| ROC AUC =", round(cal.auc_roc(*cal.confianca_acerto(Z, y, T)), 3))
fig = graficos.confiabilidade(sistemas, idioma="en")
"""))

    c.append(md(r"""
## Box 9. Uncertainty

A number of bits per byte only becomes a claim together with its uncertainty. The project's protocol uses several seeds per mode and a bootstrap that resamples works and seeds. Here, with a single seed, the interval below resamples the works only, and so it measures variation between parts of the text, not between training runs. The comparison between the two modes uses the difference paired by seed, described in the blocks of this box, and the results of the full protocol are in `docs/`, in the repository.
"""))
    c.append(code(r"""
import numpy as np
grupos = sorted({r["grupo"] for r in res["documentos"]})
por_grupo = {g: [(r["nats"], r["bytes"]) for r in res["documentos"] if r["grupo"] == g] for g in grupos}
rng = np.random.default_rng(0)
replicas = []
for _ in range(2000):
    escolha = rng.choice(grupos, size=len(grupos), replace=True)
    nats = sum(n for g in escolha for n, _ in por_grupo[g])
    nbytes = sum(b for g in escolha for _, b in por_grupo[g])
    replicas.append(nats / nbytes / np.log(2))
print(f"{MODO} | bpb = {res['bpb']:.4f} | 95% interval across works:", np.percentile(replicas, [2.5, 97.5]).round(4))
"""))

    c.append(md(r"""
## Box 10. Reading and memorisation

A model trained for many epochs on a small corpus can look coherent because it copies the training text. Before any human reading, each sample goes through two automatic filters, the share of its words found in the training lexicon (level N1) and the length of the longest span copied from the training text.
"""))
    c.append(code(r"""
from slm.leitura import IndiceTreino, lexico, taxa_lexico
textos_treino = [d["texto"] for d in ler_documentos(CORPUS, "treino")]
lex = lexico(textos_treino)
indice = IndiceTreino(textos_treino)
for a in amostras.tolist():
    t = como_texto(a)
    print("lexicon rate:", round(taxa_lexico(t, lex), 3), "| longest copied span:", indice.maior_trecho(t), "words")
"""))
    c.append(md(r"""
### Deletion curve

How much does a prediction depend on its context? For each passage of 50 words, context words are removed, starting with those that most support the decision, and the probability of the decided token is followed relative to the original one. With 100% removed the context is empty, and what remains is the model's intrinsic bias. The curve with random order serves as a reference.
"""))
    c.append(code(r"""
from slm import delecao
itens = delecao.itens([d["texto"] for d in ler_documentos(CORPUS, "teste")], tok, n_itens=20, n_palavras=50)
res_del = delecao.analisar(modelo, MODO, tok, itens, L, T=T, dispositivo=DISPOSITIVO)
print(f"AUC by importance {res_del['auc_importancia_media']:.3f} | random {res_del['auc_aleatoria_media']:.3f} | "
      f"empty context {res_del['p_vazio_relativo_medio']:.3f} | Spearman p0 x AUC {res_del['spearman_p0_auc']['rho']:.2f}")
fig = graficos.delecao({MODO: res_del}, idioma="en")
"""))

    if ar:
        exercicios = r"""
1. In box 0, set `CORPUS = "B"` and run everything again. What changes in the code and what stays the same?
2. In box 7, pass `gerar_ar` a `prefixo` with the ids of a sentence of your own. How far does the continuation stay on topic?
3. In box 0, double `EPOCAS_AULA`. Do the validation bits per byte fall in the same proportion as the training loss? What does the difference indicate?
4. In box 4, what would happen to the training loss if the input were not shifted by `[BOS]`, and why would that number say nothing about the quality of the model?
5. On the deletion curve, what does it mean for the curve to stay near 1 even with 100% of the context removed?
6. Run the notebook `{outro}` with the same configuration and compare the bits per byte. What conclusion does the asymmetric reading of box 9 allow?"""
    else:
        exercicios = r"""
1. In box 0, set `CORPUS = "B"` and run everything again. What changes in the code and what stays the same?
2. In box 7, lower `T_gen` to 8. What happens to the text, and why?
3. In box 0, double `EPOCAS_AULA`. Do the validation bits per byte fall in the same proportion as the training loss?
4. In box 6, why does the loss divide the sum of each window by its own $l$, rather than the sum of the whole batch by the total number of masks in the batch?
5. On the deletion curve, what does it mean for the curve to stay near 1 even with 100% of the context removed?
6. Run the notebook `{outro}` with the same configuration and compare the bits per byte. What conclusion does the asymmetric reading of box 9 allow?"""
    c.append(md("## Exercises\n" + exercicios.replace("{outro}", outro) + r"""

## References

The full list of references, all open access, is in `referencias/REFERENCIAS.md`, in the repository."""))
    return c


IGUAL = "Nothing. This box is identical in both notebooks."
NOTAS = {
    0: ('The configuration is the same, except for `MODO`, which here is `"autoregressivo"` and in the diffusion notebook is `"difusao"`. The technique gives the measured throughput of this mode.',
        'The configuration is the same, except for `MODO`, which here is `"difusao"` and in the autoregressive notebook is `"autoregressivo"`. The technique gives the measured throughput of this mode.'),
    1: (IGUAL, IGUAL),
    2: (IGUAL, IGUAL),
    3: ("Nothing. The predicted padding exists to make the comparison between the two modes fair, and so it is the same in both notebooks.",
        "Nothing. The predicted padding exists to make the comparison between the two modes fair, and so it is the same in both notebooks."),
    4: ("This is the box where the modes part ways. Here the window stays intact, goes in shifted by a `[BOS]` and is read with a causal mask. In the diffusion notebook, the window goes in unshifted, with $l$ positions replaced by `[MASK]`, and attention sees the whole window.",
        "This is the box where the modes part ways. Here $l$ positions of the window become `[MASK]`, and attention sees the whole window. In the autoregressive notebook, the window stays intact, goes in shifted by a `[BOS]` and is read with a causal mask."),
    5: ("The architecture is the same, with the same number of parameters, but each notebook trains its own weights from scratch. Inside the network, the only difference is the attention mask, causal here and absent in the diffusion notebook ($M = 0$ in the attention block). Outside it, the input (box 4) and the loss (box 6) change.",
        "The architecture is the same, with the same number of parameters, but each notebook trains its own weights from scratch. Inside the network, the only difference is the attention mask, absent here ($M = 0$ in the attention block) and causal in the autoregressive notebook. Outside it, the input (box 4) and the loss (box 6) change."),
    6: ("The loss is different. Here it is the exact negative log-likelihood of the window. In the diffusion notebook it is an unbiased estimator of the NELBO, an upper bound on that quantity, with weight $L/l$ per window. The optimisation (AdamW, learning-rate schedule, fp16 with GradScaler) is the same.",
        "The loss is different. Here it is an unbiased estimator of the NELBO, an upper bound on the negative log-likelihood, with weight $L/l$ per window. In the autoregressive notebook it is the exact negative log-likelihood. The optimisation (AdamW, learning-rate schedule, fp16 with GradScaler) is the same."),
    7: ("Generation is different. Here one token comes out at a time, from left to right, with one pass of the network per token. In the diffusion notebook, the window starts fully masked and is revealed in `T_gen` steps, in random order. The float64 draw is the same.",
        "Generation is different. Here the window starts fully masked and is revealed in `T_gen` steps, in random order, and `T_gen` can be smaller than the number of tokens. In the autoregressive notebook one token comes out at a time, from left to right, with one pass of the network per token. The float64 draw is the same."),
    8: ("The formula for bits per byte is the same, but its meaning changes. Here the number is exact. In the diffusion notebook it comes from the NELBO and is an upper bound, which makes the comparison asymmetric. Temperature calibration uses the same routine, on different populations, every position in the autoregressive mode and only the masked ones in diffusion.",
        "The formula for bits per byte is the same, but its meaning changes. Here the number comes from the NELBO and is an upper bound. In the autoregressive notebook it is exact, which makes the comparison asymmetric. Temperature calibration uses the same routine, on different populations, every position in the autoregressive mode and only the masked ones in diffusion."),
    9: ("Nothing. The interval across works and the rule for comparing the modes are the same in both notebooks.",
        "Nothing. The interval across works and the rule for comparing the modes are the same in both notebooks."),
    10: ("The code is identical. Only this notebook carries the measurement of memorisation over many epochs, which was made in this mode.",
         "The code is identical. The measurement of memorisation over many epochs appears only in the autoregressive notebook, because it was made in that mode."),
}


def do_modo(itens: list, modo: str) -> list:
    return [(tipo, texto) for tipo, texto, modos in itens if modos in ("ambos", modo)]


def tecer(cels: list[dict], modo: str) -> list[dict]:
    """Give every box its theory (box text and building blocks from notebooks/teoria_en.py),
    application (its code) and technique (notebooks/tecnica_en.py), as in gerar_notebook.tecer."""
    def cel(tipo, texto):
        return md(texto) if tipo == "md" else code(texto)

    def tecnica(caixa):
        itens = do_modo(TECNICA.get(caixa, []), modo)
        return [md("### Technique")] + [cel(t, x) for t, x in itens] if itens else []

    saida, caixa, aplicacao_aberta = [], None, False
    for c in cels:
        m = re.match(r"## (Box (\d+)\.|Exercises)", c["source"]) if c["cell_type"] == "markdown" else None
        if m:
            if caixa is not None:
                saida += tecnica(caixa)
            caixa = int(m.group(2)) if m.group(2) else None
            if caixa is not None:
                cabeca, _, resto = c["source"].partition("\n")
                outro = "diffusion" if modo == "autoregressivo" else "autoregressive"
                nota = NOTAS[caixa][modo != "autoregressivo"]
                c = dict(c, source=f"{cabeca}\n\n> **What changes from the {outro} notebook.** {nota}\n\n### Theory\n{resto}")
                aplicacao_aberta = True
        elif c["cell_type"] == "code" and caixa is not None and aplicacao_aberta:
            saida.append(md("### Application"))
            aplicacao_aberta = False
        saida.append(c)
        if m and caixa is not None:
            k = iter(range(1, 100))
            for t, x in do_modo(TEORIA.get(caixa, []), modo):
                saida.append(cel(t, re.sub(r"\*\*Block #,", lambda _: f"**Block {caixa}.{next(k)},", x)))
    return saida


def main():
    SAIDA.mkdir(exist_ok=True)
    for modo, nome in ARQUIVOS.items():
        nb = {"cells": tecer(celulas(modo), modo),
              "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                           "kernelspec": {"display_name": "Python 3", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        for i, c in enumerate(nb["cells"]):
            c["id"] = f"c{i:03d}"
        (SAIDA / nome).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"en/{nome}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
