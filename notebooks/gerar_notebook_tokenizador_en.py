#!/usr/bin/env python3
"""Build the English version of the tokenizer notebook, notebooks/en/slm_tokenizer.ipynb.

English counterpart of notebooks/gerar_notebook_tokenizador.py: the same seven parts and the same
code, with the prose written in English. Package identifiers stay in Portuguese, and the notebook
opens with a glossary of them.

    python notebooks/gerar_notebook_tokenizador_en.py
"""
import json
import sys
from pathlib import Path

PASTA = Path(__file__).resolve().parent
sys.path.insert(0, str(PASTA))
from gerar_notebook import REPO, code, md  # noqa: E402

SAIDA = PASTA / "en"
ARQUIVO = "slm_tokenizer.ipynb"


def celulas() -> list[dict]:
    c = []
    c.append(md(r"""
# A tokenizer from scratch, and a tokenizer of your own in its place

Every language model reads numbers, not letters. The tokenizer is the piece that turns text into numbers and numbers back into text, and its choice sets the length of the sequences, the size of the output layer and, with them, how much each epoch of training costs. This notebook builds tokenizers from scratch, from the simplest to the most elaborate, shows how to plug a tokenizer you invented into the same package and uses them to train the project's two architectures, autoregressive and masked diffusion.

Each part is told in three steps. The theory comes broken down into mathematical building blocks, the application runs the code and shows what goes in and what comes out, and the technique covers the problems the project met and the measurement that revealed them. The sibling notebooks, `slm_autoregressive.ipynb` and `slm_diffusion.ipynb`, follow the whole model box by box. Here the focus is box 2, the tokenizer, and what changes elsewhere when it changes.

Project, decisions and references at [{REPO}]({REPO}). The corpus is Brazilian Portuguese, so the texts and samples below are in Portuguese, and a Portuguese version of this notebook sits in the `notebooks/` folder. To run it in Colab, choose **Runtime, Change runtime type, T4 GPU** and run the cells in order. The runs of part 6 take a few minutes on a desktop GPU and longer on a T4.

If you find this material useful, please star the repository and cite it through `CITATION.cff`.

### A note on the code

The package `slm/` is written with Portuguese identifiers, and this notebook keeps them, because they are the exact code behind the project's results. The ones used here are `tokenizador` (tokenizer), `treino`, `validacao` (training, validation), `ler_documentos` (read documents), `tokenizar` (tokenize), `janelar` (cut into windows), `treinar` (train), `avaliar` (evaluate), `conferir_contrato` (check the contract), `escolher_vocabulario` (choose the vocabulary), `fracao_ausente` (absent fraction), `codigo` (code) and `unidades` (units). The two modes appear in the code as `autoregressivo` and `difusao`.
""".replace("{REPO}", REPO)))
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
sys.path.insert(0, os.path.join(RAIZ, "experimentos", "m10_codigo"))
os.chdir(RAIZ)
import torch
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"
EPOCAS_REF = float(os.environ.get("SLM_EPOCAS", "16"))
print("device:", DISPOSITIVO, torch.cuda.get_device_name(0) if DISPOSITIVO == "cuda" else "", "| epochs of the reference BPE:", EPOCAS_REF)
""".replace("{REPO}", REPO)))
    c.append(md(r"""
## Mathematical map

| Symbol | Meaning |
|---|---|
| $d$, $B_d$ | a document (here, a chapter) and its number of bytes in UTF-8, after NFC normalisation |
| $\tau$, $\tau^{-1}$ | the tokenizer, from text to ids, and the way back, from ids to bytes |
| $V$ | vocabulary size, with ids 0 to 3 reserved for `[PAD]`, `[BOS]`, `[EOS]` and `[MASK]` |
| $n_d = \lvert\tau(d)\rvert$ | number of symbols in the document |
| $\sigma = \sum_d n_d / \sum_d B_d$ | symbols per byte, the density of the tokenizer |
| $\mathrm{NLL}(d)$ | the model's loss on the document, the negative log-likelihood in nats |
| $\mathrm{bpb}$ | bits per byte, the yardstick that does not depend on the tokenizer |
| $L$, $n_c$, $h$ | context window in symbols, number of layers and width of the network |
| $C(V, L)$ | training cost per symbol, in FLOPs |
| $A$, $\ell_u$, $H_A$ | alphabet of a code, length of the codeword of unit $u$ and entropy in base $A$ (part 5) |
| $S_b$, $E_b$ | training symbols and epochs of the run with tokenizer $b$ (part 6) |

The chain is short. The text $d$ becomes $\tau(d)$, the model learns on $\tau(d)$, and the measurement goes back to the text through the number of bytes, so that different tokenizers sit on the same scale.

The texts come from corpus A, the prose of Machado de Assis in the spelling of the period editions, already cleaned and split into training, validation and test in the repository. Throughout the notebook, held-out text means text that does not enter training, here the validation partition.
"""))
    c.append(code(r"""
from slm.corpus import ler_documentos
treino_docs, valid_docs = ler_documentos("A", "treino"), ler_documentos("A", "validacao")
treino = [d["texto"] for d in treino_docs]
valid = [d["texto"] for d in valid_docs]
B_treino, B_valid = sum(d["bytes"] for d in treino_docs), sum(d["bytes"] for d in valid_docs)
print(f"training: {len(treino)} chapters, {B_treino / 1e6:.2f} MB | validation: {len(valid)} chapters, {B_valid / 1e6:.2f} MB")
FRASE = "Capitú tinha olhos de ressaca, disse-lhe elle n'aquella hora."
"""))

    c.append(md(r"""
## Part 1. The contract

### Theory

In this project, a tokenizer is any object that honours a short contract. The contract is what allows the tokenizer to be swapped without touching any other box of the package.

**Block 1.1, there and back.** $\tau: \text{text} \to \{4, \dots, V-1\}^*$ and $\tau^{-1}: \{0, \dots, V-1\}^* \to \text{bytes}$, with $\tau^{-1}(\tau(s)) = \mathrm{utf8}(s)$ for every string $s$. Nothing is lost, not an accent, a space or a rare character.

**Block 1.2, tests U1 to U6.** U1 (ida e volta exata) is the exact round trip, U2 determinism (the same text always gives the same ids, and the line "U2 persistência" checks the same after saving and loading the tokenizer), U3 ids within $\{0, \dots, V-1\}$, U4 specials never produced from text, even when the text contains the string `[MASK]`, U5 decoding any single id without error, and U6 the requirement that none of this presupposes BPE. Code in `conferir_contrato`, in `slm/tokenizador.py`, which prints the test names in Portuguese.

**Block 1.3, the yardstick.** The bits per byte of a model are

$$\mathrm{bpb} = \frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d B_d} = \underbrace{\frac{\sum_d \mathrm{NLL}(d)}{\ln 2 \cdot \sum_d n_d}}_{\text{bits per symbol}} \cdot \underbrace{\frac{\sum_d n_d}{\sum_d B_d}}_{\sigma}.$$

A tokenizer that shortens the sequence lowers $\sigma$ and, in return, makes each symbol harder to predict. Only the product compares tokenizers, which is why no table in this notebook compares loss per token across different tokenizers.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
import tempfile
from slm.tokenizador import TokenizadorBytes, conferir_contrato
sonda = valid[:3] + [FRASE, "a ação do coração \u2014 漢字 \U0001F642 [MASK]"]
bytes_tok = TokenizadorBytes()
for teste, ok in conferir_contrato(bytes_tok, sonda, tempfile.mkdtemp()).items():
    print(f"{teste:32s} {'passes' if ok else 'FAILS'}")
"""))
    c.append(md(r"""
### Technique

**Same letter, different bytes.** Unicode admits two forms for the same accented letter. In the composed form (NFC), "ã" is a single code point. In the decomposed form (NFD), it is an "a" followed by a combining tilde. The pre-tokenization regular expression separates the combining mark from the letter, and the word falls apart. The project's corpus is normalised to NFC once, before anything else, and the bytes of the yardstick are counted on that text.
"""))
    c.append(code(r"""
import unicodedata, regex
from slm.tokenizador import GPT2_REGEX
for forma in ("NFC", "NFD"):
    t = unicodedata.normalize(forma, "a ação do coração")
    print(forma, len(t.encode()), "bytes |", regex.findall(GPT2_REGEX, t))
"""))

    c.append(md(r"""
## Part 2. Bytes, the simplest tokenizer

### Theory

**Block 2.1, alphabet.** The 256 values of a byte plus the 4 specials give $V = 260$. Since every text is a sequence of bytes, every text has a representation, and there is no unknown token.

**Block 2.2, density.** Each byte becomes one symbol, so $\sigma = 1$. A window of $L$ symbols sees exactly $L$ bytes of text. This is the starting point against which the other tokenizers measure how much they shorten the sequence.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
ids = bytes_tok.encode(FRASE)
print(len(FRASE), "characters,", len(FRASE.encode()), "bytes,", len(ids), "ids")
print(ids[:24], "...")
print("the 'ú' of Capitú takes two ids:", bytes_tok.encode("ú"))
"""))
    c.append(md(r"""
### Technique

**Less context in the same window.** With $L = 256$, the byte model sees 256 bytes of text, and the model on the project's BPE sees about 911. A fair comparison between the two has to decide whether it equalises the window in symbols or the context in bytes, and the project's experiment measured both forms side by side. Part 7 shows that the choice matters. At 16 epochs, bytes lose 0.80 bits per byte to BPE with the same window in symbols and 1.41 with the same context in bytes.
"""))

    c.append(md(r"""
## Part 3. BPE built by hand

### Theory

Byte-level BPE starts from the byte alphabet and repeatedly merges the most frequent neighbouring pair into a new symbol (Sennrich et al., 2016).

**Block 3.1, pre-tokenization.** A regular expression cuts the text into pieces, such as words with the space before them, numbers and punctuation, and no merge crosses the boundary between two pieces. The GPT-2 pattern (Radford et al., 2019) is the one used in the project.

**Block 3.2, counting pairs.** After pre-tokenization, the corpus becomes a multiset of pieces $w$ with frequency $f(w)$, and for each neighbouring pair $(a, b)$, $c(a, b) = \sum_w f(w)\, \#\{(a, b) \text{ neighbours in } w\}$.

**Block 3.3, merging and tie-breaking.** One picks $(a^*, b^*) = \arg\max c(a, b)$, with ties decided by the smallest pair of ids, creates the symbol $a^* b^*$ with the next free id and rewrites all pieces. This repeats until there are $V$ symbols.

**Block 3.4, encoding.** Encoding applies the merges in the order they were learned, and decoding concatenates the bytes of each symbol. Since every merge concatenates bytes, the contract of part 1 holds with no extra effort. Code in `BPEDidatico`, in `slm/tokenizador.py`.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
import collections
from slm.tokenizador import ORDEM_BYTES, BPEDidatico, BPEBiblioteca
print("pre-tokenization:", regex.findall(GPT2_REGEX, FRASE))
amostra = treino[:20]
id_byte = {bytes([b]): 4 + i for i, b in enumerate(ORDEM_BYTES)}
pedacos = collections.Counter(tuple(bytes([b]) for b in p.encode()) for t in amostra for p in regex.findall(GPT2_REGEX, t))
cont = collections.Counter()
for p, f in pedacos.items():
    for par in zip(p, p[1:]):
        cont[par] += f
a_mao = max(cont, key=lambda par: (cont[par], -id_byte[par[0]], -id_byte[par[1]]))
didatico = BPEDidatico().treinar(amostra, 260 + 60)
print("first merge by hand:", a_mao, "with", cont[a_mao], "occurrences | by the package:", didatico.merges[0])
def mostrar(x):
    try:
        return repr(x.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(x)
for a, b in didatico.merges[:12]:
    print(f"{mostrar(a):>10} + {mostrar(b):<10} -> {mostrar(a + b)}")
biblioteca = BPEBiblioteca().treinar(amostra, 260 + 60)
print("the 60 merges of the didactic implementation and of the tokenizers library are equal:", didatico.merges == biblioteca.merges())
"""))
    c.append(md(r"""
### Technique

**Two implementations, one answer.** On the 440 KB sample of corpus A, in 3,463 of the 3,836 merges (90%) more than one pair tied at the maximum count, and the first tie appeared at merge 56 (`scripts/medir_empates_bpe.py`). In those cases only the tie-breaking rule chooses the pair, and two implementations without a written rule may diverge. The didactic version wrote down the convention that reproduces the library's, and the two became identical, merge by merge and id by id, with a test of the package guarding it after every change. The cell above is the same check, on a small sample.
"""))

    c.append(md(r"""
## Part 4. How many symbols, and how to cut the text

### Theory

**Block 4.1, absent tokens.** For a BPE with vocabulary $V$, the share of learned tokens that never occur in the held-out text is $a(V) = \lvert\{t \ge 260 : t \notin \tau(\text{held-out})\}\rvert / (V - 260)$. Each absent token is a row of the embedding trained on very few examples (Land and Bartolo, 2024).

**Block 4.2, the project's rule.** $V^* = \max\{V \in \{1{,}024, 2{,}048, 4{,}096, 8{,}192\} : a(V) \le 0.10\}$, decided before any model is trained. Code in `escolher_vocabulario`. The held-out text used here is the validation partition, the same one that later measures bits per byte, and the rule only looks at which tokens occur in it, not at any model result.

**Block 4.3, the price of $V$.** In the convention of Kaplan et al. (2020), training costs about 6 FLOPs per parameter per symbol, 2 forward and 4 backward. Each layer has about $12 h^2$ parameters (attention and MLP), which gives $72\, n_c\, h^2$, attention over the window adds $6\, n_c\, L\, h$, and the output layer adds $6\, h\, V$:

$$C(V, L) = 72\, n_c\, h^2 + 6\, n_c\, L\, h + 6\, h\, V.$$

The count leaves out the input embedding, which is a table lookup. A larger $V$ lowers $\sigma$ but makes each symbol more expensive through the last term, and part 6 uses this formula to equalise the budget.

**Block 4.4, the pre-tokenization pattern.** The GPT-2 pattern was written for English, with contractions such as `'s` and `'re`. In period Portuguese, elision appears in "n'aquella" and "d'elle". A variant that keeps the elision whole in one piece is measured below, for comparison only, without changing the project's pattern.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
from slm.tokenizador import escolher_vocabulario, fracao_ausente
V_escolhido, ausentes = escolher_vocabulario(treino, valid)
print(" V     absent tokens  symbols per byte")
for V, a in ausentes.items():
    tok_V = BPEBiblioteca().treinar(treino, V)
    s = sum(len(tok_V.encode(t)) for t in valid) / B_valid
    print(f"{V:5d}  {a:13.3f}  {s:16.3f}", "<- chosen" if V == V_escolhido else "")
bpe = BPEBiblioteca().treinar(treino, V_escolhido)
bpe_repo = BPEBiblioteca.carregar("tokenizador/A")
print("\nequal to the repository's tokenizer:", bpe.merges() == bpe_repo.merges())
"""))
    c.append(code(r"""
PT_REGEX = r" ?\p{L}+(?:['’]\p{L}+)*| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
print("GPT-2:", regex.findall(GPT2_REGEX, "disse-lhe elle n'aquella hora"))
print("PT   :", regex.findall(PT_REGEX, "disse-lhe elle n'aquella hora"))
bpe_pt = BPEBiblioteca(regex_pt=PT_REGEX).treinar(treino, V_escolhido)
for nome, t in (("GPT-2", bpe), ("PT", bpe_pt)):
    s = sum(len(t.encode(x)) for x in valid) / B_valid
    print(f"{nome:5s} | symbols per byte {s:.4f} | absent tokens {fracao_ausente(t, valid):.3f}")
"""))
    c.append(md(r"""
### Technique

**A vocabulary too large for the text.** With $V = 8{,}192$, 12% of the tokens learned on corpus A never occur in validation, and the rule chooses $V = 4{,}096$. On corpus B it chose $V = 1{,}024$, with a held-out text of 174 KB against 540 KB for corpus A, and a smaller held-out text leaves more tokens unseen, which pushes $V$ down. In the project's sensitivity analysis, training with $V = 2{,}048$ or $8{,}192$ did not change the sign of the comparison between the two modes, so the rule serves to avoid waste, not to decide the result.

**Measure before adopting.** A new pre-tokenization pattern would only enter the project on the bits per byte of trained models, never on the symbol counts of the cell above.
"""))

    c.append(md(r"""
## Part 5. A tokenizer of your own in the contract

### Theory

Any tokenizer that honours the contract serves the package, even one with nothing of BPE in it. The example here is a proposal by the project's author, described in `docs/decisao_tokenizador_codigo.md`. The model does not learn on the text, but on a code language $C$, designed to carry more information per bit, in the style of Morse code, where the most frequent letter, "E", gets the shortest signal.

**Block 5.1, units.** The 4,096 most frequent pre-words of the training text (the pieces of the pre-tokenization of block 3.1) stay whole, and the others are cut by a morpheme rule written by hand for Portuguese (`experimentos/m10_codigo/morfema.py`), into stem, derivational suffix and inflection.

**Block 5.2, base-$A$ Huffman code.** Each unit $u$, with frequency $p_u$ in training, gets a codeword over an alphabet of $A$ symbols, prefix-free, of length $\ell_u$. The Huffman code minimises $\bar\ell = \sum_u p_u \ell_u$ among prefix-free codes, and $H_A \le \bar\ell < H_A + 1$, with $H_A = -\sum_u p_u \log_A p_u$. The table is derived from training and frozen before any model is trained.

**Block 5.3, escape.** A unit absent from the table is spelled out between two escapes, character by character, and by bytes when the character was never seen. Thus $\tau^{-1}(\tau(s)) = \mathrm{utf8}(s)$ for any text, and the contract holds.

**Block 5.4, the three fronts.** The proposal seeks at once fewer symbols per byte, more information per symbol and fewer bits per byte, and the three are tied by the identity nominal bits per byte $= \sigma \log_2 A$. A small alphabet gives cheap symbols and long sequences, and a large one the opposite. The size of the alphabet decides which front is won and which is paid for.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
import hashlib, json, time
from codigo_morse import CodigoMorse
t0 = time.time()
codigo = CodigoMorse.construir(treino, A=256, n_topo=4096, variante="prefixo")
pasta_codigo = tempfile.mkdtemp()
contrato = conferir_contrato(codigo, sonda, pasta_codigo)
print(f"code built in {time.time() - t0:.0f} s | inventory of {len(codigo.inventario)} units | V = {codigo.vocab_size}")
for teste, ok in contrato.items():
    print(f"{teste:32s} {'passes' if ok else 'FAILS'}")
congelado = json.load(open("tokenizador/m10_C1/metadados.json"))["sha256"]
print("table equal to the one frozen in the experiment:", hashlib.sha256(open(f"{pasta_codigo}/tabela.json", "rb").read()).hexdigest() == congelado)
"""))
    c.append(code(r"""
import math
from codigo_morse import unidades
for frase in (FRASE, "As desconfianças do sobrinho, e um Zwqx que ninguém viu."):
    us = unidades(frase, codigo.topo, "prefixo")
    print("units:", us)
    print("codewords:", [codigo.cod.get(u, "escape and spelling") for u in us])
print("code symbols per byte: training", round(sum(len(codigo.encode(x)) for x in treino) / B_treino, 3),
      "| validation", round(sum(len(codigo.encode(x)) for x in valid) / B_valid, 3))
print(" tokenizer     V  symbols per byte  nominal bits per byte")
for nome, t in (("bytes", bytes_tok), ("BPE", bpe), ("code", codigo)):
    s = sum(len(t.encode(x)) for x in valid) / B_valid
    print(f"{nome:8s} {t.vocab_size:6d}  {s:16.3f}  {s * math.log2(t.vocab_size - 4):21.2f}")
"""))
    c.append(md(r"""
### Technique

**The separator is expensive.** In Morse, the boundaries between letters are pauses. In the project's measurements, on the inventory of mixed units with the suffix approximation, marking each boundary with a symbol of its own made the code 62.8% more expensive in bits per byte than the prefix-free version (5.30 against 3.26), which is why the code of this part uses no separator.

**The inventory fits the training text.** The code is denser on training (0.339 symbols per byte) than on validation (0.401), because words outside the table pay an escape and spelling. The epoch factor of part 6 uses the training density, which is the one that consumes compute.

**Density is not learning.** In the context-free measurements, the BPE units have lower entropy per byte than the morpheme inventories tested, 2.75 bits against 2.84 for the best of them and 2.92 for the inventory of this part. Counting symbols does not say whether a model learns better on the code, and only the training of part 6 answers it.
"""))

    c.append(md(r"""
## Part 6. Training both architectures with the tokenizers built here

### Theory

**Block 6.1, equal budget.** A run with tokenizer $b$ costs $S_b \cdot E_b \cdot C(V_b, L)$ FLOPs, with $S_b$ training symbols and $E_b$ epochs. To spend the same as the reference BPE over $E_{ref}$ epochs,

$$E_b = E_{ref} \cdot \frac{S_{ref}\, C(V_{ref}, L)}{S_b\, C(V_b, L)}.$$

A less dense tokenizer has more symbols and gets fewer passes over the text.

**Block 6.2, what is compared.** The runs use the same network, the same data and the hyperparameters frozen in the project's protocol, and the measurement is that of block 1.3, bits per byte of the original text on validation. In the autoregressive mode the number is exact. In diffusion it is an upper bound, estimated here with a single draw of masks per window ($K = 1$), correct on average, and a disadvantage of diffusion says nothing about its true likelihood.
"""))
    c.append(md("### Application"))
    c.append(code(r"""
def custo(V, L=256, n_c=6, h=384):
    return 72 * n_c * h * h + 6 * n_c * L * h + 6 * h * V
toks = {"BPE": bpe, "bytes": bytes_tok, "code": codigo}
S = {nome: sum(len(t.encode(x)) for x in treino) for nome, t in toks.items()}
fator = {nome: S["BPE"] * custo(bpe.vocab_size) / (S[nome] * custo(t.vocab_size)) for nome, t in toks.items()}
plano = json.load(open("resultados/m10/m10_plano.json"))["bracos"]
for nome, braco in (("BPE", "B1"), ("bytes", "B0"), ("code", "C1")):
    print(f"{nome:5s} | {S[nome] / B_treino:.3f} symbols per training byte | epoch factor {fator[nome]:.3f}"
          f" | in the experiment {plano[braco]['fator_epocas']['simbolos']:.3f}")
"""))
    c.append(code(r"""
from slm.config import Config
from slm.corpus import tokenizar
from slm.janelas import janelar
from slm.modelo import ConfigRede, Transformer
from slm.treino import treinar
from slm.avaliacao import avaliar
HP = json.load(open("preregistro/m6_delta_min.json"))["escolha"]
L = 256

def corrida(nome, modo):
    tok = toks[nome]
    tr, va = tokenizar(treino_docs, tok), tokenizar(valid_docs, tok)
    janelas, _ = janelar([d["ids"] for d in tr], L)
    cfg = Config(modo=modo, vocab_size=tok.vocab_size, L=L, epocas=EPOCAS_REF * fator[nome], papel="aula", **HP[modo],
                 precisao="fp16" if DISPOSITIVO == "cuda" else "fp32",
                 atencao="efficient" if DISPOSITIVO == "cuda" else "math").validar()
    torch.manual_seed(cfg.semente)
    modelo = Transformer(ConfigRede(vocab_size=cfg.vocab_size, L=L, n_layer=cfg.n_layer, d=cfg.d, n_head=cfg.n_head,
                                    dropout=cfg.dropout)).to(DISPOSITIVO)
    t0 = time.time()
    treinar(cfg, modelo, janelas, DISPOSITIVO)
    modelo.eval()
    bpb = avaliar(modelo, va, modo, L, K=1, dispositivo=DISPOSITIVO)["bpb"]
    print(f"{nome:5s} {modo:15s} | {cfg.epocas:6.2f} epochs | {time.time() - t0:5.0f} s | validation bpb {bpb:.4f}")
    return modelo, bpb

resultados, modelos = {}, {}
for nome, modo in (("BPE", "autoregressivo"), ("BPE", "difusao"), ("bytes", "autoregressivo"), ("code", "autoregressivo")):
    modelos[(nome, modo)], resultados[(nome, modo)] = corrida(nome, modo)
"""))
    c.append(code(r"""
from slm.geracao import gerar_ar
from slm.tokenizador import ESPECIAIS
EOS = ESPECIAIS["[EOS]"]
amostras = gerar_ar(modelos[("code", "autoregressivo")], L, n=3, semente=1, dispositivo=DISPOSITIVO)
for linha in amostras.tolist():
    ids = linha[: linha.index(EOS)] if EOS in linha else linha
    codigo.invalidos = 0
    texto = codigo.decode(ids).decode("utf-8", "replace")
    print(f"[{codigo.invalidos} invalid codewords] {texto[:300]}\n")
"""))
    c.append(md(r"""
### Technique

**A codeword cut in half.** Generation stops at the end of the window, which can fall in the middle of a codeword. Decoding discards that remainder, and in the experiment, with the code of this part, 6 and 8 of 32 samples ended that way, at 16 and at 256 epochs. A codeword that does not exist in the table becomes the replacement character, and the count appears at the start of each sample above.

**One seed does not decide.** The numbers of this part come from one seed each. The full experiment used 3 seeds per arm, two forms of window and two budgets, and part 7 reads it.
"""))

    c.append(md(r"""
## Part 7. A fair reading

### Theory

**Block 7.1, paired difference.** With seeds paired across arms, the difference $\Delta_b = \mathrm{bpb}_b - \mathrm{bpb}_{ref}$ is estimated by the mean over seeds, with a 95% interval from the $t$ distribution. An interval entirely above zero is a disadvantage of arm $b$.

**Block 7.2, criterion fixed beforehand.** The preregistered reading, written and frozen before the runs, compared the code with BPE. With 39 of 54 runs seen, the question became the viability of the code, and that amendment is recorded as made after seeing data, with the original reading reported alongside (`docs/decisao_tokenizador_codigo.md`).
"""))
    c.append(md("### Application"))
    c.append(code(r"""
res_m10 = json.load(open("resultados/m10/m10_resultado.json"))
fmt = lambda v: f"{v['media']:+.3f} [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]"
print("difference to BPE in bits per byte, mean of 3 seeds and 95% interval")
print("arm    equal in       16 epochs                256 epochs")
for braco, nome in (("C1", "code"), ("B0", "bytes")):
    for janela, rotulo in (("simbolos", "symbols"), ("bytes", "bytes")):
        e16, e256 = (res_m10[f"autoregressivo|e{p}|{braco}-{janela}"] for p in (16, 256))
        print(f"{nome:6s} {rotulo:13s}   {fmt(e16):24s} {fmt(e256)}")
for nome in ("code", "bytes"):
    aqui = resultados[(nome, "autoregressivo")] - resultados[("BPE", "autoregressivo")]
    print(f"in this notebook, {nome}, equal window in symbols, {EPOCAS_REF:g} BPE epochs, 1 seed: {aqui:+.3f}")
"""))
    c.append(md(r"""
### Technique

**The order can flip with the budget.** At equal compute, the byte model was the worst arm at 16 BPE epochs and the best at 256, about 1 bit per byte below BPE. The comparison is against a BPE that worsened from 1.954 to 2.849 bits per byte between the two budgets, because it memorised its training text. Equalising compute makes the byte model pass over the text only 0.32 times the BPE epochs, and it memorises less. That explanation was not measured separately, and the control is left to the project's paper.

**The result of the code.** By the original reading, the code of this part loses to BPE at both budgets, by 0.745 bits per byte at the best point, and the three variants tested in the experiment lose by 0.64 to 0.76. By the viability criteria, it works. The round trip is exact, the generations decode back to text and, at the 16-epoch point, the model sits below the context-free entropy of its own units (2.70 against 2.92 bits per byte). At 256 epochs it sits above, at 3.84, because it overfits, like BPE. A negative result measured with care is worth as much as a positive one, and it shows which way not to go.
"""))
    c.append(md(r"""
## Exercises

1. In part 4, change the grid of $V$ to `(512, 1024, 2048, 4096, 8192, 16384)`. Where does the share of absent tokens pass 10%, and what does that say about the size of the corpus?
2. In part 5, build the code with `A=4096` instead of 256. What happens to symbols per byte and to nominal bits per byte?
3. Write a tokenizer of your own, for example one that encodes each character by its code point number, pass it through `conferir_contrato` and train with it in part 6.
4. In part 6, equalise the context in bytes instead of the window in symbols, with `L = round(256 * σ_b / σ_BPE)`, and recompute the epoch factor with `custo(V, L)` for that window, because the attention term changes. Does the order between the tokenizers change?
5. Why does part 6 compare bits per byte and not the mean loss per symbol of each run?

## References

The full list, with pinned versions and licences, is in `referencias/REFERENCIAS.md`, in the repository.
"""))
    return c


def main():
    SAIDA.mkdir(exist_ok=True)
    nb = {"cells": celulas(),
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"c{i:03d}"
    (SAIDA / ARQUIVO).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"en/{ARQUIVO}: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
