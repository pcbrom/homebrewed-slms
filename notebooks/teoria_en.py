"""English counterpart of notebooks/teoria.py: the mathematical building blocks of the notebooks.

mapa(modo) is the global map placed after the pipeline figure. TEORIA maps the box number to a
list of (kind, text, modes) cells placed right after the box introduction, with kind "md" or
"code" and modes "ambos", "autoregressivo" or "difusao". Block titles are written "**Block #,"
and numbered per notebook by the generator. Each block gives the definition, the derivation when
there is one, the package function that implements it and the link to the next box, and code cells
redo most blocks by hand and check them against the package. Structure, keys, cell order, modes
and code identifiers mirror notebooks/teoria.py exactly, and only the prose, the printed strings
and the comments are in English.
"""

AMBOS, AR, DIF = "ambos", "autoregressivo", "difusao"


def mapa(modo: str) -> str:
    if modo == DIF:
        transforma, entrada, condicional = r"\text{mask}", "z", r"p_\theta(\cdot \mid z)"
        linha_caixa4 = "| $l$, $S$, $z$ | number of masks, masked set, masked window | box 4, `slm/mascaramento.py` |"
        linha_perda = r"| $\hat\ell_{dif}(w)$ | NELBO estimate for one window, in nats | box 6, `slm/perda.py` |"
        perda, geracao = r"\hat\ell_{dif}(w)", "from masks to text, revealing positions in random order"
    else:
        transforma, entrada, condicional = r"\text{shift}", "x", r"p_\theta(w_i \mid w_{<i})"
        linha_caixa4 = "| $x$ | window shifted by one BOS, read under a causal mask | box 4, `slm/perda.py` |"
        linha_perda = r"| $\ell_{AR}(w)$ | negative log-likelihood of one window, in nats | box 6, `slm/perda.py` |"
        perda, geracao = r"\ell_{AR}(w)", "one token at a time, from left to right"
    return rf"""
## Mathematical map, from text to measurement

Each box takes in one mathematical object and hands on another. The table fixes the notation used throughout the notebook, and the chain just below shows how the blocks fit together.

| Symbol | Meaning | Where it arises |
|---|---|---|
| $d$, $B_d$ | a document (chapter or law) and its length in bytes in UTF-8, after NFC | box 1, `slm/dados.py` |
| $\tau(d) = (t_1, \dots, t_n)$ | token ids of the document, with $t_i \in \{{4, \dots, V-1\}}$ | box 2, `slm/tokenizador.py` |
| $w \in \{{0, \dots, V-1\}}^L$ | a window of $L$ tokens | box 3, `slm/janelas.py` |
{linha_caixa4}
| $f_\theta$, $p_\theta$ | network with parameters $\theta$ and the conditional distributions it defines | box 5, `slm/modelo.py` |
{linha_perda}
| $\mathrm{{NLL}}(d)$, $\mathrm{{bpb}}$ | loss of a document and bits per byte | box 8, `slm/avaliacao.py` |
| CI | confidence interval for the bits per byte | box 9, `slm/estatistica.py` |

Some symbols carry two meanings, and the context keeps them apart. $d$ is both a document and the width of the network, $S$ is both the masked set and the GradScaler scale, $T$ is both the number of generation steps and the temperature, $K$ is both the key matrix and the number of estimates per window, and $t$ is diffusion time whereas $t_i$ is a token. The network's logits are written $o$ so that they cannot be confused with the masked window $z$.

$$
\underbrace{{d}}_{{\text{{box 1}}}}
\xrightarrow{{\ \tau\ }}
\underbrace{{(t_1,\dots,t_n)}}_{{\text{{box 2}}}}
\xrightarrow{{\ \oplus\,\mathrm{{EOS}},\ \text{{cut at }} L\ }}
\underbrace{{w}}_{{\text{{box 3}}}}
\xrightarrow{{\ {transforma}\ }}
\underbrace{{{entrada}}}_{{\text{{box 4}}}}
\xrightarrow{{\ f_\theta\ }}
\underbrace{{{condicional}}}_{{\text{{box 5}}}}
\xrightarrow{{\ -\log\ }}
\underbrace{{{perda}}}_{{\text{{box 6}}}}
\xrightarrow{{\ \sum_{{w \in d}}\ }}
\underbrace{{\mathrm{{NLL}}(d)}}_{{\text{{box 8}}}}
\xrightarrow{{\ \sum_d / (\ln 2 \sum_d B_d)\ }}
\underbrace{{\mathrm{{bpb}} \pm \mathrm{{CI}}}}_{{\text{{boxes 8 and 9}}}}
$$

Training (box 6) closes a loop over this chain, because the gradient of the loss with respect to $\theta$ flows back into box 5 and changes $f_\theta$. Generation (box 7) runs through the same $p_\theta$ in the opposite direction, {geracao}.
"""


TEORIA = {
    0: [("md", r"""
### Building blocks

**Block #, parameters excluding the embedding.** Each Transformer block has the attention projections $Q$, $K$ and $V$ plus the output projection, which together come to $4d^2$ weights, and an MLP of width $4d$, with $8d^2$ weights. Hence $N = 12\, n_{layer}\, d^2$, ignoring biases and normalisations. Code in `scripts/dimensionar.py`.

**Block #, cost of one training token.** The forward pass costs $2N$ in the matrix products of the layers, $2\, n_{layer}\, L\, d$ in the attention product $QK^\top$ and $2\, d\, V$ in the output projection. This is the convention of Kaplan et al. (2020). PaLM (Chowdhery et al., 2022) also counts the product of the attention weights with the values, which doubles the attention term. Backpropagation costs roughly twice as much as the forward pass, so that, under Kaplan's convention,

$$C_{train} = 6N + 6\, n_{layer}\, L\, d + 6\, d\, V.$$

**Block #, throughput and time.** With peak $P$ and utilisation MFU, the throughput is $P \cdot \mathrm{MFU} / C_{train}$ tokens per second, and the time needed to see $D$ tokens is $D$ divided by it. This is the number that decides how many epochs and how many seeds fit within the budget of boxes 6 and 9.
""", AMBOS), ("code", r"""
n_layer, d, L_, V_ = 6, 384, 256, 4096
N = 12 * n_layer * d * d
C = 6 * N + 6 * n_layer * L_ * d + 6 * d * V_
print(f"N = {N:,} | C_train = {C / 1e6:.1f} MFLOP per token | {C / (6 * N) - 1:.1%} above 6N")
""", AMBOS)],
    1: [("md", r"""
### Building blocks

**Block #, normalisation.** Every text goes through $u \mapsto \mathrm{NFC}(u)$, the canonical composed form of Unicode. Once it has been applied, every accented character in Portuguese is a single code point.

**Block #, bytes.** $B_d = |\mathrm{UTF8}(\mathrm{NFC}(d))|$. It is the denominator of the bits per byte in box 8, which is why it is counted here, once and only once, and never on the text that the tokenizer reconstructs.

**Block #, partition.** Each work is a sequence of chapters $c_1, \dots, c_m$. The split sends an opening block to training, one chapter to a margin, a contiguous block to validation, another chapter to a margin and the closing block to test, with every held-out block holding at least 5 chapters and 30 KB. *Papéis avulsos*, with 12 short stories, only has room for the validation block and stays out of the test set. The partition is a function of the document and is fixed before any training takes place. Code in `slm/dados.py`.
""", AMBOS), ("code", r"""
import unicodedata
from slm.corpus import ler_documentos
docs = ler_documentos(CORPUS)
d0 = docs[0]
print(d0["doc_id"], "| already in NFC:", unicodedata.is_normalized("NFC", d0["texto"]),
      "| B_d recounted:", len(d0["texto"].encode("utf-8")), "= stored bytes:", d0["bytes"])
""", AMBOS)],
    2: [("md", r"""
### Building blocks

**Block #, alphabet.** The vocabulary starts with the 256 bytes. Since every text is a sequence of bytes, every text can be represented, and there is no such thing as an unknown token.

**Block #, pair counting.** After pre-tokenization, the corpus becomes a multiset of words $w$ with frequency $f(w)$. For each adjacent pair $(a, b)$, $c(a,b) = \sum_w f(w)\, \#\{(a,b) \text{ adjacent in } w\}$.

**Block #, merge.** The pair $(a^*, b^*) = \arg\max c(a,b)$ is chosen, with ties broken by the smallest pair of ids, the symbol $a^* b^*$ is created with the next id and every word is rewritten. The process repeats until there are $V$ symbols.

**Block #, encoding and decoding.** Encoding applies the merges in the order in which they were learned, and decoding concatenates the bytes of each symbol. Because every merge concatenates bytes, $\mathrm{decode}(\mathrm{encode}(s)) = s$ for any $s$. Code in `BPEDidatico.treinar`, `encode` and `decode`, in `slm/tokenizador.py`. The output of this box, $\tau(d)$, is the input to box 3.
""", AMBOS), ("code", r"""
import collections, regex
from slm.tokenizador import GPT2_REGEX, ORDEM_BYTES, BPEDidatico
id_byte = {bytes([b]): i for i, b in enumerate(ORDEM_BYTES)}
amostra2 = [d["texto"] for d in docs if d["particao"] == "treino"][:5]
palavras = collections.Counter(tuple(bytes([b]) for b in p.encode()) for t in amostra2 for p in regex.findall(GPT2_REGEX, t))
cont = collections.Counter()
for p, f in palavras.items():
    for par in zip(p, p[1:]):
        cont[par] += f
a_mao = max(cont, key=lambda par: (cont[par], -id_byte[par[0]], -id_byte[par[1]]))
pelo_pacote = BPEDidatico().treinar(amostra2, 265).merges[0]
print("first merge by hand:", a_mao, "| by the package:", pelo_pacote, "| equal:", a_mao == pelo_pacote, "| count:", cont[a_mao])
""", AMBOS)],
    3: [("md", r"""
### Building blocks

**Block #, end of document.** $s(d) = \tau(d) \oplus (\text{EOS})$, of length $n + 1$.

**Block #, windows.** There are $N_w(d) = \lceil (n+1)/L \rceil$ windows, window $k$, counted from zero, holding the tokens $s_{kL+1}, \dots, s_{(k+1)L}$, and the last one is padded with EOS. No context crosses from one window into another.

**Block #, padding.** $\phi = \dfrac{\sum_d \big(L\, N_w(d) - n_d - 1\big)}{\sum_d L\, N_w(d)}$ is the fraction of positions that cost computation without carrying any text. Code in `slm/janelas.py`. Each window $w$ moves on to box 4.
""", AMBOS), ("code", r"""
import math
from slm.janelas import janelas_documento
ids0 = tok.encode(docs[0]["texto"])
n = len(ids0)
Nw = math.ceil((n + 1) / 256)
a_mao = ids0 + [2] * (Nw * 256 - n)
print("n =", n, "| N_w = ceil((n+1)/L) =", Nw, "| matches the package:", janelas_documento(ids0, 256).flatten().tolist() == a_mao)
""", AMBOS)],
    4: [("md", r"""
### Building blocks

**Block #, absorbing forward process.** With $\alpha_t = 1 - t$, each token survives with probability $1 - t$ or turns into `[MASK]` with probability $t$, independently of the others, that is, $q(z_i \mid w_i) = (1-t)\,\delta_{w_i}(z_i) + t\,\delta_{\mathrm{MASK}}(z_i)$.

**Block #, from time to the number of masks.** Given $t$, the number of masks is $l \sim \mathrm{Bin}(L, t)$, and given $l$ the set $S$ is uniform over the $\binom{L}{l}$ subsets. Box 6 shows why it is enough to draw $l$ uniformly from $\{1, \dots, L\}$.

**Block #, stratified sampling.** With $U \sim U(0,1)$ shared by the whole batch and a permutation $\pi$, $u_b = (\pi(b) - 1 + U)/B$ and $l_b = 1 + \lfloor u_b L \rfloor$. Each $u_b$ falls in a different stratum of $[0,1)$, which introduces no bias and generally reduces the variance of the batch mean relative to independent sampling, although that reduction is not guaranteed for every function of $l$.

**Block #, a uniform set of size $l$.** $L$ uniform numbers are drawn, and the $l$ positions of lowest rank are taken as $S$. By symmetry, every subset of size $l$ is equally likely. Then $z = w$ outside $S$ and $z = \mathrm{MASK}$ on $S$. Code in `sortear_l`, `mascara_de_l` and `mascarar`, in `slm/mascaramento.py`. The masked window $z$ moves on to the network in box 5.
""", DIF), ("code", r"""
from slm.mascaramento import sortear_l, mascara_de_l
g4 = torch.Generator().manual_seed(1)
l4 = sortear_l(8, 256, g4)
m4 = mascara_de_l(l4, 256, g4)
print("sampled l:", l4.tolist())
print("masks per row equal to l:", torch.equal(m4.sum(1), l4), "| one l per stratum of 1/8:", sorted(((l4 - 1) * 8 // 256).tolist()))
""", DIF), ("md", r"""
### Building blocks

**Block #, chain rule.** Every distribution over windows can be written as $p(w) = \prod_{i=1}^{L} p(w_i \mid w_{<i})$, with no approximation at all. The autoregressive mode models each factor.

**Block #, shifted input.** The input is $x = (\text{BOS}, w_1, \dots, w_{L-1})$, and the output at position $i$ predicts $w_i$. `[BOS]` stands in for the token that does not yet exist at the start. Code in `entrada_ar`, in `slm/perda.py`.

**Block #, causal mask.** In attention, $M_{ij} = 0$ if $j \le i$ and $-\infty$ otherwise. Added to the scores before the softmax, it sets the weight of every future position to zero, so the prediction of $w_i$ sees only $\text{BOS}, w_1, \dots, w_{i-1}$. The input $x$ moves on to the network in box 5.
""", AR), ("code", r"""
from slm.perda import entrada_ar
w4 = janelas_treino[:1, :8]
x4 = entrada_ar(w4)
print("window :", w4.tolist()[0])
print("input  :", x4.tolist()[0])
print("shifted by one BOS:", x4[0, 0].item() == 1 and torch.equal(x4[0, 1:], w4[0, :-1]))
""", AR)],
    5: [("md", r"""
### Building blocks

**Block #, input.** $h^{(0)}_i = E[x_i] + P_i$, where $E \in \mathbb{R}^{V \times d}$ is the embedding matrix and $P \in \mathbb{R}^{L \times d}$ holds the learned positions.

**Block #, attention.** With $u = \mathrm{LN}(h)$, $Q = u W_Q$, $K = u W_K$ and $V_a = u W_V$, split into $H$ heads of size $d_h = d/H$,

$$A = \mathrm{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_h}} + M\Big) V_a, \qquad h \leftarrow h + A W_O.$$

$M = 0$ in diffusion and causal in the autoregressive mode. That is the only difference between the two modes inside the network.

**Block #, MLP.** $h \leftarrow h + W_2\, \mathrm{GELU}(W_1\, \mathrm{LN}(h))$, of width $4d$.

**Block #, output.** $o = \mathrm{LN}(h^{(n_{layer})})\, E^\top \in \mathbb{R}^{L \times V}$, with the same $E$ as at the input. The logit of `[MASK]` is fixed at $-\infty$, and $p_\theta(\cdot \mid \text{input})_i = \mathrm{softmax}(o_i)$. Code in `Bloco.forward` and `Transformer.forward`, in `slm/modelo.py`. Box 6 turns $p_\theta$ into a loss.

The cell below redoes the forward pass of a small network with explicit matrices and checks the result against the module.
""", AMBOS), ("code", r"""
import torch.nn.functional as F
from slm.modelo import ConfigRede, Transformer
torch.manual_seed(0)
mini = Transformer(ConfigRede(vocab_size=tok.vocab_size, L=16, n_layer=1, d=32, n_head=4)).eval()
x5 = janelas_treino[:2, :16]
b = mini.blocos[0]
causal5 = MODO == "autoregressivo"
with torch.no_grad():
    h = mini.emb(x5) + mini.pos[:16]
    q, k, v = b.qkv(b.ln1(h)).view(2, 16, 3, 4, 8).permute(2, 0, 3, 1, 4)
    escores = q @ k.transpose(-1, -2) / 8 ** 0.5
    if causal5:
        escores = escores.masked_fill(~torch.ones(16, 16, dtype=torch.bool).tril(), float("-inf"))
    h = h + b.proj((torch.softmax(escores, dim=-1) @ v).transpose(1, 2).reshape(2, 16, 32))
    h = h + b.out(F.gelu(b.fc(b.ln2(h))))
    z5 = mini.ln(h) @ mini.emb.weight.T
    z5[..., 3] = float("-inf")
    print("forward pass by hand matches the module:", torch.allclose(z5, mini(x5, causal5), atol=1e-5))
""", AMBOS)],
    6: [("md", r"""
### Building blocks

**Block #, cross-entropy.** For logits $o$ and target $y$, $-\log p(y) = -o_y + \log \sum_v e^{o_v}$. The gradient with respect to $o$ is $p - e_y$, so the network is pushed to raise the logit of the target and to lower the others in proportion to the probability each of them receives.

**Block #, autoregressive loss.** $\ell_{AR}(w) = \sum_{i=1}^{L} -\log p_\theta(w_i \mid \text{BOS}, w_{<i})$. By the chain rule of box 4, this is exactly $-\log p_\theta(w)$, with neither a bound nor an approximation. During training, the batch loss is the mean per token. Code in `perda_ar`, in `slm/perda.py`.
""", AR), ("md", r"""
### Building blocks

**Block #, cross-entropy.** For logits $o$ and target $y$, $-\log p(y) = -o_y + \log \sum_v e^{o_v}$. The gradient with respect to $o$ is $p - e_y$, so the network is pushed to raise the logit of the target and to lower the others in proportion to the probability each of them receives.

**Block #, continuous-time diffusion NELBO.** With the linear schedule,

$$\mathrm{NELBO}(w) = \int_0^1 \frac{1}{t}\, \mathbb{E}_{z \sim q_t(\cdot \mid w)}\Big[\sum_{i:\, z_i = \mathrm{MASK}} -\log p_\theta(w_i \mid z)\Big]\, dt \;\ge\; -\log p_\theta(w).$$

In diffusion the network defines only conditionals. The likelihood of the model is the average, over the reveal orders $\sigma$, of the product of the conditionals along each order, $p_\theta(w) = \mathbb{E}_\sigma\big[\prod_{i=1}^{L} p_\theta(w_{\sigma(i)} \mid w_{\sigma(<i)})\big]$, and the NELBO is the average of $-\log \prod$ over the same orders (Ou et al., 2024, Theorem 2). By Jensen's inequality, it bounds $-\log p_\theta(w)$ from above.

**Block #, from the integral to the sum.** Let $F_l = \mathbb{E}_{|S| = l}\big[\sum_{i \in S} -\log p_\theta(w_i \mid z)\big]$. Since $l \sim \mathrm{Bin}(L, t)$ and every term is non-negative, Tonelli's theorem allows the sum and the integral to be swapped,

$$\mathrm{NELBO} = \sum_{l=1}^{L} \binom{L}{l} \Big(\int_0^1 t^{l-1}(1-t)^{L-l}\, dt\Big) F_l = \sum_{l=1}^{L} \binom{L}{l} \mathrm{B}(l, L-l+1)\, F_l = \sum_{l=1}^{L} \frac{F_l}{l},$$

because $\binom{L}{l}\mathrm{B}(l, L-l+1) = \frac{L!}{l!(L-l)!} \cdot \frac{(l-1)!(L-l)!}{L!} = \frac{1}{l}$.

**Block #, estimator.** Hence $\mathrm{NELBO} = \mathbb{E}_{l \sim U\{1..L\}}\big[\tfrac{L}{l} F_l\big]$, and the estimator for one window is $\hat\ell_{dif}(w; l, S) = \frac{L}{l}\sum_{i \in S} -\log p_\theta(w_i \mid z)$. During training, the batch loss is the mean of $\hat\ell / L$. Code in `perda_difusao`, in `slm/perda.py`.
""", DIF), ("md", r"""
**Block #, optimisation.** The gradient $g = \nabla_\theta\, \text{loss}$, clipped to norm 1.0, feeds AdamW, with $m \leftarrow \beta_1 m + (1-\beta_1) g$, $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ and $\theta \leftarrow \theta - \eta_t\big(\hat m / (\sqrt{\hat v} + \epsilon) + \lambda \theta\big)$, where $\hat m$ and $\hat v$ are bias-corrected and the decay $\lambda$ applies only to the weight matrices, leaving out biases, LayerNorm and positions. The rate $\eta_t$ rises linearly to the peak $\eta$ during warm-up and then follows $\eta_t = \eta_{min} + \tfrac{1}{2}(\eta - \eta_{min})(1 + \cos \pi\, \text{progress})$. In fp16 the loss is multiplied by a scale $S$ before backpropagation, the gradient is divided by $S$ before the step, and a step containing an infinite value is skipped while $S$ is reduced. Code in `slm/treino.py`. The trained parameters return to box 5 and go on to serve boxes 7 and 8.
""", AMBOS), ("code", r"""
from slm.perda import perda_ar
w6 = torch.randint(4, tok.vocab_size, (1, 16))
with torch.no_grad():
    z6 = mini(torch.cat([torch.ones(1, 1, dtype=torch.long), w6[:, :-1]], 1), True)
    a_mao = (-torch.log_softmax(z6.double(), -1).gather(-1, w6[..., None])).sum()
    print("l_AR by hand:", round(float(a_mao), 4), "| package:", round(float(perda_ar(mini, w6)[1]), 4))
zz = torch.zeros(5, requires_grad=True)
F.cross_entropy(zz[None], torch.tensor([2])).backward()
print("cross-entropy gradient = p - e_y:", torch.allclose(zz.grad, torch.softmax(zz.detach(), 0) - torch.eye(5)[2]))
""", AR), ("code", r"""
from scipy.special import beta, comb
from slm.mascaramento import mascarar
from slm.perda import perda_difusao
print("identity C(L,l) B(l, L-l+1) = 1/l:", all(abs(comb(256, l) * beta(l, 257 - l) - 1 / l) < 1e-9 for l in (1, 2, 17, 128, 256)))
w6 = torch.randint(4, tok.vocab_size, (1, 16))
m6 = torch.zeros(1, 16, dtype=torch.bool)
m6[0, [2, 5, 11]] = True
with torch.no_grad():
    lp = torch.log_softmax(mini(mascarar(w6, m6), False).double(), -1).gather(-1, w6[..., None])[..., 0]
    a_mao = 16 / 3 * float(-lp[m6].sum())
    print("(L/l) sum of the masked CE by hand:", round(a_mao, 4), "| package:", round(float(perda_difusao(mini, w6, m6)[1]), 4))
zz = torch.zeros(5, requires_grad=True)
F.cross_entropy(zz[None], torch.tensor([2])).backward()
print("cross-entropy gradient = p - e_y:", torch.allclose(zz.grad, torch.softmax(zz.detach(), 0) - torch.eye(5)[2]))
""", DIF)],
    7: [("md", r"""
### Building blocks

**Block #, Gumbel-max.** If $g_v = -\log(-\log u_v)$ with $u_v \sim U(0,1)$ independent, then $\arg\max_v (\log p_v + g_v)$ has distribution $p$. Sampling turns into an argmax, which is why the precision of $u$ matters.
""", AMBOS), ("md", r"""
**Block #, ancestral sampling.** Generation starts from $x = (\text{BOS})$, draws $w_1 \sim p_\theta(\cdot \mid \text{BOS})$, appends $w_1$ to the input and repeats, until `[EOS]` or the end of the window. It is the chain rule traversed in the direction of generation. Code in `sortear` and `gerar_ar`, in `slm/geracao.py`. The samples move on to box 10.
""", AR), ("md", r"""
**Block #, reveal schedule.** In the step from $t = k/T$ to $s = (k-1)/T$, each masked position is revealed with probability $\frac{\alpha_s - \alpha_t}{1 - \alpha_t} = \frac{t - s}{t} = \frac{1}{k}$, and at the last step ($k = 1$) all of them are revealed. Code in `sortear` and `gerar_difusao`, in `slm/geracao.py`. The samples move on to box 10.
""", DIF), ("code", r"""
from slm.geracao import sortear
p7 = torch.tensor([0.5, 0.3, 0.15, 0.05])
s7 = sortear(torch.log(p7).expand(200000, 4), torch.Generator().manual_seed(0))
freq = (torch.bincount(s7, minlength=4) / len(s7)).tolist()
print("target p:", [round(x, 3) for x in p7.tolist()], "| Gumbel-max frequencies:", [round(x, 3) for x in freq])
""", AMBOS)],
    8: [("md", r"""
### Building blocks

**Block #, document loss.** $\mathrm{NLL}(d) = \sum_{w \in d} \ell_{AR}(w)$, summing over the windows of box 3. In the autoregressive mode this value is exact.

**Block #, bits per byte.** $\mathrm{bpb}(D) = \dfrac{\sum_{d \in D} \mathrm{NLL}(d)}{\ln 2 \cdot \sum_{d \in D} B_d}$. The $\ln 2$ converts nats into bits, and dividing by bytes rather than tokens makes the figure comparable across tokenizers.
""", AR), ("md", r"""
### Building blocks

**Block #, document loss.** $\mathrm{NLL}(d) = \sum_{w \in d} \hat\ell_{dif}(w)$, summing over the windows of box 3, each one taken as the mean of $K$ estimates with stratified $l$. With $K = 1$, the value is an unbiased estimate of an upper bound on the negative log-likelihood, and it carries Monte Carlo noise.

**Block #, bits per byte.** $\mathrm{bpb}(D) = \dfrac{\sum_{d \in D} \mathrm{NLL}(d)}{\ln 2 \cdot \sum_{d \in D} B_d}$. The $\ln 2$ converts nats into bits, and dividing by bytes rather than tokens makes the figure comparable across tokenizers.
""", DIF), ("md", r"""
**Block #, temperature.** With $\beta = 1/T$, the mean validation loss is $\mathcal{L}(\beta) = \frac{1}{N}\sum_n \big(-\beta o_{n,y_n} + \log \sum_v e^{\beta o_{n,v}}\big)$. Its derivatives are $\mathcal{L}'(\beta) = \frac{1}{N}\sum_n (\mathbb{E}_{p_\beta}[o_n] - o_{n,y_n})$ and $\mathcal{L}''(\beta) = \frac{1}{N}\sum_n \mathrm{Var}_{p_\beta}[o_n] \ge 0$, so $\mathcal{L}$ is convex and every stationary point is a global minimum. Convexity is not enough for the pure Newton method to converge, nor does it guarantee that the minimum is finite, since $\beta$ grows without bound when every prediction is correct. The package applies the step $\beta \leftarrow \beta - \mathcal{L}'/\mathcal{L}''$, halves the step for as long as it would take $\beta$ to zero or below, and stops once the step falls below $10^{-10}$ or after 100 iterations.

**Block #, ECE.** With confidence bins $b$, $\mathrm{ECE} = \sum_b \frac{n_b}{N}\, |\bar c_b - \bar a_b|$. Code in `slm/avaliacao.py` and `slm/calibracao.py`. The bits per byte of each document move on to box 9. The cell below checks the aggregation into bits per byte starting from the nats per document returned by the package.
""", AMBOS), ("code", r"""
import math
from slm.avaliacao import avaliar
tres = validacao[:3]
r8 = avaliar(modelo, tres, MODO, L, dispositivo=DISPOSITIVO)
a_mao = sum(x["nats"] for x in r8["documentos"]) / (math.log(2) * sum(d["bytes"] for d in tres))
print("bpb by hand:", round(a_mao, 5), "| package:", round(r8["bpb"], 5))
""", AMBOS)],
    9: [("md", r"""
### Building blocks

**Block #, bootstrap by work.** Each replicate resamples the works with replacement, recomputes the bits per byte as a ratio of sums and stores the value. The interval runs from the 2.5th to the 97.5th percentile of the replicates. Resampling whole works respects the dependence between chapters of the same work.

**Block #, seeds.** A single run cannot say how much the result varies with the initialisation and the order of the data. The project protocol trains several seeds and resamples at two levels, works and seeds. With $\hat\sigma$ estimated from $m$ pilot seeds, the upper bound is $\sigma_{sup} = \hat\sigma\sqrt{(m-1)/\chi^2_{0.025;\,m-1}}$, and the number of seeds is the smallest $n$ with $n \ge (t_{0.975;n-1} + t_{0.8;n-1})^2 (\sigma_{sup}/\Delta_{min})^2$.

**Block #, comparison between the modes.** Seed $s$ fixes the initialisation and the data order in both modes, and the paired difference is $\Delta_s = \mathrm{bpb}_{dif,s} - \mathrm{bpb}_{AR,s}$. If the interval for $\Delta$ lies below zero, diffusion is better. If it lies above zero, the autoregressive mode beats the diffusion bound, which says nothing about the true likelihood of diffusion. If it contains zero, the result is inconclusive. This reading needs the runs of both notebooks, and the result of the full protocol is in `docs/`, in the repository. Code in `slm/estatistica.py`.
""", AMBOS), ("code", r"""
from slm.estatistica import fator_sigma_sup, numero_sementes
print("sigma_sup factor with 10 seeds:", round(fator_sigma_sup(10), 4))
for r in (0.5, 0.75, 1.0):
    print(f"sigma/Delta_min = {r}: n =", numero_sementes(r, 1.0, m=10).n)
""", AMBOS)],
    10: [("md", r"""
### Building blocks

**Block #, lexicon rate.** $\rho(a) = \frac{|\{x \in \text{words}(a) : x \in \Lambda_{train}\}|}{|\text{words}(a)|}$, where $\Lambda_{train}$ is the lexicon of the training partition.

**Block #, longest copied span.** $c(a) = \max\{k : \text{some span of } k \text{ words of } a \text{ occurs in the training text}\}$. The search hashes $k$-grams and runs a binary search on $k$, because whenever a span of $k$ words occurs all of its sub-spans occur as well, and the property is therefore monotone in $k$.

**Block #, deletion curve.** With the decision $\hat y$ and $p_0 = p(\hat y \mid \text{full context})$, the importance of word $j$ is $p_0 - p(\hat y \mid \text{context without } j)$. Removing the $\mathrm{round}(f n)$ most important words gives $C(f) = \min\big(p(\hat y \mid \text{remaining context})/p_0, 1\big)$, and $\mathrm{AUC} = \int_0^1 C(f)\, df$ by the trapezoidal rule. $C(1)$ measures the intrinsic bias.

**Block #, Spearman.** $\rho_S$ is the Pearson correlation between the ranks of $p_0$ and the ranks of the AUC, item by item. Code in `slm/leitura.py` and `slm/delecao.py`.
""", AMBOS), ("code", r"""
import numpy as np
from scipy.stats import rankdata, spearmanr
x10, y10 = np.array([0.2, 0.5, 0.1, 0.9]), np.array([0.7, 0.4, 0.8, 0.3])
print("Spearman by hand:", round(float(np.corrcoef(rankdata(x10), rankdata(y10))[0, 1]), 4),
      "| scipy:", round(float(spearmanr(x10, y10).statistic), 4))
grade = np.linspace(0, 1, 11)
C10 = np.array([1, .6, .5, .45, .4, .38, .36, .35, .34, .33, .33])
print("AUC by the trapezoidal rule:", round(float(np.trapezoid(C10, grade)), 4))
""", AMBOS)],
}
