"""English counterpart of notebooks/tecnica.py: the technique sections of the notebooks.

The module holds the problems met in the project, the solutions adopted and the measurement that
revealed each problem. TECNICA maps the box number to a list of (kind, text, modes) cells, with kind
"md" or "code" and modes "ambos", "autoregressivo" or "difusao", exactly as in the Portuguese
module. Code cells keep every identifier unchanged, so they share variables with the other cells,
and only printed strings and comments are translated. The project documents in docs/ hold the full
record.
"""

AMBOS, AR, DIF = "ambos", "autoregressivo", "difusao"

TECNICA = {
    0: [("md", r"""
**The short formula misleads for a small model.** The usual cost estimate, $6N$ operations per token, ignores attention and the output layer over the vocabulary. In a model with 4 layers, $d = 256$ and a vocabulary of 8 thousand tokens, these two terms add 73% to the cost, 65 points from the output layer and 8 from attention. `scripts/dimensionar.py` uses the full formula.
""", AMBOS), ("md", r"""
**Estimate, then measure.** The fraction of peak throughput that training actually achieves (MFU) cannot be read off the spec sheet. On an RTX 5090 capped at 500 W, the autoregressive mode processed 1.13 million tokens per second in fp16. That gives an MFU of 0.41 when attention counts only the $QK^\top$ product, the convention of Kaplan et al. (2020), and 0.43 when it also counts the product with the values, the PaLM convention (Chowdhery et al., 2022). The 209.5 TFLOPS peak in the denominator comes from secondary sources and has not been checked against NVIDIA's technical documentation. At the measured throughput, the research plan would fit into about 5.6 hours of training, excluding evaluation, whereas the estimate made without measurement ranged from 7.5 to 45 hours.
""", AR), ("md", r"""
**Estimate, then measure.** The fraction of peak throughput that training actually achieves (MFU) cannot be read off the spec sheet. On an RTX 5090 capped at 500 W, diffusion processed 1.04 million tokens per second in fp16, slightly below the autoregressive mode (1.13 million). That gives an MFU of 0.38 when attention counts only the $QK^\top$ product, the convention of Kaplan et al. (2020), and 0.40 when it also counts the product with the values, the PaLM convention (Chowdhery et al., 2022). The 209.5 TFLOPS peak in the denominator comes from secondary sources and has not been checked against NVIDIA's technical documentation. At the measured throughput, the research plan would fit into about 5.6 hours of training, excluding evaluation, whereas the estimate made without measurement ranged from 7.5 to 45 hours.
""", DIF)],
    1: [("md", r"""
**A work "in prose" that was not only prose.** *Relíquias de casa velha* contains a sonnet and two comedies. Mixing verse and drama into a prose corpus changes the distribution the model learns, so the work was removed, along with the *Poesias completas*. The lesson is to inspect the content rather than trust the label on the work.

**A split that leaks.** Drawing documents at random would place neighbouring chapters of the same work in both training and validation. The solution was contiguous blocks of whole chapters, with a one-chapter margin between training and validation and another between validation and test. Proper names still leak. In a sample of 4 works, examined before the corpus was built, between 46% and 97% of the names in a held-out block already appeared in the training portion of the same work. For that reason the project measures bits per byte both with and without the name spans.

**The price of a minimum block.** Requiring at least 5 chapters and 30 KB in validation and in test cut the training text to 2.22 MB, against an estimated 3.2 MB or so. In *Yayá Garcia*, the test block ended up larger than the training block. Every choice of split is paid for in training data.

**Boilerplate text.** In corpus B, 23.1% of the 728 laws from 2019 to 2022 on the Senate list open budgetary credits in standardised wording. Credit, budget and commemorative laws repeat the same formula from one law to the next, and the corpus decision removed these classes so that the measurement reflects substantive legislative text.
""", AMBOS)],
    2: [("md", r"""
**Same letter, different bytes.** In NFC, "ã" is a single code point. In NFD, it is an "a" followed by a combining tilde. The pre-tokenization regular expression separates the combining mark from the letter, and the word falls apart. The cell below shows the effect. The solution is to normalise the corpus to NFC once, before anything else, and to count the metric's bytes on that text.
""", AMBOS), ("code", r"""
import unicodedata, regex
from slm.tokenizador import GPT2_REGEX
frase = "a ação do coração"
for forma in ("NFC", "NFD"):
    t = unicodedata.normalize(forma, frase)
    print(forma, len(t.encode()), "bytes |", regex.findall(GPT2_REGEX, t))
""", AMBOS), ("md", r"""
**Two implementations, one answer.** In the 440 KB sample of corpus A, more than one pair tied for the maximum count in 3,463 of the 3,836 merges (90%), and the first tie appeared at merge 56. In those cases only the tie-breaking rule picks the pair, and two implementations without a written rule can diverge. Writing the tie-break down as a convention (highest count, then the smallest pair of ids) made the didactic BPE and the library BPE identical, merge by merge and id by id. A test enforces this on every change, and the tie count comes from `scripts/medir_empates_bpe.py`.

**A vocabulary too large for the text.** With $V = 8{,}192$, 12% of the tokens learned on corpus A never appear in the validation text. Each of them is an embedding row trained on very few examples. The project's rule picks the largest $V$ with at most 10% of such absent tokens, which gave $V = 4{,}096$ for corpus A and $V = 1{,}024$ for corpus B.
""", AMBOS)],
    3: [("md", r"""
**Length for free.** If diffusion were given the document length without paying for it, while the autoregressive mode pays for its `[EOS]`, the comparison would favour diffusion by up to $\log_2 L$ bits per document. In the package's G1 test, with exact conditionals and 4-token windows, the undue advantage is exactly the entropy of the length, 0.856 nat, or 1.235 bits per document, to within $10^{-9}$. The solution is to predict the padding in both modes. The cost is that, with $L = 256$, 12.2% of the training positions are padding.
""", AMBOS)],
    4: [("md", r"""
**Two forms of the same loss, very different variances.** The time form draws $t$ and weights the loss by $1/t$. Near $t = 0$ the weight blows up, and the estimator's variance is infinite without a cutoff $t \ge \varepsilon$, which in turn leaves the bound incomplete. The mask-count form draws $l$ and weights by $L/l$, which never exceeds $L$. By the identity $\int_0^1 \binom{L}{l} t^{l-1}(1-t)^{L-l}\,dt = 1/l$, the two share the same expectation, and the $l$ form has the lower variance, which is shown in two steps. Given $l$, the time follows $\mathrm{Beta}(l+1, L-l+1)$ and $\mathbb{E}[1/t \mid l] = (L+1)/l$, and by the Rao-Blackwell theorem replacing $1/t$ with this conditional expectation does not increase the variance. The second step, recorded in the formulation decision, compares the second moments, $L \sum_l \mathbb{E}[f_l^2]/l^2$ against $(L+1) \sum_l \mathbb{E}[f_l^2]/l^2$, and discards the wasted draw of $l = 0$. The cell below compares only the weights, which is where the difference lies. With the cutoff, the time weight approaches 1,000, and without it there is no bound at all, whereas the mask-count weight never exceeds $L$. The equality of expectations holds for the full estimator, weight times loss, and not for the weight on its own.
""", DIF), ("code", r"""
import numpy as np
rng = np.random.default_rng(0)
L_ = 256
t = rng.uniform(1e-3, 1, 100000)
l_ = rng.integers(1, L_ + 1, 100000)
for nome, peso in (("1/t, cut at 1e-3", 1 / t), ("L/l", L_ / l_)):
    print(f"{nome:18s} max weight {peso.max():8.1f} | weight spread {peso.std():7.2f}")
""", DIF), ("md", r"""
**One token fewer in the input.** The shift by `[BOS]` drops the last token of the window from the input, so it appears only as a target. Without the shift, the input at position $i$ would contain its own target, and the network could copy it instead of predicting it. Together, the shift and the causal mask ensure that the prediction of $w_i$ uses only $w_{<i}$, and the theory cell above checks the shift.
""", AR)],
    5: [("md", r"""
**The mask logit.** If the network could predict `[MASK]`, part of the probability would go to a token that is never the answer, and the distribution over texts would no longer sum to 1. The project's test enumerates every reveal order in a tiny network and checks that the sum comes to 1, with a tolerance of $10^{-9}$. With the mask logit left free, the sum fell to 0.59, and with the target visible to its own prediction it rose to 1.28, so both defects are caught. That is why the `[MASK]` logit is $-\infty$ in both modes.
""", AMBOS), ("md", r"""
**Where dropout acts.** Dropout used to act only inside attention and now also acts on the embedding and on both residual branches of each block. This is a design choice, with no measured comparison against the earlier version.
""", AMBOS), ("md", r"""
**Doesn't diffusion need its own architecture?** Some diffusion models for text use a dedicated network. SEDD (Lou, Meng and Ermon, 2023) and MDLM (Sahoo et al., 2024) use a diffusion Transformer that takes the noise level as an extra input through adaptive normalisation, the total noise $\bar\sigma(t)$ in SEDD and the time in MDLM. Ou et al. (2024, Theorem 1) showed that, when the process is one of masking, the score factorises into an analytic scalar that depends only on time multiplied by the distribution of the clean tokens given the visible context, and that distribution does not depend on time. The network need only learn this distribution, and the time input becomes unnecessary. Table 12 of MDLM reports similar perplexities with and without it. RADD (Ou et al., 2024) and LLaDA (Nie et al., 2025) use a plain Transformer, with no time input and no causal mask. The project follows this line for a reason of experimental design, since with the same architecture in both modes any measured difference between them comes from training and generation rather than from the network.
""", AMBOS)],
    6: [("md", r"""
**Overflow in the gradient, not in the loss.** The largest number fp16 can represent is 65,504. The loss is already computed in fp32 under autocast, but the gradient reaching the logits is multiplied by the `GradScaler` scale ($2^{16}$) and by the diffusion weight. In a test with one masked token per sequence, the $1/t$ form divided by the number of masks overflowed at all three batch sizes tested (2, 4 and 64), the $1/t$ form divided by $L$ reached 64,000 at batch size 4, and the $l$ form stayed at 16,384. The $l$ form caps the peak at $S/B$.
""", DIF), ("md", r"""
**Precision checked, not assumed.** A sentinel compares, tensor by tensor, the fp16 gradient with the fp32 gradient on the same batch, from the start of training to the end. On the RTX 5090, the worst relative error was 0.0025 in the autoregressive mode and 0.0099 in diffusion, against a margin of 0.05, and across 3 seeds per mode the absolute difference in bits per byte between fp16 and fp32 never exceeded 0.0063. The formal judgement of this equivalence depends on the margin $\Delta_{min}$, which is frozen only after the hyperparameter search, and is still pending. An error-ratio criterion (diffusion's error at most twice the autoregressive one) rejected differences with no practical effect, because it compared two errors close to fp16 rounding. A recorded amendment replaced it with equivalence in bits per byte.
""", AMBOS), ("md", r"""
**Many epochs over little text.** In the hyperparameter search, the autoregressive mode with 256 epochs got worse on validation, going from about 1.9 with 16 epochs to somewhere between 4.85 and 5.84 bits per byte without dropout, depending on the learning rate. With a dropout of 0.2, it landed between 2.85, at a rate of $3 \times 10^{-4}$, and 3.75, at a rate of $3 \times 10^{-3}$. The model memorises the training text, and regularisation is what holds it back.
""", AR), ("md", r"""
**Many epochs over little text.** In the hyperparameter search, with a single seed, diffusion with 256 epochs kept improving on validation, reaching 1.84 bits per byte in the configuration with a rate of $3 \times 10^{-4}$ and no dropout, while the autoregressive mode got worse over the same 256 epochs. Prabhudesai et al. (2025) report the same pattern at a larger scale and attribute it to masking different positions on each pass, which acts as data augmentation. This mechanism was not tested in the project, and confirming the pattern across several seeds is part of the protocol.
""", DIF)],
    7: [("md", r"""
**Noise truncated by precision.** The Gumbel draw uses $g = -\log(-\log u)$. Zheng et al. (2024) show that truncating this noise already distorts sampling in fp32. Truncated noise acts as a lower temperature, so the samples lose diversity and the generative perplexity looks better than it really is. In fp16 the problem gets worse, and by the project's calculation $u$ has only 2,048 possible values and the largest possible noise drops to 7.62, against 16.64 in fp32 and 36.74 in fp64. The solution is to sample in float64. The cell below measures the effect.
""", AMBOS), ("code", r"""
import math
for nome, dt, p in (("fp16", torch.float16, 11), ("fp32", torch.float32, 24), ("fp64", torch.float64, 53)):
    u = torch.rand(2_000_000, dtype=dt)
    print(f"{nome}: distinct values of u = {u.unique().numel():>9d} | largest theoretical g = {-math.log(-math.log(1 - 2.0 ** -p)):.2f}")
""", AMBOS), ("md", r"""
**Confidence versus chance.** Revealing the most confident tokens first relies on an argmax and discards the random draw. The project uses random revealing as the default, and confidence-based revealing enters only as a labelled variant.
""", DIF)],
    8: [("md", r"""
**A bound, not a value.** In diffusion, bits per byte come from the NELBO, which is an upper bound. If diffusion comes in below the autoregressive mode, it is better. If it comes in above, the gap may simply be slack in the bound. This asymmetric reading is declared before anything is run.
""", DIF), ("md", r"""
**An exact value that serves as the yardstick.** In the autoregressive mode, bits per byte are the exact log-likelihood. Diffusion has only an upper bound. In the comparison, the autoregressive figure is the fixed reference, and only a diffusion result below it supports a conclusion.
""", AR), ("md", r"""
**Probability is not confidence.** The reliability diagram shows whether the stated confidence matches the observed accuracy. The temperature, fitted by Newton's method on validation, corrects the scale of the logits without changing which token is chosen.
""", AMBOS)],
    9: [("md", r"""
**How many seeds?** With 3 seeds, the 95% interval for the standard deviation across seeds runs from 0.52 to 6.28 times the estimate. The project estimates the spread in a 10-seed pilot and derives from it the number of seeds for the final comparison, with a declared ceiling.

**Decide before looking.** The minimum effect size, the hyperparameter grid and the reading criterion are frozen with a hash before the runs they govern. Whenever a criterion had to change after the data were in, the change was recorded as an amendment, together with its justification.
""", AMBOS)],
    10: [("md", r"""
**Coherent or copied?** A model trained for many epochs can look coherent because it reproduces passages of the training text. The copy filter measures the longest copied passage and discards any sample that goes beyond what is observed between human texts. The discard rate is reported as a measure of memorisation.
""", AMBOS), ("md", r"""
**Overfitting is recitation too.** Without dropout and with a rate of $10^{-3}$, the autoregressive model trained for 256 epochs reached 5.6 bits per byte on validation, far worse than with 16 epochs. Across 16 samples of 256 tokens per run, the longest passage copied from the training text had a median of 4 words with 16 epochs and of 144 to 150 words with 256 epochs, with a maximum of 180. With a dropout of 0.2 and a rate of $3 \times 10^{-4}$, the same 256 epochs gave a median of 5 to 5.5 words, although isolated samples still copied 44 and 95 words. Overfitting showed up in both forms, as confident wrong predictions on new text and as recitation of the training text. The measurement lives in `scripts/medir_copia.py`, and the result of each run in `runs/<corrida>/copia.json`.
""", AR), ("md", r"""
**The bias left without context.** On the deletion curve, with 100% of the context removed, the network still assigns probability to the original decision. This residue measures the model's intrinsic bias, what it knows without looking at the input.
""", AMBOS)],
}
