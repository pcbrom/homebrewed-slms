# Dimensionamento do treino ao hardware

Antes de escolher o número de camadas ou o tamanho do corpus, vale responder a uma pergunta mais simples: quanto texto a sua GPU consegue processar por hora, e quanto texto o modelo que você quer treinar precisa ver. A diferença entre essas duas quantidades decide se o experimento termina numa aula, numa noite ou numa semana. Este documento dá as fórmulas, mostra a conta para duas placas de referência e indica como trocar a estimativa pelo valor medido na sua máquina. O script [`scripts/dimensionar.py`](../scripts/dimensionar.py) faz a conta para qualquer hardware.

## 1. Custo de um token de treino

Para um Transformer com $n_{layer}$ camadas, dimensão $d_{model}$, contexto $n_{ctx}$ e vocabulário $n_{vocab}$, o número de parâmetros fora das matrizes de embedding é

$$N \approx 12\, n_{layer}\, d_{model}^2,$$

e o custo do passe direto por token é a soma das linhas da Tabela 1 de Kaplan et al. (2020, seção 2.1):

$$C_{forward} \approx 2N + 2\, n_{layer}\, n_{ctx}\, d_{model} + 2\, d_{model}\, n_{vocab}.$$

O primeiro termo vem das multiplicações de matriz das camadas, o segundo da atenção e o terceiro da projeção de saída sobre o vocabulário. O passe de volta custa cerca do dobro do direto, o que leva ao custo de treino por token

$$C_{train} \approx 6N + 6\, n_{layer}\, n_{ctx}\, d_{model} + 6\, d_{model}\, n_{vocab}.$$

A aproximação $C_{train} \approx 6N$, a mais citada, despreza os dois últimos termos. Kaplan et al. justificam a omissão para modelos em que $d_{model} > n_{ctx}/12$, onde a atenção pesa pouco, e não incluem a projeção de saída em $N$. Em modelos grandes isso é razoável. Em modelos pequenos, não: com $d_{model} = 256$ e $n_{vocab} = 8.000$, a atenção e a projeção de saída acrescentam 73% ao custo estimado por $6N$, dos quais 65 pontos vêm só da projeção de saída, conforme a tabela da seção 5. Quem dimensiona um SLM pela fórmula curta subestima o tempo de treino justamente na faixa em que pretende trabalhar, e é por isso que o script usa a forma completa.

A atenção bidirecional do modo de difusão e a causal do modo autoregressivo têm o mesmo custo nesta conta, porque a implementação usual calcula a matriz inteira de atenção e só depois aplica a máscara.

## 2. Vazão

A vazão em tokens por segundo é o trabalho útil que a placa entrega dividido pelo custo de cada token:

$$\text{tokens/s} = \frac{P \cdot \text{MFU}}{C_{train}},$$

em que $P$ é o pico de operações por segundo na precisão do treino e MFU (*model FLOPs utilization*, Chowdhery et al., 2022) é a fração desse pico que o treino de fato aproveita. O pico vem da especificação do fabricante. O MFU não vem de lugar nenhum além da medida. Ele depende do tamanho das matrizes, do lote, do comprimento da sequência, da implementação da atenção e do limite de potência da placa, e em modelos pequenos tende a ser baixo porque as matrizes não ocupam a GPU inteira. As tabelas abaixo usam 30% e 50% como faixa de trabalho. Essa faixa é uma hipótese para a estimativa, não um valor medido, e deve ser substituída pelo MFU medido no seu hardware (seção 6).

## 3. Quanto texto o modelo precisa

Hoffmann et al. (2022) estimaram que, para um orçamento fixo de compute, o treino ótimo usa da ordem de 20 tokens por parâmetro. A regra serve de alvo para o dimensionamento, não de limite. Treinar com menos tokens deixa capacidade sem uso, e treinar com mais continua melhorando o modelo, só que com retorno menor por FLOP. O tempo até esse alvo é

$$t = \frac{r \cdot N_{tot}}{\text{tokens/s}}, \qquad r \approx 20,$$

em que $N_{tot}$ inclui ou não os parâmetros de embedding, conforme a convenção adotada. Em modelos pequenos a diferença pesa, porque o embedding é uma fração grande do total. O script conta com embedding por padrão e aceita `--sem-embedding` para a outra convenção.

Lido ao contrário, o mesmo raciocínio diz qual modelo cabe num orçamento de tempo. Com $C = 6 N D$ e $D = rN$, o tamanho ótimo para um compute total $C$ é

$$N^* = \sqrt{\frac{C}{6r}}.$$

Essa inversão usa a aproximação curta $6N$ e, pela seção 1, superestima o tamanho que cabe quando o modelo é pequeno.

O alvo de tokens também revela um limite que a GPU não resolve. Um modelo de 30M de parâmetros pede cerca de 600M de tokens de treino. Se o seu corpus tem bem menos que isso, o treino passa a repetir o mesmo texto por várias épocas, e a questão deixa de ser velocidade e passa a ser memorização.

## 4. Memória

O treino com AdamW em precisão mista ocupa 16 bytes por parâmetro antes de qualquer ativação. No arranjo descrito por Rajbhandari et al. (2019, seção 3.1), são pesos e gradientes em fp16, mais cópia dos pesos e dois momentos do Adam em fp32. No `autocast` do PyTorch, são pesos e gradientes em fp32 e os dois momentos em fp32. Os dois arranjos somam o mesmo total. A coluna de memória das tabelas mostra essa parcela como fração da VRAM. As ativações dependem do lote, do contexto, da dimensão e do número de camadas, e são medidas, não estimadas, pelo procedimento da seção 6. Na faixa deste projeto a memória dos parâmetros é pequena diante da VRAM das duas placas de referência, de modo que o limite prático é o tempo e o volume de texto, e não a memória.

## 5. Duas placas de referência

As tabelas abaixo foram produzidas por `scripts/dimensionar.py`, com $n_{ctx} = 256$, $n_{vocab} = 8.000$, embedding compartilhado entre entrada e saída e alvo de 20 tokens por parâmetro contando o embedding.

**RTX 5090**, pico de 209,5 TFLOPS densos em BF16 com acumulação em FP32 e 32 GB de VRAM. O valor do pico foi tomado de fontes secundárias e não foi conferido no documento técnico da NVIDIA. No ambiente de referência deste projeto a placa opera com teto de potência de 500 W, abaixo do padrão de fábrica, o que reduz o MFU alcançável.

```
python3 scripts/dimensionar.py --pico-tflops 209.5 --vram-gb 32 --arq 4x256 6x384 8x512 12x768 24x1024 --horas 8
```

| Arquitetura | $N$ | $N_{tot}$ | Custo por token | Acima de $6N$ | Memória AdamW | Tokens alvo | Tempo (MFU 0,3) | Tempo (MFU 0,5) |
|---|---|---|---|---|---|---|---|---|
| 4 × 256 | 3,1M | 5,2M | 32,7M FLOP | 73% | 0,2% | 104M | 1 min | 1 min |
| 6 × 384 | 10,6M | 13,7M | 85,7M FLOP | 34% | 0,6% | 274M | 6 min | 4 min |
| 8 × 512 | 25,2M | 29,3M | 182M FLOP | 20% | 1,4% | 585M | 28 min | 17 min |
| 12 × 768 | 84,9M | 91,1M | 561M FLOP | 10% | 4,2% | 1,8B | 4,5 h | 2,7 h |
| 24 × 1024 | 302M | 310M | 1,9B FLOP | 5% | 14,4% | 6,2B | 2,2 dias | 31 h |

Com 8 horas de treino, o tamanho ótimo fica entre 123M e 159M de parâmetros, para 2,5B a 3,2B de tokens.

**T4**, a GPU do Colab gratuito, com 65 TFLOPS em precisão mista FP16/FP32 e 16 GB de GDDR6 (NVIDIA, 2019). A T4 não oferece BF16, e o treino nela usa FP16 com escalonamento de gradiente.

```
python3 scripts/dimensionar.py --pico-tflops 65 --vram-gb 16 --arq 4x256 6x384 8x512 --horas 1
```

| Arquitetura | $N$ | $N_{tot}$ | Custo por token | Acima de $6N$ | Memória AdamW | Tokens alvo | Tempo (MFU 0,3) | Tempo (MFU 0,5) |
|---|---|---|---|---|---|---|---|---|
| 4 × 256 | 3,1M | 5,2M | 32,7M FLOP | 73% | 0,5% | 104M | 3 min | 2 min |
| 6 × 384 | 10,6M | 13,7M | 85,7M FLOP | 34% | 1,3% | 274M | 20 min | 12 min |
| 8 × 512 | 25,2M | 29,3M | 182M FLOP | 20% | 2,7% | 585M | 1,5 h | 55 min |

Com uma hora de treino na T4, o tamanho ótimo fica entre 24M e 31M de parâmetros, para cerca de 480M a 620M de tokens.

## 6. Do estimado ao medido

As tabelas acima valem como ordem de grandeza e servem para descartar configurações inviáveis, mas não substituem a medida. O procedimento que transforma a estimativa em número da sua máquina é curto:

1. Rodar alguns passos de treino da arquitetura escolhida, com o lote e o contexto reais, descartando os primeiros passos de aquecimento.
2. Medir tokens por segundo com sincronização da GPU antes de cada leitura do relógio.
3. Calcular o MFU medido como $\text{tokens/s} \times C_{train} / P$.
4. Registrar a memória de pico com `torch.cuda.max_memory_allocated()`, que inclui as ativações.
5. Rodar `scripts/dimensionar.py` de novo com `--mfu` igual ao valor medido.

O MFU medido na RTX 5090 a 500 W, em 26/10/2026, com a rede de 6 camadas, $d = 384$, $V = 4.096$, $L = 256$ e lote 64, foi de 0,41 no autoregressivo e 0,38 na difusão em fp16 (1,13 e 1,04 milhão de tokens por segundo, contra o pico de 209,5 TFLOPS), e de 0,16 nos dois modos em fp32. O MFU na T4 será registrado quando medido. O modo de difusão e o autoregressivo devem ser medidos separadamente, porque o pré-processamento do mascaramento e a amostragem de $t$ acrescentam custo fora das multiplicações de matriz.

## Referências

Versões fixadas, licenças, hashes e PDFs de acesso aberto em [referencias/REFERENCIAS.md](../referencias/REFERENCIAS.md).

CHOWDHERY, A. et al. PaLM: scaling language modeling with pathways. *arXiv*, 2204.02311, 2022.

HOFFMANN, J. et al. Training compute-optimal large language models. *arXiv*, 2203.15556, 2022.

KAPLAN, J. et al. Scaling laws for neural language models. *arXiv*, 2001.08361, 2020.

NVIDIA. *NVIDIA T4 Tensor Core GPU*: datasheet. 2019. Disponível em: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf.

RAJBHANDARI, S. et al. ZeRO: memory optimizations toward training trillion parameter models. *arXiv*, 1910.02054, 2019.
