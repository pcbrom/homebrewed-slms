# Pesquisa: protocolo de comparação entre os modos

Levantamento de 22/10/2026. Este documento reúne o que a literatura usa para comparar a difusão mascarada com o autoregressivo sob o mesmo orçamento, e o que isso exige da bancada deste projeto. Cálculos e recomendações sem fonte direta vêm marcados como do projeto.

## 1. O problema

Comparar dois modos de geração parece simples quando a rede é a mesma, mas a resposta muda conforme o que se iguala entre eles. Três trabalhos centrais da área chegam a conclusões diferentes em parte porque usam critérios diferentes de orçamento, e quem lê só o resultado não percebe. Se a bancada deste projeto não declarar o critério antes de rodar, qualquer resultado pode ser reinterpretado depois, que é exatamente o que um revisor externo vai perguntar.

## 2. Contagem de FLOPs

Kaplan et al. (2020, seção 2.1, Tabela 1) dão o custo do passe direto por token, $C_{forward} \approx 2N + 2\, n_{layer}\, n_{ctx}\, d_{attn}$, e definem o custo de treino como $C \approx 6N$ por token, com $N$ excluindo embeddings. Hoffmann et al. (2022, apêndice F) usam $6ND$ no ajuste e uma conta detalhada que inclui embeddings, a projeção de saída e cada multiplicação da atenção. A razão entre a conta detalhada e $6ND$ fica entre 1,03 e 1,10 nos modelos de 73M a 305M da Tabela A4. O PaLM (Chowdhery et al., 2022, apêndice B) soma $6N$ por token a $12 L H Q T$ para a atenção densa. A conta deste projeto está em [DIMENSIONAMENTO.md](DIMENSIONAMENTO.md).

**A atenção bidirecional muda a conta?** Nas convenções analíticas acima, não, porque elas contam a matriz de atenção inteira mesmo em modelos causais. Na execução real, muda. O FlashAttention-2 (Dao, 2023) pula cerca de metade dos blocos com máscara causal e, para benchmark, divide por 2 o custo da atenção nesse caso. A difusão gasta, portanto, até o dobro no termo de atenção, e só nele. Pela razão $n_{ctx}/(12d)$ de Kaplan, esse termo é pequeno: com $d = 384$, vale 5,6% de $2N$ em $n_{ctx} = 256$ e 22,2% em $n_{ctx} = 1024$, conforme cálculo deste projeto.

## 3. Como a literatura iguala os dois modos

| Trabalho | Critério de igualdade | Consequência |
|---|---|---|
| MDLM (Sahoo et al., 2024, suplemento D.2) | "Tokens vistos" pela difusão, contando só os mascarados, cerca de metade do total. O autoregressivo recebe metade dos passos | A difusão recebe cerca do dobro dos FLOPs do autoregressivo. A Tabela 3 informa 524B de tokens, que é o produto sem o fator 0,5 |
| Nie et al. (2024) | $C = 6ND$, "a relationship valid for both ARMs and MDMs", com $N$ sem embedding, em curvas de mesmo FLOP | Diferença constante de cerca de 16 vezes em compute a favor do autoregressivo, no regime de uma época |
| Prabhudesai et al. (2025) | Fronteira de perda contra FLOPs com dados únicos $U$ fixos e muitas épocas | Perto do ótimo de Chinchilla o autoregressivo vence. Com muitas épocas sobre dados escassos, a difusão passa à frente |

Prabhudesai et al. não explicitam a fórmula de FLOPs dos próprios experimentos, não relatam sementes nem testes estatísticos, e o artigo tem inconsistências entre texto e Tabela 1 e na unidade de uma constante. Os resultados servem de hipótese para este projeto, não de referência numérica.

## 4. Contagem de parâmetros

Kaplan exclui os embeddings de $N$ por produzir leis de escala "significantly cleaner". Hoffmann os inclui. Pearce e Song (2024) atribuem boa parte da divergência entre as duas leis a essa escolha, junto com a pequena escala, e recomendam parâmetros e compute totais. Porian et al. (2024) apontam três outras causas: o custo da última camada, a duração do aquecimento e o ajuste do otimizador dependente da escala.

Na faixa deste projeto o embedding não é pequeno. Com $d = 384$ e 6 camadas, ele equivale a 36% dos parâmetros fora do embedding com $V = 10$ mil e a 116% com $V = 32$ mil, conforme cálculo deste projeto. Com o mesmo tokenizador nos dois modos, compartilhamento de pesos ligado nos dois e a rede sem entrada de tempo, as duas redes ficam idênticas em parâmetros, exceto pela linha do token de máscara. Relatam-se o total e o valor sem embedding.

## 5. Métrica principal

O bits por byte (Gao et al., 2020, seção 3.1) é $\mathrm{BPB} = \sum_d \mathrm{NLL}_d / (\ln 2 \cdot \sum_d B_d)$, com cada documento avaliado de forma independente. Para o autoregressivo, a NLL é exata. Para a difusão, é a NELBO, um limite superior. O MDLM registra que "all perplexities for diffusion models are upper bounds", e o SEDD que "the autoregressive result is an exact likelihood, while the diffusion results are upper bounds".

Disso decorre uma assimetria que governa a leitura dos resultados. A comparação favorece o autoregressivo, porque a NLL verdadeira da difusão é menor ou igual ao valor relatado. Se o limite da difusão fica abaixo do BPB do autoregressivo, a vantagem da difusão está demonstrada. O inverso não demonstra nada: uma derrota da difusão pode ser apenas folga do limite.

A NELBO é estimada por Monte Carlo sobre $t$. O MDLM usa uma amostra por sequência com baixa discrepância, e Nie et al. usam 128. O projeto fixa e declara o número de amostras e as sementes de avaliação, relata o erro-padrão de Monte Carlo e calcula a perda de avaliação em fp32 por causa do peso $1/t$. Esta última é recomendação do projeto, sem fonte.

## 6. Inferência estatística

Koehn (2004) introduziu o bootstrap pareado na avaliação de tradução automática. Dror et al. (2018) organizam a escolha do teste: com distribuição desconhecida e amostra pequena, bootstrap ou teste de aleatorização. O teste de Wilcoxon testa simetria das diferenças em torno de zero, uma hipótese sobre mediana que não coincide com o BPB agregado, que é uma razão de somas. Nenhuma prática padronizada de teste por documento para BPB entre modelos de linguagem foi encontrada.

A variação entre sementes de treino precisa entrar na inferência. O Multi-Bootstrap do MultiBERTs (Sellam et al., 2021) reamostra separadamente as sementes de pré-treino e os itens de teste. Na prática publicada, Madaan et al. (2024) usam 10 sementes, o PolyPythias (van der Wal et al., 2025) usa 9 em modelos de 14M a 410M, e o DataDecide (Magnusson et al., 2025) usa 3.

O protocolo candidato, que é recomendação do projeto construída sobre essas fontes, usa a estatística $\Delta = \mathrm{BPB}_{AR} - \mathrm{BPB}_{dif}$ como razão de somas, reamostrando documentos pareados e sementes de forma independente, com pelo menos $10^4$ reamostras, intervalo de 95% e margem mínima de efeito fixada antes da corrida. O mínimo é de 3 sementes por modo, ou 5 se o orçamento permitir.

## 7. Avaliação das amostras

**Perplexidade gerativa.** É a perplexidade das amostras sob um modelo externo, o GPT-2 Large no SEDD e no MDLM. Zheng et al. (2024) mostram que ela engana sozinha: a precisão baixa no sorteio reduz a temperatura efetiva, a perplexidade gerativa melhora e a diversidade cai (ver [pesquisa de difusão](pesquisa_difusao.md), seção 6). Ela só se relata junto da entropia das amostras, $H = -\sum_k p_k \log p_k$ por sequência (apêndice H.1), de preferência numa varredura de temperatura.

Um projeto "do zero" não tem modelo externo pré-treinado, e nenhuma solução publicada para isso foi encontrada. As opções do projeto são um avaliador autoregressivo maior, treinado só na partição de treino, com semente própria e declarado antes, ou a perplexidade gerativa tratada como métrica secundária. Usar como avaliador o próprio autoregressivo da comparação favorece o autoregressivo.

**Diversidade.** O distinct-n de Li et al. (2016) é o "number of distinct unigrams and bigrams divided by total number of generated words". O denominador é o total de palavras, não de n-gramas, e a variante adotada precisa ser declarada.

**Memorização.** O RAVEN (McCoy et al., 2021) considera um n-grama duplicado "if it appears in the training set or the context" e relata a fração de n-gramas novos para $n$ de 1 a 10, contra a linha de base do texto humano de teste. Carlini et al. (2022, definição 3.1) chamam uma sequência $s$ de extraível com $k$ tokens se o prefixo seguido de $s$ está no treino e a decodificação gulosa a partir do prefixo produz $s$, e replicam a medida no T5, que é mascarado. A versão gulosa para a difusão, desmascarando condicionado ao prefixo, é escolha de desenho deste projeto, sem fonte. No regime de muitas épocas sobre um corpus pequeno, esta é a medida que separa geração de cópia.

## 8. Escala viável na T4

Os modelos do TinyStories (Eldan e Li, 2023) vão de cerca de 1M a 35M de parâmetros, com 1 a 8 camadas, "trained on a single V100 GPU within at most 30 hours", e produzem texto coerente abaixo de 10M. O exemplo de Shakespeare por caractere do nanoGPT, com 6 camadas, $d = 384$ e contexto 256, leva cerca de 3 minutos numa A100, segundo o README.

A T4 tem 65 TFLOPS em precisão mista FP16/FP32, 16 GB de GDDR6 a 300 GB/s e 70 W (NVIDIA, 2019). A capacidade computacional é 7.5, e o suporte a bf16 nos núcleos tensoriais começa na 8.0, segundo o guia de programação CUDA. O treino na T4 usa precisão mista em FP16 com cópia mestra em FP32 e escalonamento da perda (Micikevicius et al., 2017). A única fonte primária encontrada sobre treino de difusão mascarada em 16 bits é uma frase do RADD (Ou et al., 2024, apêndice J): "The small models are trained on multi-accelerator compute nodes with float16 precision", aplicada a modelos treinados tanto com a forma por $t$ quanto com a forma por número de máscaras. A frase não informa escalonamento de perda nem o acelerador, e não permite saber se se trata de IEEE fp16. O MDLM treina em bf16.

O Colab gratuito informa que "notebooks can run for at most 12 hours, depending on availability and your usage patterns", que sessões ociosas expiram e que "Colab does not publish these limits". Não há garantia de GPU nem tempo de ociosidade publicado. O MFU real na T4 não foi encontrado e precisa ser medido.

## 9. Protocolo candidato

O protocolo adotado, reduzido a um ponto confirmatório, está na [decisão de orçamento](decisao_orcamento.md).

1. Fixar antes de rodar: corpus, partição por documento, tokenizador único, comprimento de bloco igual, a mesma rede com compartilhamento de pesos e sem entrada de tempo.
2. Orçamento primário em FLOPs analíticos iguais, com a atenção contada como densa nos dois modos. Relatar à parte o tempo de parede. Cobrir uma grade de épocas sobre o mesmo volume de dados únicos, para alcançar o regime em que Prabhudesai et al. encontram o cruzamento. O critério do MDLM entra como análise secundária declarada.
3. Métrica primária em bits por byte por razão de somas, com a NELBO estimada por número fixo de amostras e erro de Monte Carlo relatado. A conclusão de vantagem da difusão só vale com o intervalo excluindo zero.
4. Inferência por bootstrap pareado em documentos e sementes, com 3 a 5 sementes por modo e margem mínima de efeito fixada antes.
5. Amostras geradas com sorteio em float64, perplexidade gerativa do avaliador declarado junto com a entropia, distinct-n com denominador declarado e novidade de n-gramas contra o treino e contra o texto humano de teste.
6. Calibração prévia na T4, medindo MFU e tempo por passo de cada modo, com o orçamento final dividido em sessões curtas e checkpoints.

Referências completas, versões e licenças em [referencias/REFERENCIAS.md](../referencias/REFERENCIAS.md).
