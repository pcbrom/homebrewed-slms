# Pesquisa: tokenizador de referência

Levantamento de 22/10/2026. Algoritmos conferidos nos artigos originais e nas documentações das bibliotecas, nas versões indicadas. Cálculos e testes feitos por este projeto vêm marcados como tal. Este documento serve à etapa 1, que precisa de um tokenizador de referência treinado do zero e de um contrato estável, e prepara a etapa 2, que compara tokenizadores.

## 1. Algoritmos

**BPE** (Sennrich, Haddow e Birch, 2016, seção 3.2). O vocabulário começa com os caracteres, e cada palavra vira uma sequência de caracteres seguida de um símbolo de fim de palavra. A cada iteração, contam-se os pares adjacentes ponderados pela frequência da palavra e substitui-se toda ocorrência do par mais frequente por um símbolo novo. Após $M$ fusões, o vocabulário tem o tamanho inicial mais $M$, e $M$ "is the only hyperparameter". A codificação aplica as fusões na ordem aprendida. O treino ingênuo custa $O(NM)$ para um corpus de $N$ símbolos, e Zouhar et al. (2023b) dão implementação em $O(N \log M)$. O problema de otimização subjacente é APX-completo (Kozma e Voderholzer, 2024), e variantes de tokenização ótima são NP-completas (Whittington et al., 2024). Sem tratamento de bytes, caracteres não vistos no treino ficam desconhecidos, e a ida e volta não é garantida.

**BPE em nível de byte** (Radford et al., 2019, seção 2.2). O vocabulário base são os 256 bytes, em vez dos mais de 130 mil pontos de código Unicode, e fusões entre categorias de caractere são proibidas, exceto para espaços. Uma expressão regular de pré-tokenização separa contrações, letras, números, pontuação e espaços antes das fusões. Como não há token desconhecido e o espaço é um byte comum, $\mathrm{decode}(\mathrm{encode}(s)) = s$ vale para qualquer cadeia. O caminho inverso não vale: uma sequência arbitrária de ids pode formar UTF-8 inválido.

**Unigram** (Kudo, 2018, seção 3.2). Um modelo $P(\mathbf{x}) = \prod_i p(x_i)$ sobre subpalavras, com segmentação $\mathbf{x}^* = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})$ por Viterbi. O treino parte de um vocabulário semente grande e alterna EM sobre $p(x)$ com a remoção das peças cuja retirada menos reduz a verossimilhança, até o tamanho desejado. Bostrom e Durrett (2020) encontraram o Unigram igual ou melhor que o BPE em pré-treino, em inglês e japonês.

**WordPiece.** O artigo original não tem versão aberta, e o algoritmo de treino do Google nunca foi publicado. A codificação, descrita por Song et al. (2020), casa gulosamente o maior prefixo presente no vocabulário e troca a palavra inteira por `[UNK]` se em algum ponto nada casar. A ida e volta não é garantida, e a normalização do BERT em modo sem caixa remove os acentos, o que o desqualifica para português neste projeto.

**SentencePiece** (Kudo e Richardson, 2018). Treina sobre texto bruto e escapa o espaço como `▁`. A garantia declarada é $\mathrm{Decode}(\mathrm{Encode}(\mathrm{Normalize}(t))) = \mathrm{Normalize}(t)$ (seção 3.1), isto é, sobre o texto normalizado, não sobre o original. A normalização padrão é uma variante do NFKC, que transforma "1ª" em "1a", "nº" em "no" e "m²" em "m2", conforme cálculo deste projeto com o `unicodedata` do Python. Também por padrão, espaços repetidos são colapsados. Para ida e volta exata, as opções necessárias são `normalization_rule_name=identity`, `remove_extra_whitespaces=false` e `byte_fallback=true`. A documentação avisa que a semente fixa "does not guarantee permanent reproducibility across updates".

## 2. Normalização Unicode e português

Português tem caracteres que admitem duas representações. Em NFC, "ã" é um único ponto de código, com 2 bytes em UTF-8. Em NFD, é "a" seguido de um til combinante, com 3 bytes. Num teste deste projeto com a expressão regular do GPT-2, "a ação do coração" em NFC se divide em `['a', ' ação', ' do', ' coração']`. Em NFD, a divisão fica `['a', ' ac', '̧', 'a', '̃', 'o', ' do', ' corac', '̧', 'a', '̃', 'o']`, porque a marca combinante não é classificada como letra. A normalização, portanto, muda o que o tokenizador vê e muda a contagem de bytes. A regra que sai disso é normalizar o corpus em NFC uma única vez, antes de tudo, deixar o tokenizador sem normalização própria e contar os bytes da métrica sobre esse mesmo texto.

## 3. Implementação própria ou biblioteca

| Opção | Licença | A favor | Contra |
|---|---|---|---|
| Implementação própria, no estilo do minbpe | MIT (minbpe) | Expõe o algoritmo linha a linha, que é o objetivo didático | O treino ingênuo em Python é lento em dezenas de MB. Para $N = 5 \times 10^7$ e cerca de 16 mil fusões, $N \cdot M \approx 8 \times 10^{11}$ operações elementares, estimativa de ordem deste projeto |
| Hugging Face `tokenizers` 0.23.x | Apache-2.0 | Treino em Rust, rápido, API estável | Esconde o algoritmo. O ramo 1.0, em versão candidata desde 19/10/2026, declara que as interfaces Python iniciais não expõem treino, o que obriga a fixar `tokenizers<1.0` |
| `sentencepiece` 0.2.2 | Apache-2.0 | Unigram e BPE, controle fino de especiais | Padrões que alteram o texto, como descrito na seção 1 |

O minbpe treina dois tokenizadores de 512 tokens sobre 185.768 bytes em cerca de 25 s num M1, conforme o README. O próprio autor lista como pendência uma versão para "large files and big vocabs". A implementação didática do *Dive into Deep Learning* segmenta por casamento guloso do maior subtoken, e não pela reaplicação das fusões na ordem aprendida. Serve como ilustração, não como referência de codificação.

O caminho que concilia os dois objetivos é implementar o BPE em nível de byte no notebook, sobre uma amostra do corpus, para mostrar o algoritmo, e treinar o tokenizador do experimento com a biblioteca, verificando que as duas produzem as mesmas fusões na amostra. A verificação vira um teste.

## 4. Tokens especiais

O contrato do projeto reserva `[PAD]`, `[BOS]`, `[EOS]` e `[MASK]`, e nenhum texto pode produzir esses ids. As bibliotecas tratam isso de formas diferentes:

- No `tokenizers` 0.23, a ordem de `special_tokens` no treinador define os ids, e os especiais não participam do treino. Por padrão, porém, a cadeia de um especial que apareça no texto é convertida no id dele. A opção `encode_special_tokens=True` desliga esse comportamento.
- No `sentencepiece`, os `control_symbols` apenas reservam ids e não participam da segmentação do texto, enquanto os `user_defined_symbols` são casados no texto.
- No `tiktoken`, o padrão `disallowed_special="all"` levanta erro se o texto contiver um especial.

A propriedade a testar é que $\mathrm{encode}(t)$ nunca contém um id reservado, para qualquer texto $t$, inclusive quando $t$ contém literalmente a cadeia "[PAD]".

## 5. Tamanho do vocabulário

Em modelos pequenos, o vocabulário decide uma fração grande dos parâmetros. Com $N \approx 12\, n_{layer}\, d^2$ parâmetros fora do embedding (Kaplan et al., 2020) e embedding compartilhado entre entrada e saída, a fração do embedding no total fica assim, conforme cálculo deste projeto:

| $d$, camadas | $N$ sem embedding | $V = 4.096$ | $8.192$ | $16.384$ | $32.768$ |
|---|---|---|---|---|---|
| 256, 8 | 6,29M | 14% | 25% | 40% | 57% |
| 384, 8 | 14,16M | 10% | 18% | 31% | 47% |
| 512, 6 | 18,87M | 10% | 18% | 31% | 47% |

Sem compartilhamento, com $d = 384$ e 8 camadas, as frações sobem para 18%, 31%, 47% e 64%. O compartilhamento de pesos entre embedding e saída é recomendado por Press e Wolf (2016), com redução de perplexidade, e justificado por Inan et al. (2016). Liu et al. (2024) registram que o embedding passa de 20% de um modelo de 125M e que compartilhá-lo economiza 16M de parâmetros com queda de 0,2 ponto, recuperada com mais camadas.

Tao et al. (2024) propõem a lei $N_v^{opt} \propto N_{nv}^{\gamma}$, com $\gamma \approx 0{,}83$, para o tamanho ótimo de vocabulário em função dos parâmetros fora do vocabulário. O menor grupo do estudo tem 33M de parâmetros fora do vocabulário, com $V$ de 4.096 a 96.256, e o valor ótimo para esse grupo está só em figura, sem número extraível. Nenhum ótimo publicado foi encontrado para a faixa de 10M a 30M. A referência empírica mais próxima é o llama2.c, que usa $V = 4.096$ em TinyStories. Em português, o TeenyTinyLlama, de 160M e 460M, e o Tucano usam SentencePiece com 32 mil tokens (Corrêa et al., 2024a, 2024b), numa escala acima da deste projeto.

## 6. Métricas

**Bits por byte** (Gao et al., 2020). Com $\ell$ a perda média por token em nats, $L_T$ o número de tokens e $L_B$ o número de bytes UTF-8,

$$\mathrm{BPB} = \frac{L_T}{L_B} \cdot \frac{\ell}{\ln 2} = -\frac{1}{L_B \ln 2} \sum_i \ln p(t_i \mid t_{<i}).$$

A justificativa é a "invariance to different tokenization schemes". A perplexidade por token muda com o número de tokens por documento e não é comparável entre tokenizadores. Duas consequências práticas: $L_B$ é contado sobre o texto NFC fixo do corpus, e não sobre o texto normalizado pelo tokenizador, e a soma inclui os mesmos tokens em todos os braços, inclusive o EOS quando ele for alvo.

**Diagnósticos intrínsecos.** Compressão em bytes por token, $L_B / L_T$. Fertilidade, "the average number of subword tokens produced per tokenized word", e proporção de palavras divididas em mais de um token (Rust et al., 2020). Eficiência de Rényi, $H_\alpha(p)/H_0$ sobre a distribuição unigrama dos tokens, com $H_\alpha(p) = \frac{1}{1-\alpha} \log \sum_\delta p(\delta)^\alpha$ e $H_0 = \log |\Delta|$ (Zouhar et al., 2023a), que nesse trabalho correlacionou 0,78 com BLEU em $\alpha = 2{,}5$. Uso do vocabulário, a fração de ids com ao menos uma ocorrência num corpus retido, e tokens subtreinados (Land e Bartolo, 2024).

**Limites dos diagnósticos.** Nenhuma métrica intrínseca substitui a medida no modelo treinado. Schmidt et al. (2024), com 64 modelos de 350M a 2,4B, encontraram que menos tokens não implicam melhor desempenho, com correlação de Pearson de 0,241 entre contagem de tokens do corpus e acurácia média. Cognetta et al. (2024) constroem variantes de BPE que elevam a eficiência de Rényi e pioram o desempenho. Ali et al. (2023) acham fertilidade e paridade "not always predictive". Goldman et al. (2024) defendem o contrário, com compressão correlacionada a desempenho e efeito maior em modelos menores. A literatura não fecha a questão, e por isso o bits por byte no modelo treinado é a métrica que decide.

**O que manter fixo ao comparar tokenizadores.** O TokSuite (Altıntaş et al., 2025) fixa arquitetura, dados, orçamento e inicialização e varia só o tokenizador. Tao et al. medem os dados em caracteres, e não em tokens. Para este projeto, isso se traduz em bytes de treino iguais entre braços, passos e FLOPs relatados, várias sementes, tamanho de vocabulário igual e bits por byte no mesmo conjunto retido.

## 7. Testes de conformidade

Nenhuma suíte padronizada foi encontrada. Duas bases reutilizáveis têm licença MIT: os testes do `tiktoken`, com ida e volta por propriedades geradas pelo Hypothesis, ida e volta de cada id isolado, entradas repetitivas e especiais proibidos, e os testes de BPE do curso CS336 de Stanford, com fusões iguais a uma referência, especiais fora das fusões e limite de tempo. O TokEval (Meister, 2026) reúne 18 métricas intrínsecas, entre elas fidelidade de reconstrução e integridade de tokens UTF-8.

O contrato do projeto passa a exigir, no mínimo:

1. Ida e volta em texto aleatório e no corpus em NFC.
2. Determinismo do treino, com dois treinos sobre a mesma entrada gerando arquivos idênticos por hash.
3. Ids sempre em $[0, V)$.
4. Nenhum id reservado produzido a partir de texto.
5. Decodificação de cada id isolado sem erro.
6. Igualdade das fusões entre a implementação didática e a da biblioteca sobre a amostra do notebook.

## 8. Material didático existente

O minbpe tem exercícios em `exercise.md` (MIT), e o vídeo "Let's build the GPT Tokenizer", de Karpathy, acompanha o código. O curso de LLM do Hugging Face cobre BPE, WordPiece e Unigram no capítulo 6 (Apache-2.0), com o aviso de que as implementações não são eficientes. O *Dive into Deep Learning*, na seção 15.6 (texto CC BY-SA 4.0), traz o BPE com a ressalva da segmentação gulosa. O curso CS336 e o nanochat (MIT) oferecem testes e scripts de treino e avaliação de tokenizador. Para a etapa 2 deste projeto, essas fontes servem de referência de formato. O notebook próprio segue o padrão do notebook da etapa 1.

## 9. Consequências para as decisões

As escolhas finais estão na [decisão do tokenizador](decisao_tokenizador.md), que revisou o tamanho de vocabulário e a implementação após medida.

- O tokenizador de referência candidato é o BPE em nível de byte, sem token desconhecido e com ida e volta garantida.
- O corpus é normalizado em NFC antes do tokenizador, que não normaliza nada.
- O notebook implementa o BPE para ensinar e treina o experimento com a biblioteca, com teste de igualdade das fusões.
- O vocabulário candidato fica entre 4.096 e 8.192 tokens, com embedding compartilhado, ou seja, de 10% a 25% dos parâmetros na faixa do projeto. A escolha precisa de medida, porque não há ótimo publicado nessa escala.
- O bits por byte é a métrica que decide. Compressão, fertilidade e Rényi são diagnósticos.

Referências completas, versões e licenças em [referencias/REFERENCIAS.md](../referencias/REFERENCIAS.md).
