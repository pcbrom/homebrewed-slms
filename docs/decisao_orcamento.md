# Decisão: orçamento, inferência e critério de saída coerente

**Estado:** aceita pelo autor em 22/10/2026. **Data:** 22/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente, convergido na rodada 3 para a versão reduzida descrita aqui.

## Questão

Como comparar o modo de difusão mascarada com o autoregressivo de forma justa, na mesma rede e no mesmo corpus, e como responder, com medida, quanto treino é preciso para o modelo começar a produzir saída coerente.

## Decisão

**Escopo da afirmação.** O corpus de demonstração tem da ordem de 1M de tokens, valor estimado a confirmar por medida, e fica muito abaixo do alvo de dados de Chinchilla para a rede escolhida. O treino roda, portanto, por muitas épocas sobre o mesmo texto, e esse regime é declarado como eixo do experimento, não como defeito. Toda afirmação vale para capítulos não vistos das 10 obras do corpus, até 256 épocas, e nenhuma afirmação se faz sobre obra inédita, sobre tamanho de modelo ou sobre o paradigma fora desse regime.

**Rede.** Arquitetura única de 6 camadas com $d = 384$, com 10,6M de parâmetros fora do embedding e 12,3M no total com o vocabulário de 4.096 medido na [decisão do tokenizador](decisao_tokenizador.md), sem afirmação sobre escala.

**Grade de orçamento.** Cinco pontos, 1, 4, 16, 64 e 256 épocas, cada um uma corrida própria com cronograma de taxa de aprendizado em cosseno fechado na própria duração, porque checkpoints intermediários de uma corrida longa ficam subtreinados em relação ao ponto que representam (Hoffmann et al., 2022, apêndice B). FLOPs e tokens são iguais entre os modos em cada ponto, com o orçamento contado por passo aplicado, conforme a [decisão de precisão](decisao_precisao.md).

**Ponto confirmatório único.** O teste de hipótese se faz só no ponto de 256 épocas, o mais próximo do regime em que Prabhudesai et al. (2025) relatam vantagem da difusão, com $\alpha = 0{,}05$ bilateral. Os pontos de 1, 4, 16 e 64 épocas são descritivos, com intervalo de confiança e sem teste.

**Métrica e critério.** Bits por byte na validação, como razão de somas, com a difusão avaliada pela NELBO exata na forma por número de máscaras. O critério é assimétrico e fica declarado antes da corrida. Se o intervalo da diferença $\Delta = \mathrm{bpb}_{dif} - \mathrm{bpb}_{AR}$ ficar abaixo de zero, a difusão é melhor em verossimilhança e o resultado é conclusivo. Se ficar acima de zero, o autoregressivo supera o limite da difusão, sem conclusão sobre a verossimilhança verdadeira dela. Se contiver zero, o resultado é inconclusivo.

**Margem mínima de efeito.** $\Delta_{min}$ vale um quarto do ganho de bits por byte do autoregressivo entre 4 e 16 épocas, medido na busca de hiperparâmetros e congelado com hash antes do piloto. Em escala logarítmica, equivale ao que o autoregressivo ganha com cerca de 41% mais treino. Se esse ganho for nulo ou negativo, a rede passa a 4 camadas com $d = 256$. O valor de $\Delta_{min}$ e o erro dele são publicados.

**Sementes.** Um piloto externo de 10 sementes por modo no ponto confirmatório estima o desvio $\hat\sigma$ da diferença pareada por semente, e não entra na confirmatória. O número de sementes confirmatórias vem de $t$ pareado com poder de 0,8, $\alpha = 0{,}05$ e $\sigma$ tomado no limite superior do intervalo de 95%, $1{,}83\,\hat\sigma$ com 9 graus de liberdade. Para $\hat\sigma/\Delta_{min}$ igual a 0,5, 0,75 e 1, isso dá 9, 17 e 29 sementes. O teto é de 30 sementes por modo. Acima dele, o ponto é relatado como não resolvível em $\Delta_{min}$. Os pontos descritivos usam 5 sementes.

**Unidade e inferência.** Blocos contíguos de capítulos inteiros por obra, conforme a [decisão de corpus](decisao_corpus.md), com a obra como grupo do bootstrap e a semente como unidade de pareamento. A cobertura do bootstrap é medida no piloto com 1.000 réplicas nulas por troca de sinal nos pares e precisa ser de pelo menos 0,93 no intervalo de 95%. Se falhar, vale um teste de aleatorização sobre as sementes, e a afirmação se restringe ao conjunto de teste.

**Hiperparâmetros.** A busca usa a mesma grade nos dois modos, com 2 sementes por configuração, nos pontos de 16 e 256 épocas. Se as duas melhores configurações diferirem menos que $\hat\sigma$, vence a mais regularizada.

**Reprodução na T4.** No ponto de 16 épocas, a equivalência entre T4 e 5090 é testada por TOST com margem $\Delta_{min}$ e $\alpha = 0{,}05$, com teto de 10 sementes por modo. Sem resolução dentro do teto, a equivalência é relatada como não demonstrada.

## Orçamento de GPU

Conta feita com `awk`, com $U = 10^6$ tokens (estimado), 85,7M FLOP por token (valor com vocabulário de 8 mil, conservador diante dos 72,0M com o vocabulário provisório de 2.048), pico fp16 da RTX 5090 entre 104,75 e 209,5 TFLOPS (não conferido no documento técnico) e MFU de 0,3 (não medido):

| Parcela | Tokens |
|---|---|
| Busca de hiperparâmetros, 9 configurações, 2 sementes, 2 pontos, 2 modos | 9,79B |
| Piloto, 10 sementes, 256 épocas, 2 modos | 5,12B |
| Pontos descritivos, 5 sementes, 2 modos | 0,85B |
| Confirmatória no teto, 30 sementes, 256 épocas, 2 modos | 15,36B |
| Total | 31,12B |

O total corresponde a 11,8 a 23,6 horas de GPU com MFU de 0,3, ou 35 a 71 horas com MFU de 0,1. O teto declarado é de 24 horas. Acima dele, os cortes seguem esta ordem: busca de hiperparâmetros com 3 configurações (24,59B de tokens, 9,3 a 18,6 horas com MFU de 0,3) e, depois, teto confirmatório de 20 sementes. A avaliação da NELBO não entra na conta e é medida à parte.

## Saída coerente e roteiro da aula

A leitura humana define o roteiro da aula e nunca compara os modos, que se comparam só por bits por byte.

**Níveis.** N1 é ortografia e palavras válidas. N2 é sintaxe local. N3 é frase gramatical completa. N4 é coerência de parágrafo.

**N1 automático.** Taxa de palavras presentes no léxico da partição de treino, comparada com a do texto humano de teste. Um dicionário moderno marcaria a grafia de época como erro.

**Filtro de cópia.** Uma amostra é descartada se copiar do treino trecho maior que $k$ palavras, com $k$ igual ao percentil 99 do maior trecho coincidente entre excertos humanos de teste e o treino. A taxa de descarte por ponto da grade, com intervalo, é relatada como métrica de memorização.

**N2 a N4 por leitura cega.** Três avaliadores, o autor e dois alunos, com rubrica de 0 a 2. Por ponto da grade, 12 itens por modo, 3 de cada uma de 4 sementes fixadas, com comprimento igual e ordem aleatória, mais 20 excertos humanos e 20 embaralhados como controle. O alfa de Krippendorff ordinal é calculado só nos itens de modelo e precisa ser de pelo menos 0,667, com e sem o autor. Um avaliador sai se não ordenar o texto humano acima do embaralhado em pelo menos 18 de 20 pares. Um nível conta num ponto com pelo menos 10 de 12 itens com mediana de pelo menos 2, cujo limite inferior de Clopper-Pearson unilateral de 95% é 0,56.

**Quanto treino.** A resposta é o intervalo entre o último ponto sem o nível e o primeiro com ele, por exemplo de 16 a 64 épocas. A resolução da grade é de fator 4.

**Aula na T4.** Com o MFU entre 0,1 e 0,3, a corrida de 16 épocas leva de 70 a 211 segundos, estimativa a confirmar. A aula treina ao vivo até o nível que couber no tempo e mostra os níveis seguintes a partir de checkpoints pré-treinados.

## Portões, em ordem

| Portão | Critério | Se falhar |
|---|---|---|
| G0 | Pré-registro com hash: partição, tokenizador, $U$ medido, grade, rubrica, $k$, ponto confirmatório | Nada roda |
| G1 | Vazão fp16 medida nos dois modos, na 5090 e na T4. Projeção dentro de 24 h após os cortes | Cortes na ordem declarada |
| G2 | Sanidade em 16 épocas contra referência fp32, sem estouro | [Decisão de precisão](decisao_precisao.md) |
| G3 | Busca de hiperparâmetros e $\Delta_{min}$. Ganho do autoregressivo de 4 para 16 épocas maior que zero e erro de Monte Carlo da NELBO abaixo de $\Delta_{min}/4$ | Rede de 4 × 256 |
| G4 | Piloto e cobertura de pelo menos 0,93, com $n$ de até 30 | Ponto não resolvível ou teste de aleatorização |
| G5 | Confirmatória no ponto de 256 épocas | |
| G6 | TOST entre T4 e 5090 no ponto de 16 épocas | Equivalência não demonstrada |
| G7 | Leitura humana | Nível não atingido, relatado como tal |

## Riscos declarados

- $\Delta_{min}$ depende dos dados e da curva do autoregressivo, medida com 2 sementes. Congelado com hash antes do piloto, o valor e o erro dele são publicados.
- A cobertura por troca de sinal mantém as 10 obras fixas e não testa a perda de cobertura com poucos grupos. A afirmação fica restrita a capítulos não vistos dessas obras.
- O piloto externo, o limiar de 0,667 do alfa e a suposição de perda de cobertura com poucos grupos não têm fonte em [referencias/](../referencias/REFERENCIAS.md). São escolhas do projeto.
- Avaliadores sem formação em grafia do século XIX enfraquecem os níveis N2 a N4. O filtro por controles limita o dano.
- A sessão gratuita do Colab não garante GPU nem tempo contínuo, e o TOST pode exigir mais de uma sessão.

## Fora do escopo

- Afirmação sobre obra inédita, sobre tamanho de modelo ou sobre mais de 256 épocas.
- Teste de hipótese nos pontos descritivos.
- Comparação entre modos por leitura humana.

## Registro da medida

Em 23/10/2026, com o corpus e o tokenizador construídos: o treino tem $U = 624.377$ tokens, ou 711.680 posições de janela com o preenchimento, e o custo é de 76,7M FLOPs por token com $V = 4.096$ e $L = 256$. Refeita a conta da seção de orçamento com esses valores, o total no teto de sementes é de 22,1B de tokens, ou 7,5 a 15,0 horas com MFU de 0,3 e 22,5 a 45,0 horas com MFU de 0,1, conforme o pico de fp16 da 5090, ainda não conferido. O MFU medido no portão G1 decide os cortes.

Portão G1, vazão medida em 26/10/2026 na RTX 5090 a 500 W (`scripts/medir_vazao.py`, 60 passos após 15 de aquecimento): em fp16, 1,13 milhão de tokens por segundo no autoregressivo e 1,04 milhão na difusão, com 2,2 GB de VRAM e consumo de 344 W. Em fp32, 448 mil e 430 mil tokens por segundo. Com o pico de 209,5 TFLOPS, o MFU em fp16 é de 0,41 e 0,38. Os 22,1B de tokens do teto de sementes correspondem a cerca de 5,6 horas de treino em fp16, ou cerca de 14 horas em fp32, antes da avaliação. O teto de 24 horas comporta o plano sem cortes.

## Emenda de 27/10/2026: sementes do ponto confirmatório

Aprovada pelo autor antes de qualquer corrida confirmatória e congelada em `preregistro/m7_emenda.json`. No piloto de 10 sementes, a cobertura do bootstrap em dois níveis ficou em 0,918, abaixo do mínimo de 0,93, e a inferência confirmatória passa ao teste de aleatorização por troca de sinal sobre as sementes, com a afirmação restrita ao conjunto de avaliação. A fórmula de poder, com $\hat\sigma = 0{,}0084$ e $\Delta_{min} = 0{,}0907$, pede 3 sementes, mas com 3 sementes o teste de aleatorização tem 8 trocas de sinal e o menor p bilateral é 0,25, o que impede a rejeição a 0,05 com qualquer efeito. O número de sementes confirmatórias passa a ser o maior entre o da fórmula de poder e o menor $n$ com $2 \cdot 2^{-n} < 0{,}05$, ou seja, 6 sementes, com menor p bilateral de 0,031.
