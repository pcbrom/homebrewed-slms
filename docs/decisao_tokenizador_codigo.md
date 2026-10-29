# Decisão: tokenizador por código de comprimento variável, no estilo Morse (M10)

**Estado:** experimento concluído em 27/10/2026, com emenda de viabilidade registrada depois de dados parciais. Proposta do autor, publicada com efeito didático por decisão de 29/10/2026. **Código:** `experimentos/m10_codigo/`. **Dados:** `resultados/m10/`, `tokenizador/m10_*` e `runs/m10-*`.

## Questão

Um modelo de linguagem pequeno aprende melhor, com o mesmo orçamento de computação, sobre uma língua de código desenhada para carregar mais informação por bit do que sobre o texto tokenizado por BPE? A pergunta interessa a quem treina em casa, porque o tokenizador fixa o comprimento das sequências, o tamanho da camada de saída e, com eles, o custo de cada época.

## A proposta

O modelo não aprende sobre texto humano tokenizado, e sim sobre uma língua de código $C$, que não precisa ser uma língua humana. São três peças.

- Ida determinística $T: \text{texto} \to C$, sem perda, desenhada à mão por regras fixas e não aprendida do corpus.
- Treino e inferência inteiramente em $C$.
- Volta determinística $T^{-1}: C \to \text{texto}$, com $T^{-1}(T(s)) = s$ exatamente.

A densidade é buscada em três frentes ao mesmo tempo, com sequências mais curtas (menos símbolos por texto), símbolos mais informativos (vocabulário mais rico) e menos bits no total. Como $T$ é bijeção sobre sequências discretas, $p_\theta(T(s)) = p_\theta(s)$, e os bits por byte do texto original continuam sendo a régua justa contra o BPE. O contrato do tokenizador do projeto (ida e volta exata, testes U1 a U6) é o formato de encaixe de $T$, e o treino aceita qualquer tokenizador que o cumpra pela implementação `externo` de `slm/tokenizador.py`.

O desenho segue o código Morse, um código de comprimento variável em que as unidades mais frequentes recebem as palavras de código mais curtas, como o "E" de um único ponto. Três escolhas ficam abertas no desenho: a unidade codificada (letra, sílaba, morfema ou palavra), o tamanho do alfabeto de $C$ e o tratamento das fronteiras, por separador explícito, como as pausas do Morse, ou por código livre de prefixo.

## Implementação

`experimentos/m10_codigo/codigo_morse.py` implementa $T$ sobre unidades mistas: as 4.096 pré-palavras mais frequentes do treino ficam inteiras e as demais são cortadas pela regra de morfema de `experimentos/m10_codigo/morfema.py`, escrita à mão para o português, inclusive a grafia de época do corpus A. Cada unidade recebe uma palavra de código de um código de Huffman de base $A$, livre de prefixo, construído com as frequências do treino e congelado antes de qualquer treino. Unidade ausente da tabela é soletrada entre dois escapes, por caractere do treino e por byte UTF-8 quando o caractere nunca foi visto, de modo que a ida e a volta são exatas para qualquer texto.

## Medidas de ordem zero

Medidas de `experimentos/m10_codigo/medidas_codigo.py` e `medidas_morfema.py`, corpus A, códigos construídos no treino e medidos na validação (539.572 bytes). São frequências de unidades, sem contexto, e não dizem o quanto um modelo aprende sobre cada código.

| Variante | Símbolos por byte | Bits nominais por byte | Entropia das unidades por byte | Informação por bit |
|---|---|---|---|---|
| BPE, V = 4.096 | 0,285 | 3,42 | 2,75 | 0,80 |
| BPE, V = 8.192 | 0,258 | 3,35 | 2,55 | 0,76 |
| Mista 4.096, A = 4.096 | 0,324 | 3,89 | 2,84 | 0,73 |
| Mista 4.096, A = 256 | 0,407 | 3,26 | 2,84 | 0,87 |
| Morfema (aproximação), A = 256 | 0,459 | 3,67 | 3,11 | 0,85 |
| Palavra, A = 256 | 0,475 | 3,80 | 3,31 | 0,87 |
| Mista 4.096, A = 2 | 2,841 | 2,84 | 2,84 | 1,00 |

O separador custa caro. Com fronteira marcada por pausa, a mista 4.096 com A = 256 gasta 5,30 bits por byte, 62,8% a mais que a versão livre de prefixo. As três frentes se trocam por uma identidade aritmética, bits por byte igual a símbolos por byte vezes $\log_2$ do alfabeto, e é o tamanho do alfabeto que decide qual delas se ganha. Com alfabeto de 256, a mista 4.096 vence o BPE de 4.096 em bits nominais por byte (3,26 contra 3,42) e em informação por bit (0,87 contra 0,80) e perde em comprimento, com sequência 1,43 vez mais longa. As unidades do BPE, porém, têm entropia de ordem zero menor por byte que qualquer inventário morfológico testado. A regra de morfema revista, com derivação e prefixo, chega a 2,92 bits de entropia por byte, contra 2,84 da aproximação por lista de sufixos e 2,75 do BPE, e só o treino diz se as unidades morfológicas compensam isso pelo contexto.

## Experimento treinado

Mesma rede da etapa 1 (6 camadas, $d = 384$), corpus A, mesma partição e mesma régua, bits por byte do texto original na validação, modo autorregressivo, hiperparâmetros congelados no M6 (taxa $3 \times 10^{-4}$, dropout 0,2) e 3 sementes.

| Braço | Tokenização | Alfabeto do modelo | Papel |
|---|---|---|---|
| B1 | BPE de referência | 4.096 + 4 especiais | referência |
| B0 | bytes, sem fusões | 256 + 4 | controle de alfabeto pequeno sem unidades |
| C1 | código sobre a mista 4.096 com a regra, livre de prefixo | 256 + 4 | braço principal |
| C2 | o mesmo código com alfabeto de 4.096 | 4.096 + 4 | separa unidade e código do tamanho do alfabeto |
| C3 | código sobre a mista com a aproximação de sufixos | 256 + 4 | testa se a regra linguística importa |

Cada braço recebeu o mesmo total de FLOPs que B1 em 16 épocas e, num segundo ponto, em 256 épocas, com o número de épocas ajustado pela densidade da partição de treino e pelo tamanho da camada de saída (`resultados/m10/m10_plano.json`). Duas janelas são desfecho, lado a lado, por decisão de 26/10/2026: a mesma janela em símbolos ($L = 256$ em todos os braços) e o mesmo contexto em bytes. A tabela do código foi congelada com hash antes de qualquer treino, e reconstruí-la do zero com os comandos abaixo reproduz o mesmo hash nos três códigos. Os portões foram a ida e volta exata em todas as partições dos corpora A e B (G0, aprovado para C1, C2 e C3, sem nenhuma falha), a volta das sequências geradas (G1), o orçamento igual conferido nos registros de cada corrida (G2) e a sentinela de precisão em fp16 (G3).

## Emenda de 27/10/2026: critério de viabilidade

A leitura pré-registrada comparava C1 com B1: promissor se o intervalo da diferença ficasse inteiro abaixo de zero, inconclusivo se contivesse zero e negativo se ficasse acima. Com 39 de 54 corridas já vistas, a pergunta passou de superioridade sobre o BPE para viabilidade do código. A emenda foi feita depois de dados parciais e fica registrada como tal, e a leitura original continua relatada ao lado. O código é viável nesta escala quando cumpre quatro critérios: ida e volta exata, aprendizado sobre o código (bits por byte do modelo abaixo da entropia de ordem zero das unidades e dos símbolos, na mesma validação), volta das gerações e custo frente ao BPE medido com intervalo, como preço da abordagem e não como critério de aprovação.

## Resultados

Diferença contra o BPE (B1) em bits por byte de validação, intervalo de 95% por $t$ sobre 3 sementes. B1 fica em 1,954 com 16 épocas e 2,849 com 256.

| Braço | 16 épocas, janela em símbolos | 16 épocas, janela em bytes | 256 épocas, janela em símbolos | 256 épocas, janela em bytes |
|---|---|---|---|---|
| C1, código com a regra | +0,745 [0,736, 0,754] | +0,763 [0,744, 0,781] | +0,987 [0,942, 1,031] | +0,882 [0,824, 0,941] |
| C2, alfabeto de 4.096 | +0,706 [0,691, 0,721] | +0,694 [0,677, 0,710] | +1,539 [1,471, 1,608] | +1,548 [1,456, 1,641] |
| C3, morfema aproximado | +0,642 [0,627, 0,657] | +0,666 [0,657, 0,674] | +0,749 [0,717, 0,781] | +0,676 [0,655, 0,698] |
| B0, bytes | +0,797 [0,760, 0,834] | +1,414 [1,405, 1,422] | −1,000 [−1,024, −0,976] | −1,010 [−1,026, −0,993] |

Pela leitura original pré-registrada, o resultado de C1 frente ao BPE é negativo nos dois pontos. Pelos critérios de viabilidade, o código é viável nesta escala.

1. Ida e volta exata: aprovado (G0).
2. Aprendizado sobre o código: aprovado no ponto de 16 épocas. C1 fica em 2,70 bits por byte, contra 2,95 da ordem zero dos símbolos e 2,92 da ordem zero das unidades, C3 em 2,60 contra 2,90 e 2,84, e C2 em 2,66 contra 3,05 e 2,92. A 256 épocas os três ficam acima das referências de ordem zero, por sobreajuste, como o próprio BPE (2,849 contra 2,747 da sua ordem zero).
3. Volta das gerações: em 32 amostras por modelo, palavras de código inválidas aparecem em 0 a 5 amostras conforme o braço e o ponto, e entre 3 e 9 amostras terminam com uma palavra de código incompleta, porque a geração para no limite da janela no meio dela, e a volta descarta esse resto. Os textos decodificados são português na grafia da época, com a coerência própria de um modelo deste tamanho.
4. Custo frente ao BPE no melhor ponto de cada braço (16 épocas): de +0,64 a +0,76 bit por byte.

O controle por bytes (B0), com computação igual, é o pior braço em 16 épocas e o melhor em 256, cerca de 1 bit por byte abaixo do BPE. Igualar a computação faz o modelo de bytes passar menos vezes pelo texto, 0,32 vez as épocas do BPE, e ele sobreajusta menos. Esse controle entra no artigo, na tabela do controle por bytes, e a explicação pelo número de passagens não foi medida à parte.

## Consequências

Nesta escala, a densidade de ordem zero não se converteu em aprendizado. Os três códigos custam de 0,64 a 0,76 bit por byte a mais que o BPE no melhor ponto, e a regra linguística (C1) ficou atrás da aproximação crua (C3). O material fica no repositório como exemplo didático de como encaixar um tokenizador próprio no contrato, medi-lo com orçamento igual e ler o resultado com critérios fixados antes, inclusive quando a pergunta muda no meio do caminho e a mudança precisa ficar registrada. As corridas gravam o caminho original dos tokenizadores (`privado/tokenizador_*`), que no repositório corresponde a `tokenizador/m10_*`.

## Reprodução

```bash
python experimentos/m10_codigo/codigo_morse.py construir --A 256 --topo 4096 --variante prefixo --saida tokenizador/m10_C1
python experimentos/m10_codigo/codigo_morse.py construir --A 4096 --topo 4096 --variante prefixo --saida tokenizador/m10_C2
python experimentos/m10_codigo/codigo_morse.py construir --A 256 --topo 4096 --variante aproximacao --saida tokenizador/m10_C3
python experimentos/m10_codigo/codigo_morse.py verificar tokenizador/m10_C1
python experimentos/m10_codigo/medidas_codigo.py
python experimentos/m10_codigo/medidas_morfema.py
python experimentos/m10_codigo/m10_experimento.py plano
python experimentos/m10_codigo/m10_experimento.py rodar --modo autoregressivo --max-min 60
python experimentos/m10_codigo/m10_experimento.py analisar
python experimentos/m10_codigo/m10_viabilidade.py --dispositivo cuda
```
