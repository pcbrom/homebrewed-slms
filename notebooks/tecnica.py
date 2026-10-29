"""Technique sections of the notebooks: problems met in the project and the solutions adopted.

TECNICA maps the box number to a list of (kind, text, modes) cells, with kind "md" or "code" and
modes "ambos", "autoregressivo" or "difusao". The generator opens each box section with the
heading. Every problem cites the measurement that revealed it; the project documents in docs/
hold the full record.
"""

AMBOS, AR, DIF = "ambos", "autoregressivo", "difusao"

TECNICA = {
    0: [("md", r"""
**A fórmula curta engana em modelo pequeno.** A estimativa usual de custo, $6N$ operações por token, ignora a atenção e a camada de saída sobre o vocabulário. Num modelo de 4 camadas com $d = 256$ e vocabulário de 8 mil tokens, os dois termos acrescentam 73% ao custo, 65 pontos da camada de saída e 8 da atenção. O `scripts/dimensionar.py` usa a fórmula completa.
""", AMBOS), ("md", r"""
**Estimar, depois medir.** A fração do pico que o treino realmente aproveita (MFU) não sai da ficha técnica. Numa RTX 5090 limitada a 500 W, o modo autorregressivo processou 1,13 milhão de tokens por segundo em fp16. Isso dá MFU de 0,41 quando a atenção conta só o produto $QK^\top$, convenção de Kaplan et al. (2020), e de 0,43 quando conta também o produto pelos valores, convenção do PaLM (Chowdhery et al., 2022). O pico de 209,5 TFLOPS do denominador vem de fontes secundárias e não foi conferido no documento técnico da NVIDIA. Com a vazão medida, o plano de pesquisa caberia em cerca de 5,6 horas de treino, sem contar a avaliação, quando a estimativa sem medida ia de 7,5 a 45 horas.
""", AR), ("md", r"""
**Estimar, depois medir.** A fração do pico que o treino realmente aproveita (MFU) não sai da ficha técnica. Numa RTX 5090 limitada a 500 W, a difusão processou 1,04 milhão de tokens por segundo em fp16, um pouco abaixo do modo autorregressivo (1,13 milhão). Isso dá MFU de 0,38 quando a atenção conta só o produto $QK^\top$, convenção de Kaplan et al. (2020), e de 0,40 quando conta também o produto pelos valores, convenção do PaLM (Chowdhery et al., 2022). O pico de 209,5 TFLOPS do denominador vem de fontes secundárias e não foi conferido no documento técnico da NVIDIA. Com a vazão medida, o plano de pesquisa caberia em cerca de 5,6 horas de treino, sem contar a avaliação, quando a estimativa sem medida ia de 7,5 a 45 horas.
""", DIF)],
    1: [("md", r"""
**Uma obra "em prosa" que não era só prosa.** *Relíquias de casa velha* traz um soneto e duas comédias. Misturar verso e teatro num corpus de prosa muda a distribuição que o modelo aprende, e a obra saiu, junto com as *Poesias completas*. A lição é inspecionar o conteúdo, sem confiar no rótulo da obra.

**Uma divisão que vaza.** Sortear documentos ao acaso colocaria capítulos vizinhos da mesma obra no treino e na validação. A solução foram blocos contíguos de capítulos inteiros, com um capítulo de margem entre o treino e a validação e outro entre a validação e o teste. Os nomes próprios ainda vazam. Numa amostra de 4 obras, examinada antes da construção do corpus, de 46% a 97% dos nomes de um bloco retido já apareciam na parte de treino da mesma obra. Por isso o projeto mede os bits por byte com e sem os trechos de nomes.

**O preço de um bloco mínimo.** Exigir ao menos 5 capítulos e 30 KB na validação e no teste reduziu o texto de treino a 2,22 MB, contra cerca de 3,2 MB estimados. Em *Yayá Garcia*, o bloco de teste ficou maior que o de treino. Toda escolha de divisão custa dado de treino.

**Texto de formulário.** No corpus B, 23,1% das 728 leis de 2019 a 2022 da lista do Senado abrem créditos orçamentários com redação padronizada. Leis de crédito, orçamento e homenagem repetem a mesma fórmula de uma lei para outra, e a decisão do corpus retirou essas classes para que a medida reflita texto legislativo substantivo.
""", AMBOS)],
    2: [("md", r"""
**Mesma letra, bytes diferentes.** Na NFC, "ã" é um único ponto de código. Na NFD, é um "a" seguido de um til combinante. A expressão regular da pré-tokenização separa a marca combinante da letra, e a palavra se desfaz. A célula abaixo mostra o efeito. A solução é normalizar o corpus em NFC uma única vez, antes de tudo, e contar os bytes da métrica sobre esse texto.
""", AMBOS), ("code", r"""
import unicodedata, regex
from slm.tokenizador import GPT2_REGEX
frase = "a ação do coração"
for forma in ("NFC", "NFD"):
    t = unicodedata.normalize(forma, frase)
    print(forma, len(t.encode()), "bytes |", regex.findall(GPT2_REGEX, t))
""", AMBOS), ("md", r"""
**Duas implementações, uma resposta.** Na amostra de 440 KB do corpus A, em 3.463 das 3.836 fusões (90%) mais de um par empatou na contagem máxima, e o primeiro empate apareceu na fusão 56. Nesses casos só a regra de desempate escolhe o par, e duas implementações sem regra escrita podem divergir. Escrever o desempate como convenção (maior contagem, depois o menor par de ids) tornou o BPE didático e o da biblioteca idênticos, fusão a fusão e id a id. Um teste garante isso a cada mudança, e a contagem dos empates sai de `scripts/medir_empates_bpe.py`.

**Vocabulário grande demais para o texto.** Com $V = 8.192$, 12% dos tokens aprendidos no corpus A nunca aparecem no texto de validação. Cada um deles é uma linha do embedding treinada com pouquíssimos exemplos. A regra do projeto escolhe o maior $V$ com no máximo 10% desses tokens ausentes, o que deu $V = 4.096$ no corpus A e $V = 1.024$ no corpus B.
""", AMBOS)],
    3: [("md", r"""
**Comprimento de graça.** Se a difusão recebesse o comprimento do documento sem pagar por ele, enquanto o modo autorregressivo paga pelo seu `[EOS]`, a comparação favoreceria a difusão em até $\log_2 L$ bits por documento. No teste G1 do pacote, com condicionais exatas e janelas de 4 tokens, a vantagem indevida é exatamente a entropia do comprimento, 0,856 nat, ou 1,235 bit por documento, dentro de $10^{-9}$. A solução é prever o preenchimento nos dois modos. O custo é que, com $L = 256$, 12,2% das posições de treino são preenchimento.
""", AMBOS)],
    4: [("md", r"""
**Duas formas da mesma perda, variâncias muito diferentes.** A forma em tempo sorteia $t$ e pondera a perda por $1/t$. Perto de $t = 0$ o peso explode, e a variância do estimador é infinita sem um corte $t \ge \varepsilon$, que por sua vez deixa o limite incompleto. A forma em número de máscaras sorteia $l$ e pondera por $L/l$, que nunca passa de $L$. Pela identidade $\int_0^1 \binom{L}{l} t^{l-1}(1-t)^{L-l}\,dt = 1/l$, as duas têm a mesma esperança, e a forma em $l$ tem a menor variância, o que se mostra em dois passos. Dado $l$, o tempo segue $\mathrm{Beta}(l+1, L-l+1)$ e $\mathbb{E}[1/t \mid l] = (L+1)/l$, e pelo teorema de Rao-Blackwell trocar $1/t$ por essa esperança condicional não aumenta a variância. O segundo passo, registrado na decisão de formulação, compara os segundos momentos, $L \sum_l \mathbb{E}[f_l^2]/l^2$ contra $(L+1) \sum_l \mathbb{E}[f_l^2]/l^2$, e descarta o sorteio inútil de $l = 0$. A célula abaixo compara só os pesos, que é onde está a diferença. Com o corte, o peso em tempo chega perto de 1.000, e sem ele não tem limite, enquanto o peso em número de máscaras nunca passa de $L$. A igualdade das esperanças vale para o estimador completo, peso vezes perda, e não para o peso sozinho.
""", DIF), ("code", r"""
import numpy as np
rng = np.random.default_rng(0)
L_ = 256
t = rng.uniform(1e-3, 1, 100000)
l_ = rng.integers(1, L_ + 1, 100000)
for nome, peso in (("1/t, corte em 1e-3", 1 / t), ("L/l", L_ / l_)):
    print(f"{nome:18s} maior peso {peso.max():8.1f} | dispersão do peso {peso.std():7.2f}")
""", DIF), ("md", r"""
**Um token a menos na entrada.** O deslocamento por `[BOS]` descarta o último token da janela na entrada, e ele só aparece como alvo. Sem o deslocamento, a entrada da posição $i$ conteria o próprio alvo, e a rede poderia copiá-lo em vez de prevê-lo. O deslocamento e a máscara causal, juntos, garantem que a previsão de $w_i$ usa só $w_{<i}$, e a célula de teoria acima confere o deslocamento.
""", AR)],
    5: [("md", r"""
**O logit da máscara.** Se a rede pudesse prever `[MASK]`, parte da probabilidade iria para um token que nunca é a resposta, e a distribuição sobre os textos deixaria de somar 1. O teste do projeto enumera todas as ordens de revelação numa rede minúscula e confere que a soma dá 1, com tolerância de $10^{-9}$. Com o logit da máscara livre, a soma caiu para 0,59, e com o alvo visível para a própria previsão subiu para 1,28, de modo que os dois defeitos são detectados. Por isso o logit de `[MASK]` é $-\infty$ nos dois modos.
""", AMBOS), ("md", r"""
**Onde o dropout age.** O dropout agia só dentro da atenção e passou a agir também no embedding e nos dois ramos residuais de cada bloco. É uma escolha de projeto, sem comparação medida contra a versão anterior.
""", AMBOS), ("md", r"""
**A difusão não precisa de uma arquitetura própria?** Parte dos modelos de difusão para texto usa uma rede própria. SEDD (Lou, Meng e Ermon, 2023) e MDLM (Sahoo et al., 2024) usam um Transformer de difusão que recebe o nível de ruído como entrada extra, por normalização adaptativa, o ruído total $\bar\sigma(t)$ no SEDD e o tempo no MDLM. Ou et al. (2024, Teorema 1) mostraram que, quando o processo é de máscaras, o escore se fatora num escalar analítico que depende só do tempo vezes a distribuição dos tokens limpos dado o contexto visível, e essa distribuição não depende do tempo. A rede só precisa aprender essa distribuição, e a entrada de tempo se torna dispensável. A Tabela 12 do MDLM relata perplexidades semelhantes com e sem ela. RADD (Ou et al., 2024) e LLaDA (Nie et al., 2025) usam um Transformer comum, sem entrada de tempo e sem máscara causal. O projeto segue essa linha por uma razão de desenho experimental, já que, com a mesma arquitetura nos dois modos, uma diferença medida entre eles vem do treino e da geração, e não da rede.
""", AMBOS)],
    6: [("md", r"""
**Estouro no gradiente, não na perda.** O maior número que o fp16 representa é 65.504. A perda já é calculada em fp32 sob autocast, mas o gradiente que chega aos logits é multiplicado pela escala do `GradScaler` ($2^{16}$) e pelo peso da difusão. Num teste com um token mascarado por sequência, a forma $1/t$ dividida pelo número de máscaras estourou nos três tamanhos de lote testados (2, 4 e 64), a forma $1/t$ dividida por $L$ chegou a 64.000 com lote 4, e a forma em $l$ ficou em 16.384. A forma em $l$ limita o pico a $S/B$.
""", DIF), ("md", r"""
**Precisão conferida, não presumida.** Uma sentinela compara, tensor a tensor, o gradiente em fp16 com o gradiente em fp32 no mesmo lote, do início ao fim do treino. Na RTX 5090, o pior erro relativo foi 0,0025 no modo autorregressivo e 0,0099 na difusão, contra uma margem de 0,05, e, em 3 sementes por modo, a diferença de bits por byte entre fp16 e fp32 não passou de 0,0063 em módulo. O julgamento formal dessa equivalência depende da margem $\Delta_{min}$, congelada só depois da busca de hiperparâmetros, e está pendente. Um critério de razão entre erros (o da difusão no máximo o dobro do autorregressivo) reprovava diferenças sem efeito prático, porque comparava dois erros próximos do arredondamento do fp16. Uma emenda registrada o substituiu pela equivalência em bits por byte.
""", AMBOS), ("md", r"""
**Muitas épocas sobre pouco texto.** Na busca de hiperparâmetros, o modo autorregressivo com 256 épocas piorou na validação, de cerca de 1,9 com 16 épocas para algo entre 4,85 e 5,84 bits por byte sem dropout, conforme a taxa de aprendizado. Com dropout de 0,2, ficou entre 2,85, com taxa $3 \times 10^{-4}$, e 3,75, com taxa $3 \times 10^{-3}$. O modelo decora o texto de treino, e é a regularização que o segura.
""", AR), ("md", r"""
**Muitas épocas sobre pouco texto.** Na busca de hiperparâmetros, com uma semente, a difusão com 256 épocas continuou melhorando na validação, até 1,84 bit por byte na configuração com taxa $3 \times 10^{-4}$ e sem dropout, enquanto o modo autorregressivo piorava nas mesmas 256 épocas. Prabhudesai et al. (2025) relatam o mesmo padrão em escala maior e o atribuem ao fato de mascarar posições diferentes a cada passagem funcionar como aumento de dados. Neste projeto esse mecanismo não foi testado, e a confirmação do padrão com várias sementes faz parte do protocolo.
""", DIF)],
    7: [("md", r"""
**Ruído truncado pela precisão.** O sorteio de Gumbel usa $g = -\log(-\log u)$. Zheng et al. (2024) mostram que a truncagem desse ruído já distorce a amostragem em fp32. O ruído truncado age como temperatura mais baixa, as amostras perdem diversidade e a perplexidade gerativa parece melhor do que é. Em fp16 o problema piora, e pelo cálculo do projeto $u$ tem só 2.048 valores possíveis e o maior ruído possível cai para 7,62, contra 16,64 em fp32 e 36,74 em fp64. A solução é sortear em float64. A célula abaixo mede o efeito.
""", AMBOS), ("code", r"""
import math
for nome, dt, p in (("fp16", torch.float16, 11), ("fp32", torch.float32, 24), ("fp64", torch.float64, 53)):
    u = torch.rand(2_000_000, dtype=dt)
    print(f"{nome}: valores distintos de u = {u.unique().numel():>9d} | maior g teórico = {-math.log(-math.log(1 - 2.0 ** -p)):.2f}")
""", AMBOS), ("md", r"""
**Confiança contra acaso.** Revelar primeiro os tokens de maior confiança usa um argmax e descarta o sorteio. O projeto usa a revelação aleatória como padrão, e a revelação por confiança só entra como variante rotulada.
""", DIF)],
    8: [("md", r"""
**Um limite, não um valor.** Na difusão, os bits por byte vêm da NELBO, que é um limite superior. Se a difusão ficar abaixo do modo autorregressivo, ela é melhor. Se ficar acima, a diferença pode ser folga do limite. Essa leitura assimétrica é declarada antes de rodar.
""", DIF), ("md", r"""
**Um valor exato, que serve de régua.** No modo autorregressivo, os bits por byte são a log-verossimilhança exata. A difusão só tem um limite superior. Na comparação, o número autorregressivo é a referência fixa, e só uma difusão abaixo dele produz conclusão.
""", AR), ("md", r"""
**Probabilidade não é confiança.** O diagrama de confiabilidade mostra se a confiança declarada bate com a acurácia observada. A temperatura, ajustada pelo método de Newton na validação, corrige a escala dos logits sem mudar qual token é escolhido.
""", AMBOS)],
    9: [("md", r"""
**Quantas sementes?** Com 3 sementes, o intervalo de 95% do desvio padrão entre sementes vai de 0,52 a 6,28 vezes a estimativa. O projeto estima a dispersão num piloto de 10 sementes e deriva dela o número de sementes da comparação final, com um teto declarado.

**Decidir antes de olhar.** O tamanho mínimo de efeito, a grade de hiperparâmetros e o critério de leitura são congelados com hash antes das corridas que eles governam. Quando um critério precisou mudar depois dos dados, a mudança foi registrada como emenda, com a justificativa.
""", AMBOS)],
    10: [("md", r"""
**Coerente ou copiado?** Um modelo treinado por muitas épocas pode parecer coerente porque reproduz trechos do texto de treino. O filtro de cópia mede o maior trecho copiado e descarta a amostra que passa do que se observa entre textos humanos. A taxa de descarte é relatada como medida de memorização.
""", AMBOS), ("md", r"""
**Sobreajuste também é recitação.** Sem dropout e com taxa $10^{-3}$, o modelo autorregressivo treinado por 256 épocas chegou a 5,6 bits por byte na validação, muito pior que com 16 épocas. Em 16 amostras de 256 tokens por corrida, o maior trecho copiado do texto de treino teve mediana de 4 palavras com 16 épocas e de 144 a 150 palavras com 256 épocas, com máximo de 180. Com dropout de 0,2 e taxa $3 \times 10^{-4}$, as mesmas 256 épocas deram mediana de 5 a 5,5 palavras, ainda com amostras isoladas de 44 e 95 palavras copiadas. O sobreajuste apareceu nas duas formas, como previsões erradas e confiantes no texto novo e como recitação do texto de treino. A medida está em `scripts/medir_copia.py`, e o resultado de cada corrida em `runs/<corrida>/copia.json`.
""", AR), ("md", r"""
**O viés que sobra sem contexto.** Na curva de deleção, com 100% do contexto removido, a rede ainda dá probabilidade à decisão original. Esse resto mede o viés intrínseco do modelo, o que ele sabe sem olhar a entrada.
""", AMBOS)],
}
