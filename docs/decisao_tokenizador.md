# Decisão: tokenizador de referência da etapa 1

**Estado:** aceita pelo autor em 22/10/2026. **Data:** 22/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente, convergido na rodada 3.

## Questão

Que tokenizador treinar do zero no corpus de demonstração, como garantir que a implementação didática do notebook e a do experimento sejam a mesma coisa, e que contrato adotar para que um tokenizador novo entre na etapa 2 sem alterar o pacote.

## Decisão

**Algoritmo.** BPE em nível de byte, com a expressão regular de pré-tokenização do GPT-2 e sem normalização própria. O corpus é normalizado em NFC antes, uma única vez, e os bytes da métrica são contados sobre esse texto.

**Implementação.** A biblioteca Hugging Face `tokenizers`, com versão fixada e registrada, treina o tokenizador do experimento. O BPE didático, escrito no notebook, treina uma amostra do corpus para ensinar o algoritmo. O desempate do BPE didático é uma convenção escrita: maior contagem, depois o menor par de ids na ordem do mapeamento `bytes_to_unicode`. Com essa convenção, numa amostra de cerca de 440 KB de três obras, as duas implementações produziram as mesmas fusões (1.788 com $V = 2.048$ e 3.836 com $V = 4.096$) e os mesmos ids num texto retido de outra obra (48.463 e 42.717 ids). O treino didático levou de 11 a 12 ms por fusão num núcleo de CPU.

**Especiais.** `[PAD]`, `[BOS]`, `[EOS]` e `[MASK]` nos ids 0 a 3, nunca produzidos a partir de texto, mesmo quando o texto contém literalmente a cadeia de um especial.

**Tamanho do vocabulário.** $V$ é configuração fixa da etapa 1, igual nos dois modos, e não braço comparado. Ele é escolhido por medida no retido, antes de qualquer treino do modelo: o maior valor de $\{1.024, 2.048, 4.096, 8.192\}$ com no máximo 10% dos tokens aprendidos ausentes do retido. O limiar de 10% é convenção declarada antes da medida, sem fonte, e o resultado depende do tamanho do retido, porque um retido maior reduz a ausência e empurra $V$ para cima. Na amostra, o resultado provisório é $V = 2.048$, o que dá ao modelo de 6 camadas com $d = 384$ cerca de 11,4M de parâmetros e 72,0M de FLOPs por token, conforme `scripts/dimensionar.py`. A escolha de $V$ por bits por byte fica para a etapa 2.

**Contrato.** O tokenizador é tratado como caixa-preta, com `encode(str) → ids`, `decode(ids) → bytes`, `vocab_size`, ids especiais, `save` e `load` e metadados. Os testes universais são:

| Teste | Exigência |
|---|---|
| U1 | Ida e volta exata em bytes, no corpus em NFC e em texto aleatório |
| U2 | Codificação determinística entre processos e após `save` e `load` |
| U3 | Ids em $[0, V)$ |
| U4 | Ids 0 a 3 nunca produzidos a partir de texto |
| U5 | Decodificação de cada id isolado em bytes, sem erro |
| U6 | Um tokenizador trivial de bytes, com 260 ids, passa em U1 a U5 sem alteração do pacote |

O U6 demonstra que o contrato não pressupõe BPE. Dois testes específicos de BPE valem só para a referência: igualdade de fusões e igualdade de ids no retido entre a implementação didática e a biblioteca.

## Justificativa

O BPE em nível de byte não tem token desconhecido e garante a ida e volta (Radford et al., 2019). A normalização fora do tokenizador evita que a forma NFD parta "ação" em marcas combinantes e que o NFKC padrão do SentencePiece altere "1ª" e "nº". A regex do GPT-2 parte palavras com hífen e apóstrofo, como "disse-lhe" e "n'aquella", presentes em 2,36% das palavras da amostra. Uma regex para português comprimiu de 0,71% a 1,71% melhor, mas o efeito em bits por byte não foi medido, e compressão não decide (Schmidt et al., 2024). A regex do GPT-2 fica como referência documentada da literatura, e a regex para português abre a bancada da etapa 2.

O vocabulário pesa contra o corpus. Medidas no retido limpo da amostra:

| $V$ | Bytes por token no retido | Tokens aprendidos ausentes do retido | Tokens com menos de 5 ocorrências no retido |
|---|---|---|---|
| 1.024 | 2,534 | 5,5% | 10,3% |
| 2.048 | 3,017 | 8,3% | 28,8% |
| 4.096 | 3,423 | 21,2% | 62,8% |
| 8.192 | 3,734 | 44,4% | 84,6% |

Com $V = 4.096$, a mediana de ocorrências no retido é de 3, o que deixa linhas do embedding vistas poucas vezes, o risco descrito por Land e Bartolo (2024). Escolher $V$ por bits por byte exigiria pelo menos 10 sementes por valor, porque com 3 sementes o intervalo de 95% do desvio vai de $0{,}52\sigma$ a $6{,}28\sigma$, e isso não cabe na etapa 1 multiplicado por quatro valores.

## Portões, em ordem

| Portão | Critério | Se falhar |
|---|---|---|
| G0 | Igualdade de fusões e ids e ida e volta exata na amostra | Aprovado |
| G1 | Divisão da [decisão de corpus](decisao_corpus.md) congelada por sha256, limpeza aplicada antes do tokenizador, nenhum texto de validação no treino do tokenizador | Nada segue |
| G2 | Regra de $V$ aplicada ao retido real limpo, com $V$, fração ausente e tamanho do retido registrados | $V$ provisório substituído |
| G3 | Na CPU do Colab: treino da biblioteca no treino completo em até 2 min, treino didático da amostra em até 2 min e codificação do corpus completo pelas duas | Reduz-se a amostra didática até caber |
| G4 | Igualdade de fusões e ids na versão do `tokenizers` disponível no Colab | Fica a versão que passa, ou nada segue |
| G5 | U1 a U6 aprovados para a referência e para o tokenizador trivial, e os dois testes específicos de BPE aprovados para a referência | Correção do pacote |

## Riscos declarados

- Embedding subtreinado se o $V$ final passar de 2.048.
- O desempate do BPE didático está amarrado à ordem do mapeamento `bytes_to_unicode` do GPT-2, convenção sem significado linguístico, e a equivalência com a biblioteca foi verificada só na versão 0.19.1.
- A regex anglófona mantida tem impacto em bits por byte desconhecido.
- A validação em blocos de capítulos da mesma obra torna a cobertura do vocabulário otimista. A análise com uma obra fora é apresentada junto com o resultado principal.

## Fora do escopo

- Varredura de $V$ por bits por byte e regex para português, que passam à etapa 2.
- Unigram, WordPiece e SentencePiece como referência da etapa 1.

## Registro da medida

Portão G2, em 23/10/2026, no retido real limpo (validação do corpus A, 109 capítulos e 0,54 MB): a fração de tokens aprendidos ausentes do retido foi de 0,4% com $V = 1.024$, 0,9% com 2.048, 3,5% com 4.096 e 12,0% com 8.192. Pela regra, $V = 4.096$. O risco declarado de embedding subtreinado acima de 2.048 se aplica. Na amostra didática de 440 KB do treino, o BPE didático e a biblioteca `tokenizers` 0.23.3 produziram as mesmas fusões e os mesmos ids no retido, com 51 s de treino didático. O tokenizador comprime o treino a 3,56 bytes por token.
