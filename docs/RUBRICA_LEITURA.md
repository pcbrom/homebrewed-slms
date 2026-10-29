# Rubrica da leitura cega

**Estado:** aprovada pelo autor em 26/10/2026 e congelada no pré-registro `preregistro/m9_leitura.json`, parte do portão G0 da [decisão de orçamento](decisao_orcamento.md). Não muda até o fim da leitura.

A leitura cega mede até onde o texto gerado chega em três níveis de coerência, N2 a N4, em cada ponto da grade de treino. O nível N1, ortografia e palavras válidas, é medido automaticamente pela taxa de léxico e não entra na leitura. A leitura não compara os modos de geração, que se comparam só por bits por byte. Ela serve para definir o roteiro da aula, isto é, quanto treino é preciso para cada nível aparecer.

## O que o avaliador recebe

Um arquivo `lote_<n>.csv` com duas colunas, `item_id` e `texto`. Cada texto tem 60 palavras, recortadas a partir de uma posição sorteada, e por isso pode começar e terminar no meio de uma frase. O lote mistura, em ordem aleatória, textos gerados por modelos, excertos de Machado de Assis e excertos de Machado com as palavras embaralhadas. Nada no texto, no identificador ou na posição indica a origem do item, e o avaliador não deve tentar adivinhá-la. Todos os itens se avaliam da mesma forma.

## Regras gerais

- A grafia é a das edições de época ("elle", "annos", "n'aquella", "Capitú"). Grafia antiga não é erro.
- O corte nas pontas não conta. A frase incompleta no começo e a frase incompleta no fim do texto não se avaliam em N3, e as palavras partidas pelo corte não pesam em N2.
- Cada nível se avalia de forma independente. Um texto pode ter nota 2 em N2 e nota 0 em N4.
- Na dúvida entre duas notas, a menor.
- Uma célula fica em branco só quando o texto não permite avaliar aquele nível, por exemplo um texto ilegível por caracteres corrompidos. Célula em branco conta como falha no item.

## N2, sintaxe local

**Pergunta.** os grupos curtos de palavras vizinhas, de três a cinco palavras, são possíveis em português?

| Nota | Critério |
|---|---|
| 0 | A maior parte dos grupos curtos não é possível em português, como "de a o que lhe tinha". |
| 1 | Há grupos possíveis e grupos impossíveis em proporções parecidas. |
| 2 | Quase todos os grupos curtos são possíveis, com no máximo uma ou duas falhas isoladas de concordância ou de regência. |

## N3, frase gramatical completa

**Pergunta.** as frases que começam e terminam dentro do texto são gramaticais?

| Nota | Critério |
|---|---|
| 0 | Nenhuma frase completa do texto é gramatical, ou o texto não tem frase completa. |
| 1 | Pelo menos uma frase completa é gramatical, mas outras têm erros que impedem a leitura. |
| 2 | Todas as frases completas do texto são gramaticais. Falhas leves de pontuação não tiram a nota. |

## N4, coerência de parágrafo

**Pergunta.** as frases se ligam num fio de sentido, como num parágrafo escrito por alguém?

| Nota | Critério |
|---|---|
| 0 | As frases não têm relação entre si, ou o texto não se lê como sequência de ideias. |
| 1 | Há relação entre algumas frases, com saltos de assunto, personagens que surgem e somem sem motivo ou referências que não se resolvem. |
| 2 | O texto se lê como um trecho de parágrafo, com assunto, personagens e referências mantidos do começo ao fim. |

## Resposta

Um arquivo `respostas_<avaliador>_<n>.csv` por lote, com as colunas `item_id`, `N2`, `N3` e `N4` e notas 0, 1 ou 2. O nome do avaliador é o mesmo em todos os lotes. A chave que liga cada item à origem fica fora do alcance dos avaliadores até o fim da leitura.

## Como as notas são usadas

O documento de decisão fixa as regras, aqui resumidas para o avaliador. Os pares de excerto humano e embaralhado servem de controle de atenção, e um avaliador sai da análise se não der ao excerto humano soma de notas estritamente maior que ao embaralhado em pelo menos 18 de 20 pares. Entre os avaliadores que ficam, a concordância é medida pelo alfa de Krippendorff ordinal, com limiar de 0,667, com e sem o autor. Um nível conta num ponto da grade quando pelo menos 10 de 12 itens de um modo têm mediana das notas de pelo menos 2.
