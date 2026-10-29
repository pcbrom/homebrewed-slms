# Pesquisa: corpora em português

Levantamento de 22/10/2026. Licenças, tamanhos e formatos conferidos nas páginas de origem listadas ao final, salvo quando marcado como incerto ou estimado. A escolha final dos corpora é objeto de decisão própria, e este documento registra o terreno sobre o qual ela será tomada.

## 1. O que se procura

O projeto precisa de dois corpora. O corpus A, de demonstração, deve ter estilo reconhecível, para que quem roda o notebook perceba quando o modelo começa a aprender, e licença que permita redistribuição com a cláusula não comercial do repositório. O corpus B serve para verificar que a troca de domínio só exige trocar o adaptador, e por isso precisa ser de um domínio bem distinto do A.

## 2. Base legal

No Brasil, a proteção patrimonial dura setenta anos "contados de 1° de janeiro do ano subseqüente ao de seu falecimento" (Lei 9.610/1998, art. 41). A mesma lei não protege "textos de tratados ou convenções, leis, decretos, regulamentos, decisões judiciais e demais atos oficiais" (art. 8º, IV), e atribui ao Estado "a defesa da integridade e autoria da obra caída em domínio público" (art. 24, §2º), o que na prática obriga a atribuir a autoria corretamente.

| Autor | Morte | Domínio público no Brasil desde |
|---|---|---|
| José de Alencar | 1877 | 1948 |
| Machado de Assis | 1908 | 1979 |
| Aluísio Azevedo | 1913 | 1984 |
| Lima Barreto | 1922 | 1993 |
| Júlia Lopes de Almeida | 1934 | 2005 |
| Oswald de Andrade | 1954 | 2025 |

Os anos de morte foram tomados da listagem do Project Gutenberg, e o início do domínio público é o ano de morte mais 71. Ilustrações seguem regra própria: no ebook 79326 (*Pau brasil*), as ilustrações de Tarsila do Amaral, morta em 1973, continuam protegidas.

Os termos do Project Gutenberg distinguem a obra da marca. O §1.C diz que o projeto não reivindica o direito de impedir cópia e distribuição "as long as all references to Project Gutenberg are removed". Mantidas as referências, a licença do projeto precisa acompanhar o arquivo (§1.E.1 e §1.E.4). O crédito em seção de agradecimentos, referências ou colofão não conta como uso da marca.

## 3. Candidatos

| Candidato | Licença | Redistribuível | Tamanho | Observação |
|---|---|---|---|---|
| Project Gutenberg, Machado de Assis (12 obras em português) | Obras em domínio público no Brasil e nos EUA. A licença do Gutenberg rege a marca | Sim, com as referências ao Gutenberg removidas | 4,34 MB brutos. Prosa sem poesia, cerca de 3,8 MB (estimado) | Sem login. A política de robôs declara o site "intended for human users only" |
| Project Gutenberg, outros romancistas brasileiros | Idem | Idem | 11,09 MB brutos em 34 arquivos, Machado incluído | Alencar, Aluísio Azevedo, Lima Barreto, Pompéia, Taunay, Júlia Lopes de Almeida e outros |
| Hugging Face `ronenfilho/machado-de-assis-corpus` | CC0 1.0 declarada pelo autor do conjunto | Pela declaração, sim | 14,58 MB em JSONL, 1.793 textos | Criado em 27/09/2026 por um único autor. Junta Gutenberg e Wikisource, e a compatibilidade da CC0 com a origem Wikisource não foi conferida |
| Wikisource em português | Site em CC BY-SA 4.0. Obras em domínio público seguem em domínio público, e a política não trata explicitamente a transcrição (incerto) | Sim para as obras, incerto quanto à transcrição | Não medido | O texto transclui páginas de digitalização e precisa ser extraído pela API de parse |
| Portal Domínio Público (MEC) | Licença geral não conferida. Algumas obras em CC BY-SA 2.5 BR | Depende da obra | Não medido | Respondeu HTTP 403 com desafio Cloudflare a cliente sem navegador. Acervo em PDF |
| Brasiliana Guita e José Mindlin | A biblioteca digital não declara licença | Incerto | 2.874 livros | PDF digitalizado, OCR não verificado |
| Hugging Face `wrpbr/legislacao-federal-brasileira` | ODbL 1.0 sobre a base, com crédito a "OKF Index". Os textos são atos oficiais (art. 8º, IV) | Sim. Subconjunto publicado herda a ODbL | 44,5 MB em JSONL comprimido, 29.413 normas de 1822 a 2026 | Criado em 18/10/2026. Revisão a fixar: `49e34866a416568546dcb83b16f4158215dfd007` |
| Corpus Carolina, subcorpus `leg` | Licença por documento no cabeçalho TEI. Só os cabeçalhos estão em CC BY 4.0 | Por documento, não conferido | 4,2 GB | A etiqueta CC BY 4.0 do Hugging Face não descreve o conteúdo |
| Wikipédia em português (`wikimedia/wikipedia`, 20231101.pt) | CC BY-SA 4.0 e GFDL pelos Termos de Uso da Wikimedia | Sim, com atribuição e mesma licença | 1.112.246 artigos, 1,58 GB | Mistura pt-PT e pt-BR |
| BrWaC | Sem licença declarada. Restrito a pesquisa acadêmica, sem uso comercial | Não | 3,53 M documentos | Download manual por formulário |
| Quati | CC BY 4.0 declarada | Incerto | 1,17 GB na versão menor | A origem no ClueWeb22 sugere licença de terceiros, não conferida |
| Agência Brasil | Reprodução autorizada a veículos de comunicação com fins jornalísticos | Não, para corpus público | Não empacotado | Parte do conteúdo de agências de terceiros proíbe reprodução |

Dois candidatos foram descartados por falta de licença declarada ou por acesso mediante aprovação manual: `br-llm-data/books-gutenberg-project-pt-br` e `arubenruben/brazilian_literature`.

## 4. Qualidade dos textos

**Ortografia.** As edições em domínio público preservam a grafia anterior às reformas ortográficas. No *Dom Casmurro* do Gutenberg, que reproduz a edição Garnier de 1899, aparecem "elle" (228 ocorrências), "ella" (177), "annos" (73), "egreja" (22) e "theatro" (16). Na legislação federal, a grafia muda ao longo do tempo dentro da própria base, com "sôbre" e "Fôrças" em textos de 1955.

**Formato do Gutenberg.** UTF-8 com quebra de linha CRLF e linhas de cerca de 72 caracteres. O texto fica entre as marcas `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` e `*** END OF ...`, com cerca de 20 KB de licença e créditos fora delas. Itálico vem como `_texto_`, diálogo como `--`, e há folha de rosto antes do primeiro capítulo. Os capítulos aparecem como algarismos romanos sozinhos numa linha, o que permite cortar por capítulo.

**Duplicatas.** O *Dom Casmurro* do Wikisource transcreve a mesma edição do Gutenberg. A lista de páginas do `ronenfilho/machado-de-assis-corpus` sugere capítulos de romances que também estão no Gutenberg, sem verificação no conteúdo.

**Legislação.** O texto da norma está na coluna `texto`, que começa com o título repetido e um marcador interno (`###...@@@`). Números de CPF e RG foram substituídos por `[CPF removido]`. O comprimento vai de 115 a 907.830 caracteres, com mediana de 1.381. Leis que alteram outras reproduzem artigos do texto alterado, o que pode gerar quase-duplicatas entre documentos, hipótese ainda não medida.

## 5. Volume de texto diante do dimensionamento

O corpus de Machado de Assis no Gutenberg é pequeno diante do que um modelo na faixa do projeto pede. Pelo [dimensionamento](DIMENSIONAMENTO.md), um modelo de cerca de 30M de parâmetros tem alvo de cerca de 600M de tokens. Num tokenizador em nível de byte cada token cobre ao menos um byte, de modo que esse alvo corresponde a pelo menos 600 MB de texto. Com cerca de 3,8 MB de prosa, o treino nesse corpus fica pelo menos duas ordens de grandeza abaixo do alvo e passa obrigatoriamente por muitas épocas sobre o mesmo texto. O regime de dados escassos, e não a velocidade da GPU, é a condição de trabalho do corpus A, e a medida de memorização deixa de ser acessória.

O corpus B tem folga: a legislação federal tem cerca de 156 milhões de caracteres na coluna de texto, estimados pela média do comprimento vezes o número de linhas.

## 6. Opções levantadas para a decisão

As escolhas finais estão na [decisão de corpus](decisao_corpus.md), que substituiu a fonte do corpus B pela fonte primária.

**Corpus A.** Machado de Assis do Gutenberg, limpo e versionado no repositório sem as referências ao Gutenberg, com crédito no README. Para ampliar o volume, há duas vias: contos e crônicas do Wikisource, com deduplicação contra o Gutenberg, ou outros romancistas brasileiros do Gutenberg, o que dilui o estilo de um único autor. A divisão por obra mede generalização para obra inédita do mesmo autor, e a divisão por capítulo mede só dentro da distribuição.

Limpeza mínima levantada: cortar entre as marcas START e END, remover a folha de rosto, converter CRLF em LF e normalizar em NFC, juntar as linhas de cada parágrafo, remover marcas de itálico e notas de transcrição, tratar `--` de forma consistente e excluir traduções e poesia. A modernização ortográfica não foi considerada, por não haver ferramenta verificada e porque a grafia faz parte do estilo.

**Corpus B.** Legislação federal, com revisão fixada e baixada pelo notebook, sem versionar o texto no repositório, o que evita criar base derivada sob ODbL. Limpeza levantada: usar a coluna `texto`, remover título repetido e marcador interno, normalizar em NFC, deduplicar por hash, filtrar por período para reduzir a mistura ortográfica e amostrar até volume comparável ao do A, para que a comparação entre domínios não se confunda com a quantidade de dados. Divisão por norma, por hash do identificador, ou temporal, com o teste nos anos mais recentes. A alternativa ao B é a Wikipédia em português, também baixada pelo notebook por causa da CC BY-SA.

## Fontes

Lei 9.610/1998: https://www.planalto.gov.br/ccivil_03/leis/l9610.htm

Licença do Project Gutenberg: https://www.gutenberg.org/policy/license.html

Política de robôs do Project Gutenberg: https://www.gutenberg.org/policy/robot_access.html

Livros em português no Project Gutenberg: https://www.gutenberg.org/browse/languages/pt

Direitos de autor no Wikisource: https://pt.wikisource.org/wiki/Wikisource:Direitos_de_autor

Termos de Uso da Wikimedia: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use

ODbL 1.0: https://opendatacommons.org/licenses/odbl/1-0/

Legislação federal brasileira: https://huggingface.co/datasets/wrpbr/legislacao-federal-brasileira

Corpus de Machado de Assis: https://huggingface.co/datasets/ronenfilho/machado-de-assis-corpus

Corpus Carolina: https://huggingface.co/datasets/carolina-c4ai/corpus-carolina

Wikipédia em português: https://huggingface.co/datasets/wikimedia/wikipedia

BrWaC: https://huggingface.co/datasets/UFRGS/brwac

Quati: https://huggingface.co/datasets/unicamp-dl/quati

Biblioteca digital da Brasiliana Guita e José Mindlin: https://digital.bbm.usp.br/handle/bbm/1

Agência Brasil, página institucional: https://agenciabrasil.ebc.com.br/sobre

Números de ebook de Machado de Assis em português no Gutenberg: 55752 (*Dom Casmurro*), 54829 (*Braz Cubas*), 55682 (*Quincas Borba*), 57001 (*Papeis avulsos*), 61653 (*Poesias completas*), 33056 (*Historias sem data*), 53101 (*A mão e a luva*), 67162 (*Helena*), 56737 (*Esaú e Jacob*), 55797 (*Memorial de Ayres*), 67780 (*Yayá Garcia*), 67935 (*Reliquias de casa velha*).
