# Decisão: corpora da etapa 1

**Estado:** aceita pelo autor em 22/10/2026. **Data:** 22/10/2026. **Veredito do debate adversarial:** condicional, sem objeção fatal remanescente, convergido na rodada 3.

## Questão

Que textos usar para o corpus de demonstração (A), que precisa de estilo reconhecível, e para o corpus de verificação de troca de domínio (B), e como dividi-los sem vazamento entre treino, validação e teste.

## Decisão

### Corpus A: Machado de Assis

**Composição.** As obras de Machado de Assis em português no Project Gutenberg, em prosa, sem *Poesias completas* (61653) e sem *Relíquias de casa velha* (67935), que contém um soneto e duas comédias. Ficam 10 obras. Traduções feitas por Machado e o volume em inglês ficam fora. Obras que não cumprirem o bloco mínimo saem do teste, e a exclusão entra neste documento.

**Limpeza.** Corte entre as marcas `START` e `END` do Gutenberg, o que remove todas as referências ao projeto, conforme o §1.C da licença dele. Folha de rosto removida, CRLF convertido em LF, normalização NFC, linhas do parágrafo unidas, marcas de itálico removidas e grafia original mantida.

**Segmentação.** Regra de corte por obra, para capítulo em algarismo romano isolado, para "CAPITULO" e para as entradas de data do *Memorial de Ayres*.

**Divisão.** Blocos contíguos de capítulos inteiros em cada obra, um para validação e um para teste, com um capítulo descartado como margem entre eles e fronteira sempre de capítulo. Cada bloco tem pelo menos 5 capítulos e 30 KB por obra e partição, cerca de 30 janelas de 256 tokens. As 10 obras viram os grupos do bootstrap da [decisão de orçamento](decisao_orcamento.md). A afirmação vale para capítulos não vistos dessas obras. A divisão deixando uma obra inteira fora fica como análise secundária, num único orçamento.

**Vazamento de nomes.** A lista de nomes próprios de cada obra é extraída só da partição de treino, depois da divisão, e versionada. Os bits por byte são relatados com e sem os trechos de nome, e a fração dos nomes do bloco já vistos no treino é declarada por obra. Numa amostra de 4 obras, essa fração foi de 0,459 a 0,966, com os nomes ocupando de 1,6% a 2,6% dos bytes do bloco.

### Corpus B: legislação federal

**Composição.** Leis ordinárias de 2019 a 2022, só das classes substantiva e alteradora. As classes de crédito, orçamento e homenagem saem, por serem texto de formulário. Na lista do Senado desses anos, com 728 leis numeradas, a classificação pela ementa deu crédito 23,1%, orçamento 1,1%, homenagem 11,7%, alteradora 32,3% e substantiva 31,9%. A regra de classe é ampliada para "Acrescenta" e "Revoga", e a fração final de cada classe é publicada.

**Obtenção.** A lista vem do serviço de dados abertos do Senado, e o texto vem das páginas compiladas do Planalto, no padrão de URL `_ato2019-2022/{ano}/lei/l{n}.htm`, que funcionou em 40 de 40 páginas de teste. A limpeza remove o texto revogado (`<strike>`) e as anotações editoriais de redação. O volume estimado das duas classes é de cerca de 3,0 MB, comparável aos cerca de 3,2 MB de treino do A, com incerteza larga por vir de 10 páginas por classe.

**Vazamento.** Cada lei entra no grupo da primeira lei citada na ementa, e o grupo inteiro vai para uma única partição. Quase-cópias são detectadas por MinHash. Leis substantivas que alteram outras só no corpo do texto escapam desse agrupamento, e o tamanho do maior grupo e a parcela de texto citado são medidos e declarados.

### Caminho único de obtenção e licença

Os dois corpora limpos são versionados no repositório, cada um com manifesto de URL, sha256 e data de coleta. O notebook só clona o repositório e não baixa nada da fonte. A construção roda uma única vez, localmente.

O texto do A está em domínio público no Brasil e nos Estados Unidos, e o do B é ato oficial sem proteção (Lei 9.610/1998, art. 8º, IV). A licença CC BY-NC-SA 4.0 do repositório não recai sobre esses textos. Cada corpus fica em diretório próprio, com aviso de estatuto e atribuição: o crédito ao Project Gutenberg no A, sem uso da marca, e a fonte Planalto e Senado no B. O manifesto do B registra a data de coleta porque o Planalto reescreve as páginas compiladas.

## Portões, em ordem

| Portão | Critério | Se falhar |
|---|---|---|
| G0 | Tokens do A medidos com o tokenizador da partição de treino | Grade da [decisão de orçamento](decisao_orcamento.md) recalculada |
| G1 | Segmentação de todas as obras e lista de nomes. Todas as obras no teste cumprem o bloco mínimo | Obra sai do teste, com registro neste documento |
| G2 | Auditoria manual da regra de classe em 72 ementas sorteadas. Passa com limite superior de Clopper-Pearson de 95% abaixo de 5%, o que exige zero erros | A regra é revista e auditada em nova amostra |
| G3 | Construção do B: pelo menos 99% das páginas, volume de pelo menos 2,5 MB, maior grupo com no máximo 10% dos bytes | Inclusão de 2015 a 2018, com novo teste de URL, registrada como mudança de amostra |
| G4 | Teste no Colab, feito pelo autor: clone, sha256 do A e do B, uma época curta | Correção do empacotamento |
| G5 | Comparação automática entre as configurações de A e de B: só a configuração e o adaptador mudam | Correção do pacote |

A auditoria do G2 com 50 ementas não poderia passar: mesmo com zero erros, o limite superior de Clopper-Pearson de 95% seria 7,1%. Com 72 ementas e zero erros, o limite fica abaixo de 5%. A conta foi feita com `awk`, como $1 - 0{,}025^{1/n}$.

## Registro da construção

Portão G0, em 23/10/2026, com o tokenizador de $V = 4.096$: o treino do corpus A tem 624.377 tokens (712 mil com o preenchimento das janelas de 256), a validação 153.756 e o teste 133.520.

*Papeis avulsos* (57001) sai do teste. A obra tem 12 contos, e a unidade de corte de uma coletânea é o conto, de modo que os blocos de validação e de teste, com 5 contos cada e as duas margens, consumiriam a obra inteira. Ela fica só com treino e validação. As outras 9 obras ficam no teste e são os grupos do bootstrap. A divisão exige que o treino de cada obra também cumpra o bloco mínimo, o que nas obras de capítulos longos (*Yayá Garcia*, *A mão e a luva*, *Historias sem data*) deixa o treino com 5 a 7 capítulos. O treino do A fica com 2,22 MB, abaixo dos cerca de 3,2 MB estimados antes da construção. Os números por obra e partição estão em [RELATORIO_CONSTRUCAO.md](../corpus/RELATORIO_CONSTRUCAO.md).

## Riscos declarados

- O vazamento de nomes no desenho por blocos é medido, não removido. A generalização para obra inédita depende só da análise secundária.
- O volume do B foi estimado com 10 páginas por classe, e a média de comprimento de leis tende a ser puxada por poucas leis longas.
- O agrupamento pela primeira lei citada não pega alterações feitas no corpo do texto.
- Os termos de uso do portal do Planalto não puderam ser conferidos. O fundamento do versionamento é o art. 8º, IV da Lei 9.610/1998, que cobre o texto do ato.

## Fora do escopo

- O Hugging Face `wrpbr/legislacao-federal-brasileira`, substituído pela fonte primária.
- Ampliação do A com outros romancistas, que diluiria o estilo de um único autor e, pela medida da mesa de corpus, levaria a cerca de 2,4M de tokens, ainda no regime escasso.
- Modernização ortográfica.
