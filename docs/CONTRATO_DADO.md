# Contrato de dados

Este documento fixa o formato de cada artefato que passa de um componente para outro. Um componente só lê o que está descrito aqui, e mudança de formato exige mudança deste documento no mesmo commit. Todo texto é UTF-8 em NFC, todo JSON usa chaves em português sem acento, e todo hash é sha256 em hexadecimal minúsculo.

## 1. Corpus (caixa 1)

### `corpus/<A|B>/manifesto.json`

Um objeto por fonte baixada.

| Campo | Tipo | Conteúdo |
|---|---|---|
| `fonte_id` | texto | Número do ebook (A) ou identificador da lei (B) |
| `url` | texto | URL exata baixada |
| `data_coleta` | texto | Data ISO 8601 |
| `sha256_bruto` | texto | Hash do arquivo baixado |
| `sha256_limpo` | texto | Hash do texto limpo produzido |
| `estatuto` | texto | `dominio_publico` (A) ou `ato_oficial_art8_IV` (B) |
| `classe` | texto | Só no B: `substantiva` ou `alteradora` |
| `excluida` | texto ou nulo | Motivo de exclusão, se houver |

### `corpus/<A|B>/documentos.jsonl`

Uma linha por documento.

| Campo | Tipo | Conteúdo |
|---|---|---|
| `doc_id` | texto | Identificador único e estável, por exemplo `A-55752-c017` |
| `grupo` | texto | Obra (A) ou grupo de leis relacionadas (B), unidade do bootstrap |
| `ordem` | inteiro | Posição do documento dentro do grupo |
| `particao` | texto | `treino`, `margem`, `validacao` ou `teste` |
| `texto` | texto | Texto limpo em NFC |
| `bytes` | inteiro | $B_d$, número de bytes UTF-8 de `texto` |

O arquivo `corpus/<A|B>/divisao.json` registra a regra de divisão aplicada, o número de documentos e de bytes por partição e por grupo, e o hash de `documentos.jsonl`. Os documentos marcados `margem` não entram em nenhuma partição usada.

### `corpus/A/nomes/<obra>.txt`

Um nome próprio por linha, extraído só dos documentos de treino da obra.

### `corpus/<A|B>/AVISO.md`

Estatuto do texto, atribuição e a observação de que a licença do repositório não recai sobre ele.

## 2. Tokenizador (caixa 2)

### Interface

```python
class Tokenizador(Protocol):
    vocab_size: int
    especiais: dict[str, int]          # {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[MASK]": 3}
    def encode(self, texto: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    def salvar(self, pasta: Path) -> None: ...
    @classmethod
    def carregar(cls, pasta: Path) -> "Tokenizador": ...
    def metadados(self) -> dict: ...
```

### `tokenizador/<corpus>/metadados.json`

Um tokenizador por corpus, ao lado do `tokenizer.json` da biblioteca. A troca de domínio carrega o tokenizador do corpus escolhido. Qualquer tokenizador que cumpra o contrato é carregado por `carregar_tokenizador`, que escolhe a implementação pelo campo `implementacao`. Um tokenizador mantido fora do pacote usa `implementacao = externo`, com `modulo`, o arquivo Python relativo à pasta do tokenizador, e `classe`, a classe que implementa o contrato. Quando o treino recebe `--tokenizador`, o caminho fica gravado em `extra.tokenizador` no `config.json` da corrida, e as análises posteriores carregam esse tokenizador em vez do tokenizador do corpus.

| Campo | Conteúdo |
|---|---|
| `implementacao` | `biblioteca`, `didatico`, `bytes` ou `externo` |
| `versao_biblioteca` | Versão exata do `tokenizers` |
| `vocab_size` | $V$ |
| `regex` | Expressão regular de pré-tokenização |
| `sha256_treino` | Hash do texto de treino do tokenizador |
| `fracao_ausente_retido` | Fração de tokens aprendidos ausentes do retido |
| `bytes_retido` | Tamanho do retido usado na regra de $V$ |
| `sha256_tokenizer_json` | Hash do arquivo do tokenizador |

## 3. Configuração de corrida (caixa 0)

### `runs/<run_id>/config.json`

| Campo | Conteúdo |
|---|---|
| `run_id` | Identificador único |
| `modo` | `difusao` ou `autoregressivo` |
| `corpus` | `A` ou `B`, mais o hash de `documentos.jsonl` |
| `tokenizador` | Hash de `tokenizer.json` |
| `rede` | `n_layer`, `d`, `n_head`, `L`, `V`, compartilhamento de pesos |
| `otimizacao` | Taxa, decaimento, cronograma, lote $B$, épocas-alvo |
| `precisao` | `fp16` ou `fp32`, kernel de atenção |
| `semente` | Inteiro que fixa inicialização e ordem dos dados |
| `maquina` | GPU, driver, versões de PyTorch e CUDA, teto de potência |
| `papel` | `hp`, `piloto`, `descritivo`, `confirmatoria`, `tost` ou `calibracao` |

## 4. Resultados de corrida (caixas 6 a 8)

### `runs/<run_id>/passos.jsonl`

Uma linha a cada $n$ passos aplicados: `passo`, `epoca`, `perda`, `taxa`, `escala_gradscaler`, `passos_pulados`, `grad_max_logits`, `tempo_s`.

### `runs/<run_id>/sentinela.jsonl`

Uma linha por medida: `passo`, `faixa_l`, `tensor`, `erro_relativo_p95`, `underflow`, `ativacao_max`.

### `runs/<run_id>/avaliacao.json`

| Campo | Conteúdo |
|---|---|
| `conjunto` | `validacao` ou `teste` |
| `K` | Amostras por janela (1 no autoregressivo) |
| `documentos` | Lista com `doc_id`, `grupo`, `nats`, `nats_sem_nomes`, `bytes`, `bytes_sem_nomes`, `janelas` |
| `bpb` | Razão de somas |
| `erro_mc` | Erro-padrão de Monte Carlo da NELBO (difusão) |

### `runs/<run_id>/amostras.jsonl`

Uma linha por amostra: `amostra_id`, `prefixo`, `texto`, `passos_geracao`, `precisao_sorteio`, `maior_trecho_copiado`, `descartada`.

Os pesos ficam em `runs/<run_id>/pesos.pt`, fora do git.

## 5. Pré-registro

### `preregistro/<nome>.json` e `preregistro/<nome>.sha256`

O arquivo registra, antes da corrida que governa, os valores congelados: partição (hash), tokenizador (hash), $U$ medido, grade de épocas, ponto confirmatório, $\Delta_{min}$, teto de sementes, rubrica, $k$ do filtro de cópia, $T_{gen}$ e critério de leitura. O `.sha256` contém o hash do `.json` e é commitado antes da primeira semente que ele governa. Qualquer alteração posterior gera um pré-registro novo, nunca a edição do anterior.

## 6. Leitura cega (caixa 10)

### `leitura/lote_<n>.csv`

Colunas `item_id` e `texto`, em ordem aleatória, sem indicação de modo, ponto, semente ou controle.

### `leitura/chave_<n>.json`

Mapa de `item_id` para `tipo` (`modelo`, `humano` ou `embaralhado`), com `modo`, `ponto` e `semente` nos itens de modelo e `par` nos controles, que liga cada excerto humano ao embaralhado correspondente. Fica fora do alcance dos avaliadores até o fim da leitura e só é commitado depois.

### `leitura/respostas_<avaliador>_<n>.csv`

Colunas `item_id`, `N2`, `N3` e `N4`, com notas de 0 a 2.

## 7. Análises de calibração e deleção

### `runs/<run_id>/calibracao.json`

| Campo | Conteúdo |
|---|---|
| `T` | Temperatura ajustada na validação |
| `n_previsoes` | Número de previsões de token usadas |
| `ece_T1`, `ece_calibrado` | ECE sem e com temperatura |
| `auc_roc` | Área sob a curva ROC da confiança |
| `confiabilidade` | Saída de `calibracao.confiabilidade`, com e sem temperatura |
| `seletiva` | Saída de `calibracao.curva_seletiva` |
| `portao_beta` | Saída de `calibracao.portao_beta` por alvo |

### `runs/<run_id>/delecao.json`

Saída de `delecao.analisar`: `modo`, `T`, `grade`, `itens` (por item, `decisao`, `p0`, `acerto`, as duas curvas, as duas AUC e `p_vazio_relativo`), curvas médias, AUC médias, `p_vazio_relativo_medio` e as correlações de Spearman e de Pearson entre $p_0$ e a AUC.
