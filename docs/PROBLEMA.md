# Enquadramento do problema

## Definição

Construir do zero um modelo de linguagem pequeno (SLM), com dois modos de geração sobre a mesma rede, difusão mascarada e autoregressivo, em um pacote enxuto e um notebook Colab que qualquer pessoa adapta a um corpus próprio trocando apenas a configuração e o adaptador de domínio.

## Motivação

Quem estuda modelos de linguagem hoje aprende quase sempre por meio de modelos prontos, carregados de um repositório e ajustados por uma biblioteca que esconde cada etapa. O aluno que termina esse percurso sabe chamar um modelo, mas raramente sabe dizer de onde vem a perda que ele minimiza, por que a atenção é causal num caso e bidirecional no outro, ou se o texto gerado vem de generalização ou de memorização. Diante de um corpus próprio, de um domínio que nenhum modelo pronto cobre bem, essa pessoa não tem por onde começar.

O projeto responde a essa lacuna expondo a cadeia inteira, do texto bruto às amostras geradas, com cada etapa visível e medida. A escolha de dois modos de geração sobre a mesma rede tem função didática e científica. A única diferença entre eles é a máscara de atenção e a perda, de modo que a comparação isola o paradigma de geração dos demais fatores, em vez de comparar modelos que diferem em tudo.

## Público

- Alunos de pós-graduação em computação, em aula prática, com conhecimento de redes neurais e PyTorch e sem experiência prévia em treinar modelo de linguagem.
- Pesquisadores que precisem de um SLM treinado num domínio específico e queiram um ponto de partida auditável.
- O próprio projeto, na etapa 2, como bancada de pesquisa sobre tokenização.

## Produto pretendido

1. Pacote `slm/` com os componentes separados por responsabilidade: adaptador de domínio, tokenizador, processo de mascaramento, rede, perda, laço de treino, geração e avaliação.
2. Notebook Colab didático, gerado por script versionado no repositório, que executa o pacote com ilustração de entrada e saída em cada etapa e com o formalismo matemático de cada componente. São dois notebooks em português do Brasil, um por modo de geração, cada um legível sozinho, mesmo que repitam as caixas comuns aos dois. Cada notebook é fracionado segundo as caixas do diagrama de arquitetura, uma seção por caixa, para que o pesquisador acompanhe o que entra, o que acontece e o que sai de cada componente antes de passar ao seguinte. O notebook é a referência das aulas, e cada caixa apresenta sempre três partes: teoria, com o formalismo, aplicação, com o código executado e a entrada e a saída ilustradas, e técnica, com os problemas encontrados no projeto e as soluções adotadas, cada um sustentado pela medida que o revelou. Toda a matemática aparece em blocos de construção: cada bloco traz a definição, a dedução quando houver, a função do pacote que o implementa e a ligação com a caixa seguinte, e um mapa inicial mostra como os blocos se encadeiam do texto bruto até a comparação final. Sempre que o custo permite, uma célula refaz o bloco à mão e confere o resultado contra o pacote.
3. Bancada de comparação entre os dois modos, com orçamento declarado e teste estatístico.
4. Pasta `cookbook/` com receitas reproduzíveis.
5. Artigo no estilo tutorial para o arXiv, escrito depois das etapas 1 e 2 e antes da abertura do repositório.

## Escopo por etapa

### Etapa 1 (esta)

- Tokenizador de referência treinado do zero no corpus, restrito à partição de treino.
- Contrato do tokenizador documentado, com interface e testes de conformidade, de forma que outra implementação possa substituí-lo sem alterar o restante do pacote.
- Os dois modos de geração, treinados e comparados sob o mesmo orçamento.
- Adaptação a domínio verificada com um segundo corpus, distinto do de demonstração.

### Etapa 2

- Proposta de tokenizador original, inserida pelo contrato definido na etapa 1.
- Bancada comparativa contra tokenizadores de referência da literatura, com métricas intrínsecas e extrínsecas.
- Material didático para que pesquisadores e estudantes treinem e avaliem por conta própria um tokenizador do zero, no mesmo formato do notebook da etapa 1, com o algoritmo exposto, as métricas explicadas e a bancada pronta para receber uma implementação nova.
- A implementação da proposta permanece fora de qualquer repositório público e de qualquer material distribuído a terceiros até depósito ou submissão. O material didático usa apenas tokenizadores de referência e o contrato público.

### Etapa 3

- Pós-treino por preferência, na sequência dos grandes laboratórios em versão mini e didática: otimização direta de preferências (DPO) nos dois modos, com a NELBO no lugar da verossimilhança exata na difusão, como no LLaDA 1.5.
- Medida antes e depois da coerência, pela leitura cega, e da calibração, por ECE, diagrama de confiabilidade e curva de deleção. A perda de calibração depois do pós-treino foi relatada para calibração de resposta em modelos treinados por PPO (relatório técnico do GPT-4, Figura 8), e Tian et al. (2023) mencionam estudos nessa direção. Para DPO e calibração de próximo token não há evidência nas fontes do projeto, e a etapa estima o efeito nos dois sentidos, além de verificar se a temperatura o corrige.
- A fonte de preferência é humana, com recompensa programática como braço comparativo e controle com rótulos embaralhados, conforme a [decisão de preferência](decisao_preferencia.md). A recompensa programática só passa a fonte principal se a capacidade humana medida não cobrir a carga, e nesse caso a metodologia declara a limitação.

## Restrições duras

- Do zero: nenhum peso pré-treinado e nenhum tokenizador pré-treinado. A rede é inicializada aleatoriamente e o tokenizador é treinado no corpus do próprio experimento.
- O notebook executa de ponta a ponta numa GPU T4 do Colab, com precisão fp16 e escalonamento de gradiente.
- Toda calibração local roda na RTX 5090 sob o teto de 500 W, uma corrida por vez.
- O corpus de demonstração tem licença compatível com redistribuição em repositório público, com atribuição registrada.
- Divisão treino, validação e teste feita por documento e antes do treino do tokenizador, para que nenhuma estatística da avaliação vaze para o vocabulário.
- Comparações entre configurações usam o mesmo orçamento, declarado antes da corrida (bytes de texto vistos ou FLOPs), e o critério de decisão também é declarado antes.
- Sementes fixas e versões de dependências registradas em cada resultado.

## Licença e citação

- Uso livre para pesquisa e ensino, uso comercial vedado. Licença CC BY-NC-SA 4.0 para código, notebooks, documentos, figuras e pesos treinados, conforme o padrão de publicação do autor.
- Dados de terceiros mantêm a licença de origem, com atribuição em arquivo próprio. Corpus cuja licença não seja compatível com a cláusula não comercial, como CC BY-SA, não é redistribuído no repositório e é baixado pelo notebook a partir da fonte original.
- Toda fonte científica citada é de acesso aberto, baixada e registrada em `referencias/` com versão fixada, licença e sha256. O PDF entra no repositório apenas quando a licença da obra permite redistribuição por terceiros. Nos demais casos, o repositório guarda a URL e o hash, e `scripts/referencias.py` baixa e confere o arquivo.
- Ao concluir a etapa 1, o repositório recebe DOI pelo Zenodo. O README abre com o selo do DOI e o bloco de citação em BibTeX, e o repositório inclui `CITATION.cff` e `.zenodo.json`.

## Critérios de sucesso

- O notebook roda sem erro na T4, com o tempo de cada etapa medido e exibido.
- Os dois modos treinam sob o mesmo orçamento e são comparados em bits por byte na validação, pareados por documento, com intervalo de confiança.
- As amostras de cada modo são exibidas lado a lado, junto com a sobreposição de n-gramas com o treino, para distinguir geração de memorização.
- A troca de domínio exige editar apenas a célula de configuração e o adaptador, condição verificada com o segundo corpus.
- O contrato do tokenizador tem testes que uma implementação nova executa sem alterar o pacote.

## Fora de escopo

- Ajuste por instrução e interface de conversa. O alinhamento por preferência fica fora da etapa 1 e é tratado na etapa 3.
- Modelos acima da faixa que treina em tempo de aula na T4.
- Entrada multimodal.
- Serving em produção.

## Questões em aberto

Cada questão abaixo segue para pesquisa com fonte e, quando de peso, para decisão por debate adversarial.

- Formulação da difusão discreta: máscara absorvente (MDLM, LLaDA) contra as alternativas da literatura.
- Tokenizador de referência: implementação própria de BPE, que expõe o algoritmo, ou biblioteca, que é mais rápida e menos sujeita a erro.
- Unidade do orçamento de comparação: bytes de texto vistos ou FLOPs, e como controlar o número de parâmetros.
- Corpus de demonstração e segundo corpus de verificação, com licença confirmada.
- Duração-alvo de execução do notebook em aula, que fixa o tamanho do modelo e do corpus.
