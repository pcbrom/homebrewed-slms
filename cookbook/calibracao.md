# Calibrar precisão e vazão

Esta receita reproduz o marco M5 numa GPU, conforme a [decisão de precisão](../docs/decisao_precisao.md) e o [dimensionamento](../docs/DIMENSIONAMENTO.md). Ela responde a duas perguntas antes de qualquer corrida longa. Quantos tokens por segundo o treino realmente processa na sua placa, e o treino em fp16 chega ao mesmo resultado que o treino em fp32? Sem a primeira resposta, o orçamento de épocas e sementes é chute. Sem a segunda, qualquer diferença entre os modos pode ser efeito da precisão numérica.

Antes de começar, confirme que nenhuma outra corrida pesada ocupa a GPU. Medir vazão com a placa dividida produz um número que não descreve a placa. No ambiente de referência a RTX 5090 opera com teto de 500 W, e os números abaixo valem só para esse teto.

## 1. Vazão e MFU

```bash
.venv/bin/python scripts/medir_vazao.py --pico-tflops 209.5 --arq 6x384 --vocab 4096 --precisao fp16 --dispositivo cuda
.venv/bin/python scripts/medir_vazao.py --pico-tflops 209.5 --arq 6x384 --vocab 4096 --precisao fp32 --dispositivo cuda
```

O script mede os dois modos com sincronização da GPU antes de cada leitura do relógio, descartando os passos de aquecimento, e imprime tokens por segundo, MFU e memória de pico. Troque `--pico-tflops` pelo pico da sua placa na precisão do treino. Na RTX 5090 a 500 W, com $V = 4.096$, $L = 256$ e lote 64, o MFU medido foi de 0,41 no modo autorregressivo e 0,38 na difusão em fp16, e de 0,16 nos dois modos em fp32. Registre o seu resultado na seção 6 do dimensionamento e refaça a estimativa de orçamento com ele.

## 2. Portão G1, sentinela de gradiente

```bash
for modo in autoregressivo difusao; do
  for precisao in fp16 fp32; do
    for s in 0 1 2; do
      .venv/bin/python scripts/treinar.py --modo $modo --precisao $precisao --semente $s --epocas 16 \
        --papel calibracao --atencao efficient --dispositivo cuda --sentinela \
        --run-id calibracao-$modo-$precisao-s$s
    done
  done
done
```

São 12 corridas de 16 épocas, rodadas uma por vez. Na RTX 5090 a 500 W, o treino de cada uma levou cerca de 11 s em fp16 e 27 s em fp32, sem contar a avaliação e a sentinela. Com `--sentinela`, o treino compara, tensor a tensor, o gradiente em fp16 (desescalado) com o gradiente em fp32 sobre lotes-sonda fixos, um por faixa de $l/L$ na difusão e uma faixa única no autoregressivo, no início, aos 20% do orçamento, a cada 500 passos e no fim, e grava `runs/<run_id>/sentinela.jsonl`. O aceite é por tensor, no percentil 95 entre lotes. O portão G1 exige erro relativo de até 0,05, com piso de norma, underflow em até 1% dos elementos, ativação máxima de até $2^{12}$ por camada, passos pulados em até 1% depois do passo 50 e nenhum NaN. As figuras da sentinela saem de

```bash
.venv/bin/python scripts/gerar_figuras.py --saida figuras/resultados
```

## 3. Portão G2, fp16 contra fp32 em bits por byte

```bash
.venv/bin/python scripts/portao_precisao.py
```

O script lê as 12 corridas, calcula por modo a diferença de bits por byte de validação entre fp16 e fp32, pareada por semente, e o intervalo de 95% por $t$ com 2 graus de liberdade. O critério compara esse intervalo com $\pm m_{prec}$, em que $m_{prec} = \Delta_{min}/4$. Como $\Delta_{min}$ só é congelado no marco M6 (`scripts/m6.py selecionar`), até lá o script imprime as diferenças e os intervalos com `"aprovado": null`. Depois do congelamento, ele confere o hash de `preregistro/m6_delta_min.json` e devolve o veredito por modo. As diferenças medidas no ambiente de referência estão no registro da decisão de precisão, e o script as reproduz.

## O que fazer se um portão falhar

Uma falha do G1 só no modo autorregressivo indica defeito de código, não de precisão. Uma falha na difusão em $l$ pequeno, ou uma reprovação do G2, aciona os remédios da decisão de precisão em ordem, cada um retestado: `lm_head` e perda em fp32, com custo de cerca de 2,5 vezes, depois fp32 puro no modelo de 4 camadas com $d = 256$, depois redução do modelo. A cada remédio, o orçamento é refeito com a vazão medida no passo 1. Em nenhum dos casos o critério é ajustado depois de ver os dados sem registro. Uma mudança de critério entra como emenda datada na decisão de precisão, com a justificativa, como foi feito com o critério de razão entre erros da sentinela.
