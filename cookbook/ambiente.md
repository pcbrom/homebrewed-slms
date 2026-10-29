# Preparar o ambiente

O projeto usa Python 3.12 e um ambiente virtual próprio, com as versões fixadas em `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
```

A roda `cu128` do PyTorch 2.10 traz kernels para a arquitetura Blackwell (sm_120). Para conferir sem iniciar nenhuma corrida:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list()[-2:])"
```

No Colab, o PyTorch já vem instalado, e o notebook instala só `tokenizers` na versão fixada e `regex`.

## Testes

```bash
.venv/bin/python -m pytest -q tests/
```

A suíte roda inteira em CPU, em poucos segundos, e inclui os portões G0 e G1 da [decisão de formulação](../docs/decisao_formulacao.md).
