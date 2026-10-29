import math

import pytest
import torch

from slm.avaliacao import avaliar, logp_tokens_ar
from slm.modelo import ConfigRede, Transformer
from slm.tokenizador import ESPECIAIS


def test_logp_tokens_ar_cobre_ate_o_eos_e_bate_com_a_avaliacao():
    torch.manual_seed(0)
    m = Transformer(ConfigRede(vocab_size=12, L=8)).eval()
    docs = [[5, 6, 7], list(range(4, 12)) * 2]
    lp = logp_tokens_ar(m, docs, 8)
    assert [len(x) for x in lp] == [4, 17]
    assert all((x < 0).all() for x in lp)
    # with no padding after [EOS] the window total equals the evaluation nats
    doc = list(range(4, 11))
    res = avaliar(m, [{"doc_id": "d", "grupo": "g", "ids": doc, "bytes": 1}], "autoregressivo", 8)
    assert -logp_tokens_ar(m, [doc], 8)[0].sum() == pytest.approx(res["documentos"][0]["nats"], rel=1e-6)
    assert ESPECIAIS["[EOS]"] == 2 and math.isfinite(res["bpb"])
