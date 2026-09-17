import torch
from adarank_rec.config import AdaRankConfig
from adarank_rec.model import AdaRankRec
from adarank_rec.modules import OrderedLowRankFusion


def test_ordered_scales_and_shapes():
    fusion = OrderedLowRankFusion(16, 16)
    scales = fusion.scales()
    assert torch.all(scales[:-1] >= scales[1:]) and scales[0].item() == 1
    z = fusion(torch.randn(2, 3, 16), torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool), 8)
    assert z.shape == (2, 3, 16)


def test_model_all_paper_ranks():
    cfg = AdaRankConfig(dim=32, decoder_dim=64, visual_dim=6, text_dim=7, num_queries=2,
                        ranks=(4, 8, 16, 32), controller_hidden=16, max_history=8)
    sid = torch.randint(0, 1024, (11, 4)); model = AdaRankRec(10, sid, cfg)
    batch = {"history": torch.tensor([[1,2,0],[3,4,5]]), "history_mask": torch.tensor([[1,1,0],[1,1,1]], dtype=torch.bool),
             "target": torch.tensor([2,5]), "visual": torch.randn(2,6), "text": torch.randn(2,7)}
    for rank in cfg.ranks: assert model.logits_for_rank(batch, rank).shape == (2, 4, 1024)
    assert model.sufficient_ranks(batch).shape == (2,)
