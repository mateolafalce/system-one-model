import torch

from student import DecisionHead


def test_head_scores_mask_positions():
    bsz, seq, k, d = 2, 16, 3, 64
    head = DecisionHead(d, n_layers=2, dropout=0.0)
    hidden = torch.randn(bsz, seq, d)
    mask = torch.ones(bsz, seq, dtype=torch.long)
    type_ids = torch.tensor([0, 2])
    option_positions = torch.tensor([[1, 4, 7], [2, 5, 9]])
    option_mask = torch.ones(bsz, k, dtype=torch.bool)
    logits = head(hidden, mask, type_ids, option_positions, option_mask)
    assert logits.shape == (bsz, k)
    assert torch.isfinite(logits).all()


def test_padded_options_are_masked():
    head = DecisionHead(32, n_layers=1, dropout=0.0)
    hidden = torch.randn(1, 8, 32)
    mask = torch.ones(1, 8, dtype=torch.long)
    type_ids = torch.zeros(1, dtype=torch.long)
    option_positions = torch.tensor([[1, 3, -1]])
    option_mask = torch.tensor([[True, True, False]])
    logits = head(hidden, mask, type_ids, option_positions, option_mask)
    assert torch.isfinite(logits[0, :2]).all()
    assert logits[0, 2] < -1e3
