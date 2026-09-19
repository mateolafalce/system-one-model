import torch

from loss import ranked_probability_score, student_loss
from metrics import entropy_confidence, expected_value


def _batch(logits_k, target, primitive="choice", soft=False, teacher=None):
    bsz, k = 1, logits_k
    return {
        "option_mask": torch.ones(bsz, k, dtype=torch.bool),
        "targets": torch.tensor([target]),
        "is_soft": torch.tensor([soft]),
        "is_score": torch.tensor([primitive == "score"]),
        "teacher_probs": torch.tensor([teacher or [0.0] * k]),
    }


def test_perfect_hard_ce_near_zero():
    logits = torch.tensor([[10.0, -10.0, -10.0]])
    out = student_loss(logits, _batch(3, 0))
    assert float(out["ce"]) < 0.01


def test_rps_zero_when_one_hot_on_target():
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    rps = ranked_probability_score(probs, torch.tensor([2]), mask)
    assert float(rps) < 1e-6


def test_rps_far_worse_than_near():
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.tensor([0])
    near = ranked_probability_score(torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]), target, mask)
    far = ranked_probability_score(torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]), target, mask)
    assert float(far) > float(near)


def test_confidence_one_hot_and_uniform():
    assert entropy_confidence([1, 0, 0]) > 0.99
    u = entropy_confidence([1 / 3, 1 / 3, 1 / 3])
    assert u < 0.02


def test_expected_value():
    assert abs(expected_value([0, 0, 0, 1, 0]) - 3.0) < 1e-9
    assert abs(expected_value([0.5, 0.5]) - 0.5) < 1e-9


def test_kl_prefers_matching_teacher():
    teacher = [0.7, 0.2, 0.1]
    batch = _batch(3, 0, soft=True, teacher=teacher)
    good = student_loss(torch.tensor([[2.0, 0.5, 0.0]]), batch)
    bad = student_loss(torch.tensor([[0.0, 0.5, 2.0]]), batch)
    assert float(good["kl"]) < float(bad["kl"])
