"""CE for gold, KL for teacher soft labels, RPS for ordered score."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    fill = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~option_mask, fill)
    return F.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    return masked_log_softmax(logits, option_mask).exp()


def ranked_probability_score(
    probs: torch.Tensor,
    target: torch.Tensor,
    option_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean RPS over rows with a valid target. Uses bins 0..K-2 (last CDF is 1)."""
    valid_row = (target >= 0) & option_mask.any(dim=-1)
    if not valid_row.any():
        return probs.new_zeros(())
    bsz, max_k = probs.shape
    levels = torch.arange(max_k, device=probs.device).unsqueeze(0).expand(bsz, max_k)
    cdf_p = probs.cumsum(dim=-1)
    cdf_t = (levels >= target.unsqueeze(1)).to(probs.dtype)
    # Ignore padded option slots: once the valid suffix starts, CDF should already be 1.
    k_len = option_mask.sum(dim=-1).clamp(min=1)
    bin_mask = option_mask.clone()
    # Drop the last valid bin (always 1 vs 1).
    last = (k_len - 1).unsqueeze(1)
    bin_mask.scatter_(1, last, False)
    sq = (cdf_p - cdf_t).pow(2) * bin_mask.to(probs.dtype)
    rps = sq.sum(dim=-1) / bin_mask.sum(dim=-1).clamp(min=1).to(probs.dtype)
    return rps[valid_row].mean()


def student_loss(
    logits: torch.Tensor,
    batch: dict,
    score_rps_weight: float = 0.5,
    aux_ce_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """logits: [B, max_K]. Returns a dict with 'loss' plus detached logs."""
    option_mask = batch["option_mask"]
    log_probs = masked_log_softmax(logits, option_mask)
    probs = log_probs.exp()
    targets = batch["targets"]
    is_soft = batch["is_soft"]
    is_score = batch["is_score"]
    teacher = batch["teacher_probs"]

    hard = ~is_soft & (targets >= 0)
    parts: list[torch.Tensor] = []
    logs: dict[str, torch.Tensor] = {}

    if hard.any():
        ce = F.nll_loss(log_probs[hard], targets[hard], reduction="mean")
        parts.append(ce)
        logs["ce"] = ce.detach()
        score_rows = hard & is_score
        if score_rows.any() and score_rps_weight:
            rps = ranked_probability_score(probs[score_rows], targets[score_rows], option_mask[score_rows])
            parts.append(score_rps_weight * rps)
            logs["rps"] = rps.detach()

    if is_soft.any():
        # KL(teacher || student) with teacher already a distribution over valid options.
        t = teacher[is_soft].clamp(min=1e-8)
        t = t * option_mask[is_soft].to(t.dtype)
        t = t / t.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        kl = F.kl_div(log_probs[is_soft], t, reduction="batchmean")
        parts.append(kl)
        logs["kl"] = kl.detach()
        if aux_ce_weight:
            hard_t = t.argmax(dim=-1)
            aux = F.nll_loss(log_probs[is_soft], hard_t, reduction="mean")
            parts.append(aux_ce_weight * aux)
            logs["aux_ce"] = aux.detach()

    if not parts:
        loss = logits.mean() * 0.0
    else:
        loss = parts[0]
        for p in parts[1:]:
            loss = loss + p
    return {"loss": loss, **logs}
