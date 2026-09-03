"""Shared DPO math used by preference training and evaluation."""

import torch
import torch.nn.functional as F

# ==============================================================================
# DPO Math
# ==============================================================================

def get_sequence_logprob(logits, labels, mask):
    """Return the summed completion log-probability for each sequence."""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = mask[:, 1:].float()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    policy_gap = policy_chosen - policy_rejected
    ref_gap = ref_chosen - ref_rejected
    logits = beta * (policy_gap - ref_gap)
    loss = -F.logsigmoid(logits)
    accuracy = (logits > 0).float().mean()
    return loss.mean(), accuracy


def dpo_collate(batch, pad_id):
    """Pad chosen/rejected branches while keeping completion masks aligned."""
    result = {}
    for prefix in ("chosen", "rejected"):
        result[f"{prefix}_ids"] = torch.nn.utils.rnn.pad_sequence(
            [item[f"{prefix}_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        result[f"{prefix}_mask"] = torch.nn.utils.rnn.pad_sequence(
            [item[f"{prefix}_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    return result
