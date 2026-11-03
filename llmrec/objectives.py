"""Scalar objectives and token alignment; no optimizer or model training."""

import math

IGNORE_INDEX = -100


def build_sequence(prompt, reasoning, sid_tokens, mask_target=False):
    """Inputs are already-tokenized segments; spans must come from tokenizer.

    Prompt includes BOS/chat prefix. Original SFT supervises reasoning + SID.
    Added RFT supervises reasoning only. EOS/packing/padding are omitted here.
    """
    if not prompt or not sid_tokens:
        raise ValueError("Prompt and SID must be nonempty")
    if mask_target and not reasoning:
        raise ValueError("Masked RFT requires reasoning tokens")
    tokens = [*prompt, *reasoning, *sid_tokens]
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Token IDs must be nonnegative integers")
    labels = [IGNORE_INDEX] * len(prompt) + list(reasoning)
    labels += [IGNORE_INDEX] * len(sid_tokens) if mask_target else list(sid_tokens)
    return {"input_ids": tokens, "labels": labels}


def causal_cross_entropy(logits, labels):
    """Mean valid-token CE: logits[t] predicts labels[t+1], exactly one shift."""
    if len(logits) != len(labels) or len(labels) < 2:
        raise ValueError("Need equal sequence lengths >= 2")
    losses = []
    width = len(logits[0])
    if not width or any(len(row) != width for row in logits):
        raise ValueError("Logits must be rectangular and nonempty")
    if any(not math.isfinite(value) for row in logits for value in row):
        raise ValueError("Logits must be finite")
    for scores, target in zip(logits[:-1], labels[1:]):
        if target == IGNORE_INDEX:
            continue
        if type(target) is not int or not 0 <= target < width:
            raise ValueError("Target outside vocabulary")
        largest = max(scores)
        log_z = largest + math.log(sum(math.exp(x - largest) for x in scores))
        losses.append(log_z - scores[target])
    if not losses:
        raise ValueError("No supervised tokens after causal shift")
    return sum(losses) / len(losses)


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Vanilla DPO on completion log-prob SUMS, with a fixed reference policy.

    No label smoothing or auxiliary generation loss in this compact example.
    All values are scalars: this function cannot backpropagate.
    """
    if beta <= 0 or not all(math.isfinite(x) for x in
                           (policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)):
        raise ValueError("Finite log probabilities and positive beta required")
    margin = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return softplus(-margin)


def reinforce_loss(log_probs, rewards, baseline=0.0):
    """Basic REINFORCE objective only; NOT PPO/GRPO or a full RL trainer."""
    if not log_probs or len(log_probs) != len(rewards):
        raise ValueError("Nonempty aligned log probabilities and rewards required")
    if not all(math.isfinite(x) for x in [*log_probs, *rewards, baseline]):
        raise ValueError("Inputs must be finite")
    return -sum(lp * (reward - baseline) for lp, reward in zip(log_probs, rewards)) / len(rewards)


def behavior_f1(predicted, expected):
    predicted, expected = set(predicted), set(expected)
    # Explicit demo convention: both empty -> 0. Real scorer may differ.
    denominator = len(predicted) + len(expected)
    return 2 * len(predicted & expected) / denominator if denominator else 0.0
