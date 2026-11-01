"""One GRPO update step, written by hand. This is the part you must understand, not just run.

``GRPOTrainer`` hides the algorithm behind a ``.train()`` call. Here you reimplement its
core on plain tensors, following the slides step by step:

    Step 2: sample a group of G outputs for the same question   (done by the caller)
    Step 3: advantages   A_i = (r_i - mean(r)) / (std(r) + eps)
    Step 4: policy ratio ρ_{i,t} = π_θ(o_{i,t}) / π_{θ_old}(o_{i,t})
            clipped objective  min(ρ·A, clip(ρ, 1-ε, 1+ε)·A)
            optional KL penalty  D_KL(π_θ || π_ref) ≈ exp(logp_ref - logp) - (logp_ref - logp) - 1
            loss = -( mean over tokens of clipped objective  -  β · KL )

Everything works on *log-probabilities per token*, shaped ``(G, T)``, with a mask that is
1 on completion tokens and 0 on padding. Do not touch the model here: the caller computes
the log-probs; you compute the loss.

Run the tests to check your implementation::

    uv run pytest tests/test_grpo_step.py -v

They are marked as expected failures until you implement the functions.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale: bool = True) -> torch.Tensor:
    """Step 3 of the slides: normalise rewards within the group.

    Args:
        rewards: shape ``(G,)``, one scalar reward per sampled output.
        eps: numerical guard for the standard deviation.
        scale: if False, return ``r_i - mean(r)`` without dividing by the std.

    Returns:
        Advantages with shape ``(G,)``. A positive advantage means "better than the group".
    """
    # Tu turno.
    centered = rewards - rewards.mean()
    if not scale:
        return centered
    return centered / (rewards.std() + eps)


def policy_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    """ρ_{i,t} = π_θ / π_{θ_old}, computed in log space for numerical stability.

    Both inputs have shape ``(G, T)``. Return a tensor of the same shape.
    """
    # Tu turno.
    return torch.exp(logp_new - logp_old)


def clipped_objective(
    ratio: torch.Tensor, advantages: torch.Tensor, epsilon: float = 0.2
) -> torch.Tensor:
    """Per-token GRPO surrogate with clipping (variation 1 in the slides).

    Args:
        ratio: shape ``(G, T)``.
        advantages: shape ``(G,)``; broadcast over the token dimension.
        epsilon: clipping range.

    Returns:
        Per-token objective, shape ``(G, T)``, *before* masking and averaging.
    """
    # Tu turno.
    adv = advantages.unsqueeze(-1)  # (G,) -> (G, 1), broadcast sobre T
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    return torch.minimum(unclipped, clipped)


def kl_penalty(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of D_KL(π_θ || π_ref) used by DeepSeekMath (variation 2).

    exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1. Always >= 0. Shape ``(G, T)``.
    """
    # Tu turno.
    diff = logp_ref - logp_new
    return torch.exp(diff) - diff - 1


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.0,
    logp_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Put the pieces together and return the scalar loss to minimise.

    Average the per-token objective over the valid tokens of each output (``1/|o_i|`` in the
    formula), then over the group (``1/G``). Subtract β times the masked-mean KL when
    ``beta > 0``. Return ``(-objective, stats)`` where ``stats`` holds at least
    ``mean_advantage``, ``clip_fraction`` (share of tokens where clipping was active) and
    ``kl`` for logging.
    """
    # Tu turno.
    advantages = group_advantages(rewards)
    ratio = policy_ratio(logp_new, logp_old)
    objective = clipped_objective(ratio, advantages, epsilon)

    tokens_per_seq = mask.sum(dim=-1).clamp(min=1.0)
    per_seq_objective = (objective * mask).sum(dim=-1) / tokens_per_seq
    mean_objective = per_seq_objective.mean()

    unclipped = ratio * advantages.unsqueeze(-1)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.unsqueeze(-1)
    was_clipped = (unclipped != clipped).float()
    clip_fraction = (was_clipped * mask).sum() / mask.sum().clamp(min=1.0)

    kl_term = torch.zeros((), device=logp_new.device)
    if beta > 0:
        if logp_ref is None:
            raise ValueError("beta > 0 requires logp_ref")
        kl = kl_penalty(logp_new, logp_ref)
        per_seq_kl = (kl * mask).sum(dim=-1) / tokens_per_seq
        kl_term = per_seq_kl.mean()

    loss = -(mean_objective - beta * kl_term)

    stats = {
        "mean_advantage": advantages.mean().item(),
        "clip_fraction": clip_fraction.item(),
        "kl": kl_term.item(),
    }
    return loss, stats
if __name__ == "__main__":
    pass
