"""Checks for the hand-written GRPO step. They pass once you implement rlm/grpo_step.py."""

import torch

from rlm import grpo_step


def test_advantages_are_centered_and_scaled():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = grpo_step.group_advantages(rewards)
    assert torch.allclose(adv.mean(), torch.tensor(0.0), atol=1e-6)
    assert adv[0] > 0 and adv[1] < 0
    unscaled = grpo_step.group_advantages(rewards, scale=False)
    assert torch.allclose(unscaled, torch.tensor([0.5, -0.5, 0.5, -0.5]))


def test_uniform_rewards_give_zero_advantage():
    adv = grpo_step.group_advantages(torch.ones(8))
    assert torch.allclose(adv, torch.zeros(8), atol=1e-6)


def test_ratio_is_one_when_policies_match():
    logp = torch.randn(4, 6)
    assert torch.allclose(grpo_step.policy_ratio(logp, logp.clone()), torch.ones(4, 6))


def test_clipping_limits_positive_and_negative_advantages():
    ratio = torch.tensor([[2.0], [0.2]])
    adv = torch.tensor([1.0, -1.0])
    obj = grpo_step.clipped_objective(ratio, adv, epsilon=0.2)
    # Positive advantage, ratio too high: clipped to 1.2 * 1.0
    assert torch.isclose(obj[0, 0], torch.tensor(1.2))
    # Negative advantage, ratio too low: min(0.2 * -1, 0.8 * -1) = -0.8
    assert torch.isclose(obj[1, 0], torch.tensor(-0.8))


def test_kl_penalty_is_nonnegative_and_zero_at_equality():
    logp = torch.randn(3, 5)
    assert torch.allclose(grpo_step.kl_penalty(logp, logp), torch.zeros(3, 5), atol=1e-6)
    assert (grpo_step.kl_penalty(logp, logp + 0.3) >= 0).all()


def test_loss_pushes_probability_of_better_outputs_up():
    torch.manual_seed(0)
    logp_old = torch.log(torch.full((4, 5), 0.5))
    logp_new = logp_old.clone().requires_grad_(True)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(4, 5)
    loss, stats = grpo_step.grpo_loss(logp_new, logp_old, rewards, mask, epsilon=0.2, beta=0.0)
    loss.backward()
    # Gradient descent on the loss must *increase* log-probs of rewarded outputs (negative grad)
    # and decrease those of unrewarded ones.
    assert (logp_new.grad[0] < 0).all() and (logp_new.grad[1] > 0).all()
    assert {"mean_advantage", "clip_fraction", "kl"} <= set(stats)
