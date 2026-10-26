"""Actual ACT loss and weighted-update qualification; no assets or downloads."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


@pytest.fixture(params=[False, True], ids=["deterministic", "vae"])
def training_case(request):
    torch.manual_seed(42)
    policy = ACTPolicy(
        ACTConfig(
            chunk_size=3,
            n_action_steps=3,
            dim_model=16,
            n_heads=2,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            latent_dim=4,
            dropout=0.0,
            use_vae=request.param,
            kl_weight=0.7,
            pretrained_backbone_weights=None,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        )
    ).train()
    batch = {
        OBS_STATE: torch.randn(3, 2),
        OBS_ENV_STATE: torch.randn(3, 3),
        ACTION: torch.randn(3, 3, 2),
        "action_is_pad": torch.tensor([[False, False, False], [False, True, True], [True, True, True]]),
    }
    return policy, batch


def test_actual_act_per_sample_and_scalar_losses_match_masked_l1_and_kl(training_case):
    policy, batch = training_case
    # Reset the RNG before each genuine model invocation to replay the VAE draw.
    torch.manual_seed(123)
    predictions, (mu, log_variance) = policy.model(batch)
    l1 = torch.stack(
        [
            (predictions[i, :count] - batch[ACTION][i, :count]).abs().sum() / max(count * 2, 1)
            for i, count in enumerate((3, 1, 0))
        ]
    )
    kl = torch.zeros(3)
    if policy.config.use_vae:
        kl = (mu.square() + log_variance.exp() - 1 - log_variance).sum(dim=-1) / 2
    expected = l1 + policy.config.kl_weight * kl

    torch.manual_seed(123)
    actual, metrics = policy.forward(batch, reduction="none")
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected)
    assert metrics["l1_loss"] == pytest.approx(l1.mean().item())
    assert l1[2].item() == 0.0  # No reconstruction contribution from the all-padded example.
    if policy.config.use_vae:
        assert metrics["kld_loss"] == pytest.approx(kl.mean().item())
        torch.testing.assert_close(actual[2], policy.config.kl_weight * kl[2])
    else:
        assert "kld_loss" not in metrics

    # Scalar reconstruction weights each valid action entry equally, rather
    # than each example equally; uneven padding makes these different means.
    global_l1 = (l1[0] * 6 + l1[1] * 2) / 8
    torch.manual_seed(123)
    scalar, scalar_metrics = policy.forward(batch)
    assert scalar.ndim == 0
    torch.testing.assert_close(scalar, global_l1 + policy.config.kl_weight * kl.mean())
    assert scalar_metrics["l1_loss"] == pytest.approx(global_l1.item())
    assert not torch.isclose(actual.mean(), scalar)


def test_actual_act_weighted_optimizer_step_has_finite_gradients_and_updates(training_case):
    policy, batch = training_case
    # Use a real optimizer to isolate the loss patch from distributed training
    # orchestration. This is the trainer's weighted mean, with valid weights.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    losses, _ = policy.forward(batch, reduction="none")
    weights = torch.tensor([0.25, 2.0, 0.0])
    loss = (losses * weights).sum() / (weights.sum() + 1e-8)
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    assert any(
        not torch.equal(parameter.detach(), before[name]) for name, parameter in policy.named_parameters()
    )
