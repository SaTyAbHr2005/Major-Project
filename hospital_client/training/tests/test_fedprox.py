"""
FedProx proximal term in Module 7: proximal_mu > 0 keeps local training close to the
round's base model; 0 leaves training unchanged. Synthetic images only.
"""
import math
import os

import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters
from hospital_client.model_management.registry import build_model
from hospital_client.model_management.store import ModelStore
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.federation import _compute_parameters_checksum
from hospital_client.training.trainer import Trainer

MODEL_CONFIG = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))


def _config(dataset_dir, output_dir, **overrides):
    # Plain SGD (no momentum) so lr*mu stays below 1; validate_every > epochs so both runs keep
    # their final weights rather than a validation-selected epoch.
    defaults = dict(model_id="m", model_config=MODEL_CONFIG, dataset_dir=dataset_dir, output_dir=output_dir,
                    epochs=5, batch_size=4, learning_rate=0.01, optimizer="sgd", momentum=0.0,
                    validate_every=100, require_validation=False)
    defaults.update(overrides)
    return TrainingConfig(**defaults)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.1", None])
def test_invalid_proximal_mu_is_rejected(small_dataset_dir, tmp_path, bad):
    with pytest.raises(ValueError, match="proximal_mu"):
        validate_training_config(_config(small_dataset_dir, str(tmp_path), proximal_mu=bad))


def test_default_is_zero_and_recorded(small_dataset_dir, tmp_path):
    config = _config(small_dataset_dir, str(tmp_path))
    assert config.proximal_mu == 0.0 and config.to_dict()["proximal_mu"] == 0.0


def _train_from_base(dataset_dir, output_dir, mu):
    """Train from one fixed base checkpoint; return (base weights, trained weights, result)."""
    import torch
    torch.manual_seed(0)
    base = build_model(MODEL_CONFIG)
    store = ModelStore(os.path.join(output_dir, "models"))
    store.save_checkpoint(base, MODEL_CONFIG, "BASE", status="available", version=1)
    base_params = get_parameters(base)
    base_weights = [w.detach().clone() for w in base.parameters()]
    result = Trainer(_config(dataset_dir, output_dir, proximal_mu=mu, canonical_base_model_id="BASE",
                             canonical_base_model_version=1,
                             canonical_base_model_checksum=_compute_parameters_checksum(base_params))).run()
    trained, _ = store.load_checkpoint(result.checkpoint_model_id, result.checkpoint_version)
    return base_weights, [w.detach() for w in trained.parameters()], result


def _distance(a, b):
    """L2 distance over trainable weights only; BatchNorm running statistics are buffers that the
    proximal term does not (and should not) constrain."""
    return math.sqrt(sum(float(((x.double() - y.double()) ** 2).sum()) for x, y in zip(a, b)))


def test_proximal_term_keeps_weights_close_to_the_base_model(small_dataset_dir, tmp_path):
    base, plain, plain_result = _train_from_base(small_dataset_dir, str(tmp_path / "plain"), 0.0)
    _, prox, prox_result = _train_from_base(small_dataset_dir, str(tmp_path / "prox"), 90.0)  # lr*mu = 0.9: strong pull, still stable

    d_plain, d_prox = _distance(base, plain), _distance(base, prox)
    print(f"distance from base: plain={d_plain:.4f} fedprox={d_prox:.4f}")
    assert d_prox < 0.75 * d_plain                              # the proximal term holds them back
    assert prox_result.training_configuration["proximal_mu"] == 90.0  # reaches the federation handoff
    # Reported loss is the plain classification loss (comparable across runs), not loss + proximal term.
    assert 0.0 <= prox_result.training_metrics["loss"] < 10.0
