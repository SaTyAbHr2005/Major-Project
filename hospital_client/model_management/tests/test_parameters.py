"""
FL-ready parameter interface tests: get_parameters() -> set_parameters()
round-trip, deterministic ordering, and shape/count mismatch rejection.
"""
import numpy as np
import pytest
import torch

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters, set_parameters
from hospital_client.model_management.registry import build_model


def test_get_parameters_returns_ordered_numpy_arrays():
    model = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    params = get_parameters(model)
    assert isinstance(params, list)
    assert all(isinstance(p, np.ndarray) for p in params)
    assert len(params) == len(list(model.state_dict().keys()))


def test_get_parameters_ordering_is_deterministic():
    model = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    p1 = get_parameters(model)
    p2 = get_parameters(model)
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)


def test_round_trip_model_to_params_to_model_preserves_parameters():
    config = ModelConfig(architecture="resnet18", num_classes=3, input_size=(64, 64))
    model_a = build_model(config)
    model_b = build_model(config)  # different random init

    params_a = get_parameters(model_a)
    set_parameters(model_b, params_a)

    for key_a, key_b in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        assert torch.equal(key_a, key_b)

    # Forward pass now produces identical output too.
    x = torch.randn(2, 3, 64, 64)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        out_a, out_b = model_a(x), model_b(x)
    assert torch.equal(out_a, out_b)


def test_set_parameters_wrong_count_rejected():
    model = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    params = get_parameters(model)
    with pytest.raises(ValueError, match="count mismatch"):
        set_parameters(model, params[:-1])


def test_set_parameters_wrong_shape_rejected():
    model = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    params = get_parameters(model)
    params[0] = np.zeros((999, 999))
    with pytest.raises(ValueError, match="Shape mismatch"):
        set_parameters(model, params)


def test_parameters_are_detached_copies_not_live_references():
    model = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)))
    params = get_parameters(model)
    params[0][:] = 0  # mutate the returned array
    # Model's actual parameter tensor must be unaffected.
    original = list(model.state_dict().values())[0]
    assert not np.all(original.detach().cpu().numpy() == 0)
