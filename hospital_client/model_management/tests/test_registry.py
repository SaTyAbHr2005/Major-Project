"""
Architecture registry / catalog tests.

Actually instantiates every one of the 8 registered architectures and runs a
real forward pass on a synthetic tensor - never just checks that a string
key exists in a dict.
"""
import pytest
import torch

from hospital_client.model_management.architectures import count_parameters
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import (
    ARCHITECTURE_CATALOG,
    ARCHITECTURE_REGISTRY,
    ResourceTier,
    build_model,
    get_architecture_info,
    list_architectures,
)

EXPECTED_ARCHITECTURES = {
    "vit_b16", "efficientnet_b4", "efficientnet_b0", "resnet50",
    "resnet18", "mobilenet_v2", "mobilenet_v3_small", "mobilevit_xxs",
}

EXPECTED_RESOURCE_TIER = {
    "vit_b16": ResourceTier.HIGH_END,
    "efficientnet_b4": ResourceTier.HIGH_END,
    "efficientnet_b0": ResourceTier.MEDIUM_END,
    "resnet50": ResourceTier.MEDIUM_END,
    "resnet18": ResourceTier.LOW_END,
    "mobilenet_v2": ResourceTier.LOW_END,
    "mobilenet_v3_small": ResourceTier.VERY_LOW_END,
    "mobilevit_xxs": ResourceTier.VERY_LOW_END,
}

# Small, fast-but-valid synthetic input sizes per architecture (must be
# divisible by 16 for vit_b16's patch size; must survive 5 stride-2 stages
# for mobilevit_xxs).
TEST_INPUT_SIZE = {name: 64 for name in EXPECTED_ARCHITECTURES}


def test_all_eight_architectures_registered():
    assert set(ARCHITECTURE_REGISTRY.keys()) == EXPECTED_ARCHITECTURES
    assert set(ARCHITECTURE_CATALOG.keys()) == EXPECTED_ARCHITECTURES
    assert set(list_architectures()) == EXPECTED_ARCHITECTURES


def test_unknown_architecture_info_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_architecture_info("not_real")


def test_unknown_architecture_build_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(ModelConfig(architecture="not_real", num_classes=2))


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCHITECTURES))
def test_resource_tier_matches_project_catalog(name):
    info = get_architecture_info(name)
    assert info.resource_tier == EXPECTED_RESOURCE_TIER[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCHITECTURES))
def test_architecture_constructs_and_forward_pass_produces_correct_shape(name):
    size = TEST_INPUT_SIZE[name]
    num_classes = 4
    config = ModelConfig(architecture=name, num_classes=num_classes, input_size=(size, size), color_mode="RGB")
    model = build_model(config)
    assert isinstance(model, torch.nn.Module)

    model.eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, num_classes)


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCHITECTURES))
def test_architecture_supports_grayscale_input(name):
    size = TEST_INPUT_SIZE[name]
    config = ModelConfig(architecture=name, num_classes=3, input_size=(size, size), color_mode="L")
    model = build_model(config)
    model.eval()
    x = torch.randn(2, 1, size, size)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3)


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCHITECTURES))
def test_architecture_respects_configured_num_classes(name):
    size = TEST_INPUT_SIZE[name]
    for num_classes in (2, 7):
        config = ModelConfig(architecture=name, num_classes=num_classes, input_size=(size, size))
        model = build_model(config)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size))
        assert out.shape == (1, num_classes)


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCHITECTURES))
def test_parameter_counts_are_measured_from_real_instance(name):
    size = TEST_INPUT_SIZE[name]
    config = ModelConfig(architecture=name, num_classes=2, input_size=(size, size))
    model = build_model(config)
    params = count_parameters(model)
    assert params["total"] > 0
    assert params["trainable"] > 0
    assert params["trainable"] <= params["total"]


def test_vit_b16_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        build_model(ModelConfig(architecture="vit_b16", num_classes=2, input_size=(64, 96)))


def test_vit_b16_pretrained_requires_224():
    with pytest.raises(ValueError, match="224"):
        build_model(ModelConfig(architecture="vit_b16", num_classes=2, input_size=(64, 64), pretrained=True))


def test_mobilevit_xxs_does_not_support_pretrained():
    with pytest.raises(ValueError, match="pretrained"):
        build_model(ModelConfig(architecture="mobilevit_xxs", num_classes=2, pretrained=True))
    assert get_architecture_info("mobilevit_xxs").supports_pretrained is False


def test_registry_never_silently_substitutes_architecture():
    # Building an unsupported architecture must raise, never quietly return
    # a different (e.g. default) architecture's model.
    with pytest.raises(ValueError):
        build_model(ModelConfig(architecture="resnet19_typo", num_classes=2))
