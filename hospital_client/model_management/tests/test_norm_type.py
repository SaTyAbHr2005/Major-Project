import os
import pytest
import torch
from PIL import Image

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import ARCHITECTURE_CATALOG, build_model
from hospital_client.model_management.store import ModelStore
from hospital_client.inference.engine import LocalInferenceEngine, InferenceConfig
from hospital_client.inference.provisioning import LocalStoreModelProvider
from opacus.validators import ModuleValidator


def test_all_eight_architectures_pass_module_validator_after_group_norm_conversion():
    """Every supported architecture must be valid for Opacus after norm_type='group' conversion."""
    for arch in sorted(ARCHITECTURE_CATALOG.keys()):
        config = ModelConfig(
            architecture=arch,
            num_classes=2,
            input_size=(64, 64),
            norm_type="group",
        )
        model = build_model(config)
        assert ModuleValidator.is_valid(model), f"Architecture {arch} failed ModuleValidator.is_valid with norm_type='group'"
        # Check no BatchNorm remains
        for _, module in model.named_modules():
            assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)), (
                f"Architecture {arch} still contains a BatchNorm layer after GroupNorm conversion"
            )


def test_checkpoint_without_norm_type_loads_as_batch_norm(tmp_path):
    """Old checkpoints that lack norm_type in metadata must deserialize with norm_type='batch'."""
    store = ModelStore(str(tmp_path / "models"))
    config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    model = build_model(config)
    meta = store.save_checkpoint(model, config, "legacy_model", status="available")

    # Manually remove norm_type from the metadata JSON on disk to simulate an older checkpoint
    meta_path = os.path.join(tmp_path / "models" / "legacy_model" / "v1", "metadata.json")
    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "norm_type" in data["config"]
    del data["config"]["norm_type"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded_model, loaded_meta = store.load_checkpoint("legacy_model", 1)
    loaded_config = ModelConfig.from_dict(loaded_meta.config)
    assert loaded_config.norm_type == "batch"
    has_bn = any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)) for m in loaded_model.modules())
    assert has_bn, "Expected loaded model to contain BatchNorm layers"


def test_group_norm_model_saves_and_loads_in_inference_engine(tmp_path):
    """A GroupNorm model saved in ModelStore must load in LocalInferenceEngine and predict."""
    store_dir = str(tmp_path / "models")
    store = ModelStore(store_dir)
    config = ModelConfig(architecture="mobilenet_v3_small", num_classes=2, input_size=(64, 64), norm_type="group")
    model = build_model(config)
    store.save_checkpoint(
        model, config, "gn_model", status="available", version=1,
        extra_metadata={"class_mapping": {"benign": 0, "malignant": 1}},
    )

    provider = LocalStoreModelProvider(store_dir, "gn_model", 1)
    inf_config = InferenceConfig(device="cpu", deterministic=True)
    engine = LocalInferenceEngine(provider, inf_config)

    # Test inference on a saved image
    img = Image.new("RGB", (64, 64), color="white")
    img_path = str(tmp_path / "test.png")
    img.save(img_path)
    result = engine.predict(img_path)
    assert result.status.value == "COMPLETED"
    assert result.output.predicted_label in ["benign", "malignant"]
    assert len(result.output.class_probabilities) == 2


def test_group_norm_conversion_is_instance_level_not_class_level():
    """
    The GroupNorm/inplace=False patches in build_model must be applied to the specific
    model instance only (via types.MethodType bound to that instance), NOT to the class.
    A separately-built BatchNorm model in the same process must be completely unaffected.
    """
    import torchvision.models.resnet as _resnet

    # Record the unmodified class-level forward before triggering any conversion
    original_basic_block_forward = _resnet.BasicBlock.forward
    original_bottleneck_forward = _resnet.Bottleneck.forward

    # Build a GroupNorm instance — triggers all per-instance patching
    gn_config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64), norm_type="group")
    gn_model = build_model(gn_config)

    # Class-level forward must be UNCHANGED
    assert _resnet.BasicBlock.forward is original_basic_block_forward, (
        "build_model patched BasicBlock.forward at CLASS level — must only patch the instance"
    )
    assert _resnet.Bottleneck.forward is original_bottleneck_forward, (
        "build_model patched Bottleneck.forward at CLASS level — must only patch the instance"
    )

    # A separate non-DP ResNet18 built afterwards must still have BatchNorm layers
    bn_config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64))
    bn_model = build_model(bn_config)
    bn_layers = [m for m in bn_model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn_layers) > 0, (
        "Non-DP model built after GroupNorm conversion has no BatchNorm — class was patched globally"
    )

    # GroupNorm model must be free of BatchNorm and pass ModuleValidator
    gn_bn_layers = [m for m in gn_model.modules()
                    if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
    assert len(gn_bn_layers) == 0, "GroupNorm model still has BatchNorm layers"
    assert ModuleValidator.is_valid(gn_model)

    # Non-DP model must NOT pass ModuleValidator (it still has cross-sample BatchNorm)
    assert not ModuleValidator.is_valid(bn_model), (
        "BatchNorm model unexpectedly passes ModuleValidator after GroupNorm conversion — possible global mutation"
    )
