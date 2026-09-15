"""
Pretrained-weight behavior: Module 6 must never silently attempt a network
download. Every torchvision-backed architecture is checked here; a genuine
network download attempt would take far longer than the threshold used
below (or hang entirely on an offline machine), so an instant raise is
strong evidence no network I/O was attempted.
"""
import time

import pytest
import torch

from hospital_client.model_management.architectures import _is_weights_cached_locally, _require_local_weights
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import ARCHITECTURE_CATALOG, build_model

TORCHVISION_BACKED_ARCHITECTURES = [
    name for name, info in ARCHITECTURE_CATALOG.items() if info.supports_pretrained
]


@pytest.mark.parametrize("name", TORCHVISION_BACKED_ARCHITECTURES)
def test_pretrained_true_resolves_purely_from_local_cache_state(name):
    """
    Whatever this machine's torch hub cache actually contains, pretrained=True
    must resolve instantly (no network attempt): either it fails immediately
    with our explicit "no download" error (nothing cached), or it succeeds
    immediately using genuinely already-cached weights. Either outcome is
    fine; a slow/hanging outcome would indicate a network attempt occurred.
    """
    start = time.monotonic()
    try:
        model = build_model(ModelConfig(architecture=name, num_classes=2, pretrained=True))
        succeeded = True
    except ValueError as e:
        assert "does not download" in str(e)
        succeeded = False
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, (
        f"'{name}' with pretrained=True took {elapsed:.1f}s - "
        "expected an instant local-cache-only check, not a network attempt."
    )
    if succeeded:
        assert isinstance(model, torch.nn.Module)


def test_pretrained_true_succeeds_when_genuinely_cached_locally():
    """
    This developer machine already has genuine ImageNet-pretrained weights
    cached for efficientnet_b0 (pre-existing, unrelated to this task). Use
    that real cache to verify the positive path: pretrained=True must
    actually load real pretrained weights (not random init) when they are
    legitimately available locally, with zero network access.
    """
    from hospital_client.model_management.architectures import _is_weights_cached_locally
    import torchvision.models as tv_models

    if not _is_weights_cached_locally(tv_models.EfficientNet_B0_Weights.DEFAULT.url):
        pytest.skip("efficientnet_b0 pretrained weights are not cached on this machine")

    model_pretrained = build_model(ModelConfig(architecture="efficientnet_b0", num_classes=2, pretrained=True))
    model_random = build_model(ModelConfig(architecture="efficientnet_b0", num_classes=2, pretrained=False))

    # Backbone weights differ from a fresh random init (proves real weights loaded).
    key = "features.0.0.weight"
    assert not torch.equal(model_pretrained.state_dict()[key], model_random.state_dict()[key])


def test_weights_cache_check_is_pure_local_filesystem_lookup(tmp_path, monkeypatch):
    import torch as _torch
    monkeypatch.setattr(_torch.hub, "get_dir", lambda: str(tmp_path))

    fake_url = "https://example-not-a-real-host.invalid/weights-abc123.pth"
    assert _is_weights_cached_locally(fake_url) is False

    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "weights-abc123.pth").write_bytes(b"fake")
    assert _is_weights_cached_locally(fake_url) is True


def test_require_local_weights_raises_actionable_message_when_missing(tmp_path, monkeypatch):
    import torch as _torch
    monkeypatch.setattr(_torch.hub, "get_dir", lambda: str(tmp_path))

    class _FakeWeights:
        url = "https://example-not-a-real-host.invalid/does-not-exist.pth"

    with pytest.raises(ValueError, match="does not download"):
        _require_local_weights(_FakeWeights(), "fake_arch")


def test_mobilevit_xxs_pretrained_rejected_without_any_download_attempt():
    start = time.monotonic()
    with pytest.raises(ValueError, match="pretrained"):
        build_model(ModelConfig(architecture="mobilevit_xxs", num_classes=2, pretrained=True))
    assert time.monotonic() - start < 1.0
