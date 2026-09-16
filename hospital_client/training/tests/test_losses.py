import pytest
import torch
import torch.nn as nn

from hospital_client.training.dataloader import PreparedData
from hospital_client.training.losses import build_loss


def _prepared(class_mapping, train_label_counts, balancing_meta=None):
    return PreparedData(
        train_loader=None, validation_loader=None, test_loader=None,
        class_mapping=class_mapping,
        train_manifest_metadata={"balancing": balancing_meta} if balancing_meta is not None else {},
        train_label_counts=train_label_counts,
        train_sample_count=sum(train_label_counts.values()),
        validation_sample_count=0, test_sample_count=0,
    )


def test_none_weighting_returns_plain_cross_entropy():
    prepared = _prepared({"cat": 0, "dog": 1}, {"cat": 5, "dog": 5})
    loss = build_loss("none", prepared, torch.device("cpu"))
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert loss.weight is None


def test_balanced_weighting_computes_inverse_frequency():
    prepared = _prepared({"normal": 0, "disease": 1}, {"normal": 20, "disease": 4})
    loss = build_loss("balanced", prepared, torch.device("cpu"))
    assert loss.weight is not None
    # disease (minority) must get a higher weight than normal (majority)
    assert loss.weight[1].item() > loss.weight[0].item()


def test_manifest_weighting_uses_module5_precomputed_weights():
    balancing = {"strategy": "class_weights", "training_counts": {"cat": 8, "dog": 2}, "class_weights": {"cat": 0.625, "dog": 2.5}}
    prepared = _prepared({"cat": 0, "dog": 1}, {"cat": 8, "dog": 2}, balancing_meta=balancing)
    loss = build_loss("manifest", prepared, torch.device("cpu"))
    assert loss.weight[0].item() == pytest.approx(0.625)
    assert loss.weight[1].item() == pytest.approx(2.5)


def test_manifest_weighting_raises_when_module5_did_not_compute_weights():
    prepared = _prepared({"cat": 0, "dog": 1}, {"cat": 5, "dog": 5}, balancing_meta={"strategy": "none", "training_counts": {}})
    with pytest.raises(ValueError, match="has no balancing.class_weights"):
        build_loss("manifest", prepared, torch.device("cpu"))


def test_validation_and_test_distributions_are_never_touched_by_weighting():
    # Class weighting only affects the loss function object; it must never
    # be applied to reshape/resample validation or test data (verified by
    # construction here - build_loss takes no validation/test arguments).
    import inspect
    assert "validation" not in inspect.signature(build_loss).parameters
    assert "test" not in inspect.signature(build_loss).parameters
