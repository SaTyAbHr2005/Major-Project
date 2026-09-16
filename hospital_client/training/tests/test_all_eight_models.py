"""
Verifies all 8 Module 6 architectures through the SAME Module 7 training
engine (hospital_client.training.trainer.Trainer) - not eight duplicated
training implementations. Uses tiny synthetic datasets and 1 epoch each to
stay practical; this establishes functional correctness, not medical
accuracy (see module7_readme.md).
"""
import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import list_architectures
from hospital_client.training.config import TrainingConfig
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer

from hospital_client.training.tests.conftest import make_preprocessed_dataset

# vit_b16 needs a square size divisible by 16; mobilevit_xxs needs enough
# spatial resolution to survive 5 stride-2 stages. 64 works safely for all 8.
INPUT_SIZE = (64, 64)


@pytest.mark.parametrize("architecture", sorted(list_architectures()))
def test_architecture_trains_one_epoch_through_common_engine(architecture, tmp_path):
    dataset_dir = make_preprocessed_dataset(tmp_path, {"cat": 6, "dog": 6}, image_size=64)

    model_config = ModelConfig(architecture=architecture, num_classes=2, input_size=INPUT_SIZE, pretrained=False)
    config = TrainingConfig(
        model_id=architecture,
        model_config=model_config,
        dataset_dir=dataset_dir,
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        require_validation=True,
    )

    result = Trainer(config).run()

    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.epochs_run == 1
    assert result.architecture == architecture
    assert result.checkpoint_model_id == architecture
    assert result.checkpoint_version == 1
    assert "accuracy" in result.validation_metrics
    assert result.errors == []


@pytest.mark.parametrize("architecture", sorted(list_architectures()))
def test_architecture_parameters_extractable_after_training(architecture, tmp_path):
    from hospital_client.training.trainer import build_federation_handoff

    dataset_dir = make_preprocessed_dataset(tmp_path, {"cat": 6, "dog": 6}, image_size=64)
    model_config = ModelConfig(architecture=architecture, num_classes=2, input_size=INPUT_SIZE)
    config = TrainingConfig(
        model_id=architecture, model_config=model_config, dataset_dir=dataset_dir,
        output_dir=str(tmp_path / "out"), epochs=1, batch_size=2,
    )
    result = Trainer(config).run()
    handoff = build_federation_handoff(result, config.output_dir)
    assert len(handoff.parameters) > 0
    assert handoff.architecture == architecture
