"""
Resolves a Module 8 RecommendedConfig into Module 7's own TrainingConfig -
the "ResolvedTrainingConfig" the project's ML<->FL boundary describes.
Module 8 does not invent a parallel config schema (CLAUDE.md §9): once a
recommendation is selected, it becomes exactly the object Module 7 already
knows how to execute unchanged.

    Module 8 (this package)
        |
        v
    RecommendedConfig.to_training_config()
        |
        v
    hospital_client.training.config.TrainingConfig
        |
        v
    hospital_client.training.trainer.Trainer  (Module 7, unmodified)
"""
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import get_architecture_info
from hospital_client.resource_training.recommender import RecommendedConfig
from hospital_client.training.config import TrainingConfig


def to_training_config(
    recommendation: RecommendedConfig, model_id: str, dataset_dir: str, output_dir: str,
    num_classes: int, color_mode: str = "RGB", pretrained: bool = False,
) -> TrainingConfig:
    input_size = get_architecture_info(recommendation.architecture).default_input_size
    model_config = ModelConfig(
        architecture=recommendation.architecture, num_classes=num_classes,
        input_size=input_size, color_mode=color_mode, pretrained=pretrained,
    )
    return TrainingConfig(
        model_id=model_id, model_config=model_config, dataset_dir=dataset_dir, output_dir=output_dir,
        device=recommendation.device, precision=recommendation.precision,
        epochs=recommendation.epochs, batch_size=recommendation.batch_size,
        num_workers=recommendation.num_workers, run_test_evaluation=False,
    )
