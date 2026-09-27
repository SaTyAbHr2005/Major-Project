from hospital_client.resource_training.recommender import generate_recommendations
from hospital_client.resource_training.resolved_config import to_training_config
from hospital_client.training.config import validate_training_config


def test_every_recommendation_resolves_to_a_valid_training_config(all_simulated_profiles, policy, small_dataset_characteristics, tmp_path):
    for profile in all_simulated_profiles.values():
        rec_set = generate_recommendations(profile, policy, small_dataset_characteristics)
        for rec in rec_set.recommendations:
            config = to_training_config(
                rec, model_id="test_model", dataset_dir=str(tmp_path / "data"),
                output_dir=str(tmp_path / "out"), num_classes=small_dataset_characteristics.num_classes,
            )
            validate_training_config(config)  # must not raise
            assert config.device == rec.device
            assert config.precision == rec.precision
            assert config.batch_size == rec.batch_size
            assert config.epochs == rec.epochs
            assert config.model_config.architecture == rec.architecture


def test_resolved_config_input_size_matches_architecture_catalog(low_profile, policy, small_dataset_characteristics, tmp_path):
    from hospital_client.model_management.registry import get_architecture_info

    rec_set = generate_recommendations(low_profile, policy, small_dataset_characteristics)
    rec = rec_set.recommendations[0]
    config = to_training_config(rec, "m", str(tmp_path / "d"), str(tmp_path / "o"), small_dataset_characteristics.num_classes)
    expected = get_architecture_info(rec.architecture).default_input_size
    assert tuple(config.model_config.input_size) == expected
