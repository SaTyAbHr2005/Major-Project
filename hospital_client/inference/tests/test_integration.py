"""
I. Real cross-module integration - no mocks of Modules 4-8:

    Module 4 (ingest) -> Module 5 (preprocess) -> Module 7 (REAL training run,
    saving a Module 6 artifact) -> Module 16 (provision + local inference on a NEW image)

plus Module 5 ImageConfig <-> Module 6 compatibility and Module 8 device selection.
Modules 9-15 are not needed (and not used).
"""
import numpy as np
import pytest
import torch
from PIL import Image

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import ModelCompatibilityError
from hospital_client.inference.provisioning import LocalStoreModelProvider
from hospital_client.inference.tests.conftest import make_image
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters
from hospital_client.model_management.store import ModelStore
from hospital_client.preprocessing.config import ImageConfig, PreprocessingConfig
from hospital_client.training.config import TrainingConfig
from hospital_client.training.dataset import ManifestImageDataset
from hospital_client.training.result import TrainingStatus
from hospital_client.training.tests.conftest import make_preprocessed_dataset
from hospital_client.training.trainer import Trainer, build_federation_handoff

INPUT_SIZE = (64, 64)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m4_m5_m7")
    dataset_dir = make_preprocessed_dataset(tmp, {"cat": 8, "dog": 8}, image_size=64)  # real Module 4 -> Module 5 output
    output_dir = str(tmp / "training_out")
    config = TrainingConfig(
        model_id="chest_model", model_config=ModelConfig(architecture="mobilenet_v3_small", num_classes=2, input_size=INPUT_SIZE),
        dataset_dir=dataset_dir, output_dir=output_dir, epochs=1, batch_size=4,
    )
    result = Trainer(config).run()  # real Module 7 training
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    return {"result": result, "output_dir": output_dir, "dataset_dir": dataset_dir, "tmp": tmp}


def _new_image(path, seed=7):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)).save(str(path))
    return str(path)


def test_module7_trained_artifact_is_provisioned_and_used_for_local_inference(trained):
    provider = LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"])
    engine = LocalInferenceEngine(provider)
    image = _new_image(trained["tmp"] / "new_scan.png")
    result = engine.predict(image, reference="e2e-1")

    train = trained["result"]
    assert result.succeeded and result.reference == "e2e-1"
    assert (result.model_id, result.model_version, result.architecture) == ("chest_model", train.checkpoint_version, "mobilenet_v3_small")
    assert set(result.class_probabilities) == set(train.class_mapping) == {"cat", "dog"}  # labels come from Module 7's recorded mapping
    stored = ModelStore(provider.store_root).get_version("chest_model", train.checkpoint_version)
    assert result.model_info["artifact_sha256"] == stored.artifact_sha256
    assert result.model_info["model_status"] == "trained" and result.model_info["approval"]["platform_approved"] is False
    engine.close()


def test_inference_sees_exactly_the_tensor_module7_evaluation_uses(trained):
    engine = LocalInferenceEngine(LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]))
    image = _new_image(trained["tmp"] / "same_as_m7.png", seed=11)
    result = engine.predict(image)

    dataset = ManifestImageDataset({"records": [{"source_path": image, "processed_path": None, "label": "cat"}]},
                                   trained["result"].class_mapping, "RGB", INPUT_SIZE)
    m7_tensor = dataset[0][0].unsqueeze(0)  # what Module 7 would feed the model at evaluation time
    with torch.no_grad():
        m7_probs = torch.softmax(engine.model(m7_tensor), dim=1)[0]
    for label, index in trained["result"].class_mapping.items():
        assert result.class_probabilities[label] == pytest.approx(float(m7_probs[index]), abs=1e-6)
    engine.close()


def test_loaded_weights_are_exactly_what_module7_would_hand_to_federation(trained):
    engine = LocalInferenceEngine(LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]))
    handoff = build_federation_handoff(trained["result"], trained["output_dir"])  # Module 7 -> (future) Module 9 contract
    ours = get_parameters(engine.model)
    assert len(ours) == len(handoff.parameters) == handoff.parameter_count
    assert all(np.array_equal(a, b) for a, b in zip(ours, handoff.parameters))
    engine.close()  # the handoff itself is NOT an approved model source - only the Module 6 artifact is provisioned


def test_module5_image_config_compatibility_is_enforced_with_real_module5_config(trained):
    matching = PreprocessingConfig()
    matching.image = ImageConfig(target_size=INPUT_SIZE, color_mode="RGB")
    engine = LocalInferenceEngine(
        LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]),
        InferenceConfig(image_config=matching.image),
    )
    assert engine.predict(_new_image(trained["tmp"] / "c.png")).succeeded
    engine.close()

    wrong = ImageConfig(target_size=(128, 128), color_mode="RGB")
    with pytest.raises(ModelCompatibilityError, match="input_size mismatch"):
        LocalInferenceEngine(LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]),
                             InferenceConfig(image_config=wrong))


def test_module8_device_selection_drives_inference(trained):
    from hospital_client.resource_training.dataset_characteristics import inspect_dataset
    from hospital_client.resource_training.evaluator import select_device
    from hospital_client.resource_training.policy import default_policy
    from hospital_client.resource_training.recommender import generate_recommendations
    from hospital_client.resource_training.resource_profile import build_resource_profile

    profile = build_resource_profile(check_network=False)
    policy = default_policy()
    policy.enable_dry_run_measurement = False  # keep this integration test quick
    recommendations = generate_recommendations(profile, policy, inspect_dataset(trained["dataset_dir"])).recommendations
    assert recommendations
    module8_device, _ = select_device(profile)
    assert recommendations[0].device == module8_device

    for config in (InferenceConfig(device=recommendations[0].device), InferenceConfig(device="auto")):
        engine = LocalInferenceEngine(LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]), config)
        result = engine.predict(_new_image(trained["tmp"] / "d.png"))
        assert result.succeeded and result.device == module8_device
        engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_real_end_to_end_cuda_inference_of_a_cpu_trained_module7_model(trained):
    engine = LocalInferenceEngine(LocalStoreModelProvider.from_training_result(trained["result"], trained["output_dir"]),
                                  InferenceConfig(device="cuda"))
    result = engine.predict(_new_image(trained["tmp"] / "cuda_new.png"))
    assert result.succeeded and result.device == "cuda" and result.peak_memory_mb > 0
    assert set(result.class_probabilities) == {"cat", "dog"}
    engine.close()
