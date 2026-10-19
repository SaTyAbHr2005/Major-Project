import json
import os
import pytest

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.dp import PrivacyBudgetExceededError
from hospital_client.training.result import TrainingStatus
from hospital_client.training.trainer import Trainer, build_federation_handoff


def _dp_config(dataset_dir, output_dir, **overrides):
    defaults = dict(
        model_id="dp_model",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64), norm_type="group"),
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=2,
        batch_size=4,
        dp_enabled=True,
        dp_noise_multiplier=1.0,
        dp_max_grad_norm=1.0,
        dp_target_delta=1e-5,
        dp_target_epsilon=100.0,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_dp_pre_training_budget_refusal(small_dataset_dir, tmp_path):
    """The hospital must project epsilon and refuse before training if the cap would be exceeded."""
    output_dir = str(tmp_path / "out_refuse")
    # A tiny target epsilon (0.001) that cannot possibly cover 2 epochs of training
    config = _dp_config(small_dataset_dir, output_dir, dp_target_epsilon=0.001)
    with pytest.raises(PrivacyBudgetExceededError, match="Privacy budget exceeded"):
        Trainer(config).run()


def test_dp_training_execution_and_accounting(small_dataset_dir, tmp_path):
    """Real DP training with Opacus Poisson sampling, BatchMemoryManager, and RDP accounting."""
    output_dir = str(tmp_path / "out_dp")
    config = _dp_config(small_dataset_dir, output_dir, epochs=1, batch_size=4)
    result = Trainer(config).run()

    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    tc = result.training_configuration
    assert tc["dp_enabled"] is True
    assert tc["dp_noise_multiplier"] == 1.0
    assert tc["dp_max_grad_norm"] == 1.0
    assert tc["dp_target_delta"] == 1e-5
    assert tc["dp_sample_rate"] > 0
    assert tc["dp_steps"] > 0
    assert tc["dp_epsilon"] > 0
    assert tc["dp_cumulative_epsilon"] >= tc["dp_epsilon"]

    # Verify local dp_state.json persistence
    state_path = os.path.join(output_dir, "dp_state_dp_model.json")
    assert os.path.exists(state_path)
    with open(state_path, "r", encoding="utf-8") as f:
        dp_state = json.load(f)
    assert dp_state["job_id"] == "dp_model"
    assert len(dp_state["history"]) == 1
    entry = dp_state["history"][0]
    assert entry["noise_multiplier"] == 1.0
    assert entry["delta"] == 1e-5
    assert entry["steps"] == tc["dp_steps"]

    # Verify FederationHandoff includes all DP fields
    handoff = build_federation_handoff(result, output_dir, round_id="R1", client_id="H1")
    assert handoff.training_configuration["dp_enabled"] is True
    assert handoff.training_configuration["dp_epsilon"] == tc["dp_epsilon"]
    assert handoff.training_configuration["dp_cumulative_epsilon"] == tc["dp_cumulative_epsilon"]


def test_batch_memory_manager_physical_batch_adaptation(small_dataset_dir, tmp_path):
    """The logical batch defines q; Module 8 chooses the physical batch for BatchMemoryManager."""
    output_dir = str(tmp_path / "out_bmm")
    # Logical batch 8 with physical batch size 2
    config = _dp_config(small_dataset_dir, output_dir, epochs=1, batch_size=2, dp_logical_batch_size=8)
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    tc = result.training_configuration
    # q is based on logical batch 8, not physical batch 2
    # In small_dataset_dir, train count is ~16 (70% of 24)
    # q = 8 / N
    assert tc["dp_logical_batch_size"] == 8
    assert tc["batch_size"] == 2
    assert tc["dp_sample_rate"] == 8.0 / result.training_metrics["sample_count"]


def test_dp_rejects_fedprox_proximal_mu(small_dataset_dir, tmp_path):
    """TrainingConfig validation must refuse proximal_mu > 0 together with dp_enabled."""
    config = _dp_config(small_dataset_dir, str(tmp_path / "out"), proximal_mu=0.1)
    with pytest.raises(ValueError, match="Differential privacy cannot be combined with proximal_mu"):
        validate_training_config(config)


def test_dp_rejects_non_fp32_precision(small_dataset_dir, tmp_path):
    """validate_training_config must reject precision != 'fp32' when dp_enabled is True."""
    config = _dp_config(small_dataset_dir, str(tmp_path / "out_fp16"), precision="fp16")
    with pytest.raises(ValueError, match="Differential privacy requires precision='fp32'"):
        validate_training_config(config)


def test_dp_multi_round_state_persistence_across_runs(small_dataset_dir, tmp_path):
    """
    Two consecutive rounds of the same federation job with different model IDs and output dirs
    must persist history in the shared dp_state.json. Round 2's pre-check and cumulative
    epsilon must include Round 1.
    """
    job_id = "FED_JOB_MULTI_TEST"
    dp_state_path = str(tmp_path / "federation_runs" / job_id / "dp_state.json")

    # Round 1
    out_r1 = str(tmp_path / "runs" / "FED_JOB_MULTI_TEST_20261019_010000")
    config_r1 = _dp_config(
        small_dataset_dir,
        out_r1,
        model_id="FED_JOB_MULTI_TEST_20261019_010000",
        epochs=1,
        batch_size=4,
        dp_state_path=dp_state_path,
        dp_state_job_id=job_id,
        dp_target_epsilon=50.0,
    )
    result_r1 = Trainer(config_r1).run()
    assert result_r1.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    eps_r1 = result_r1.training_configuration["dp_epsilon"]
    cum_r1 = result_r1.training_configuration["dp_cumulative_epsilon"]
    assert cum_r1 == eps_r1

    # Verify state file on disk
    assert os.path.exists(dp_state_path)
    with open(dp_state_path, "r", encoding="utf-8") as f:
        st = json.load(f)
    assert st["job_id"] == job_id
    assert len(st["history"]) == 1

    # Round 2: Different timestamped model ID and output directory, same dp_state_path & job_id
    out_r2 = str(tmp_path / "runs" / "FED_JOB_MULTI_TEST_20261019_020000")
    config_r2 = _dp_config(
        small_dataset_dir,
        out_r2,
        model_id="FED_JOB_MULTI_TEST_20261019_020000",
        epochs=1,
        batch_size=4,
        dp_state_path=dp_state_path,
        dp_state_job_id=job_id,
        dp_target_epsilon=50.0,
    )
    result_r2 = Trainer(config_r2).run()
    assert result_r2.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    eps_r2 = result_r2.training_configuration["dp_epsilon"]
    cum_r2 = result_r2.training_configuration["dp_cumulative_epsilon"]

    # Round 2 cumulative epsilon must be greater than both round 1 and round 2 standalone epsilons
    assert cum_r2 > eps_r1
    assert cum_r2 > eps_r2

    with open(dp_state_path, "r", encoding="utf-8") as f:
        st2 = json.load(f)
    assert len(st2["history"]) == 2
    assert st2["history"][0]["steps"] > 0
    assert st2["history"][1]["steps"] > 0

    # If round 3 would exceed target cap, pre-check refuses before starting training
    out_r3 = str(tmp_path / "runs" / "FED_JOB_MULTI_TEST_20261019_030000")
    config_r3 = _dp_config(
        small_dataset_dir,
        out_r3,
        model_id="FED_JOB_MULTI_TEST_20261019_030000",
        epochs=1,
        batch_size=4,
        dp_state_path=dp_state_path,
        dp_state_job_id=job_id,
        dp_target_epsilon=cum_r2 + 0.0001,  # tight cap that round 3 will exceed
    )
    with pytest.raises(PrivacyBudgetExceededError, match="Privacy budget exceeded"):
        Trainer(config_r3).run()


def test_non_dp_training_unchanged(small_dataset_dir, tmp_path):
    """Non-DP training runs without any Opacus overhead or DP state."""
    config = TrainingConfig(
        model_id="non_dp",
        model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(64, 64)),
        dataset_dir=small_dataset_dir,
        output_dir=str(tmp_path / "out_non_dp"),
        epochs=1,
        batch_size=4,
    )
    result = Trainer(config).run()
    assert result.status == TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
    assert result.training_configuration.get("dp_enabled") is False
    assert not os.path.exists(os.path.join(tmp_path / "out_non_dp", "dp_state_non_dp.json"))

