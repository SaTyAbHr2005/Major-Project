"""The admin job list carries a DP job's settings and each hospital's epsilon from the server ledger (DP UI)."""
from federation.api_adapter import get_all_jobs, job_dp
from federation.jobs import JobManager
from federation.storage import FederationStorage

MAPPING = {"Parasitized": 0, "Uninfected": 1}
DP_REQS = {"dp_enabled": True, "dp_noise_multiplier": 1.0, "dp_max_grad_norm": 1.0, "dp_target_delta": 1e-5,
           "dp_target_epsilon": 5.0, "dp_logical_batch_size": 256}


def _job(jm, job_id, reqs):
    jm.create_job(federation_job_id=job_id, task="Malaria", task_type="image_classification",
                  architecture="mobilenet_v3_small", num_classes=2, class_mapping=MAPPING,
                  expected_participants=[], minimum_participants=2, deadline="", training_requirements=reqs,
                  aggregation_configuration={"protocol": "flower", "total_rounds": 2, "round_duration_hours": 24})


def test_job_list_reports_dp_settings_and_latest_epsilon_per_hospital(tmp_path):
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    jm = JobManager(storage)
    _job(jm, "JOB_DP", DP_REQS)
    _job(jm, "JOB_PLAIN", {})
    for round_id, hospital, cumulative in (("R1", "H1", 1.56), ("R1", "H2", 2.29), ("R2", "H1", 1.90)):
        storage.record_dp_round(federation_job_id="JOB_DP", round_id=round_id, participant_id=hospital,
                                noise_multiplier=1.0, sample_rate=0.0116, steps=261, delta=1e-5,
                                round_epsilon=1.56, cumulative_epsilon=cumulative)

    jobs = {j["federation_job_id"]: j for j in get_all_jobs(storage)}
    dp = job_dp(storage, jobs["JOB_DP"])
    assert dp["noise_multiplier"] == 1.0 and dp["max_grad_norm"] == 1.0 and dp["delta"] == 1e-5
    assert dp["epsilon_cap"] == 5.0 and dp["logical_batch_size"] == 256
    # The running total is the latest ledger row per hospital, with the number of accepted rounds.
    assert dp["spent"] == {"H1": {"rounds": 2, "epsilon": 1.90}, "H2": {"rounds": 1, "epsilon": 2.29}}
    assert job_dp(storage, jobs["JOB_PLAIN"]) is None
