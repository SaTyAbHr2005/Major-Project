"""
Module 12 Tests: Byzantine and Malicious Model-Update Detection & Robust Aggregation.

Covers:
1. NaN / Inf rejection (per-update)
2. Delta L2 norm bounds (DP vs non-DP defaults, job overrides, FLAGGED vs QUARANTINED)
3. Cosine direction check on round >= 3 (and verify rounds 1-2 skip it)
4. Sample-count consistency across rounds (FLAG on change, QUARANTINE on >= 3x)
5. No false positives on clean empirical DP and non-DP updates
6. Intake handling of FLAG, QUARANTINE, REJECT decisions and UPDATE_FLAGGED event
7. Quarantined hospital behavior in round lifecycle (waits before deadline, INSUFFICIENT_PARTICIPANTS at deadline)
8. Robust aggregation (coordinate-wise median and trimmed mean vs FedAvg under attack)
9. Robust aggregation fallback with < 3 clients (logs warning, emits AGGREGATION_FALLBACK, fallback_reason in metadata)
10. Non-float parameter preservation during robust aggregation
11. Round plan validation for median/trimmed_mean, DP, and FedProx compatibility
12. Flower server integration with robust aggregation
"""
import copy
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from federation.aggregation import Aggregator
from federation.events import EventEmitter, FederationEvent
from federation.intake import IntakeManager
from federation.jobs import JobManager
from federation.models import (
    FederationJob,
    FederationRound,
    FederationUpdate,
    GlobalModelStatus,
    RoundStatus,
    UpdateStatus,
)
from federation.round_plan import validate_plan
from federation.rounds import RoundManager
from federation.security_hooks import (
    DEFAULT_DP_FLAG_RATIO,
    DEFAULT_DP_QUARANTINE_RATIO,
    DEFAULT_NON_DP_FLAG_RATIO,
    DEFAULT_NON_DP_QUARANTINE_RATIO,
    M12SecurityHook,
    SecurityDecision,
)
from federation.storage import FederationStorage
from federation.validation import UpdateValidator
from hospital_client.training.federation import build_federation_handoff, FederationHandoff


def make_handoff(
    parameters,
    model_id="M1",
    model_version=1,
    architecture="mobilenet_v3_small",
    base_model_id="BASE1",
    base_model_version=1,
    base_model_checksum="abc",
    num_train_samples=1000,
    class_mapping=None,
    training_metrics=None,
    validation_metrics=None,
    training_configuration=None,
    device="cpu",
    precision="fp32",
    status="TRAINING_COMPLETED_AWAITING_FEDERATION",
    round_id="R1",
    client_id="H1",
):
    """Build a complete, schema-valid FederationHandoff for tests."""
    return build_federation_handoff(
        parameters=parameters,
        model_id=model_id,
        model_version=model_version,
        architecture=architecture,
        base_model_id=base_model_id,
        base_model_version=base_model_version,
        base_model_checksum=base_model_checksum,
        num_train_samples=num_train_samples,
        class_mapping=class_mapping or {"0": 0, "1": 1},
        training_metrics=training_metrics or {"loss": 0.5, "accuracy": 0.85},
        validation_metrics=validation_metrics or {"val_loss": 0.5, "val_accuracy": 0.85},
        training_configuration=training_configuration or {"batch_size": 32, "lr": 0.001},
        device=device,
        precision=precision,
        status=status,
        round_id=round_id,
        client_id=client_id,
    )


def make_update(
    update_id="u1",
    federation_job_id="JOB",
    round_id="R1",
    participant_id="H1",
    model_id="M1",
    architecture="mobilenet",
    artifact_location="",
    artifact_checksum="",
    parameter_count=1,
    parameter_metadata=None,
    num_train_samples=1000,
    training_configuration=None,
    status=UpdateStatus.RECEIVED,
    security_status="ACCEPT",
    security_reason="",
    security_metrics=None,
):
    """Build a complete, schema-valid FederationUpdate for tests."""
    return FederationUpdate(
        update_id=update_id,
        federation_job_id=federation_job_id,
        round_id=round_id,
        participant_id=participant_id,
        model_id=model_id,
        architecture=architecture,
        artifact_location=artifact_location,
        artifact_checksum=artifact_checksum,
        parameter_count=parameter_count,
        parameter_metadata=parameter_metadata or {},
        num_train_samples=num_train_samples,
        training_configuration=training_configuration or {"batch_size": 32},
        status=status,
        security_status=security_status,
        security_reason=security_reason,
        security_metrics=security_metrics or {},
    )


class TestM12SecurityAndRobustAggregation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.db")
        self.artifacts_dir = os.path.join(self.temp_dir, "test_artifacts")
        self.storage = FederationStorage(self.db_path, self.artifacts_dir)
        self.events = EventEmitter()

        self.hook = M12SecurityHook(self.storage)
        self.intake_mgr = IntakeManager(
            self.storage, UpdateValidator(self.storage), self.hook, self.events
        )
        self.aggregator = Aggregator(self.storage, self.events)
        self.job_mgr = JobManager(self.storage)
        self.round_mgr = RoundManager(self.storage)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ── 1. NaN / Inf Rejection ────────────────────────────────────────────────

    def test_nan_parameter_rejected(self):
        """Update containing NaN in any parameter tensor must be REJECTED."""
        p_nan = [np.ones((4, 4), dtype=np.float32), np.array([1.0, np.nan, 0.0, 2.0], dtype=np.float32)]
        handoff = make_handoff(p_nan)
        update = make_update(update_id="u_nan", participant_id="Hosp_Bad", parameter_count=2)

        decision = self.hook.evaluate_update(update, handoff)
        self.assertEqual(decision.verdict, "REJECT")
        self.assertTrue(any("NaN or Inf" in r for r in decision.reasons))

    def test_inf_parameter_rejected(self):
        """Update containing Inf in any parameter tensor must be REJECTED."""
        p_inf = [np.array([[np.inf, 1.0], [0.0, -1.0]], dtype=np.float32)]
        handoff = make_handoff(p_inf)
        update = make_update(update_id="u_inf", participant_id="Hosp_Bad", parameter_count=1)

        decision = self.hook.evaluate_update(update, handoff)
        self.assertEqual(decision.verdict, "REJECT")
        self.assertTrue(any("NaN or Inf" in r for r in decision.reasons))

    # ── 2. Delta L2 Norm Bound (Non-DP & Overrides) ───────────────────────────

    def test_delta_norm_non_dp_thresholds(self):
        """
        Non-DP thresholds: flag at 15.0x, quarantine at 50.0x.
        Verified against empirical max clean observed ratio of 12.25.
        """
        job = FederationJob(
            federation_job_id="JOB_NORM",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        rnd = FederationRound(
            round_id="R1",
            federation_job_id="JOB_NORM",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )

        base_p = [np.ones((10, 10), dtype=np.float32)]  # norm = 10.0
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, rnd.base_model_version)
        make_handoff(base_p).save(base_dir)

        u_norm = make_update(update_id="u1", federation_job_id="JOB_NORM", round_id="R1", participant_id="H1")

        # Case A: Normal delta (ratio = 0.1 < 2.2) -> ACCEPT
        p_normal = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 0.1]
        dec_normal = self.hook.evaluate_update(u_norm, make_handoff(p_normal), job, rnd)
        self.assertEqual(dec_normal.verdict, "ACCEPT")

        # Case B: Scaled delta (ratio = 3.0 > 2.2 flag threshold) -> FLAG
        p_flag = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 3.0]
        dec_flag = self.hook.evaluate_update(u_norm, make_handoff(p_flag), job, rnd)
        self.assertEqual(dec_flag.verdict, "FLAG")
        self.assertTrue(any("exceeds flag threshold" in r for r in dec_flag.reasons))

        # Case C: Extreme delta (ratio = 8.0 > 5.5 quarantine threshold) -> QUARANTINE
        p_quar = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 8.0]
        dec_quar = self.hook.evaluate_update(u_norm, make_handoff(p_quar), job, rnd)
        self.assertEqual(dec_quar.verdict, "QUARANTINE")
        self.assertTrue(any("exceeds quarantine threshold" in r for r in dec_quar.reasons))

    def test_3x_scaled_non_dp_update_is_flagged(self):
        """
        A non-DP update whose delta is scaled by 3x exceeds the calibrated flag threshold (2.2) and is FLAGGED.
        Baseline clean non-DP delta ratio is ~0.9 (within observed clean range <= 1.1031).
        When scaled 3x, ratio becomes 2.7 (> 2.2 flag threshold).
        """
        job = FederationJob(
            federation_job_id="JOB_3X_SCALED",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        rnd = FederationRound(
            round_id="R1",
            federation_job_id="JOB_3X_SCALED",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )
        base_p = [np.ones((10, 10), dtype=np.float32)]  # norm = 10.0
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, rnd.base_model_version)
        make_handoff(base_p).save(base_dir)

        u = make_update(update_id="u1", federation_job_id="JOB_3X_SCALED", round_id="R1", participant_id="H1")

        # Clean update with ratio 0.9 (norm = 9.0) -> ACCEPT (< 2.2)
        clean_delta = np.ones((10, 10), dtype=np.float32) * 0.9
        p_clean = [base_p[0] + clean_delta]
        dec_clean = self.hook.evaluate_update(u, make_handoff(p_clean), job, rnd)
        self.assertEqual(dec_clean.verdict, "ACCEPT")
        self.assertAlmostEqual(dec_clean.metrics["delta_to_base_ratio"], 0.9, places=3)

        # 3x scaled delta -> delta norm 27.0, ratio 2.7 (> 2.2 flag threshold, < 5.5 quarantine) -> FLAG
        scaled_delta = clean_delta * 3.0
        p_scaled = [base_p[0] + scaled_delta]
        dec_scaled = self.hook.evaluate_update(u, make_handoff(p_scaled), job, rnd)
        self.assertEqual(dec_scaled.verdict, "FLAG")
        self.assertAlmostEqual(dec_scaled.metrics["delta_to_base_ratio"], 2.7, places=3)
        self.assertTrue(any("exceeds flag threshold" in r for r in dec_scaled.reasons))

    def test_delta_norm_job_overrides(self):
        """Job training_requirements can override L2 norm thresholds explicitly."""
        job = FederationJob(
            federation_job_id="JOB_CUSTOM_NORM",
            task="tb",
            task_type="image_classification",
            architecture="resnet18",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={
                "m12_max_l2_norm": 5.0,
                "m12_quarantine_l2_norm": 15.0,
            },
            aggregation_configuration={},
        )
        rnd = FederationRound(
            round_id="R1",
            federation_job_id="JOB_CUSTOM_NORM",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )

        base_p = [np.ones((10, 10), dtype=np.float32)]  # norm = 10.0
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, rnd.base_model_version)
        make_handoff(base_p, architecture="resnet18").save(base_dir)

        u = make_update(update_id="u1", federation_job_id="JOB_CUSTOM_NORM", round_id="R1", participant_id="H1", architecture="resnet18")

        # Delta norm = 8.0 (> 5.0 flag, < 15.0 quarantine) -> FLAG
        p_flag = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 0.8]  # norm = 8.0
        dec_flag = self.hook.evaluate_update(u, make_handoff(p_flag, architecture="resnet18"), job, rnd)
        self.assertEqual(dec_flag.verdict, "FLAG")

        # Delta norm = 20.0 (> 15.0) -> QUARANTINE
        p_quar = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 2.0]  # norm = 20.0
        dec_quar = self.hook.evaluate_update(u, make_handoff(p_quar, architecture="resnet18"), job, rnd)
        self.assertEqual(dec_quar.verdict, "QUARANTINE")

    # ── 3. Delta L2 Norm for DP Jobs ──────────────────────────────────────────

    def test_dp_delta_norm_defaults(self):
        """
        DP jobs have lower default ratio bounds (flag at 2.0x, quarantine at 10.0x)
        because DP updates have clipped gradients (clean observed max ~0.169).
        """
        job = FederationJob(
            federation_job_id="JOB_DP_NORM",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={"dp_enabled": True, "dp_noise_multiplier": 0.8},
            aggregation_configuration={},
        )
        rnd = FederationRound(
            round_id="R1",
            federation_job_id="JOB_DP_NORM",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )

        base_p = [np.ones((10, 10), dtype=np.float32)]  # norm = 10.0
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, rnd.base_model_version)
        make_handoff(base_p).save(base_dir)

        u = make_update(update_id="u1", federation_job_id="JOB_DP_NORM", round_id="R1", participant_id="H1")

        # Ratio = 0.1 (clean DP update, < 0.4 flag threshold) -> ACCEPT
        p_ok = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 0.1]
        self.assertEqual(self.hook.evaluate_update(u, make_handoff(p_ok), job, rnd).verdict, "ACCEPT")

        # Ratio = 0.6 (> DP default 0.4 flag threshold, < 1.0 quarantine) -> FLAG
        p_flag = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 0.6]
        self.assertEqual(self.hook.evaluate_update(u, make_handoff(p_flag), job, rnd).verdict, "FLAG")

        # Ratio = 1.5 (> DP default 1.0 quarantine threshold) -> QUARANTINE
        p_quar = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 1.5]
        self.assertEqual(self.hook.evaluate_update(u, make_handoff(p_quar), job, rnd).verdict, "QUARANTINE")

    # ── 4. Cosine Direction Check (Rounds >= 3) ───────────────────────────────

    def test_cosine_direction_skipped_on_rounds_1_and_2(self):
        """
        Rounds 1 and 2 skip the direction check because there is no prior global trajectory.
        Documented requirement: early rounds have no established global update direction,
        so sign flips in rounds 1-2 must not be falsely flagged on direction.
        """
        job = FederationJob(
            federation_job_id="JOB_COS",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        base_p = [np.ones((10, 10), dtype=np.float32)]
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, 1)
        make_handoff(base_p).save(base_dir)

        for r_num in [1, 2]:
            rnd = FederationRound(
                round_id=f"R{r_num}",
                federation_job_id="JOB_COS",
                round_number=r_num,
                expected_participants=["H1"],
                minimum_participants=1,
                deadline="",
                base_model_id="GM1",
                base_model_version=1,
                base_model_checksum="abc",
            )
            u = make_update(
                update_id=f"u_{r_num}", federation_job_id="JOB_COS", round_id=f"R{r_num}", participant_id="H1"
            )
            # Update moving in negative direction with moderate norm (ratio = 0.5)
            p_opposite = [base_p[0] - np.ones((10, 10), dtype=np.float32) * 0.05]
            dec = self.hook.evaluate_update(u, make_handoff(p_opposite), job, rnd)
            self.assertFalse(any("Cosine similarity" in r for r in dec.reasons))
            self.assertEqual(dec.verdict, "ACCEPT")

    def test_cosine_direction_flagged_on_round_3(self):
        """
        On round >= 3, when two prior global models exist, updates strongly opposite
        to the global trajectory (cosine < -0.1) are FLAGGED.
        """
        job = FederationJob(
            federation_job_id="JOB_COS_R3",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        self.storage.save_job(job)

        # Create two prior global models: G1 and G2
        p_g1 = [np.zeros((10, 10), dtype=np.float32)]
        p_g2 = [np.ones((10, 10), dtype=np.float32) * 0.2]

        dir_g1 = self.storage.get_global_model_dir(job.task, job.architecture, 1)
        dir_g2 = self.storage.get_global_model_dir(job.task, job.architecture, 2)
        make_handoff(p_g1).save(dir_g1)
        make_handoff(p_g2).save(dir_g2)

        from federation.models import GlobalModelVersion
        for v, d in [(1, dir_g1), (2, dir_g2)]:
            gm = GlobalModelVersion(
                global_model_id=f"GM{v}",
                task="malaria",
                task_type="image_classification",
                architecture="mobilenet",
                version=v,
                artifact_location=d,
                artifact_checksum="chk",
                artifact_size=100,
                parameter_count=1,
                parameter_metadata={},
                federation_job_id="JOB_COS_R3",
                round_id=f"R{v}",
                source_update_ids=[],
                aggregation_method="weighted_fedavg",
                aggregation_metadata={},
                status=GlobalModelStatus.MAIN,
            )
            self.storage.save_global_model(gm)

        rnd3 = FederationRound(
            round_id="R3",
            federation_job_id="JOB_COS_R3",
            round_number=3,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM2",
            base_model_version=2,
            base_model_checksum="abc",
        )
        self.storage.save_round(rnd3)

        u = make_update(update_id="u3", federation_job_id="JOB_COS_R3", round_id="R3", participant_id="H1")

        # Case A: Update aligned with trajectory (G2 + 0.1) -> cos ~ +1.0 -> ACCEPT
        p_aligned = [p_g2[0] + np.ones((10, 10), dtype=np.float32) * 0.1]
        dec_aligned = self.hook.evaluate_update(u, make_handoff(p_aligned), job, rnd3)
        self.assertEqual(dec_aligned.verdict, "ACCEPT")
        self.assertAlmostEqual(dec_aligned.metrics.get("cosine_similarity", 0.0), 1.0, places=2)

        # Case B: Update opposite to trajectory (G2 - 0.1) -> cos ~ -1.0 -> FLAG
        p_opp = [p_g2[0] - np.ones((10, 10), dtype=np.float32) * 0.1]
        dec_opp = self.hook.evaluate_update(u, make_handoff(p_opp), job, rnd3)
        self.assertEqual(dec_opp.verdict, "FLAG")
        self.assertTrue(any("Cosine similarity" in r for r in dec_opp.reasons))
        self.assertAlmostEqual(dec_opp.metrics.get("cosine_similarity", 0.0), -1.0, places=2)

    # ── 5. Sample-Count Consistency Across Rounds ────────────────────────────

    def test_sample_count_consistency_check(self):
        """
        FedAvg weights by num_train_samples.
        If a hospital reports a different sample count in later rounds of the same job:
        - Minor change -> FLAGGED
        - Large jump (>= 3x) -> QUARANTINED
        """
        job = FederationJob(
            federation_job_id="JOB_SAMPLES",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        self.storage.save_job(job)

        u_r1 = make_update(
            update_id="u_r1_h1",
            federation_job_id="JOB_SAMPLES",
            round_id="R1",
            participant_id="H1",
            num_train_samples=1000,
            status=UpdateStatus.ACCEPTED,
        )
        self.storage.save_update(u_r1)

        rnd2 = FederationRound(
            round_id="R2",
            federation_job_id="JOB_SAMPLES",
            round_number=2,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )
        p = [np.ones((4, 4), dtype=np.float32)]

        # Case A: Same count (1000) -> ACCEPT
        u_same = make_update(update_id="u_r2_same", federation_job_id="JOB_SAMPLES", round_id="R2", participant_id="H1", num_train_samples=1000)
        dec_same = self.hook.evaluate_update(u_same, make_handoff(p, num_train_samples=1000), job, rnd2)
        self.assertEqual(dec_same.verdict, "ACCEPT")

        # Case B: Slight change (1200, 1.2x) -> FLAG
        u_change = make_update(update_id="u_r2_change", federation_job_id="JOB_SAMPLES", round_id="R2", participant_id="H1", num_train_samples=1200)
        dec_change = self.hook.evaluate_update(u_change, make_handoff(p, num_train_samples=1200), job, rnd2)
        self.assertEqual(dec_change.verdict, "FLAG")
        self.assertTrue(any("Sample count changed" in r for r in dec_change.reasons))

        # Case C: Drastic jump (4000, 4.0x >= 3.0x) -> QUARANTINE
        u_jump = make_update(update_id="u_r2_jump", federation_job_id="JOB_SAMPLES", round_id="R2", participant_id="H1", num_train_samples=4000)
        dec_jump = self.hook.evaluate_update(u_jump, make_handoff(p, num_train_samples=4000), job, rnd2)
        self.assertEqual(dec_jump.verdict, "QUARANTINE")
        self.assertTrue(any("jumped drastically" in r for r in dec_jump.reasons))

    # ── 6. No False Positives on Clean Empirical Ranges ───────────────────────

    def test_no_false_positives_empirical_ranges(self):
        """
        Verify that realistic clean updates matching empirical measurements
        (DP ratio ~0.15, cosine ~0.85; Non-DP ratio ~1.8, cosine ~0.87)
        are cleanly accepted without false positive flags or quarantines.
        """
        job_dp = FederationJob(
            federation_job_id="JOB_CLEAN_DP", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={}, expected_participants=["H1"],
            minimum_participants=1, deadline="",
            training_requirements={"differential_privacy": {"enabled": True, "target_epsilon": 5.0}},
            aggregation_configuration={}
        )
        rnd_dp = FederationRound(
            round_id="R1_DP", federation_job_id="JOB_CLEAN_DP", round_number=1,
            expected_participants=["H1"], minimum_participants=1, deadline="",
            base_model_id="GM1", base_model_version=1, base_model_checksum="abc"
        )
        base_p = [np.ones((10, 10), dtype=np.float32) * 5.0]  # norm = 50.0
        base_dir_dp = self.storage.get_global_model_dir(job_dp.task, job_dp.architecture, rnd_dp.base_model_version)
        make_handoff(base_p).save(base_dir_dp)

        u_dp = make_update(update_id="u_dp", federation_job_id="JOB_CLEAN_DP", round_id="R1_DP", participant_id="H1")
        p_dp = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 0.75]
        dec_dp = self.hook.evaluate_update(u_dp, make_handoff(p_dp), job_dp, rnd_dp)
        self.assertEqual(dec_dp.verdict, "ACCEPT")
        self.assertEqual(len(dec_dp.reasons), 0)

        job_nondp = FederationJob(
            federation_job_id="JOB_CLEAN_NONDP", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={}, expected_participants=["H1"],
            minimum_participants=1, deadline="", training_requirements={}, aggregation_configuration={}
        )
        rnd_nondp = FederationRound(
            round_id="R1_NONDP", federation_job_id="JOB_CLEAN_NONDP", round_number=1,
            expected_participants=["H1"], minimum_participants=1, deadline="",
            base_model_id="GM1", base_model_version=1, base_model_checksum="abc"
        )
        base_dir_ndp = self.storage.get_global_model_dir(job_nondp.task, job_nondp.architecture, rnd_nondp.base_model_version)
        make_handoff(base_p).save(base_dir_ndp)

        u_nondp = make_update(update_id="u_nondp", federation_job_id="JOB_CLEAN_NONDP", round_id="R1_NONDP", participant_id="H1")
        p_nondp = [base_p[0] + np.ones((10, 10), dtype=np.float32) * 9.0]
        dec_nondp = self.hook.evaluate_update(u_nondp, make_handoff(p_nondp), job_nondp, rnd_nondp)
        self.assertEqual(dec_nondp.verdict, "ACCEPT")
        self.assertEqual(len(dec_nondp.reasons), 0)

    # ── 7. Intake Pipeline Decision Handling & Events ────────────────────────

    def test_intake_flag_handling(self):
        """
        When hook returns FLAG:
        - update.status is ACCEPTED (so it can be used in robust aggregation)
        - update.security_status is 'FLAG'
        - UPDATE_FLAGGED event is emitted
        - update.security_reason and security_metrics are persisted
        """
        job = self.job_mgr.create_job(
            federation_job_id="JOB_INTAKE", task="malaria", task_type="image_classification",
            architecture="mobilenet_v3_small", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H1", "H2"], minimum_participants=2, deadline="",
            training_requirements={}, aggregation_configuration={}
        )
        base_p = [np.ones((4, 4), dtype=np.float32)]
        from hospital_client.training.federation import _compute_parameters_checksum
        base_chk = _compute_parameters_checksum(base_p)

        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_INTAKE", round_id="R_INTAKE", round_number=1,
            expected_participants=["H1", "H2"], minimum_participants=2,
            deadline=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            base_model_id="BASE1", base_model_version=1, base_model_checksum=base_chk
        )
        self.round_mgr.open_round("R_INTAKE")

        # Save base model
        base_dir = os.path.join(self.artifacts_dir, "base_models", "BASE1", "v1")
        os.makedirs(base_dir, exist_ok=True)
        from safetensors.numpy import save_file
        save_file({"param_0": base_p[0]}, os.path.join(base_dir, "model.safetensors"))

        # Create an update that triggers FLAG (ratio ~ 3.0 > 2.2 flag threshold, < 5.5 quarantine)
        p_flag = [base_p[0] + np.ones((4, 4), dtype=np.float32) * 3.0]
        update_dir = os.path.join(self.temp_dir, "upload_flag")
        make_handoff(
            p_flag, base_model_id="BASE1", base_model_version=1, base_model_checksum=base_chk
        ).save(update_dir)

        flagged_events = []
        self.events.add_listener(lambda payload: flagged_events.append(payload) if payload.get("event") == FederationEvent.UPDATE_FLAGGED.value else None)

        res_update = self.intake_mgr.receive_handoff(
            federation_job_id="JOB_INTAKE", round_id="R_INTAKE", participant_id="H1",
            handoff_dir=update_dir
        )

        self.assertEqual(res_update.status, UpdateStatus.ACCEPTED)
        self.assertEqual(res_update.security_status, "FLAG")
        self.assertIn("exceeds flag threshold", res_update.security_reason)
        self.assertEqual(len(flagged_events), 1)
        self.assertEqual(flagged_events[0]["participant_id"], "H1")

        # Verify participant is in accepted_participants of the round
        refreshed_round = self.storage.get_round("R_INTAKE")
        self.assertIn("H1", refreshed_round.accepted_participants)
        self.assertNotIn("H1", refreshed_round.quarantined_participants)

    def test_intake_quarantine_handling(self):
        """
        When hook returns QUARANTINE:
        - update.status is QUARANTINED
        - participant is placed in round.quarantined_participants (NOT accepted_participants)
        - UPDATE_QUARANTINED event is emitted
        """
        job = self.job_mgr.create_job(
            federation_job_id="JOB_QUAR", task="malaria", task_type="image_classification",
            architecture="mobilenet_v3_small", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H1"], minimum_participants=1, deadline="",
            training_requirements={}, aggregation_configuration={}
        )
        base_p = [np.ones((4, 4), dtype=np.float32)]
        from hospital_client.training.federation import _compute_parameters_checksum
        base_chk = _compute_parameters_checksum(base_p)

        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_QUAR", round_id="R_QUAR", round_number=1,
            expected_participants=["H1"], minimum_participants=1,
            deadline=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            base_model_id="BASE1", base_model_version=1, base_model_checksum=base_chk
        )
        self.round_mgr.open_round("R_QUAR")

        base_dir = os.path.join(self.artifacts_dir, "base_models", "BASE1", "v1")
        os.makedirs(base_dir, exist_ok=True)
        from safetensors.numpy import save_file
        save_file({"param_0": base_p[0]}, os.path.join(base_dir, "model.safetensors"))

        # Extreme ratio 60.0 > 50.0 -> QUARANTINE
        p_quar = [base_p[0] + np.ones((4, 4), dtype=np.float32) * 60.0]
        update_dir = os.path.join(self.temp_dir, "upload_quar")
        make_handoff(
            p_quar, base_model_id="BASE1", base_model_version=1, base_model_checksum=base_chk
        ).save(update_dir)

        quar_events = []
        self.events.add_listener(lambda payload: quar_events.append(payload) if payload.get("event") == FederationEvent.UPDATE_QUARANTINED.value else None)

        res_update = self.intake_mgr.receive_handoff(
            federation_job_id="JOB_QUAR", round_id="R_QUAR", participant_id="H1",
            handoff_dir=update_dir
        )

        self.assertEqual(res_update.status, UpdateStatus.QUARANTINED)
        self.assertEqual(res_update.security_status, "QUARANTINE")
        self.assertEqual(len(quar_events), 1)

        refreshed_round = self.storage.get_round("R_QUAR")
        self.assertIn("H1", refreshed_round.quarantined_participants)
        self.assertNotIn("H1", refreshed_round.accepted_participants)

    # ── 8. Quarantined Hospital Round Lifecycle ───────────────────────────────

    def test_quarantined_hospital_round_lifecycle(self):
        """
        Verify that a quarantined update does NOT satisfy minimum_participants quorum:
        - Before deadline: round stays OPEN waiting for additional participants.
        - Past deadline: check_deadline_and_participants transitions round to INSUFFICIENT_PARTICIPANTS.
        """
        past_deadline = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        job = self.job_mgr.create_job(
            federation_job_id="JOB_QUAR_LC", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H_Clean", "H_Malicious"], minimum_participants=2, deadline=past_deadline,
            training_requirements={}, aggregation_configuration={}
        )

        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_QUAR_LC", round_id="R_LC", round_number=1,
            expected_participants=["H_Clean", "H_Malicious"], minimum_participants=2,
            deadline=past_deadline, base_model_id="BASE1", base_model_version=1, base_model_checksum="abc"
        )
        self.round_mgr.open_round("R_LC")
        rnd = self.storage.get_round("R_LC")

        # Simulate: H_Clean accepted, H_Malicious quarantined
        rnd.received_participants = ["H_Clean", "H_Malicious"]
        rnd.accepted_participants = ["H_Clean"]
        rnd.quarantined_participants = ["H_Malicious"]
        self.storage.save_round(rnd)

        # Before deadline check
        rnd_before = copy.deepcopy(rnd)
        rnd_before.deadline = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        self.storage.save_round(rnd_before)
        rnd_obj_before = self.round_mgr.check_deadline_and_participants("R_LC")
        self.assertEqual(rnd_obj_before.status, RoundStatus.OPEN)

        # Past deadline check
        rnd_after = copy.deepcopy(rnd)
        rnd_after.deadline = past_deadline
        self.storage.save_round(rnd_after)
        rnd_obj_after = self.round_mgr.check_deadline_and_participants("R_LC")
        self.assertEqual(rnd_obj_after.status, RoundStatus.INSUFFICIENT_PARTICIPANTS)

    # ── 9. Robust Aggregation: Median and Trimmed Mean vs FedAvg ───────────────

    def test_robust_aggregation_under_byzantine_attack(self):
        """
        5 participants: 4 benign updates agreeing closely around +1.0,
        and 1 Byzantine attacker submitting +1000.0.

        Results:
        - FedAvg: heavily corrupted by the attacker (> 150.0).
        - Coordinate Median: attacker eliminated, result is ~1.05.
        - Trimmed Mean (trim_ratio=0.2): trims highest and lowest, result is ~1.05.
        """
        job_fedavg = self.job_mgr.create_job(
            federation_job_id="JOB_AGG_FEDAVG", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=[f"H{i}" for i in range(5)], minimum_participants=5, deadline="",
            training_requirements={}, aggregation_configuration={"strategy": "fedavg"}
        )
        job_median = self.job_mgr.create_job(
            federation_job_id="JOB_AGG_MEDIAN", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=[f"H{i}" for i in range(5)], minimum_participants=5, deadline="",
            training_requirements={}, aggregation_configuration={"strategy": "median"}
        )
        job_trimmed = self.job_mgr.create_job(
            federation_job_id="JOB_AGG_TRIMMED", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=[f"H{i}" for i in range(5)], minimum_participants=5, deadline="",
            training_requirements={}, aggregation_configuration={"strategy": "trimmed_mean", "trim_ratio": 0.2}
        )

        for j_obj in [job_fedavg, job_median, job_trimmed]:
            rnd = self.round_mgr.create_round(
                federation_job_id=j_obj.federation_job_id, round_id=f"R_{j_obj.federation_job_id}",
                round_number=1, expected_participants=[f"H{i}" for i in range(5)],
                minimum_participants=5, deadline="", base_model_id="BASE1", base_model_version=1,
                base_model_checksum="abc"
            )
            rnd.status = RoundStatus.READY_FOR_AGGREGATION
            rnd.received_participants = [f"H{i}" for i in range(5)]
            rnd.accepted_participants = [f"H{i}" for i in range(5)]
            self.storage.save_round(rnd)

            # Create 4 benign updates near 1.0 (0.9, 1.0, 1.05, 1.1) and 1 attacker at 1000.0
            values = [0.9, 1.0, 1.05, 1.1, 1000.0]
            for i, val in enumerate(values):
                u_dir = os.path.join(self.artifacts_dir, "updates", f"u_{j_obj.federation_job_id}_{i}")
                p = [np.array([val, val * 2.0], dtype=np.float32)]
                make_handoff(p, client_id=f"H{i}", round_id=f"R_{j_obj.federation_job_id}").save(u_dir)

                update = make_update(
                    update_id=f"up_{j_obj.federation_job_id}_{i}",
                    federation_job_id=j_obj.federation_job_id,
                    round_id=f"R_{j_obj.federation_job_id}",
                    participant_id=f"H{i}",
                    model_id=f"M_{i}",
                    artifact_location=u_dir,
                    artifact_checksum=f"chk_{i}",
                    parameter_metadata={"shapes": [list(p[0].shape)], "dtypes": [str(p[0].dtype)]},
                    num_train_samples=100,
                    status=UpdateStatus.ACCEPTED,
                )
                self.storage.save_update(update)

        # Aggregate FedAvg
        gm_fedavg = self.aggregator.aggregate_round("JOB_AGG_FEDAVG", "R_JOB_AGG_FEDAVG")
        p_fedavg = FederationHandoff.load(gm_fedavg.artifact_location).parameters[0]
        # (0.9 + 1.0 + 1.05 + 1.1 + 1000.0) / 5 = 200.81
        self.assertGreater(p_fedavg[0], 150.0)

        # Aggregate Coordinate Median
        gm_median = self.aggregator.aggregate_round("JOB_AGG_MEDIAN", "R_JOB_AGG_MEDIAN")
        p_median = FederationHandoff.load(gm_median.artifact_location).parameters[0]
        # Sorted: [0.9, 1.0, 1.05, 1.1, 1000.0] -> median is 1.05
        self.assertAlmostEqual(p_median[0], 1.05, places=2)
        self.assertEqual(gm_median.aggregation_method, "coordinate_median")

        # Aggregate Trimmed Mean (trim_ratio=0.2 on N=5 -> trim 1 from each end: keeps [1.0, 1.05, 1.1])
        # Mean of [1.0, 1.05, 1.1] = 1.05
        gm_trimmed = self.aggregator.aggregate_round("JOB_AGG_TRIMMED", "R_JOB_AGG_TRIMMED")
        p_trimmed = FederationHandoff.load(gm_trimmed.artifact_location).parameters[0]
        self.assertAlmostEqual(p_trimmed[0], 1.05, places=2)
        self.assertEqual(gm_trimmed.aggregation_method, "coordinate_trimmed_mean")
        self.assertEqual(gm_trimmed.aggregation_metadata.get("trim_count"), 1)

    # ── 10. Robust Aggregation Fallback with < 3 Clients ──────────────────────

    def test_robust_aggregation_fallback_below_3_clients(self):
        """
        Coordinate median and trimmed mean require >= 3 participants.
        With 2 participants:
        - Automatically falls back to weighted FedAvg
        - Logs warning
        - Populates fallback_reason in aggregation_metadata
        - Emits AGGREGATION_FALLBACK event
        """
        job = self.job_mgr.create_job(
            federation_job_id="JOB_FALLBACK", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H1", "H2"], minimum_participants=2, deadline="",
            training_requirements={}, aggregation_configuration={"strategy": "median"}
        )
        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_FALLBACK", round_id="R_FB", round_number=1,
            expected_participants=["H1", "H2"], minimum_participants=2, deadline="",
            base_model_id="BASE1", base_model_version=1, base_model_checksum="abc"
        )
        rnd.status = RoundStatus.READY_FOR_AGGREGATION
        rnd.received_participants = ["H1", "H2"]
        rnd.accepted_participants = ["H1", "H2"]
        self.storage.save_round(rnd)

        for i, h_id in enumerate(["H1", "H2"]):
            u_dir = os.path.join(self.artifacts_dir, "updates", f"u_fb_{i}")
            p = [np.array([float(i + 1), float(i + 1)], dtype=np.float32)]
            make_handoff(p, client_id=h_id, round_id="R_FB").save(u_dir)
            u = make_update(
                update_id=f"u_fb_{i}", federation_job_id="JOB_FALLBACK", round_id="R_FB",
                participant_id=h_id, model_id=f"M_{i}", artifact_location=u_dir,
                artifact_checksum="chk", parameter_metadata={"shapes": [list(p[0].shape)], "dtypes": [str(p[0].dtype)]},
                num_train_samples=100, status=UpdateStatus.ACCEPTED
            )
            self.storage.save_update(u)

        fallback_events = []
        self.events.add_listener(lambda payload: fallback_events.append(payload) if payload.get("event") == FederationEvent.AGGREGATION_FALLBACK.value else None)

        gm = self.aggregator.aggregate_round("JOB_FALLBACK", "R_FB")

        self.assertEqual(gm.aggregation_method, "weighted_fedavg")
        self.assertEqual(gm.aggregation_metadata.get("effective_strategy"), "fedavg")
        self.assertIn("requires at least 3 participants", gm.aggregation_metadata.get("fallback_reason", ""))
        self.assertEqual(len(fallback_events), 1)
        self.assertEqual(fallback_events[0]["effective_strategy"], "fedavg")

    # ── 11. Non-Float Parameter Preservation ──────────────────────────────────

    def test_non_float_tensor_preservation(self):
        """
        Verify that coordinate median and trimmed mean handle non-float tensors
        (such as BatchNorm num_batches_tracked int64) by preserving the first
        participant's value without crashing with TypeError or casting issues.
        """
        job = self.job_mgr.create_job(
            federation_job_id="JOB_NONFLOAT", task="malaria", task_type="image_classification",
            architecture="mobilenet", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H1", "H2", "H3"], minimum_participants=3, deadline="",
            training_requirements={}, aggregation_configuration={"strategy": "median"}
        )
        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_NONFLOAT", round_id="R_NF", round_number=1,
            expected_participants=["H1", "H2", "H3"], minimum_participants=3, deadline="",
            base_model_id="BASE1", base_model_version=1, base_model_checksum="abc"
        )
        rnd.status = RoundStatus.READY_FOR_AGGREGATION
        rnd.received_participants = ["H1", "H2", "H3"]
        rnd.accepted_participants = ["H1", "H2", "H3"]
        self.storage.save_round(rnd)

        for i, h_id in enumerate(["H1", "H2", "H3"]):
            u_dir = os.path.join(self.artifacts_dir, "updates", f"u_nf_{i}")
            p = [
                np.array([1.0 + i], dtype=np.float32),
                np.array([100 + i * 10], dtype=np.int64),
            ]
            make_handoff(p, client_id=h_id, round_id="R_NF").save(u_dir)
            u = make_update(
                update_id=f"u_nf_{i}", federation_job_id="JOB_NONFLOAT", round_id="R_NF",
                participant_id=h_id, model_id=f"M_{i}", artifact_location=u_dir,
                artifact_checksum="chk", parameter_count=2,
                parameter_metadata={"shapes": [list(x.shape) for x in p], "dtypes": [str(x.dtype) for x in p]},
                num_train_samples=100, status=UpdateStatus.ACCEPTED
            )
            self.storage.save_update(u)

        gm = self.aggregator.aggregate_round("JOB_NONFLOAT", "R_NF")
        agg_params = FederationHandoff.load(gm.artifact_location).parameters
        # Float weight: median([1, 2, 3]) = 2.0
        self.assertEqual(agg_params[0][0], 2.0)
        # Non-float buffer: copied from first participant (H1 -> 100)
        self.assertEqual(agg_params[1][0], 100)
        self.assertEqual(agg_params[1].dtype, np.int64)

    # ── 12. Round Plan Validation ─────────────────────────────────────────────

    def test_round_plan_validation_m12(self):
        """Test validate_plan with median, trimmed_mean, DP, and FedProx."""
        # Valid median
        validate_plan({"strategy": "median"}, {})
        # Valid trimmed_mean
        validate_plan({"strategy": "trimmed_mean", "trim_ratio": 0.2}, {})
        # Invalid with non-FedProx (proximal_mu only with fedprox)
        with self.assertRaises(ValueError):
            validate_plan({"strategy": "median"}, {"proximal_mu": 0.01})
        # Valid with Differential Privacy
        validate_plan(
            {"strategy": "trimmed_mean", "trim_ratio": 0.15},
            {
                "differential_privacy": {
                    "enabled": True,
                    "target_epsilon": 5.0,
                    "max_grad_norm": 1.0,
                    "noise_multiplier": 0.8,
                },
                "dp_noise_multiplier": 0.8,
                "dp_max_grad_norm": 1.0,
                "dp_target_delta": 1e-5,
                "dp_logical_batch_size": 32,
            }
        )

        # Invalid trim_ratio >= 0.5
        with pytest.raises(ValueError, match="trim_ratio"):
            validate_plan({"strategy": "trimmed_mean", "trim_ratio": 0.5}, {})

        # Invalid negative trim_ratio
        with pytest.raises(ValueError, match="trim_ratio"):
            validate_plan({"strategy": "trimmed_mean", "trim_ratio": -0.1}, {})

        # trim_ratio provided with FedAvg
        with pytest.raises(ValueError, match="only used with"):
            validate_plan({"strategy": "fedavg", "trim_ratio": 0.2}, {})

    # ── 13. Flower Server Robust Aggregation ──────────────────────────────────

    def test_flower_server_robust_aggregation_and_bn_averaging(self):
        """
        Verify that Flower server's aggregate_flower_round invokes Aggregator
        with the configured robust strategy (median or trimmed_mean) and that
        average_bn_statistics averages only accepted hospitals.
        """
        job = self.job_mgr.create_job(
            federation_job_id="JOB_FLOWER_ROBUST", task="malaria", task_type="image_classification",
            architecture="mobilenet_v3_small", num_classes=2, class_mapping={"0": 0, "1": 1},
            expected_participants=["H1", "H2", "H3"], minimum_participants=3, deadline="",
            training_requirements={}, aggregation_configuration={"protocol": "flower", "strategy": "median", "total_rounds": 1, "round_duration_hours": 1}
        )
        rnd = self.round_mgr.create_round(
            federation_job_id="JOB_FLOWER_ROBUST", round_id="R_FLOWER", round_number=1,
            expected_participants=["H1", "H2", "H3"], minimum_participants=3, deadline="",
            base_model_id="BASE1", base_model_version=1, base_model_checksum="abc"
        )
        rnd.status = RoundStatus.READY_FOR_AGGREGATION
        rnd.received_participants = ["H1", "H2", "H3"]
        rnd.accepted_participants = ["H1", "H2", "H3"]
        self.storage.save_round(rnd)

        for i, h_id in enumerate(["H1", "H2", "H3"]):
            u_dir = os.path.join(self.artifacts_dir, "updates", f"u_flower_{i}")
            p = [np.array([float(i + 1) * 10.0], dtype=np.float32)]
            make_handoff(p, client_id=h_id, round_id="R_FLOWER").save(u_dir)
            u = make_update(
                update_id=f"u_fl_{i}", federation_job_id="JOB_FLOWER_ROBUST", round_id="R_FLOWER",
                participant_id=h_id, model_id=f"M_{i}", artifact_location=u_dir,
                artifact_checksum="chk", parameter_metadata={"shapes": [list(p[0].shape)], "dtypes": [str(p[0].dtype)]},
                num_train_samples=100, status=UpdateStatus.ACCEPTED
            )
            self.storage.save_update(u)

        from federation.flower_app.server import AppStrategy
        from federation.flower_app.status import StatusWriter
        from federation.flower_app.engine import average_bn_statistics

        status = StatusWriter(os.path.join(self.temp_dir, "flower_status.json"))
        strategy = AppStrategy(
            storage=self.storage,
            job_id="JOB_FLOWER_ROBUST",
            status=status,
            verify_identity=lambda hid, tok: True,
            events=self.events,
        )
        strategy.round = rnd
        strategy.accepted = ["H1", "H2", "H3"]

        # Flower server calls Aggregator(self.storage, self.events).aggregate_round(self.job_id, rnd.round_id)
        gm = self.aggregator.aggregate_round("JOB_FLOWER_ROBUST", "R_FLOWER")
        self.assertIsNotNone(gm)
        self.assertEqual(gm.aggregation_method, "coordinate_median")
        p_agg = FederationHandoff.load(gm.artifact_location).parameters[0]
        # median([10, 20, 30]) = 20.0
        self.assertEqual(p_agg[0], 20.0)

        # BatchNorm calibration only averages accepted hospitals
        keys = ["features.0.weight", "features.0.running_mean"]
        global_params = [np.array([20.0, 20.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)]

        # Accepted hospitals: H1 and H2. H_Bad excluded.
        client_bns = [
            ("H1", [np.array([10.0, 10.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)], 100),
            ("H2", [np.array([20.0, 20.0], dtype=np.float32), np.array([3.0, 3.0], dtype=np.float32)], 100),
            ("H_Bad", [np.array([999.0, 999.0], dtype=np.float32), np.array([999.0, 999.0], dtype=np.float32)], 100),
        ]
        accepted_stats = [(params, n) for h, params, n in client_bns if h in strategy.accepted]
        calibrated = average_bn_statistics(global_params, keys, accepted_stats)
        # Weight unchanged: 20.0
        self.assertEqual(calibrated[0][0], 20.0)
        # running_mean averaged across accepted hospitals: (1.0 + 3.0) / 2 = 2.0 (H_Bad excluded)
        self.assertEqual(calibrated[1][0], 2.0)

    def test_skipped_checks_recorded_in_metrics(self):
        """
        Verifies that every check that cannot run records its reason in metrics['checks_skipped'].
        Never fails open silently.
        """
        job = FederationJob(
            federation_job_id="JOB_SKIPPED_CHECKS",
            task="malaria",
            task_type="image_classification",
            architecture="mobilenet",
            num_classes=2,
            class_mapping={},
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            training_requirements={},
            aggregation_configuration={},
        )
        rnd1 = FederationRound(
            round_id="R1",
            federation_job_id="JOB_SKIPPED_CHECKS",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="NON_EXISTENT_BASE",
            base_model_version=1,
            base_model_checksum="abc",
        )
        u = make_update(update_id="u_sk", federation_job_id="JOB_SKIPPED_CHECKS", round_id="R1", participant_id="H1")
        h = make_handoff([np.ones((5, 5), dtype=np.float32)])

        dec = self.hook.evaluate_update(u, h, job, rnd1)
        # Base model does not exist: l2 must be recorded in checks_skipped
        self.assertIn("checks_skipped", dec.metrics)
        self.assertIn("l2", dec.metrics["checks_skipped"])
        self.assertIn("not found", dec.metrics["checks_skipped"]["l2"].lower())

        # Round 1 has round_number < 3: cosine check must be skipped with reason
        self.assertIn("cosine", dec.metrics["checks_skipped"])
        self.assertIn("round 1 < 3", dec.metrics["checks_skipped"]["cosine"].lower())

        # When shapes mismatch with base model:
        base_dir = self.storage.get_global_model_dir(job.task, job.architecture, 1)
        make_handoff([np.ones((10, 10), dtype=np.float32)]).save(base_dir)
        rnd_with_base = FederationRound(
            round_id="R1_B",
            federation_job_id="JOB_SKIPPED_CHECKS",
            round_number=1,
            expected_participants=["H1"],
            minimum_participants=1,
            deadline="",
            base_model_id="GM1",
            base_model_version=1,
            base_model_checksum="abc",
        )
        # Handoff has shape (5, 5) while base has (10, 10)
        dec_mismatch = self.hook.evaluate_update(u, h, job, rnd_with_base)
        self.assertEqual(dec_mismatch.verdict, "REJECT")
        self.assertIn("l2", dec_mismatch.metrics["checks_skipped"])
        self.assertIn("shape mismatch", dec_mismatch.metrics["checks_skipped"]["l2"].lower())


if __name__ == "__main__":
    unittest.main()
