import os
from typing import Dict, Any, List
import numpy as np

# This is imported from the hospital client if available.
# We will use it for validation.
from hospital_client.training.federation import FederationHandoff

from .models import FederationJob, FederationRound, FederationUpdate
from .storage import FederationStorage

# M1-M3 TrainingRequest dependency removed.
# M9 participant authorization is now based natively on the M9 FederationJob and FederationRound's expected_participants lists.


class UpdateValidator:
    def __init__(self, storage: FederationStorage):
        self.storage = storage

    def validate(self, job: FederationJob, rnd: FederationRound, handoff: FederationHandoff, participant_id: str, current_update_id: str = "") -> None:
        # 0. Check Authorization against M9 Job and Round (Native)
        if not job.expected_participants:
            raise ValueError(f"Job {job.federation_job_id} has no expected participants configured (fails closed).")
        if participant_id not in job.expected_participants:
            raise ValueError(f"Participant {participant_id} is not authorized for job {job.federation_job_id}.")
            
        if not rnd.expected_participants:
            raise ValueError(f"Round {rnd.round_id} has no expected participants configured (fails closed).")
        if participant_id not in rnd.expected_participants:
            raise ValueError(f"Participant {participant_id} is not authorized for round {rnd.round_id}.")
        
        # 1. Check job match
        if handoff.federation_job_id and handoff.federation_job_id != job.federation_job_id:
            raise ValueError(f"Handoff job {handoff.federation_job_id} does not match target job {job.federation_job_id}")
            
        # 2. Check round match
        if handoff.round_id and handoff.round_id != rnd.round_id:
            raise ValueError(f"Handoff round {handoff.round_id} does not match target round {rnd.round_id}")
            
        # 2.5 Check base model provenance — all three fields are independent;
        # each is checked and rejected on its own merit.
        if handoff.base_model_id != rnd.base_model_id:
            raise ValueError(f"Base model ID mismatch: expected {rnd.base_model_id}, got {handoff.base_model_id}")
        if handoff.base_model_version != rnd.base_model_version:
            raise ValueError(f"Base model version mismatch: expected {rnd.base_model_version}, got {handoff.base_model_version}")
        if handoff.base_model_checksum != rnd.base_model_checksum:
            raise ValueError(f"Base model checksum mismatch: expected {rnd.base_model_checksum}, got {handoff.base_model_checksum}")
            
        # 3. Check architecture
        if handoff.architecture != job.architecture:
            raise ValueError(f"Architecture mismatch: job requires {job.architecture}, got {handoff.architecture}")
            
        # 4. Check classes
        if handoff.num_classes != job.num_classes:
            raise ValueError(f"Classes mismatch: job requires {job.num_classes}, got {handoff.num_classes}")
            
        if handoff.class_mapping != job.class_mapping:
            raise ValueError(f"Class mapping mismatch: job requires {job.class_mapping}, got {handoff.class_mapping}")
            
        # 5. Check training status
        if handoff.status != "TRAINING_COMPLETED_AWAITING_FEDERATION":
            raise ValueError(f"Invalid training status: {handoff.status}")
            
        # 6. Duplicate checking
        # Assuming the storage can tell if this participant already submitted for this round
        updates = self.storage.list_updates_for_round(rnd.round_id)
        for u in updates:
            if u.participant_id == participant_id and u.update_id != current_update_id and u.status.value != "REJECTED":
                raise ValueError(f"Duplicate submission: Participant {participant_id} already submitted for round {rnd.round_id}")
                
        # 7. Training configuration check (simplified)
        for key, expected_val in job.training_requirements.items():
            if key in handoff.training_configuration:
                actual_val = handoff.training_configuration[key]
                if actual_val != expected_val:
                    raise ValueError(f"Training requirement mismatch for {key}: expected {expected_val}, got {actual_val}")

        # 8. FedProx: the update must report training with the job's proximal term (a missing value is not enough).
        mu = job.training_requirements.get("proximal_mu")
        if mu and handoff.training_configuration.get("proximal_mu") != mu:
            raise ValueError(f"FedProx job requires proximal_mu={mu}; the update was trained with "
                             f"{handoff.training_configuration.get('proximal_mu')}.")

        # 9. Differential Privacy (Module 10)
        from hospital_client.training.dp import is_dp_job
        if is_dp_job(job):
            # Enforce FP32 for DP jobs
            resolved_precision = handoff.precision or (handoff.training_configuration or {}).get("precision")
            if resolved_precision != "fp32":
                raise ValueError(f"DP job requires precision='fp32', but update used {resolved_precision!r}.")

            reqs = job.training_requirements or {}
            tc = handoff.training_configuration or {}
            if not tc.get("dp_enabled") and "dp_noise_multiplier" not in tc:
                raise ValueError("DP job requires update to be trained with differential privacy enabled.")

            sigma = float(tc.get("dp_noise_multiplier", 0))
            expected_sigma = float(reqs.get("dp_noise_multiplier", 0))
            if abs(sigma - expected_sigma) > 1e-6:
                raise ValueError(f"DP noise multiplier mismatch: expected {expected_sigma}, got {sigma}")

            c = float(tc.get("dp_max_grad_norm", 0))
            expected_c = float(reqs.get("dp_max_grad_norm", 0))
            if abs(c - expected_c) > 1e-6:
                raise ValueError(f"DP max grad norm mismatch: expected {expected_c}, got {c}")

            delta = float(tc.get("dp_target_delta", 0))
            expected_delta = float(reqs.get("dp_target_delta", 0))
            if abs(delta - expected_delta) > 1e-6:
                raise ValueError(f"DP target delta mismatch: expected {expected_delta}, got {delta}")

            q = float(tc.get("dp_sample_rate", 0))
            if q <= 0 or q > 1.0:
                raise ValueError(f"Invalid DP sample rate reported: {q}")

            logical_b = reqs.get("dp_logical_batch_size")
            if logical_b is not None and handoff.num_train_samples > 0:
                expected_q = float(logical_b) / float(handoff.num_train_samples)
                if abs(q - expected_q) > 1e-4:
                    raise ValueError(
                        f"DP sample rate mismatch: expected {expected_q:.6f} "
                        f"({logical_b}/{handoff.num_train_samples}), got {q:.6f}"
                    )

            steps = int(tc.get("dp_steps", 0))
            if steps <= 0:
                raise ValueError(f"Invalid DP steps reported: {steps}")

            reported_eps = float(tc.get("dp_epsilon", 0))
            from hospital_client.training.dp import compute_rdp_epsilon
            recomputed_round_eps = compute_rdp_epsilon([(sigma, q, steps)], delta)
            if abs(reported_eps - recomputed_round_eps) > 1e-3:
                raise ValueError(f"DP epsilon mismatch: reported {reported_eps:.4f}, recomputed {recomputed_round_eps:.4f}")

            # History accounting across all rounds in the database ledger
            history_rows = self.storage.get_dp_ledger(job.federation_job_id, participant_id)
            history = [(float(r["noise_multiplier"]), float(r["sample_rate"]), int(r["steps"])) for r in history_rows]
            combined_history = history + [(sigma, q, steps)]
            cumulative_eps = compute_rdp_epsilon(combined_history, delta)

            target_eps = reqs.get("dp_target_epsilon")
            if target_eps is not None and float(target_eps) > 0:
                if cumulative_eps > float(target_eps) + 1e-4:
                    raise ValueError(
                        f"Privacy budget exceeded: cumulative epsilon {cumulative_eps:.4f} "
                        f"exceeds target cap {float(target_eps):.4f} (delta={delta})"
                    )
