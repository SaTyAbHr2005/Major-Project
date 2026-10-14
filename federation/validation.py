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
