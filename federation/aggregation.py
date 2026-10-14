import os
import uuid
import hashlib
import json
import numpy as np
from typing import List

from hospital_client.training.federation import FederationHandoff
from .models import (
    FederationJob, FederationRound, GlobalModelVersion,
    GlobalModelStatus, UpdateStatus, now_iso, RoundStatus
)
from .storage import FederationStorage
from .events import EventEmitter, FederationEvent

class Aggregator:
    def __init__(self, storage: FederationStorage, event_emitter: EventEmitter):
        self.storage = storage
        self.event_emitter = event_emitter

    def aggregate_round(self, federation_job_id: str, round_id: str) -> GlobalModelVersion:
        job = self.storage.get_job(federation_job_id)
        rnd = self.storage.get_round(round_id)
        
        if not job or not rnd:
            raise ValueError("Job or Round not found.")
            
        if rnd.status not in (RoundStatus.READY_FOR_AGGREGATION, RoundStatus.AGGREGATING):
            raise ValueError(f"Round {round_id} is not ready for aggregation (status: {rnd.status})")
            
        if rnd.status != RoundStatus.AGGREGATING:
            rnd.status = RoundStatus.AGGREGATING
            rnd.aggregation_started_at = now_iso()
            self.storage.save_round(rnd)
            self.event_emitter.emit(FederationEvent.AGGREGATION_STARTED, round_id=round_id)
            
        updates = self.storage.list_updates_for_round(round_id, status=UpdateStatus.ACCEPTED)
        if len(updates) < rnd.minimum_participants:
            rnd.status = RoundStatus.FAILED
            rnd.aggregation_completed_at = now_iso()
            self.storage.save_round(rnd)
            self.event_emitter.emit(FederationEvent.AGGREGATION_FAILED, round_id=round_id, reason="Insufficient accepted updates")
            raise ValueError("Insufficient accepted updates for aggregation.")
            
        # Load handoffs
        handoffs = []
        for u in updates:
            handoff = FederationHandoff.load(u.artifact_location)
            handoffs.append((u, handoff))
            
        # 1. Verification
        ref_handoff = handoffs[0][1]
        for u, h in handoffs[1:]:
            if h.parameter_count != ref_handoff.parameter_count:
                raise ValueError("Mismatch in parameter count during aggregation.")
            if h.parameter_shapes != ref_handoff.parameter_shapes:
                raise ValueError("Mismatch in parameter shapes during aggregation.")
            if h.parameter_dtypes != ref_handoff.parameter_dtypes:
                raise ValueError("Mismatch in parameter dtypes during aggregation.")
                
        # 2. Weighting Strategy: Sample-size weighting
        total_samples = sum(h.num_train_samples for _, h in handoffs)
        
        aggregated_parameters = []
        
        # 3. Aggregation execution
        # Iterate over parameters
        for param_idx in range(ref_handoff.parameter_count):
            dtype_str = ref_handoff.parameter_dtypes[param_idx]
            is_float = dtype_str.startswith("float")
            
            if is_float:
                # Weighted average for float parameters
                avg_param = np.zeros(ref_handoff.parameter_shapes[param_idx], dtype=dtype_str)
                for _, h in handoffs:
                    weight = h.num_train_samples / total_samples
                    param_val = h.parameters[param_idx]
                    
                    if np.isnan(param_val).any() or np.isinf(param_val).any():
                        raise ValueError(f"NaN or Inf found in parameter {param_idx} from client {h.client_id or 'unknown'}")
                        
                    avg_param += weight * param_val
                aggregated_parameters.append(avg_param)
            else:
                # Non-float state explicit policy:
                # For non-float PyTorch parameters (like num_batches_tracked in BatchNorm), 
                # we copy the value from the first participant. Blindly averaging integer counters is not mathematically sound.
                # Documented policy: "First-participant copy for non-float state".
                first_param = handoffs[0][1].parameters[param_idx]
                aggregated_parameters.append(np.copy(first_param))
                
        # 4. Generate Global Model Version
        canonical_str = json.dumps(job.class_mapping, sort_keys=True)
        schema_hash = hashlib.sha256(f"{job.task}|{job.architecture}|{canonical_str}".encode('utf-8')).hexdigest()
        latest_version = self.storage.get_latest_global_model_version(job.task, job.architecture, schema_hash)
        new_version = latest_version + 1
        
        global_model_id = f"GM-{job.task}-{job.architecture}-V{new_version}".upper()
        
        # Compute checksum
        hasher = hashlib.sha256()
        for array in aggregated_parameters:
            hasher.update(np.ascontiguousarray(array).tobytes())
        checksum = hasher.hexdigest()
        
        artifact_dir = self.storage.get_global_model_dir(job.task, job.architecture, new_version)
        
        # Save aggregated parameters in the same FederationHandoff style (npz + json)
        # We reuse the FederationHandoff abstraction to write the global model to disk
        global_handoff = FederationHandoff(
            protocol_version=ref_handoff.protocol_version,
            model_id=global_model_id,
            model_version=new_version,
            architecture=job.architecture,
            base_model_id=rnd.base_model_id,
            base_model_version=rnd.base_model_version,
            base_model_checksum=rnd.base_model_checksum,
            parameters=aggregated_parameters,
            parameter_count=len(aggregated_parameters),
            parameter_shapes=ref_handoff.parameter_shapes,
            parameter_dtypes=ref_handoff.parameter_dtypes,
            parameters_checksum=checksum,
            num_train_samples=total_samples,
            num_classes=job.num_classes,
            class_mapping=job.class_mapping,
            training_metrics={},
            validation_metrics={},
            training_configuration={},
            device="cpu",
            precision="fp32",
            status="GLOBAL_MODEL_CREATED"
        )
        global_handoff.save(artifact_dir)
        
        # Calculate size
        npz_size = os.path.getsize(os.path.join(artifact_dir, "handoff_parameters.npz"))
        
        gm = GlobalModelVersion(
            global_model_id=global_model_id,
            task=job.task,
            task_type=job.task_type,
            architecture=job.architecture,
            version=new_version,
            artifact_location=artifact_dir,
            artifact_checksum=checksum,
            artifact_size=npz_size,
            parameter_count=len(aggregated_parameters),
            parameter_metadata={"shapes": ref_handoff.parameter_shapes, "dtypes": ref_handoff.parameter_dtypes},
            federation_job_id=job.federation_job_id,
            round_id=round_id,
            source_update_ids=[u.update_id for u, _ in handoffs],
            aggregation_method="weighted_fedavg",
            # FedProx changes only the hospitals' local objective; the server average is the same.
            aggregation_metadata={
                "total_samples": total_samples, "non_float_policy": "first_participant",
                "strategy": (job.aggregation_configuration or {}).get("strategy", "fedavg"),
                "proximal_mu": (job.training_requirements or {}).get("proximal_mu"),
            },
            class_mapping=job.class_mapping,
            schema_hash=schema_hash,
            status=GlobalModelStatus.EVALUATION_PENDING
        )
        self.storage.save_global_model(gm)
        
        # Update round and updates
        rnd.status = RoundStatus.GLOBAL_MODEL_CREATED
        rnd.aggregation_completed_at = now_iso()
        self.storage.save_round(rnd)
        
        for u, _ in handoffs:
            u.status = UpdateStatus.USED_IN_AGGREGATION
            u.used_in_aggregation_at = now_iso()
            self.storage.save_update(u)
            
        self.event_emitter.emit(FederationEvent.AGGREGATION_COMPLETED, round_id=round_id, global_model_id=global_model_id)
        self.event_emitter.emit(FederationEvent.GLOBAL_MODEL_CREATED, global_model_id=global_model_id)
        self.event_emitter.emit(FederationEvent.EVALUATION_READY, global_model_id=global_model_id, architecture=job.architecture, task=job.task)
        
        return gm
