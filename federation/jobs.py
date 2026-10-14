from typing import Dict, List, Any, Optional
from .models import FederationJob, JobStatus, now_iso
from .storage import FederationStorage
from .round_plan import validate_plan

class JobManager:
    def __init__(self, storage: FederationStorage):
        self.storage = storage

    def create_job(self, 
                   federation_job_id: str,
                   task: str,
                   task_type: str,
                   architecture: str,
                   num_classes: int,
                   class_mapping: Dict[str, int],
                   expected_participants: List[str],
                   minimum_participants: int,
                   deadline: str,
                   training_requirements: Dict[str, Any],
                   aggregation_configuration: Dict[str, Any]) -> FederationJob:
        
        if self.storage.get_job(federation_job_id) is not None:
            raise ValueError(f"Job {federation_job_id} already exists.")
        validate_plan(aggregation_configuration, training_requirements)
            
        job = FederationJob(
            federation_job_id=federation_job_id,
            task=task,
            task_type=task_type,
            architecture=architecture,
            num_classes=num_classes,
            class_mapping=class_mapping,
            expected_participants=expected_participants,
            minimum_participants=minimum_participants,
            deadline=deadline,
            training_requirements=training_requirements,
            aggregation_configuration=aggregation_configuration,
            status=JobStatus.CREATED
        )
        self.storage.save_job(job)
        return job

    def get_job(self, federation_job_id: str) -> Optional[FederationJob]:
        return self.storage.get_job(federation_job_id)

