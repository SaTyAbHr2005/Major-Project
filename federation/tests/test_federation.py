import json
import os
import shutil
import unittest
import numpy as np
import tempfile
import sqlite3
from unittest.mock import patch
from PIL import Image

from hospital_client.model_management.registry import build_model
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import get_parameters
from hospital_client.training.federation import build_federation_handoff

from federation.storage import FederationStorage
from federation.jobs import JobManager
from federation.rounds import RoundManager
from federation.intake import IntakeManager
from federation.validation import UpdateValidator
from federation.security_hooks import M12SecurityHook
from federation.events import EventEmitter, FederationEvent
from federation.aggregation import Aggregator
from federation.evaluation import EvaluationManager
from federation.promotion import PromotionManager, PromotionPolicy
from federation.models import GlobalModelStatus

class TestSecureFederation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_metadata.db")
        self.artifacts_dir = os.path.join(self.temp_dir, "test_artifacts")
        self.eval_dir = os.path.join(self.temp_dir, "test_eval_datasets")
        
        self.storage = FederationStorage(self.db_path, self.artifacts_dir)
        self.events = EventEmitter()
        
        # Managers
        self.job_mgr = JobManager(self.storage)
        self.round_mgr = RoundManager(self.storage)
        self.intake_mgr = IntakeManager(self.storage, UpdateValidator(self.storage), M12SecurityHook(), self.events)
        self.agg_mgr = Aggregator(self.storage, self.events)
        self.eval_mgr = EvaluationManager(self.storage, self.events, self.eval_dir)
        self.promo_mgr = PromotionManager(self.storage, self.events, PromotionPolicy(accuracy_threshold=0.0, require_improvement=True))

        # Setup evaluation dataset
        os.makedirs(os.path.join(self.eval_dir, "cancer", "v1", "0"))
        os.makedirs(os.path.join(self.eval_dir, "cancer", "v1", "1"))
        
        # Create dummy images
        img = Image.new('RGB', (224, 224), color = 'red')
        img.save(os.path.join(self.eval_dir, "cancer", "v1", "0", "test1.jpg"))
        img.save(os.path.join(self.eval_dir, "cancer", "v1", "1", "test2.jpg"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_federation(self):
        # 1. Create Job
        job_id = "FED-CANCER-RESNET18"
        class_map = {"0": 0, "1": 1}
        job = self.job_mgr.create_job(
            federation_job_id=job_id,
            task="cancer",
            task_type="image_classification",
            architecture="resnet18",
            num_classes=2,
            class_mapping=class_map,
            expected_participants=["HospA", "HospB", "HospC"],
            minimum_participants=2,
            deadline="",
            training_requirements={"epochs": 1},
            aggregation_configuration={}
        )
        self.assertEqual(job.federation_job_id, job_id)
        
        from datetime import datetime, timedelta, timezone
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        
        # 2. Create Round
        round_id = "R01"
        
        # First we need the base model checksum for the round!
        # Just create a dummy model to get the initial params checksum
        dummy = build_model(ModelConfig(architecture="resnet18", num_classes=2, input_size=(224, 224), color_mode="RGB"))
        from hospital_client.training.federation import _compute_parameters_checksum
        base_chk = _compute_parameters_checksum(get_parameters(dummy))
        
        rnd = self.round_mgr.create_round(
            federation_job_id=job_id,
            round_id=round_id,
            round_number=1,
            expected_participants=["HospA", "HospB", "HospC"],
            minimum_participants=2,
            deadline=future,
            base_model_id="BASE-RESNET18",
            base_model_version=1,
            base_model_checksum=base_chk
        )
        self.round_mgr.open_round(round_id)
        
        # 3. Simulate 3 hospitals training models and generating handoffs
        config = ModelConfig(architecture="resnet18", num_classes=2, input_size=(224, 224), color_mode="RGB")
        
        for participant in ["HospA", "HospB", "HospC"]:
            model = build_model(config)
            params = get_parameters(model)
            
            from hospital_client.training.federation import _compute_parameters_checksum
            
            handoff = build_federation_handoff(
                parameters=params,
                model_id=f"LOCAL-{participant}",
                model_version=1,
                architecture="resnet18",
                base_model_id="BASE-RESNET18",
                base_model_version=1,
                base_model_checksum=base_chk,
                num_train_samples=100,
                class_mapping=class_map,
                training_metrics={},
                validation_metrics={},
                training_configuration={"epochs": 1},
                device="cpu",
                precision="fp32",
                status="TRAINING_COMPLETED_AWAITING_FEDERATION"
            )
            
            handoff_dir = os.path.join(self.temp_dir, f"handoff_{participant}")
            handoff.save(handoff_dir)
            
            # Submit
            update = self.intake_mgr.receive_handoff(job_id, round_id, participant, handoff_dir)
            self.assertEqual(update.status.value, "ACCEPTED", f"Failed with reason: {update.rejection_reason}")

        # 4. Aggregate
        # Fast-forward time to simulate deadline passed
        rnd_record = self.round_mgr.storage.get_round(round_id)
        rnd_record.deadline = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.round_mgr.storage.save_round(rnd_record)
        
        self.round_mgr.check_deadline_and_participants(round_id)
        self.round_mgr.start_aggregation(round_id)
        gm = self.agg_mgr.aggregate_round(job_id, round_id)
        
        self.assertEqual(gm.status.value, "EVALUATION_PENDING")
        self.assertEqual(gm.version, 1)

        # 5. Evaluate
        eval_record = self.eval_mgr.evaluate(gm.global_model_id, dataset_version="v1")
        self.assertEqual(eval_record.status, "COMPLETED")
        self.assertTrue(0.0 <= eval_record.accuracy <= 1.0)
        # The admin screen polls this file while the evaluation runs.
        with open(self.eval_mgr.progress_path(gm.global_model_id)) as f:
            progress = json.load(f)
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["done"], eval_record.sample_count)
        self.assertEqual(progress["total"], eval_record.sample_count)

        gm_updated = self.storage.get_global_model(gm.global_model_id)
        self.assertEqual(gm_updated.status.value, "EVALUATED")

        # 6. Promote v1 (Should become MAIN since there's no previous MAIN)
        promo_decision = self.promo_mgr.evaluate_promotion(gm.global_model_id, eval_record.evaluation_id)
        self.assertEqual(promo_decision.decision.value, "PROMOTED")
        
        gm_final = self.storage.get_global_model(gm.global_model_id)
        self.assertEqual(gm_final.status.value, "MAIN")
        
        # 7. Second round (v2)
        round_id_2 = "R02"
        self.round_mgr.create_round(
            federation_job_id=job_id,
            round_id=round_id_2,
            round_number=2,
            expected_participants=["HospA", "HospB", "HospC"],
            minimum_participants=2,
            deadline=future,
            base_model_id=gm_final.global_model_id,
            base_model_version=gm_final.version,
            base_model_checksum=gm_final.artifact_checksum
        )
        self.round_mgr.open_round(round_id_2)
        
        for participant in ["HospA", "HospB", "HospC"]:
            model = build_model(config)
            params = get_parameters(model)
            handoff = build_federation_handoff(
                parameters=params,
                model_id=f"LOCAL-{participant}",
                model_version=2,
                architecture="resnet18",
                base_model_id=gm_final.global_model_id,
                base_model_version=gm_final.version,
                base_model_checksum=gm_final.artifact_checksum,
                num_train_samples=100,
                class_mapping=class_map,
                training_metrics={},
                validation_metrics={},
                training_configuration={"epochs": 1},
                device="cpu",
                precision="fp32",
                status="TRAINING_COMPLETED_AWAITING_FEDERATION"
            )
            handoff_dir = os.path.join(self.temp_dir, f"handoff_{participant}_v2")
            handoff.save(handoff_dir)
            self.intake_mgr.receive_handoff(job_id, round_id_2, participant, handoff_dir)
            
        # Fast-forward time to simulate deadline passed
        rnd_record2 = self.round_mgr.storage.get_round(round_id_2)
        rnd_record2.deadline = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.round_mgr.storage.save_round(rnd_record2)

        self.round_mgr.check_deadline_and_participants(round_id_2)
        self.round_mgr.start_aggregation(round_id_2)
        gm2 = self.agg_mgr.aggregate_round(job_id, round_id_2)
        
        eval_record2 = self.eval_mgr.evaluate(gm2.global_model_id, dataset_version="v1")
        
        # Promote v2 (Will not promote if accuracy isn't strictly better due to our simple policy)
        promo_decision2 = self.promo_mgr.evaluate_promotion(gm2.global_model_id, eval_record2.evaluation_id)
        
        # It might or might not promote depending on random initialization.
        # But we ensure it didn't crash and status is correctly managed.
        gm2_final = self.storage.get_global_model(gm2.global_model_id)
        self.assertIn(gm2_final.status.value, ["MAIN", "EVALUATED_NOT_PROMOTED"])
        
        gm1_final = self.storage.get_global_model(gm.global_model_id)
        if gm2_final.status.value == "MAIN":
            self.assertEqual(gm1_final.status.value, "HISTORICAL")
        else:
            self.assertEqual(gm1_final.status.value, "MAIN")

if __name__ == "__main__":
    unittest.main()
