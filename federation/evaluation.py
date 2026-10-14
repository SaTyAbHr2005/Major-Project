import os
import uuid
from typing import Dict, Any, List
import json
import glob
import time

import torch

from hospital_client.model_management.registry import build_model, get_architecture_info
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.parameters import set_parameters
from hospital_client.model_management.store import ModelStore
from hospital_client.inference.engine import LocalInferenceEngine, InferenceConfig
from hospital_client.inference.provisioning import LocalStoreModelProvider
from hospital_client.training.federation import FederationHandoff

from .models import GlobalModelVersion, GlobalModelStatus, EvaluationRecord, now_iso
from .storage import FederationStorage
from .events import EventEmitter, FederationEvent

class EvaluationManager:
    def __init__(self, storage: FederationStorage, event_emitter: EventEmitter, eval_datasets_dir: str = "evaluation_datasets"):
        self.storage = storage
        self.event_emitter = event_emitter
        self.eval_datasets_dir = eval_datasets_dir

    def evaluate(self, global_model_id: str, dataset_version: str = "v1") -> EvaluationRecord:
        gm = self.storage.get_global_model(global_model_id)
        if not gm:
            raise ValueError(f"Global model {global_model_id} not found.")
            
        if gm.status not in (GlobalModelStatus.EVALUATION_PENDING, GlobalModelStatus.EVALUATED_NOT_PROMOTED, GlobalModelStatus.MAIN, GlobalModelStatus.HISTORICAL):
            raise ValueError(f"Cannot evaluate model in status {gm.status}")
            
        # 1. Load the global model handoff (npz)
        handoff = FederationHandoff.load(gm.artifact_location)
        
        # 2. Convert to a Module 6 Artifact for M16 Engine
        arch_info = get_architecture_info(gm.architecture)
        
        config = ModelConfig(
            architecture=gm.architecture,
            num_classes=gm.parameter_metadata.get("num_classes", handoff.num_classes),
            input_size=arch_info.default_input_size,
            color_mode="RGB"
        )
        
        model = build_model(config)
        set_parameters(model, handoff.parameters)
        
        # Save to a temporary Admin Module 6 store
        m6_store_dir = os.path.join(self.storage.artifacts_dir, "admin_m6_store")
        store = ModelStore(m6_store_dir)
        
        # Determine unique M6 model id based on GM id
        m6_model_id = gm.global_model_id
        
        # Delete if exists to avoid conflicts in this eval
        try:
            store.delete_version(m6_model_id, gm.version)
        except FileNotFoundError:
            pass
            
        store.save_checkpoint(
            model=model,
            config=config,
            model_id=m6_model_id,
            status="available",
            version=gm.version,
            extra_metadata={"class_mapping": handoff.class_mapping}
        )
        
        # 3. Use M16 LocalInferenceEngine
        provider = LocalStoreModelProvider(m6_store_dir, m6_model_id, gm.version)
        inf_config = InferenceConfig(device="cpu", deterministic=True) # Run eval on CPU by default for stability
        
        engine = LocalInferenceEngine(provider, inf_config)
        
        # 4. Iterate evaluation dataset
        dataset_path = os.path.join(self.eval_datasets_dir, gm.task, dataset_version)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Evaluation dataset missing at {dataset_path}")
            
        y_true = []
        y_pred = []
        class_names = []
        failed = []   # (image, reason) for images the engine could not process
        
        # Expected structure: evaluation_datasets/{task}/{version}/{class_name}/img.jpg
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        for root, dirs, files in os.walk(dataset_path):
            class_name = os.path.basename(root)
            if class_name == dataset_version or not files:
                continue
            
            if class_name not in class_names:
                class_names.append(class_name)
                
            for file in files:
                if file.lower().endswith(image_extensions):
                    img_path = os.path.join(root, file)
                    result = engine.predict(img_path)
                    
                    if result.status.value == "COMPLETED":
                        pred = result.output.predicted_label
                        y_pred.append(pred)
                        y_true.append(class_name)
                    else:
                        failed.append((os.path.relpath(img_path, dataset_path), result.error_category or result.status.value))

        engine.close()

        # Metrics over only the images that worked would overstate (or understate) the model on the
        # evaluation set, so any failed image stops the evaluation and is reported.
        if failed:
            examples = "; ".join(f"{name} ({reason})" for name, reason in failed[:5])
            raise ValueError(
                f"Evaluation stopped: {len(failed)} of {len(failed) + len(y_true)} evaluation images could not be "
                f"processed, so no metrics were recorded. Fix or remove them from {dataset_path}. "
                f"First failures: {examples}")
        
        # 5. Calculate Metrics
        sample_count = len(y_true)
        if sample_count == 0:
            raise ValueError("No valid images evaluated.")
            
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if str(yt) == str(yp))
        accuracy = correct / sample_count
        
        # Very simple precision/recall/f1 for macro average (assuming classes are strings)
        classes_unique = list(set(y_true))
        per_class_metrics = {}
        confusion_matrix = [[0 for _ in classes_unique] for _ in classes_unique]
        
        total_precision = 0.0
        total_recall = 0.0
        
        for idx_t, cls_t in enumerate(classes_unique):
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_t and yp == cls_t)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls_t and yp == cls_t)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_t and yp != cls_t)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            per_class_metrics[cls_t] = {"precision": p, "recall": r, "f1": f, "support": tp+fn}
            total_precision += p
            total_recall += r
            
            for idx_p, cls_p in enumerate(classes_unique):
                confusion_matrix[idx_t][idx_p] = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls_t and yp == cls_p)
                
        macro_precision = total_precision / len(classes_unique)
        macro_recall = total_recall / len(classes_unique)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0

        eval_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"
        
        # 6. Save outputs
        eval_dir = os.path.join(gm.artifact_location, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        
        metrics = {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        record = EvaluationRecord(
            evaluation_id=eval_id,
            global_model_id=gm.global_model_id,
            model_version=gm.version,
            task=gm.task,
            architecture=gm.architecture,
            evaluation_dataset_id=f"EVAL-{gm.task.upper()}",
            evaluation_dataset_version=dataset_version,
            sample_count=sample_count,
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1=macro_f1,
            per_class_metrics=per_class_metrics,
            confusion_matrix=confusion_matrix,
            predictions_location=os.path.join(eval_dir, "predictions.json"),
            metrics_location=os.path.join(eval_dir, "metrics.json"),
            inference_configuration={"device": "cpu"},
            status="COMPLETED",
            completed_at=now_iso()
        )
        self.storage.save_evaluation_record(record)
        
        if gm.status == GlobalModelStatus.EVALUATION_PENDING:
            gm.status = GlobalModelStatus.EVALUATED
            self.storage.save_global_model(gm)
            
        self.event_emitter.emit(FederationEvent.EVALUATION_COMPLETED, global_model_id=gm.global_model_id, evaluation_id=eval_id)
        
        return record
