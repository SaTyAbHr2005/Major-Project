"""
The single common training engine for Module 7. Every one of Module 6's 8
architectures runs through this exact loop - architecture choice only
changes which Module 6 factory function builds the model
(hospital_client.model_management.registry.build_model). There is
deliberately no per-architecture training loop.

Resource boundary: this module NEVER inspects hardware to choose device,
batch size, or epochs - it only executes the TrainingConfig it is given
(Module 8's future job is to populate that config).
"""
import os
import time
from typing import Optional

import torch

from hospital_client.model_management.device import resolve_device
from hospital_client.model_management.parameters import get_parameters
from hospital_client.model_management.registry import build_model
from hospital_client.model_management.store import ModelStore

from hospital_client.training.checkpoint import (
    load_training_state,
    restore_rng_state,
    save_training_state,
)
from hospital_client.training.config import TrainingConfig, validate_training_config
from hospital_client.training.dataloader import build_dataloaders
from hospital_client.training.dataset import TASK_TYPE, build_preprocessing_spec
from hospital_client.training.federation import FederationHandoff, build_federation_handoff as _build_federation_handoff
from hospital_client.training.losses import build_loss
from hospital_client.training.metrics import compute_classification_metrics
from hospital_client.training.result import TrainingResult, TrainingStatus


def _models_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "models")


def resolve_precision(precision: str, device: torch.device) -> str:
    """
    Resolves the CONFIGURED precision ("fp32"/"fp16"/"auto") against the
    already-resolved device into the ACTUAL precision to train with
    ("fp32" or "fp16"). Never silently changes device, and "fp16" on a
    non-CUDA device always raises (config validation already rejects this
    for device="cpu"; this is the runtime backstop).
    """
    if precision == "fp32":
        return "fp32"
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError(
                f"precision='fp16' requires a CUDA device, but the resolved device is "
                f"'{device.type}'. Module 7 does not pretend FP16 is available outside CUDA."
            )
        return "fp16"
    if precision == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    raise ValueError(f"Unknown precision {precision!r}")


class Trainer:
    def __init__(self, config: TrainingConfig):
        validate_training_config(config)
        self.config = config

    def run(self) -> TrainingResult:
        config = self.config
        warnings, errors = [], []
        started_at = time.time()

        torch.manual_seed(config.random_seed)
        device = resolve_device(config.device)  # raises clearly if CUDA unavailable - never falls back
        precision = resolve_precision(config.precision, device)

        prepared = build_dataloaders(config)
        warnings.extend(prepared.warnings)

        model = build_model(config.model_config).to(device)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        # GradScaler with enabled=False is a documented no-op passthrough (PyTorch AMP docs),
        # so FP32 training goes through the exact same code path with byte-identical behavior.
        scaler = torch.amp.GradScaler(device=device.type, enabled=(precision == "fp16"))
        loss_fn = build_loss(config.class_weighting, prepared, device)
        model_store = ModelStore(_models_dir(config.output_dir))

        start_epoch = 1
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        checkpoint_model_id: Optional[str] = None
        checkpoint_version: Optional[int] = None

        if config.resume:
            state = load_training_state(config.output_dir, config.model_id)
            self._validate_resume_state(state, config, prepared.class_mapping)
            checkpoint_model_id = state.get("checkpoint_model_id")
            checkpoint_version = state.get("checkpoint_version")
            if checkpoint_model_id is not None and checkpoint_version is not None:
                loaded_model, _ = model_store.load_checkpoint(checkpoint_model_id, checkpoint_version, device=config.device)
                model.load_state_dict(loaded_model.state_dict())
            if state.get("optimizer_state") is not None:
                optimizer.load_state_dict(state["optimizer_state"])
            if scheduler is not None and state.get("scheduler_state") is not None:
                scheduler.load_state_dict(state["scheduler_state"])
            if state.get("scaler_state") is not None:
                scaler.load_state_dict(state["scaler_state"])
            best_metric = state.get("best_metric")
            best_epoch = state.get("best_epoch")
            start_epoch = state["epoch"] + 1
            restore_rng_state(state)

        epochs_run = 0
        train_metrics_dict, val_metrics_dict = {}, {}
        early_stop_counter = 0
        interrupted = False
        failed_error = None

        try:
            for epoch in range(start_epoch, config.epochs + 1):
                train_metrics = self._run_epoch(
                    model, prepared.train_loader, loss_fn, optimizer, device, prepared.class_mapping,
                    train_mode=True, precision=precision, scaler=scaler,
                )
                epochs_run += 1
                train_metrics_dict = train_metrics.to_dict()

                do_validate = prepared.validation_loader is not None and (epoch % config.validate_every == 0)
                if do_validate:
                    val_metrics = self._run_epoch(
                        model, prepared.validation_loader, loss_fn, None, device, prepared.class_mapping,
                        train_mode=False, precision=precision, scaler=scaler,
                    )
                    val_metrics_dict = val_metrics.to_dict()

                    if scheduler is not None and config.scheduler == "reduce_on_plateau":
                        scheduler.step(val_metrics.loss)

                    metric_name = config.early_stopping_metric if config.early_stopping else "val_loss"
                    mode = config.early_stopping_mode if config.early_stopping else "min"
                    current = self._selected_metric(val_metrics, metric_name)

                    if self._is_better(current, best_metric, mode):
                        best_metric, best_epoch, early_stop_counter = current, epoch, 0
                        metadata = model_store.save_checkpoint(
                            model, config.model_config, config.model_id, status="trained",
                            extra_metadata={
                                "epochs": epoch,
                                "optimizer": config.optimizer,
                                "learning_rate": config.learning_rate,
                                "random_seed": config.random_seed,
                                "class_mapping": prepared.class_mapping,
                                "num_classes": len(prepared.class_mapping),
                                "task_type": TASK_TYPE,
                                "preprocessing_spec": build_preprocessing_spec(
                                    config.model_config.color_mode, tuple(config.model_config.input_size),
                                    prepared.train_manifest_metadata.get("image_preprocessing")),
                            },
                        )
                        checkpoint_model_id, checkpoint_version = config.model_id, metadata.version
                    else:
                        early_stop_counter += 1

                if scheduler is not None and config.scheduler in ("step", "cosine"):
                    scheduler.step()  # stepped every epoch, independent of validation cadence

                save_training_state(
                    config.output_dir, config.model_id, epoch, optimizer, scheduler,
                    best_metric, best_epoch, prepared.class_mapping, config.model_config.to_dict(),
                    checkpoint_model_id, checkpoint_version, config.random_seed, scaler=scaler,
                )

                if config.early_stopping and do_validate and early_stop_counter >= config.early_stopping_patience:
                    warnings.append(f"Early stopping triggered at epoch {epoch} (patience={config.early_stopping_patience}).")
                    break

        except KeyboardInterrupt:
            interrupted = True
        except RuntimeError as e:
            if device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                errors.append(f"CUDA out of memory: {e}")
                failed_error = e
            else:
                raise

        duration = time.time() - started_at

        # No validation-driven "best" checkpoint was ever saved (e.g. validation
        # disabled) but training did complete epochs - save the final state so a
        # checkpoint always exists after a completed run.
        if checkpoint_model_id is None and epochs_run > 0 and failed_error is None:
            metadata = model_store.save_checkpoint(
                model, config.model_config, config.model_id, status="trained",
                extra_metadata={
                    "epochs": epochs_run, "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate, "random_seed": config.random_seed,
                    "class_mapping": prepared.class_mapping, "num_classes": len(prepared.class_mapping),
                    "task_type": TASK_TYPE,
                    "preprocessing_spec": build_preprocessing_spec(
                        config.model_config.color_mode, tuple(config.model_config.input_size),
                        prepared.train_manifest_metadata.get("image_preprocessing")),
                },
            )
            checkpoint_model_id, checkpoint_version, best_epoch = config.model_id, metadata.version, epochs_run

        test_metrics_dict = None
        if prepared.test_loader is not None and failed_error is None and not interrupted:
            test_metrics = self._run_epoch(
                model, prepared.test_loader, loss_fn, None, device, prepared.class_mapping,
                train_mode=False, precision=precision, scaler=scaler,
            )
            test_metrics_dict = test_metrics.to_dict()

        if failed_error is not None:
            status = TrainingStatus.TRAINING_FAILED
        elif interrupted:
            status = TrainingStatus.TRAINING_INTERRUPTED
        elif checkpoint_model_id is not None:
            status = TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
        else:
            status = TrainingStatus.TRAINING_COMPLETED

        return TrainingResult(
            status=status,
            model_id=config.model_id,
            architecture=config.model_config.architecture,
            checkpoint_model_id=checkpoint_model_id,
            checkpoint_version=checkpoint_version,
            best_epoch=best_epoch,
            epochs_run=epochs_run,
            training_duration_seconds=duration,
            training_metrics=train_metrics_dict,
            validation_metrics=val_metrics_dict,
            test_metrics=test_metrics_dict,
            class_mapping=prepared.class_mapping,
            training_configuration=config.to_dict(),
            random_seed=config.random_seed,
            resolved_precision=precision,
            warnings=warnings,
            errors=errors,
        )

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        config = self.config
        if config.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if config.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
            )
        raise ValueError(f"Unknown optimizer {config.optimizer!r}")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        config = self.config
        name = config.scheduler
        if name in (None, "none"):
            return None
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=config.scheduler_patience, factor=config.scheduler_factor
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
            )
        if name == "cosine":
            t_max = config.scheduler_t_max or config.epochs
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.scheduler_eta_min)
        raise ValueError(f"Unknown scheduler {name!r}")

    def _run_epoch(self, model, loader, loss_fn, optimizer, device, class_mapping, train_mode: bool, precision: str, scaler):
        model.train(train_mode)
        total_loss, n_samples = 0.0, 0
        all_true, all_pred = [], []
        autocast_enabled = precision == "fp16"
        with torch.set_grad_enabled(train_mode):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                if train_mode:
                    optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                if train_mode:
                    # scaler with enabled=False (FP32) is a documented no-op passthrough:
                    # scale()/step()/update() behave exactly like plain backward()/step().
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                n_samples += batch_size
                preds = outputs.argmax(dim=1)
                all_true.extend(labels.detach().cpu().tolist())
                all_pred.extend(preds.detach().cpu().tolist())
        avg_loss = total_loss / n_samples if n_samples else 0.0
        return compute_classification_metrics(all_true, all_pred, avg_loss, class_mapping)

    def _selected_metric(self, metrics, metric_name: str) -> float:
        return {"val_loss": metrics.loss, "val_accuracy": metrics.accuracy, "val_f1": metrics.f1}[metric_name]

    def _is_better(self, current: float, best: Optional[float], mode: str) -> bool:
        if best is None:
            return True
        return current < best if mode == "min" else current > best

    def _validate_resume_state(self, state, config: TrainingConfig, class_mapping) -> None:
        if state["model_config"]["architecture"] != config.model_config.architecture:
            raise ValueError(
                f"Cannot resume: saved state architecture "
                f"'{state['model_config']['architecture']}' does not match requested "
                f"'{config.model_config.architecture}'."
            )
        if state["class_mapping"] != class_mapping:
            raise ValueError("Cannot resume: saved class_mapping does not match the current dataset's class mapping.")


def get_model(model_id: str, version: int, output_dir: str, device: str = "cpu"):
    """Thin wrapper over Module 6's load - Module 7 never reimplements artifact loading."""
    store = ModelStore(_models_dir(output_dir))
    return store.load_checkpoint(model_id, version, device=device)


def build_federation_handoff(
    result: TrainingResult,
    output_dir: str,
    round_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> FederationHandoff:
    """
    Builds the versioned Module 7 -> Module 9 contract (see federation.py)
    from a completed TrainingResult. round_id/client_id are only ever the
    caller-supplied real values from a future federation/identity
    integration - Module 7 has no such context and never fabricates them.
    """
    if not result.ready_for_federation or result.checkpoint_model_id is None or result.checkpoint_version is None:
        raise ValueError(
            "Cannot build a federation handoff: training did not reach "
            "TRAINING_COMPLETED_AWAITING_FEDERATION with a valid checkpoint."
        )
    model, _ = get_model(result.checkpoint_model_id, result.checkpoint_version, output_dir, device="cpu")
    parameters = get_parameters(model)
    return _build_federation_handoff(
        parameters=parameters,
        model_id=result.checkpoint_model_id,
        model_version=result.checkpoint_version,
        architecture=result.architecture,
        num_train_samples=result.training_metrics.get("sample_count", 0),
        class_mapping=result.class_mapping,
        training_metrics=result.training_metrics,
        validation_metrics=result.validation_metrics,
        training_configuration=result.training_configuration,
        device=result.training_configuration.get("device", "cpu"),
        precision=result.resolved_precision,
        status=result.status.value,
        round_id=round_id,
        client_id=client_id,
    )
