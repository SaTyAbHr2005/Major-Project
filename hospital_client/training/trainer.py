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
import math
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
from hospital_client.training.federation import FederationHandoff, build_federation_handoff as _build_federation_handoff, _compute_parameters_checksum
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
        # Filled by run() when a test set is evaluated: (y_true, class probabilities,
        # class_order) from the SAME pass that produced result.test_metrics. Used by
        # training.plots for ROC/PR curves; never written into TrainingResult.
        self.test_outputs = None
        # FedProx anchor: the weights training started from (set in run() when proximal_mu > 0).
        self._prox_anchor = None

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

        # ── Canonical base model loading + verification (federation Round N) ──
        # When a FederationRound prescribes a specific base model, the hospital
        # must start from that exact artifact — not from a freshly initialised model.
        # We load it, compute its parameter checksum, and compare against the
        # round-prescribed checksum before a single training step executes.
        if config.canonical_base_model_id is not None:
            if config.canonical_base_model_version is None or config.canonical_base_model_checksum is None:
                raise ValueError(
                    "canonical_base_model_id is set but canonical_base_model_version or "
                    "canonical_base_model_checksum is missing. All three fields are required together."
                )
            model_store_base = ModelStore(_models_dir(config.output_dir))
            try:
                loaded_base, _ = model_store_base.load_checkpoint(
                    config.canonical_base_model_id,
                    config.canonical_base_model_version,
                    device=config.device,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"Canonical base model '{config.canonical_base_model_id}' v{config.canonical_base_model_version} "
                    f"is not available in the local model store. "
                    f"Download it from the federation server before training. Detail: {exc}"
                ) from exc
            # Cryptographic verification: compute SHA-256 of the stored parameters
            # and compare against the round-prescribed checksum.
            stored_params = get_parameters(loaded_base)
            stored_checksum = _compute_parameters_checksum(stored_params)
            if stored_checksum != config.canonical_base_model_checksum:
                raise ValueError(
                    f"Canonical base model '{config.canonical_base_model_id}' v{config.canonical_base_model_version} "
                    f"failed checksum verification.\n"
                    f"  Expected (from round): {config.canonical_base_model_checksum}\n"
                    f"  Computed (from store): {stored_checksum}\n"
                    "Training refused: do not train from an unverified base model."
                )
            # Load the verified weights into the model that will be trained.
            model.load_state_dict(loaded_base.state_dict())
            del loaded_base, stored_params  # free memory before training starts
        # ── End canonical base model section ──
        # FedProx: remember the starting (round base) weights before any resume or training step;
        # _run_epoch adds (mu/2)*||w - anchor||^2 to the training objective.
        if config.proximal_mu > 0:
            self._prox_anchor = [p.detach().clone() for p in model.parameters()]
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        # GradScaler with enabled=False is a documented no-op passthrough (PyTorch AMP docs),
        # so FP32 training goes through the exact same code path with byte-identical behavior.
        scaler = torch.amp.GradScaler(device=device.type, enabled=(precision == "fp16"))
        loss_fn = build_loss(config.class_weighting, prepared, device)
        model_store = ModelStore(_models_dir(config.output_dir))

        # ── Differential Privacy (Module 10) ──────────────────────────────────
        dp_engine = None
        dp_sample_rate = None
        dp_existing_history = []
        dp_state_path = None
        dp_state = None
        if config.dp_enabled:
            from hospital_client.training.dp import (
                load_dp_state,
                project_and_check_budget,
                prepare_dp_training,
            )
            dataset_size = len(prepared.train_loader.dataset)
            logical_batch_size = config.dp_logical_batch_size or config.batch_size
            dp_sample_rate = logical_batch_size / max(dataset_size, 1)
            planned_steps = int(math.ceil(dataset_size / logical_batch_size)) * config.epochs
            dp_state_path = config.dp_state_path or os.path.join(config.output_dir, f"dp_state_{config.model_id}.json")
            dp_state = load_dp_state(dp_state_path, config.model_id)
            dp_existing_history = [
                (float(entry["noise_multiplier"]), float(entry["sample_rate"]), int(entry["steps"]))
                for entry in dp_state.get("history", [])
            ]
            project_and_check_budget(
                existing_history=dp_existing_history,
                noise_multiplier=config.dp_noise_multiplier,
                sample_rate=dp_sample_rate,
                planned_steps=planned_steps,
                target_delta=config.dp_target_delta,
                target_epsilon=config.dp_target_epsilon,
            )
            dp_engine, model, optimizer, prepared.train_loader = prepare_dp_training(
                model=model,
                optimizer=optimizer,
                train_loader=prepared.train_loader,
                noise_multiplier=config.dp_noise_multiplier,
                max_grad_norm=config.dp_max_grad_norm,
                poisson_sampling=True,
            )

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
                if config.dp_enabled:
                    train_metrics_dict["sample_count"] = len(prepared.train_loader.dataset)

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
                        save_model = getattr(model, "_module", model)
                        metadata = model_store.save_checkpoint(
                            save_model, config.model_config, config.model_id, status="trained",
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

                # One progress line per epoch (aggregate metrics only). The hospital
                # desktop's local ML service parses this exact format for live charts.
                line = (f"[epoch {epoch}/{config.epochs}] train_loss={train_metrics.loss:.4f} "
                        f"train_acc={train_metrics.accuracy:.4f}")
                if do_validate:
                    line += f" val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
                print(f"{line} elapsed={time.time() - started_at:.1f}s", flush=True)

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
            save_model = getattr(model, "_module", model)
            metadata = model_store.save_checkpoint(
                save_model, config.model_config, config.model_id, status="trained",
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
            collected = {}
            test_metrics = self._run_epoch(
                model, prepared.test_loader, loss_fn, None, device, prepared.class_mapping,
                train_mode=False, precision=precision, scaler=scaler, collect=collected,
            )
            test_metrics_dict = test_metrics.to_dict()
            self.test_outputs = (collected["y_true"], collected["probs"], test_metrics.class_order)

        if failed_error is not None:
            status = TrainingStatus.TRAINING_FAILED
        elif interrupted:
            status = TrainingStatus.TRAINING_INTERRUPTED
        elif checkpoint_model_id is not None:
            status = TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION
        else:
            status = TrainingStatus.TRAINING_COMPLETED

        training_config_dict = config.to_dict()
        if config.dp_enabled and dp_engine is not None and dp_state is not None:
            from hospital_client.training.dp import compute_rdp_epsilon, save_dp_state
            steps_taken = sum(int(entry[2]) for entry in dp_engine.accountant.history)
            round_history = [(config.dp_noise_multiplier, dp_sample_rate, steps_taken)]
            round_eps = compute_rdp_epsilon(round_history, config.dp_target_delta)
            cumulative_eps = compute_rdp_epsilon(dp_existing_history + round_history, config.dp_target_delta)
            round_identifier = config.canonical_base_model_id or f"round_{config.model_id}"
            dp_state["history"].append({
                "round_id": round_identifier,
                "noise_multiplier": config.dp_noise_multiplier,
                "sample_rate": dp_sample_rate,
                "steps": steps_taken,
                "delta": config.dp_target_delta,
                "round_epsilon": round_eps,
                "cumulative_epsilon": cumulative_eps,
            })
            save_dp_state(dp_state_path, dp_state)
            training_config_dict["dp_sample_rate"] = dp_sample_rate
            training_config_dict["dp_steps"] = steps_taken
            training_config_dict["dp_epsilon"] = round_eps
            training_config_dict["dp_cumulative_epsilon"] = cumulative_eps

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
            training_configuration=training_config_dict,
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

    def _run_epoch(self, model, loader, loss_fn, optimizer, device, class_mapping, train_mode: bool, precision: str, scaler, collect=None):
        model.train(train_mode)
        total_loss, n_samples = 0.0, 0
        all_true, all_pred, all_probs = [], [], []
        autocast_enabled = precision == "fp16"

        if train_mode and self.config.dp_enabled:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            with torch.set_grad_enabled(train_mode):
                with BatchMemoryManager(data_loader=loader, max_physical_batch_size=self.config.batch_size, optimizer=optimizer) as bmm:
                    for images, labels in bmm:
                        if images.size(0) == 0:
                            continue
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                            outputs = model(images)
                            loss = loss_fn(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        batch_size = labels.size(0)
                        total_loss += loss.item() * batch_size
                        n_samples += batch_size
                        preds = outputs.argmax(dim=1)
                        all_true.extend(labels.detach().cpu().tolist())
                        all_pred.extend(preds.detach().cpu().tolist())
                        if collect is not None:
                            all_probs.extend(torch.softmax(outputs.detach().float(), dim=1).cpu().tolist())
        else:
            with torch.set_grad_enabled(train_mode):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    if train_mode:
                        optimizer.zero_grad()
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
                    if train_mode:
                        objective = loss
                        if self._prox_anchor is not None:
                            # FedProx proximal term, in fp32 outside autocast. Reported loss/metrics stay
                            # the plain classification loss so runs with and without FedProx compare directly.
                            objective = loss + (self.config.proximal_mu / 2) * sum(
                                (p.float() - a.float()).pow(2).sum() for p, a in zip(model.parameters(), self._prox_anchor))
                        # scaler with enabled=False (FP32) is a documented no-op passthrough:
                        # scale()/step()/update() behave exactly like plain backward()/step().
                        scaler.scale(objective).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    batch_size = labels.size(0)
                    total_loss += loss.item() * batch_size
                    n_samples += batch_size
                    preds = outputs.argmax(dim=1)
                    all_true.extend(labels.detach().cpu().tolist())
                    all_pred.extend(preds.detach().cpu().tolist())
                    if collect is not None:
                        all_probs.extend(torch.softmax(outputs.detach().float(), dim=1).cpu().tolist())
        if collect is not None:
            collect["y_true"], collect["probs"] = all_true, all_probs
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
    from a completed TrainingResult.

    Checkpoint identity (what M7 produced locally):
        model_id / model_version  <- result.checkpoint_model_id / version

    Base model provenance (what M9 requires to match the round):
        base_model_id / version / checksum <- taken from
        result.training_configuration["canonical_base_model_*"] when present.
        If those fields are absent (standalone, non-federated training) the
        old behaviour is preserved: base_model_id is set to the checkpoint ID
        so that a handoff can still be constructed (it will be rejected by M9
        if a canonical ID is required but missing, which is the correct outcome).

    round_id/client_id are only ever the caller-supplied real values from the
    federation integration - M7 has no such context and never fabricates them.
    """
    if not result.ready_for_federation or result.checkpoint_model_id is None or result.checkpoint_version is None:
        raise ValueError(
            "Cannot build a federation handoff: training did not reach "
            "TRAINING_COMPLETED_AWAITING_FEDERATION with a valid checkpoint."
        )
    model, _ = get_model(result.checkpoint_model_id, result.checkpoint_version, output_dir, device="cpu")
    parameters = get_parameters(model)
    num_train_samples = result.training_metrics.get("sample_count", 0)
    dataset_dir = result.training_configuration.get("dataset_dir")
    if dataset_dir:
        import os
        import json
        train_manifest_path = os.path.join(dataset_dir, "train_manifest.json")
        if os.path.exists(train_manifest_path):
            with open(train_manifest_path, "r") as f:
                train_manifest = json.load(f)
                expected_count = len(train_manifest.get("records", []))
                if num_train_samples != expected_count:
                    raise ValueError(
                        f"Invariant violation: num_train_samples ({num_train_samples}) "
                        f"does not equal the number of records in train_manifest.json ({expected_count})."
                    )

    # Resolve canonical base model provenance.  This is the round-prescribed
    # starting point, NOT the locally-produced checkpoint.
    tc = result.training_configuration
    base_model_id       = tc.get("canonical_base_model_id")      or result.checkpoint_model_id
    base_model_version  = tc.get("canonical_base_model_version") or result.checkpoint_version
    base_model_checksum = tc.get("canonical_base_model_checksum") or _compute_parameters_checksum(parameters)

    return _build_federation_handoff(
        parameters=parameters,
        model_id=result.checkpoint_model_id,
        model_version=result.checkpoint_version,
        architecture=result.architecture,
        base_model_id=base_model_id,
        base_model_version=base_model_version,
        base_model_checksum=base_model_checksum,
        num_train_samples=num_train_samples,
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
