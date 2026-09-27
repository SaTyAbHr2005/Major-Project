"""
Training configuration for Module 7 (Local Deep Learning Training).

Follows the dataclass + to_dict()/validate_*() convention already used by
Module 5 (preprocessing/config.py) and Module 6 (model_management/config.py).

Resource-aware boundary (Module 8): device/batch_size/epochs/learning_rate
are plain fields here that Module 7 EXECUTES; Module 7 never inspects
hardware to choose them. A future Module 8 "ResolvedTrainingConfig" is
expected to just populate these same fields before handing this object (or
an equivalent one) to Module 7.
"""
from dataclasses import dataclass, field
from typing import Optional

from hospital_client.model_management.config import ModelConfig, validate_model_config

VALID_OPTIMIZERS = {"adam", "sgd"}
VALID_SCHEDULERS = {None, "none", "reduce_on_plateau", "step", "cosine"}
VALID_CLASS_WEIGHTING = {"none", "manifest", "balanced"}
VALID_EARLY_STOPPING_METRICS = {"val_loss", "val_accuracy", "val_f1"}
VALID_EARLY_STOPPING_MODES = {"min", "max"}
VALID_DEVICES = {"cpu", "cuda"}
VALID_PRECISIONS = {"fp32", "fp16", "auto"}


@dataclass
class TrainingConfig:
    # Identity / Module 6 integration
    model_id: str
    model_config: ModelConfig

    # Data (Module 5 output directory containing {train,validation,test}_manifest.json)
    dataset_dir: str
    require_validation: bool = True
    run_test_evaluation: bool = False

    # Output (Module 6 artifact store + Module 7 private training state live under here)
    output_dir: str = ""

    # Device - explicitly requested only; Module 7 never auto-selects (Module 8's job)
    device: str = "cpu"

    # Mixed precision: "fp32" (default, unchanged baseline behavior), "fp16"
    # (CUDA-only autocast + gradient scaling), "auto" (resolves to fp16 on a
    # resolved CUDA device, fp32 on CPU - never silently changes device).
    precision: str = "fp32"

    # Training loop
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9  # only used when optimizer == "sgd"

    # Scheduler: None/"none" (default, unchanged), "reduce_on_plateau" (existing
    # behavior, unchanged), "step" (StepLR), "cosine" (CosineAnnealingLR).
    scheduler: Optional[str] = None
    scheduler_patience: int = 2       # reduce_on_plateau
    scheduler_factor: float = 0.5     # reduce_on_plateau
    scheduler_step_size: int = 10     # step
    scheduler_gamma: float = 0.1      # step
    scheduler_t_max: Optional[int] = None    # cosine; defaults to `epochs` when unset
    scheduler_eta_min: float = 0.0           # cosine

    # Early stopping (never driven by test data)
    early_stopping: bool = False
    early_stopping_metric: str = "val_loss"
    early_stopping_patience: int = 5
    early_stopping_mode: str = "min"

    # Loss / class imbalance
    class_weighting: str = "none"

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = False

    # Reproducibility
    random_seed: int = 42

    # Resume
    resume: bool = False

    # Validation cadence (in epochs)
    validate_every: int = 1

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_config": self.model_config.to_dict(),
            "dataset_dir": self.dataset_dir,
            "require_validation": self.require_validation,
            "run_test_evaluation": self.run_test_evaluation,
            "output_dir": self.output_dir,
            "device": self.device,
            "precision": self.precision,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "scheduler": self.scheduler,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_factor": self.scheduler_factor,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma": self.scheduler_gamma,
            "scheduler_t_max": self.scheduler_t_max,
            "scheduler_eta_min": self.scheduler_eta_min,
            "early_stopping": self.early_stopping,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_mode": self.early_stopping_mode,
            "class_weighting": self.class_weighting,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "random_seed": self.random_seed,
            "resume": self.resume,
            "validate_every": self.validate_every,
        }


def validate_training_config(config: TrainingConfig) -> None:
    """
    Raises ValueError describing the first invalid field found. Never
    silently corrects a value or substitutes a default for an invalid one.
    """
    validate_model_config(config.model_config)  # reuse Module 6's validation, don't duplicate it

    if not config.model_id or not isinstance(config.model_id, str):
        raise ValueError(f"model_id must be a non-empty string, got {config.model_id!r}")
    if not config.dataset_dir:
        raise ValueError("dataset_dir must be set to a Module 5 output directory.")
    if not config.output_dir:
        raise ValueError("output_dir must be set.")

    if config.device not in VALID_DEVICES:
        raise ValueError(f"device must be one of {sorted(VALID_DEVICES)}, got {config.device!r}")

    if config.precision not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(VALID_PRECISIONS)}, got {config.precision!r}")
    if config.precision == "fp16" and config.device == "cpu":
        raise ValueError(
            "precision='fp16' was requested with device='cpu'. Module 7 does not pretend FP16 "
            "is available for CPU training. Use device='cuda' with precision='fp16', or "
            "precision='fp32'/'auto' for CPU training."
        )

    if not isinstance(config.epochs, int) or isinstance(config.epochs, bool) or config.epochs < 1:
        raise ValueError(f"epochs must be an integer >= 1, got {config.epochs!r}")
    if not isinstance(config.batch_size, int) or isinstance(config.batch_size, bool) or config.batch_size < 1:
        raise ValueError(f"batch_size must be an integer >= 1, got {config.batch_size!r}")
    if not isinstance(config.learning_rate, (int, float)) or config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be a positive number, got {config.learning_rate!r}")
    if not isinstance(config.weight_decay, (int, float)) or config.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay!r}")
    if config.optimizer not in VALID_OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got {config.optimizer!r}")

    if config.scheduler not in VALID_SCHEDULERS:
        raise ValueError(f"scheduler must be one of {sorted(str(s) for s in VALID_SCHEDULERS)}, got {config.scheduler!r}")
    if config.scheduler == "step":
        if not isinstance(config.scheduler_step_size, int) or isinstance(config.scheduler_step_size, bool) or config.scheduler_step_size < 1:
            raise ValueError(f"scheduler_step_size must be an integer >= 1, got {config.scheduler_step_size!r}")
        if not isinstance(config.scheduler_gamma, (int, float)) or config.scheduler_gamma <= 0:
            raise ValueError(f"scheduler_gamma must be a positive number, got {config.scheduler_gamma!r}")
    if config.scheduler == "cosine":
        if config.scheduler_t_max is not None and (
            not isinstance(config.scheduler_t_max, int) or isinstance(config.scheduler_t_max, bool) or config.scheduler_t_max < 1
        ):
            raise ValueError(f"scheduler_t_max must be None or an integer >= 1, got {config.scheduler_t_max!r}")
        if not isinstance(config.scheduler_eta_min, (int, float)) or config.scheduler_eta_min < 0:
            raise ValueError(f"scheduler_eta_min must be >= 0, got {config.scheduler_eta_min!r}")

    if config.class_weighting not in VALID_CLASS_WEIGHTING:
        raise ValueError(f"class_weighting must be one of {sorted(VALID_CLASS_WEIGHTING)}, got {config.class_weighting!r}")

    if config.early_stopping:
        if config.early_stopping_metric not in VALID_EARLY_STOPPING_METRICS:
            raise ValueError(
                f"early_stopping_metric must be one of {sorted(VALID_EARLY_STOPPING_METRICS)}, "
                f"got {config.early_stopping_metric!r}"
            )
        if config.early_stopping_mode not in VALID_EARLY_STOPPING_MODES:
            raise ValueError(
                f"early_stopping_mode must be one of {sorted(VALID_EARLY_STOPPING_MODES)}, "
                f"got {config.early_stopping_mode!r}"
            )
        if not isinstance(config.early_stopping_patience, int) or config.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be an integer >= 1, got {config.early_stopping_patience!r}")
        if not config.require_validation:
            raise ValueError("early_stopping requires require_validation=True (never driven by test data).")

    if not isinstance(config.num_workers, int) or config.num_workers < 0:
        raise ValueError(f"num_workers must be an integer >= 0, got {config.num_workers!r}")
    if not isinstance(config.validate_every, int) or config.validate_every < 1:
        raise ValueError(f"validate_every must be an integer >= 1, got {config.validate_every!r}")
    if not isinstance(config.random_seed, int):
        raise ValueError(f"random_seed must be an integer, got {config.random_seed!r}")
