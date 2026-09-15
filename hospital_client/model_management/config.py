"""
Model configuration schema for Module 6 (Model Management).

Follows the same dataclass + to_dict()/from_dict() convention already used by
hospital_client.preprocessing.config and hospital_client.dataset.models.dataset_profile.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

SUPPORTED_COLOR_MODES = {"RGB", "L"}


@dataclass
class ModelConfig:
    architecture: str
    num_classes: int
    input_size: Tuple[int, int] = (224, 224)
    color_mode: str = "RGB"
    pretrained: bool = False
    dropout: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "input_size": list(self.input_size),
            "color_mode": self.color_mode,
            "pretrained": self.pretrained,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        input_size = data.get("input_size", (224, 224))
        return cls(
            architecture=data["architecture"],
            num_classes=data["num_classes"],
            input_size=tuple(input_size),
            color_mode=data.get("color_mode", "RGB"),
            pretrained=data.get("pretrained", False),
            dropout=data.get("dropout"),
        )


def validate_model_config(config: ModelConfig) -> None:
    """
    Raises ValueError describing the first invalid field found. Never
    silently fills in a default for an invalid value and never substitutes a
    different architecture.
    """
    # Local import: registry.py imports architectures that are typed against
    # ModelConfig, so importing registry at module load time here would be
    # circular. Importing inside the function breaks the cycle safely.
    from hospital_client.model_management.registry import ARCHITECTURE_REGISTRY

    if config.architecture not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{config.architecture}'. "
            f"Supported architectures: {sorted(ARCHITECTURE_REGISTRY)}"
        )

    if not isinstance(config.num_classes, int) or isinstance(config.num_classes, bool) or config.num_classes < 2:
        raise ValueError(f"num_classes must be an integer >= 2, got {config.num_classes!r}")

    valid_input_size = (
        isinstance(config.input_size, (tuple, list))
        and len(config.input_size) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in config.input_size)
    )
    if not valid_input_size:
        raise ValueError(
            f"input_size must be a (height, width) tuple of positive integers, got {config.input_size!r}"
        )

    if config.color_mode not in SUPPORTED_COLOR_MODES:
        raise ValueError(
            f"color_mode must be one of {sorted(SUPPORTED_COLOR_MODES)}, got {config.color_mode!r}"
        )

    if not isinstance(config.pretrained, bool):
        raise ValueError(f"pretrained must be a bool, got {config.pretrained!r}")

    if config.dropout is not None:
        if isinstance(config.dropout, bool) or not isinstance(config.dropout, (int, float)) or not (0.0 <= config.dropout < 1.0):
            raise ValueError(f"dropout must be a float in [0.0, 1.0), got {config.dropout!r}")
