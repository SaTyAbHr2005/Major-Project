"""
Measures REAL parameter counts for Module 6's 8 architectures by actually
constructing each model (CPU, pretrained=False, no gradients) via Module 6's
own build_model()/count_parameters() - never a guessed or hard-coded number.
Results are cached per (architecture, num_classes, input_size, color_mode)
since construction is deterministic for a given config and re-building on
every recommendation call would be wasteful.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from hospital_client.model_management.architectures import count_parameters
from hospital_client.model_management.config import ModelConfig
from hospital_client.model_management.registry import ARCHITECTURE_CATALOG, build_model, get_architecture_info

_CACHE: Dict[Tuple, "MeasuredModelProfile"] = {}


@dataclass
class MeasuredModelProfile:
    architecture: str
    display_name: str
    resource_tier: str
    param_count: int
    default_input_size: Tuple[int, int]
    supports_pretrained: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "display_name": self.display_name,
            "resource_tier": self.resource_tier,
            "param_count": self.param_count,
            "default_input_size": list(self.default_input_size),
            "supports_pretrained": self.supports_pretrained,
        }


def measure_model_profile(architecture: str, num_classes: int, color_mode: str = "RGB") -> MeasuredModelProfile:
    info = get_architecture_info(architecture)
    input_size = info.default_input_size
    cache_key = (architecture, num_classes, input_size, color_mode)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    config = ModelConfig(
        architecture=architecture, num_classes=num_classes, input_size=input_size,
        color_mode=color_mode, pretrained=False,
    )
    model = build_model(config)
    param_count = count_parameters(model)["total"]

    profile = MeasuredModelProfile(
        architecture=architecture,
        display_name=info.display_name,
        resource_tier=info.resource_tier.value,
        param_count=param_count,
        default_input_size=input_size,
        supports_pretrained=info.supports_pretrained,
    )
    _CACHE[cache_key] = profile
    return profile


def measure_all_model_profiles(num_classes: int, color_mode: str = "RGB") -> Dict[str, MeasuredModelProfile]:
    return {arch: measure_model_profile(arch, num_classes, color_mode) for arch in ARCHITECTURE_CATALOG}
