"""
Explicit model architecture registry and catalog for Module 6.

Design intent (per architecture decision): a plain, explicit dict - no
plugin auto-discovery, no metaprogramming. Adding a future architecture
means writing one factory function in architectures.py and adding one
registry/catalog entry here; nothing else in the project needs to change.

Resource tiers (HIGH/MEDIUM/LOW/VERY_LOW) are project-defined
RECOMMENDATION labels only. Module 6 never uses them to make a hardware
decision - that is Module 8's responsibility. Module 6 only exposes them as
metadata for Module 8 to consume later.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Tuple

import torch.nn as nn

from hospital_client.model_management import architectures as _arch
from hospital_client.model_management.config import ModelConfig


class ResourceTier(Enum):
    HIGH_END = "high_end"
    MEDIUM_END = "medium_end"
    LOW_END = "low_end"
    VERY_LOW_END = "very_low_end"


@dataclass
class ArchitectureInfo:
    name: str
    display_name: str
    family: str
    resource_tier: ResourceTier
    supports_pretrained: bool
    default_input_size: Tuple[int, int]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "family": self.family,
            "resource_tier": self.resource_tier.value,
            "supports_pretrained": self.supports_pretrained,
            "default_input_size": list(self.default_input_size),
            "notes": self.notes,
        }


ARCHITECTURE_REGISTRY: Dict[str, Callable[[ModelConfig], nn.Module]] = {
    "vit_b16": _arch.build_vit_b16,
    "efficientnet_b4": _arch.build_efficientnet_b4,
    "efficientnet_b0": _arch.build_efficientnet_b0,
    "resnet50": _arch.build_resnet50,
    "resnet18": _arch.build_resnet18,
    "mobilenet_v2": _arch.build_mobilenet_v2,
    "mobilenet_v3_small": _arch.build_mobilenet_v3_small,
    "mobilevit_xxs": _arch.build_mobilevit_xxs,
}

ARCHITECTURE_CATALOG: Dict[str, ArchitectureInfo] = {
    "vit_b16": ArchitectureInfo(
        "vit_b16", "ViT-B/16", "Vision Transformer", ResourceTier.HIGH_END,
        supports_pretrained=True, default_input_size=(224, 224),
        notes="Requires a square input_size divisible by the 16x16 patch size; pretrained weights require input_size=224.",
    ),
    "efficientnet_b4": ArchitectureInfo(
        "efficientnet_b4", "EfficientNet-B4", "EfficientNet", ResourceTier.HIGH_END,
        supports_pretrained=True, default_input_size=(380, 380),
        notes="torchvision reference weights were trained at 380x380.",
    ),
    "efficientnet_b0": ArchitectureInfo(
        "efficientnet_b0", "EfficientNet-B0", "EfficientNet", ResourceTier.MEDIUM_END,
        supports_pretrained=True, default_input_size=(224, 224),
        notes="Project's initial primary architecture (see module6_readme.md).",
    ),
    "resnet50": ArchitectureInfo(
        "resnet50", "ResNet-50", "ResNet", ResourceTier.MEDIUM_END,
        supports_pretrained=True, default_input_size=(224, 224),
    ),
    "resnet18": ArchitectureInfo(
        "resnet18", "ResNet-18", "ResNet", ResourceTier.LOW_END,
        supports_pretrained=True, default_input_size=(224, 224),
    ),
    "mobilenet_v2": ArchitectureInfo(
        "mobilenet_v2", "MobileNetV2", "MobileNet", ResourceTier.LOW_END,
        supports_pretrained=True, default_input_size=(224, 224),
    ),
    "mobilenet_v3_small": ArchitectureInfo(
        "mobilenet_v3_small", "MobileNetV3-Small", "MobileNet", ResourceTier.VERY_LOW_END,
        supports_pretrained=True, default_input_size=(224, 224),
    ),
    "mobilevit_xxs": ArchitectureInfo(
        "mobilevit_xxs", "MobileViT-XXS", "MobileViT", ResourceTier.VERY_LOW_END,
        supports_pretrained=False, default_input_size=(256, 256),
        notes="From-scratch implementation (torchvision has no MobileViT). No pretrained weights available.",
    ),
}


def list_architectures() -> List[str]:
    return sorted(ARCHITECTURE_REGISTRY.keys())


def get_architecture_info(name: str) -> ArchitectureInfo:
    if name not in ARCHITECTURE_CATALOG:
        raise ValueError(f"Unknown architecture '{name}'. Supported architectures: {list_architectures()}")
    return ARCHITECTURE_CATALOG[name]


def build_model(config: ModelConfig) -> nn.Module:
    """
    Constructs a real torch.nn.Module for config.architecture. Raises
    ValueError for an unknown architecture rather than silently falling back
    to a different one. Device placement is NOT done here - see device.py;
    this function always returns a CPU module, matching PyTorch's own
    default construction behavior.
    """
    from hospital_client.model_management.config import validate_model_config
    validate_model_config(config)
    factory = ARCHITECTURE_REGISTRY[config.architecture]
    return factory(config)
