"""
Compatibility checks between a Module 6 model and Module 5's preprocessing
contract. Reads Module 5's existing ImageConfig (target_size/color_mode)
without modifying Module 5 in any way. A hard mismatch is always reported
and never silently allowed through.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hospital_client.model_management.metadata import ModelMetadata


@dataclass
class CompatibilityResult:
    compatible: bool
    mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compatible": self.compatible,
            "mismatches": self.mismatches,
            "warnings": self.warnings,
            "reasons": self.reasons,
        }


def check_compatibility(
    model_metadata: ModelMetadata,
    image_config: Any = None,
    expected_num_classes: Optional[int] = None,
    expected_class_mapping: Optional[Dict[str, int]] = None,
) -> CompatibilityResult:
    """
    image_config: an instance of hospital_client.preprocessing.config.ImageConfig
    (or anything exposing .target_size / .color_mode) - imported by the
    caller, not by this module, to avoid an unnecessary hard dependency here.
    """
    result = CompatibilityResult(compatible=True)

    if image_config is not None:
        model_input_size = tuple(model_metadata.input_spec.get("input_size", []) or [])
        target_size = getattr(image_config, "target_size", None)
        if target_size is not None and model_input_size and tuple(target_size) != model_input_size:
            result.mismatches.append(
                f"input_size mismatch: model expects {model_input_size}, preprocessing produces {tuple(target_size)}"
            )

        model_color_mode = model_metadata.input_spec.get("color_mode")
        target_color_mode = getattr(image_config, "color_mode", None)
        if target_color_mode is not None and model_color_mode is not None and target_color_mode != model_color_mode:
            result.mismatches.append(
                f"color_mode mismatch: model expects '{model_color_mode}', preprocessing produces '{target_color_mode}'"
            )
        if target_size is None:
            result.warnings.append("Preprocessing ImageConfig.target_size is not set; input_size could not be checked.")
        if target_color_mode is None:
            result.warnings.append("Preprocessing ImageConfig.color_mode is not set; color_mode could not be checked.")

    if expected_num_classes is not None and model_metadata.num_classes != expected_num_classes:
        result.mismatches.append(
            f"num_classes mismatch: model has {model_metadata.num_classes}, dataset/task has {expected_num_classes}"
        )

    if expected_class_mapping is not None:
        if model_metadata.class_mapping and dict(expected_class_mapping) != dict(model_metadata.class_mapping):
            result.mismatches.append(
                f"class_mapping mismatch: model expects {model_metadata.class_mapping}, dataset has {expected_class_mapping}"
            )
        elif not model_metadata.class_mapping:
            result.warnings.append("Model metadata has no class_mapping recorded; class_mapping could not be checked.")

    if result.mismatches:
        result.compatible = False
        result.reasons = list(result.mismatches)

    return result
