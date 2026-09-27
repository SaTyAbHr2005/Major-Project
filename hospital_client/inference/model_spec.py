"""
Module 16's adapter over Module 6's ModelMetadata (Module 6 is not modified).

Module 6/7 artifacts do not record everything Module 16 would ideally know
(no explicit `task`, no normalization, `preprocessing_reference` is optional
and usually unset). This adapter never invents those values:

  * fields Module 6 DOES record (architecture, input_spec, num_classes,
    class_mapping, artifact hash, local status) are validated strictly and
    a missing/inconsistent one rejects the model;
  * fields it does NOT record are surfaced explicitly as UNAVAILABLE (or, for
    `task`, marked as inferred and why) - never silently guessed.

Class mapping is the one thing that is never assumed: without a stored,
complete label<->index mapping Module 16 refuses to interpret model outputs.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from hospital_client.inference.errors import ModelCompatibilityError
from hospital_client.model_management.config import SUPPORTED_COLOR_MODES
from hospital_client.model_management.metadata import ModelMetadata
from hospital_client.model_management.registry import ARCHITECTURE_REGISTRY

UNAVAILABLE = "unavailable"
ARTIFACT_FORMAT = "safetensors"
DEFAULT_ALLOWED_STATUSES = ("trained", "available")

SUPPORTED_TASKS = ("image_classification",)  # extension point: add a task here plus a handler in engine.py
SUPPORTED_PREPROCESSING_SPEC_VERSIONS = (1,)
TASK_SOURCE_RECORDED = "recorded: stored in Module 6 metadata by Module 7 at training time"

TASK_SOURCE_INFERRED = (
    "inferred: Module 6 metadata has no explicit task field and every Module 6 "
    "architecture is an image-classification network with a num_classes output head"
)


@dataclass
class InferenceModelSpec:
    model_id: str
    version: int
    architecture: str
    artifact_format: str
    artifact_sha256: str
    input_size: Tuple[int, int]
    color_mode: str
    num_classes: int
    class_mapping: Dict[str, int]
    index_to_label: Dict[int, str]
    task: str
    task_source: str
    model_status: str
    preprocessing_reference: Optional[str]
    preprocessing_spec: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.version,
            "architecture": self.architecture,
            "artifact_format": self.artifact_format,
            "artifact_sha256": self.artifact_sha256,
            "input_size": list(self.input_size),
            "color_mode": self.color_mode,
            "num_classes": self.num_classes,
            "class_mapping": self.class_mapping,
            "task": self.task,
            "task_source": self.task_source,
            "model_status": self.model_status,
            "preprocessing_reference": self.preprocessing_reference or UNAVAILABLE,
            "preprocessing_spec": self.preprocessing_spec or UNAVAILABLE,
            "warnings": self.warnings,
        }


def _valid_size(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    )


def _invert_class_mapping(class_mapping: Any, num_classes: int) -> Dict[int, str]:
    if not isinstance(class_mapping, dict) or not class_mapping:
        raise ModelCompatibilityError(
            "Model metadata has no class_mapping. Module 16 never assumes which output index "
            "means which class - the approved model must record its label<->index mapping."
        )
    labels_ok = all(isinstance(k, str) and k for k in class_mapping)
    indices_ok = all(isinstance(v, int) and not isinstance(v, bool) for v in class_mapping.values())
    if not labels_ok or not indices_ok:
        raise ModelCompatibilityError("class_mapping must map non-empty string labels to integer indices.")
    if len(class_mapping) != num_classes or sorted(class_mapping.values()) != list(range(num_classes)):
        raise ModelCompatibilityError(
            f"class_mapping is not a complete, unique 0..{num_classes - 1} index assignment "
            f"for num_classes={num_classes}: {class_mapping}"
        )
    return {idx: label for label, idx in class_mapping.items()}


def _validate_preprocessing_spec(spec: Any, input_size, color_mode: str) -> None:
    """The recorded spec must describe exactly the pipeline Module 16 runs; otherwise refuse."""
    if not isinstance(spec, dict):
        raise ModelCompatibilityError("Stored preprocessing_spec is not a mapping.")
    if spec.get("version") not in SUPPORTED_PREPROCESSING_SPEC_VERSIONS:
        raise ModelCompatibilityError(
            f"Unsupported preprocessing_spec version {spec.get('version')!r} "
            f"(supported: {list(SUPPORTED_PREPROCESSING_SPEC_VERSIONS)}).")
    problems = []
    if tuple(spec.get("input_size") or ()) != tuple(input_size):
        problems.append("input_size")
    if spec.get("color_mode") != color_mode:
        problems.append("color_mode")
    if spec.get("normalization") is not None:
        problems.append("normalization (Module 16 applies none)")
    if spec.get("interpolation") != "bilinear" or spec.get("resize_method") != "torchvision.transforms.Resize":
        problems.append("resize method/interpolation")
    if spec.get("channel_order") != "CHW" or spec.get("dtype") != "float32" or list(spec.get("value_range") or ()) != [0.0, 1.0]:
        problems.append("tensor layout/dtype/value range")
    upstream = spec.get("upstream")
    if upstream is not None:
        ok = isinstance(upstream, dict) and upstream.get("output_mode") in ("lazy", "materialized")
        if ok and upstream["output_mode"] == "materialized":
            size = upstream.get("target_size")
            ok = (size is None or _valid_size(size)) and upstream.get("color_mode") in (None, *SUPPORTED_COLOR_MODES)
        if not ok:
            problems.append("upstream Module 5 settings are malformed")
    if problems:
        raise ModelCompatibilityError(
            f"Stored preprocessing_spec does not match the pipeline Module 16 reproduces: {', '.join(problems)}. "
            "Inference is refused rather than feeding the model differently-prepared input.")


def build_model_spec(
    metadata: ModelMetadata,
    allowed_statuses: Sequence[str] = DEFAULT_ALLOWED_STATUSES,
    require_preprocessing_reference: bool = False,
    expected_task: Optional[str] = None,
) -> InferenceModelSpec:
    if metadata.status not in allowed_statuses:
        raise ModelCompatibilityError(
            f"Model status '{metadata.status}' is not usable for inference (allowed: "
            f"{list(allowed_statuses)}). Note: this is Module 6's LOCAL lifecycle status, "
            "not a platform approval decision (Module 15)."
        )

    if metadata.architecture not in ARCHITECTURE_REGISTRY:
        raise ModelCompatibilityError(f"Unsupported architecture '{metadata.architecture}'.")
    config = metadata.config or {}
    if config.get("architecture") != metadata.architecture:
        raise ModelCompatibilityError(
            f"Metadata architecture '{metadata.architecture}' does not match its stored model "
            f"configuration ('{config.get('architecture')}')."
        )

    input_spec = metadata.input_spec or {}
    input_size, color_mode = input_spec.get("input_size"), input_spec.get("color_mode")
    if not _valid_size(input_size):
        raise ModelCompatibilityError(
            f"Model metadata has no valid input_size (got {input_size!r}); required preprocessing "
            "information is missing, so inference is refused rather than guessed."
        )
    if color_mode not in SUPPORTED_COLOR_MODES:
        raise ModelCompatibilityError(
            f"Model metadata has no valid color_mode (got {color_mode!r}; expected one of "
            f"{sorted(SUPPORTED_COLOR_MODES)})."
        )
    if tuple(config.get("input_size", ())) != tuple(input_size) or config.get("color_mode") != color_mode:
        raise ModelCompatibilityError("Metadata input_spec disagrees with the stored model configuration.")

    num_classes = metadata.num_classes
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ModelCompatibilityError(f"Invalid num_classes in metadata: {num_classes!r}")
    if config.get("num_classes") != num_classes:
        raise ModelCompatibilityError("Metadata num_classes disagrees with the stored model configuration.")
    index_to_label = _invert_class_mapping(metadata.class_mapping, num_classes)

    if not metadata.artifact_sha256:
        raise ModelCompatibilityError("Model metadata has no artifact hash; integrity cannot be established.")

    recorded_task = getattr(metadata, "task_type", None)
    if recorded_task is not None:
        if recorded_task not in SUPPORTED_TASKS:
            raise ModelCompatibilityError(f"Model task '{recorded_task}' is not supported (supported: {list(SUPPORTED_TASKS)}).")
        task, task_source = recorded_task, TASK_SOURCE_RECORDED
    else:
        task, task_source = "image_classification", TASK_SOURCE_INFERRED
    if expected_task is not None and expected_task != task:
        raise ModelCompatibilityError(f"Model task is '{task}', but '{expected_task}' was expected.")

    warnings: List[str] = []
    preprocessing_spec = getattr(metadata, "preprocessing_spec", None)
    if preprocessing_spec is not None:
        _validate_preprocessing_spec(preprocessing_spec, input_size, color_mode)
        if "upstream" not in preprocessing_spec:
            warnings.append(
                "Stored preprocessing_spec has no upstream Module 5 settings (dataset manifest predates them); "
                "whether the model was trained on lazy or materialized images is unknown.")
    elif not metadata.preprocessing_reference:
        if require_preprocessing_reference:
            raise ModelCompatibilityError(
                "Model metadata has no preprocessing_spec/preprocessing_reference and require_preprocessing_reference=True."
            )
        warnings.append(
            "Model metadata records no preprocessing_spec and no preprocessing_reference (artifact predates them); preprocessing is the fixed Module 7 "
            "training pipeline (convert -> resize -> ToTensor) for the recorded input_size/color_mode, not verified against a stored spec."
        )
    if recorded_task is None:
        warnings.append("Model metadata records no task_type; the task is inferred (image_classification).")

    return InferenceModelSpec(
        model_id=metadata.model_id, version=metadata.version, architecture=metadata.architecture,
        artifact_format=ARTIFACT_FORMAT, artifact_sha256=metadata.artifact_sha256,
        input_size=(input_size[0], input_size[1]), color_mode=color_mode, num_classes=num_classes,
        class_mapping=dict(metadata.class_mapping), index_to_label=index_to_label,
        task=task, task_source=task_source, model_status=metadata.status,
        preprocessing_reference=metadata.preprocessing_reference, preprocessing_spec=preprocessing_spec, warnings=warnings,
    )
