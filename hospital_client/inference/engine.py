"""
The generic local inference engine. One implementation serves every Module 6
architecture: the architecture only decides which Module 6 factory builds the
model, exactly as in Module 7's Trainer.

Loading is delegated entirely to Module 6 (`ModelStore.load_checkpoint`:
SHA-256 integrity check, strict architecture/key/shape validation, safetensors
only - never pickle). The engine has no network dependency of any kind and
never writes anything: the image is decoded in memory and discarded.

    provision -> Module 6 load+validate -> spec/compat checks -> eval mode
    predict: validate image -> preprocess (Module 7 transform) -> forward under
             torch.inference_mode -> softmax -> map index via stored class_mapping
"""
import logging
import os
import contextlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import torch

from hospital_client.inference.device import VALID_DEVICE_REQUESTS, cuda_has_headroom, resolve_inference_device
from hospital_client.inference.errors import (
    ImageValidationError, InferenceConfigError, InferenceError, ModelCompatibilityError, ModelProvisioningError,
)
from hospital_client.inference.image_input import DEFAULT_MAX_IMAGE_BYTES, DEFAULT_MAX_IMAGE_PIXELS, load_validated_image
from hospital_client.inference.model_spec import DEFAULT_ALLOWED_STATUSES, InferenceModelSpec, build_model_spec
from hospital_client.inference.preprocessing import InferencePreprocessor
from hospital_client.inference.provisioning import ApprovedModelProvider, LocalModelArtifact
from hospital_client.inference.result import ClassificationOutput, InferenceResult, InferenceStatus
from hospital_client.model_management.compatibility import check_compatibility
from hospital_client.model_management.store import ModelStore

logger = logging.getLogger("hospital_client.inference")  # only safe operational fields are ever logged


@dataclass
class InferenceConfig:
    device: str = "cpu"  # "cpu" | "cuda" | "auto" (auto delegates to Module 8; never a silent fallback)
    allowed_model_statuses: Tuple[str, ...] = DEFAULT_ALLOWED_STATUSES
    require_preprocessing_reference: bool = False
    expected_task: Optional[str] = None
    image_config: Any = None  # Module 5 ImageConfig; validated against the model when supplied
    emulate_module5_materialization: Optional[bool] = None  # None = follow the upstream mode recorded with the model
    deterministic: bool = True  # deterministic cuDNN/algorithms during the forward pass (state restored afterwards)
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    extension_metadata: Dict[str, Any] = field(default_factory=dict)  # passed through to results untouched

    def validate(self) -> None:
        if self.device not in VALID_DEVICE_REQUESTS:
            raise InferenceConfigError(f"device must be one of {list(VALID_DEVICE_REQUESTS)}, got {self.device!r}")
        if self.max_image_bytes < 1 or self.max_image_pixels < 1:
            raise InferenceConfigError("max_image_bytes and max_image_pixels must be positive.")


_REFERENCE_PATTERN = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


def sanitize_reference(reference: Optional[str]) -> Optional[str]:
    """`reference` is echoed into results and may reach logs/UI: only a short opaque token is accepted
    (no whitespace, control characters or path separators)."""
    if reference is None:
        return None
    if not isinstance(reference, str) or not _REFERENCE_PATTERN.fullmatch(reference):
        raise InferenceConfigError("reference must be 1-128 characters from [A-Za-z0-9._:-] (an opaque identifier, not a path).")
    return reference


def _scrub_paths(text: str, root: str) -> str:
    for variant in {root, os.path.abspath(root), os.path.realpath(root)}:
        for form in (variant, variant.replace("\\", "/"), variant.replace("/", "\\"), variant.replace("\\", "\\\\")):
            if form:
                text = text.replace(form, "<models-root>")
    return text


@contextlib.contextmanager
def _deterministic_mode(enabled: bool):
    """Same-machine run-to-run determinism; process-wide flags are restored on exit."""
    if not enabled:
        yield
        return
    previous = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[0], previous[1]
        torch.use_deterministic_algorithms(previous[2])


def _environment(device: torch.device, spec: InferenceModelSpec, deterministic: bool = True) -> Dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": "fp32",
        "deterministic_algorithms_requested": deterministic,
        "preprocessing_spec_version": (spec.preprocessing_spec or {}).get("version", "unavailable"),
    }


def _classification_output(logits: torch.Tensor, spec: InferenceModelSpec) -> ClassificationOutput:
    # Module 6 heads are trained with CrossEntropyLoss (Module 7), so raw outputs are logits:
    # softmax gives the probability distribution. Outputs are never assumed to be probabilities.
    probabilities = torch.softmax(logits, dim=1)[0]
    if abs(float(probabilities.sum()) - 1.0) > 1e-4:
        raise ModelCompatibilityError("Model output did not form a valid probability distribution.")
    index = int(torch.argmax(probabilities))
    return ClassificationOutput(
        predicted_index=index,
        predicted_label=spec.index_to_label[index],
        confidence=float(probabilities[index]),
        class_probabilities={spec.index_to_label[i]: float(p) for i, p in enumerate(probabilities)},
    )


_TASK_HANDLERS = {"image_classification": _classification_output}  # extension point for future task types


class LocalInferenceEngine:
    def __init__(self, source: Union[ApprovedModelProvider, LocalModelArtifact], config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.config.validate()
        artifact = source.provision() if isinstance(source, ApprovedModelProvider) else source
        if not isinstance(artifact, LocalModelArtifact):
            raise InferenceConfigError("source must be an ApprovedModelProvider or a LocalModelArtifact.")
        self.artifact = artifact
        self._lock = threading.RLock()  # one prediction at a time per engine (the model and global flags are shared)

        self.device, self.device_note = resolve_inference_device(self.config.device)
        model, metadata = self._load_model(artifact)
        model = self._place_model(model, metadata)

        self.spec = build_model_spec(
            metadata, self.config.allowed_model_statuses,
            self.config.require_preprocessing_reference, self.config.expected_task,
        )
        if self.config.image_config is not None:
            compat = check_compatibility(metadata, self.config.image_config)  # Module 6 <-> Module 5 contract
            if not compat.compatible:
                raise ModelCompatibilityError(f"Model/preprocessing mismatch: {compat.mismatches}")
        self._preprocessor = InferencePreprocessor(
            self.spec, self.config.image_config, self.config.emulate_module5_materialization,
        )

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self._handler = _TASK_HANDLERS[self.spec.task]
        logger.info("model loaded model_id=%s version=%s device=%s", self.spec.model_id, self.spec.version, self.device.type)

    def _load_model(self, artifact: LocalModelArtifact):
        # Always load on CPU first; placement (and the auto-device VRAM check) happens in _place_model.
        try:
            return ModelStore(artifact.store_root).load_checkpoint(artifact.model_id, artifact.version, device="cpu")
        except (ValueError, FileNotFoundError, OSError) as e:
            # Module 6 raises ValueError for an invalid id/version and for any failed integrity/architecture/weights validation.
            raise ModelProvisioningError(
                f"Model artifact could not be loaded: {_scrub_paths(str(e), artifact.store_root)}") from None

    def _place_model(self, model: torch.nn.Module, metadata) -> torch.nn.Module:
        if self.device.type != "cuda":
            return model
        if self.config.device == "auto":
            fits, why = cuda_has_headroom(metadata.artifact_size_bytes, self.device)
            if fits:
                try:
                    return model.to(self.device)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    why = "CUDA out of memory while placing the model"
            self.device = torch.device("cpu")
            self.device_note = f"{self.device_note}; CUDA rejected ({why}); explicitly fell back to CPU"
            return model
        return model.to(self.device)

    def model_info(self) -> Dict[str, Any]:
        info = self.spec.to_dict()
        info.update({
            "integrity_verified": True,  # Module 6 refuses to load unless the SHA-256 matched
            "approval": self.artifact.approval_info(),
        })
        return info

    def _result(self, status: InferenceStatus, **kwargs) -> InferenceResult:
        return InferenceResult(
            status=status, model_id=self.spec.model_id, model_version=self.spec.version,
            architecture=self.spec.architecture, task=self.spec.task, task_source=self.spec.task_source,
            device=self.device.type, device_note=self.device_note, model_info=self.model_info(),
            extension_metadata=dict(self.config.extension_metadata),
            environment=_environment(self.device, self.spec, self.config.deterministic), **kwargs,
        )

    def _fail(self, error: InferenceError, started: float, reference: Optional[str], message: Optional[str] = None,
              preprocessing: Optional[dict] = None) -> InferenceResult:
        logger.warning("inference failed model_id=%s version=%s category=%s", self.spec.model_id, self.spec.version, error.category)
        return self._result(
            InferenceStatus.FAILED, errors=[message or str(error)], error_category=error.category,
            total_duration_seconds=time.perf_counter() - started, reference=reference,
            preprocessing=preprocessing or {},
        )

    def predict(self, image_path: str, reference: Optional[str] = None) -> InferenceResult:
        with self._lock:
            return self._predict_locked(image_path, reference)

    def _predict_locked(self, image_path: str, reference: Optional[str]) -> InferenceResult:
        if self.model is None:
            raise InferenceError("This engine has been closed; create a new LocalInferenceEngine.")
        reference = sanitize_reference(reference)
        started = time.perf_counter()
        try:
            loaded = load_validated_image(image_path, self.config.max_image_bytes, self.config.max_image_pixels)
            tensor, preprocessing = self._preprocessor.prepare(loaded)
        except (ImageValidationError, ModelCompatibilityError) as e:
            return self._fail(e, started, reference)

        try:
            logits, inference_seconds, peak_mb = self._forward(tensor)
            output = self._handler(logits, self.spec)
        except ModelCompatibilityError as e:
            return self._fail(e, started, reference, preprocessing=preprocessing)
        except RuntimeError as e:
            if self.device.type == "cuda" and "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                oom = InferenceError("CUDA out of memory during inference.")
                oom.category = "out_of_memory"
                return self._fail(oom, started, reference, preprocessing=preprocessing)
            raise

        total = time.perf_counter() - started
        logger.info("inference completed model_id=%s version=%s device=%s duration=%.4f",
                    self.spec.model_id, self.spec.version, self.device.type, inference_seconds)
        return self._result(
            InferenceStatus.COMPLETED, output=output, inference_duration_seconds=inference_seconds,
            total_duration_seconds=total, peak_memory_mb=peak_mb, preprocessing=preprocessing,
            warnings=self.spec.warnings + loaded.warnings, reference=reference,
        )

    def _forward(self, tensor: torch.Tensor):
        cuda = self.device.type == "cuda"
        x = tensor.to(self.device)
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with _deterministic_mode(self.config.deterministic), torch.inference_mode():  # no autograd graph, weights untouched
            logits = self.model(x)
        if cuda:
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if cuda else None
        logits = logits.detach().float().cpu()
        del x
        if tuple(logits.shape) != (1, self.spec.num_classes):
            raise ModelCompatibilityError(f"Model output shape {tuple(logits.shape)} does not match num_classes={self.spec.num_classes}.")
        if not torch.isfinite(logits).all():
            raise ModelCompatibilityError("Model produced non-finite output.")
        return logits, seconds, peak_mb

    def close(self) -> None:
        """Release the model (and CUDA memory) held by this engine."""
        with self._lock:
            self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
