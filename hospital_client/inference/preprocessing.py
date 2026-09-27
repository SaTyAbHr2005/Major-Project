"""
Inference preprocessing = the SAME transform the model was trained with.

Module 7 trains and evaluates on `hospital_client.training.dataset.build_transform`
(convert to color_mode -> torchvision Resize(input_size) -> ToTensor, no
normalization, no augmentation). That function is reused directly - there is
no second implementation of it here.

Module 5's `ImageConfig.normalization` is never applied by Module 5 or Module 7,
so any non-null value is rejected instead of being silently ignored or invented.

`emulate_module5_materialization=True` additionally reproduces Module 5's
materialized-output step (`BaseTransformer`: optional color-mode convert, then
LANCZOS resize to `ImageConfig.target_size`, then the lossy JPEG/WEBP re-encode that
Pillow's default `save()` performs) for models that were trained on
Module 5 materialized images. Module 5's transformer only works file->file, so
its (3-line) logic is mirrored in memory; tests/test_preprocessing.py proves the
result equals Module 5's real BaseTransformer output fed through Module 7.
"""
import io
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

from hospital_client.inference.errors import ImageValidationError, ModelCompatibilityError
from hospital_client.inference.image_input import LoadedImage
from hospital_client.inference.model_spec import UNAVAILABLE, InferenceModelSpec
from hospital_client.training.dataset import build_transform

PIPELINE_ID = "hospital_client.training.dataset.build_transform (convert -> resize -> to_tensor)"


LOSSY_REENCODED_FORMATS = ("JPEG", "WEBP")  # Module 5's BaseTransformer saves with Pillow defaults, which are lossy for these


def _module5_reencode(img: Image.Image, fmt: str) -> Image.Image:
    """Reproduce Module 5's `img.save(output_path)` round trip for lossy formats, in memory."""
    buffer = io.BytesIO()
    try:
        img.save(buffer, format=fmt)
    except Exception:
        raise ImageValidationError(
            f"Image cannot be re-encoded as {fmt} the way Module 5 materialization would (Module 5 would have rejected it)."
        ) from None
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        reopened.load()
        return reopened.copy()


class InferencePreprocessor:
    def __init__(self, spec: InferenceModelSpec, image_config: Any = None, emulate_module5_materialization: Optional[bool] = None):
        self.spec = spec
        self.image_config = image_config
        upstream = (spec.preprocessing_spec or {}).get("upstream")
        self.emulation_source = "explicit"
        if emulate_module5_materialization is None:
            # Auto: follow what Module 5 recorded at training time (never guessed when nothing was recorded).
            emulate_module5_materialization = bool(upstream and upstream.get("output_mode") == "materialized")
            self.emulation_source = "recorded_upstream" if upstream else "default_off"
            if emulate_module5_materialization and image_config is None:
                image_config = SimpleNamespace(
                    color_mode=upstream.get("color_mode"),
                    target_size=tuple(upstream["target_size"]) if upstream.get("target_size") else None,
                    normalization=None,
                )
        self.image_config = image_config
        self.emulate_module5_materialization = emulate_module5_materialization

        normalization = getattr(image_config, "normalization", None)
        if normalization is not None:
            raise ModelCompatibilityError(
                f"Preprocessing ImageConfig requests normalization '{normalization}', but no Module 5/7 "
                "code applies normalization and the model was not trained with it. Inference is refused "
                "rather than silently ignoring or inventing a normalization."
            )
        if emulate_module5_materialization and image_config is None:
            raise ModelCompatibilityError("emulate_module5_materialization=True requires the Module 5 ImageConfig.")

        self._transform = build_transform(spec.color_mode, spec.input_size, None)  # deterministic: never augmented

    def prepare(self, loaded: LoadedImage) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img = loaded.image
        conversions = []
        if self.emulate_module5_materialization:
            cfg = self.image_config
            if cfg.color_mode and img.mode != cfg.color_mode:
                img = img.convert(cfg.color_mode)
                conversions.append(f"module5_color_mode:{cfg.color_mode}")
            if cfg.target_size:
                img = img.resize(tuple(cfg.target_size), Image.Resampling.LANCZOS)
                conversions.append(f"module5_lanczos_resize:{tuple(cfg.target_size)}")
            if loaded.source_format in LOSSY_REENCODED_FORMATS:
                img = _module5_reencode(img, loaded.source_format)
                conversions.append(f"module5_{loaded.source_format.lower()}_reencode")
        if img.mode != self.spec.color_mode:
            conversions.append(f"{img.mode}->{self.spec.color_mode}")
        img = img.convert(self.spec.color_mode)

        tensor = self._transform(img).unsqueeze(0)
        channels = 3 if self.spec.color_mode == "RGB" else 1
        expected = (1, channels, self.spec.input_size[0], self.spec.input_size[1])
        if tuple(tensor.shape) != expected or tensor.dtype != torch.float32:
            raise ImageValidationError(f"Preprocessed tensor {tuple(tensor.shape)}/{tensor.dtype} does not match the model input {expected}.")
        if not torch.isfinite(tensor).all():
            raise ImageValidationError("Preprocessed image contains non-finite values.")

        metadata = {
            "pipeline": PIPELINE_ID,
            "input_size": list(self.spec.input_size),
            "color_mode": self.spec.color_mode,
            "normalization": "none (ToTensor 0-1 scaling only)",
            "preprocessing_spec_version": (self.spec.preprocessing_spec or {}).get("version", UNAVAILABLE),
            "preprocessing_spec_verified": self.spec.preprocessing_spec is not None,
            "resize_interpolation": "bilinear (torchvision Resize default)",
            "module5_materialization_emulated": self.emulate_module5_materialization,
            "module5_emulation_source": self.emulation_source,
            "preprocessing_reference": self.spec.preprocessing_reference or UNAVAILABLE,
            "source_format": loaded.source_format,
            "source_mode": loaded.source_mode,
            "source_size": list(loaded.source_size),
            "conversions": conversions,
            "tensor_shape": list(tensor.shape),
        }
        return tensor, metadata
