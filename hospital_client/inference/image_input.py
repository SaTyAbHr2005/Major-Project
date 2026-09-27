"""
Validated, read-only loading of a NEW local image. Reuses Module 4's
supported-extension list and file-readability check. The image is decoded into
memory only - it is never written, copied, moved, or modified, and no error
message contains its path or filename (both can carry patient identifiers).
"""
import io
import os
import warnings
from dataclasses import dataclass, field
from typing import List

from PIL import Image

from hospital_client.dataset.ingestion.image_ingestor import SUPPORTED_EXTENSIONS
from hospital_client.dataset.ingestion.validation import check_file_readability
from hospital_client.inference.errors import ImageValidationError

# Modes whose conversion to RGB/L behaves like Module 7's training conversion.
# High-bit-depth/float modes (I, I;16, F) are rejected: PIL's convert() clips
# their values, which would silently differ from an 8-bit training image.
SAFE_SOURCE_MODES = {"1", "L", "LA", "P", "PA", "RGB", "RGBA", "CMYK", "YCbCr"}
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}
DEFAULT_MAX_IMAGE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 100_000_000


@dataclass
class LoadedImage:
    image: Image.Image
    source_format: str
    source_mode: str
    source_size: tuple  # (width, height)
    size_bytes: int
    warnings: List[str] = field(default_factory=list)


def load_validated_image(
    path: str, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> LoadedImage:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ImageValidationError("Image path must be a non-empty string.")
    real = os.path.realpath(path)
    if os.path.isdir(real):
        raise ImageValidationError("Image path refers to a directory, not a file.")

    ok, reason = check_file_readability(real)  # exists / readable / non-zero-byte (Module 4)
    if not ok:
        raise ImageValidationError(f"Image file is not usable: {reason.lower()}.")
    ext = os.path.splitext(real)[1].lower()
    if ext in (".dcm", ".dicom"):
        raise ImageValidationError("DICOM is not supported by Module 16 (standard 8-bit image formats only).")
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(f"Unsupported image format '{ext or '(no extension)'}'; supported: {sorted(SUPPORTED_EXTENSIONS)}.")

    # The file is opened and read exactly once; every later check and the decode use these bytes,
    # so the file cannot change between validation and use (no check-then-use gap).
    try:
        with open(real, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        raise ImageValidationError("Image file could not be read.") from None
    size_bytes = len(data)
    if size_bytes > max_bytes:
        raise ImageValidationError(f"Image file is larger than the configured limit ({max_bytes} bytes).")
    if size_bytes == 0:
        raise ImageValidationError("Image file is not usable: zero-byte file.")

    notes: List[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                probe.verify()  # header / structural integrity (invalidates this handle)
            with Image.open(io.BytesIO(data)) as img:
                width, height = img.size
                if width * height > max_pixels:  # PIL itself refuses zero-sized images at open()
                    raise ImageValidationError(f"Image exceeds the configured pixel limit ({max_pixels}).")
                fmt, mode = img.format, img.mode
                if fmt not in ALLOWED_FORMATS:
                    raise ImageValidationError(f"Unsupported image content format '{fmt}'.")
                if mode not in SAFE_SOURCE_MODES:
                    raise ImageValidationError(
                        f"Image mode '{mode}' cannot be converted safely to the model's input mode "
                        "(high-bit-depth/float images would be silently clipped)."
                    )
                if getattr(img, "n_frames", 1) > 1:
                    notes.append("Multi-frame image: only the first frame is used.")
                img.load()  # full decode - surfaces truncated/corrupt pixel data
                decoded = img.copy()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError("Image exceeds the decompression safety limit.") from None
    except Exception as e:  # PIL raises many types for corrupt files (OSError, SyntaxError, ValueError, ...)
        raise ImageValidationError(f"Image could not be decoded ({type(e).__name__}).") from None

    expected_by_ext = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP"}
    if expected_by_ext.get(ext) != fmt:
        notes.append(f"File extension suggests {expected_by_ext.get(ext)} but the content is {fmt}.")

    return LoadedImage(image=decoded, source_format=fmt, source_mode=mode, source_size=(width, height),
                       size_bytes=size_bytes, warnings=notes)
