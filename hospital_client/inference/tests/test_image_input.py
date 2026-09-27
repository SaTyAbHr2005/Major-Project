"""C. Image validation: valid formats load; corrupt/empty/unsupported/unsafe images are rejected safely and never modified."""
import hashlib
import os

import numpy as np
import pytest
from PIL import Image

from hospital_client.inference.errors import ImageValidationError
from hospital_client.inference.image_input import load_validated_image
from hospital_client.inference.tests.conftest import make_image


@pytest.mark.parametrize("name, fmt", [
    ("a.jpg", "JPEG"), ("a.jpeg", "JPEG"), ("a.png", "PNG"), ("a.bmp", "BMP"),
    ("a.tif", "TIFF"), ("a.tiff", "TIFF"), ("a.webp", "WEBP"),
])
def test_every_supported_format_loads(tmp_path, name, fmt):
    path = make_image(tmp_path / name, size=(37, 29), fmt=fmt)
    loaded = load_validated_image(path)
    assert loaded.source_format == fmt and loaded.source_size == (37, 29)
    assert loaded.size_bytes == os.path.getsize(path) and loaded.warnings == []


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "1", "CMYK"])
def test_safe_source_modes_accepted(tmp_path, mode):
    fmt = "TIFF" if mode == "CMYK" else "PNG"
    ext = ".tif" if mode == "CMYK" else ".png"
    loaded = load_validated_image(make_image(tmp_path / f"m{ext}", mode=mode, fmt=fmt))
    assert loaded.source_mode == mode


def test_high_bit_depth_image_rejected_rather_than_silently_clipped(tmp_path):
    path = tmp_path / "scan16.tif"
    Image.fromarray(np.full((16, 16), 40000, dtype=np.uint16)).save(str(path))
    with pytest.raises(ImageValidationError, match="cannot be converted safely"):
        load_validated_image(str(path))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ImageValidationError, match="does not exist"):
        load_validated_image(str(tmp_path / "ghost.png"))


def test_zero_byte_file_rejected(tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ImageValidationError, match="zero-byte"):
        load_validated_image(str(path))


def test_directory_rejected(tmp_path):
    with pytest.raises(ImageValidationError, match="directory"):
        load_validated_image(str(tmp_path))


@pytest.mark.parametrize("bad", [None, "", 123, "a\x00b.png"])
def test_invalid_path_values_rejected(bad):
    with pytest.raises(ImageValidationError):
        load_validated_image(bad)


@pytest.mark.parametrize("name", ["x.gif", "x.txt", "x.pdf", "x"])
def test_unsupported_extension_rejected(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"some bytes")
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        load_validated_image(str(path))


def test_non_image_bytes_with_valid_extension_rejected(tmp_path):
    path = tmp_path / "fake.png"
    path.write_bytes(b"this is definitely not an image" * 10)
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        load_validated_image(str(path))


def test_corrupted_image_data_rejected(tmp_path):
    good = make_image(tmp_path / "good.png", size=(64, 64))
    data = open(good, "rb").read()
    path = tmp_path / "corrupt.png"
    path.write_bytes(data[: len(data) // 2])  # truncated pixel data
    with pytest.raises(ImageValidationError, match="could not be decoded"):
        load_validated_image(str(path))
    header_only = tmp_path / "header_damaged.png"
    header_only.write_bytes(b"\x00" * 16 + data[16:])
    with pytest.raises(ImageValidationError):
        load_validated_image(str(header_only))


def test_gif_content_disguised_as_png_rejected(tmp_path):
    path = tmp_path / "disguised.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(str(path), format="GIF")
    with pytest.raises(ImageValidationError, match="Unsupported image content format"):
        load_validated_image(str(path))


def test_extension_content_mismatch_is_allowed_with_a_warning(tmp_path):
    path = tmp_path / "actually_jpeg.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(str(path), format="JPEG")
    loaded = load_validated_image(str(path))
    assert loaded.source_format == "JPEG" and any("extension suggests" in w for w in loaded.warnings)


def test_pixel_and_byte_limits_enforced(tmp_path):
    path = make_image(tmp_path / "big.png", size=(50, 40))
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(path, max_pixels=1000)
    with pytest.raises(ImageValidationError, match="larger than"):
        load_validated_image(path, max_bytes=10)


def test_decompression_bomb_rejected(tmp_path, monkeypatch):
    path = make_image(tmp_path / "bomb.png", size=(50, 40))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # PIL warns above 100px and errors above 200px
    with pytest.raises(ImageValidationError):
        load_validated_image(path)


def test_multi_frame_image_uses_first_frame_with_warning(tmp_path):
    path = tmp_path / "multi.tif"
    frames = [Image.new("RGB", (10, 10), c) for c in [(255, 0, 0), (0, 255, 0)]]
    frames[0].save(str(path), save_all=True, append_images=frames[1:])
    loaded = load_validated_image(str(path))
    assert any("first frame" in w for w in loaded.warnings)
    assert loaded.image.getpixel((0, 0))[:3] == (255, 0, 0)


def test_loading_never_modifies_or_creates_files(tmp_path):
    path = make_image(tmp_path / "keep.png")
    before_bytes = hashlib.sha256(open(path, "rb").read()).hexdigest()
    before_stat = os.stat(path)
    before_listing = sorted(os.listdir(tmp_path))
    load_validated_image(path)
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == before_bytes
    assert os.stat(path).st_mtime_ns == before_stat.st_mtime_ns
    assert sorted(os.listdir(tmp_path)) == before_listing


def test_error_messages_never_contain_the_path_or_filename(tmp_path):
    secret = "John_Doe_MRN123456_chest"
    cases = []
    cases.append(str(tmp_path / f"{secret}.png"))  # missing
    empty = tmp_path / f"{secret}_empty.png"; empty.write_bytes(b""); cases.append(str(empty))
    junk = tmp_path / f"{secret}_junk.png"; junk.write_bytes(b"junk" * 50); cases.append(str(junk))
    wrong_ext = tmp_path / f"{secret}.gif"; wrong_ext.write_bytes(b"x"); cases.append(str(wrong_ext))
    for case in cases:
        with pytest.raises(ImageValidationError) as info:
            load_validated_image(case)
        assert secret not in str(info.value) and str(tmp_path) not in str(info.value)


def test_smallest_valid_image_loads(tmp_path):
    assert load_validated_image(make_image(tmp_path / "one.png", size=(1, 1))).source_size == (1, 1)
