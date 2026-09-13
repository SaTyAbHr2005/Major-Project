import os
import pytest
import numpy as np
from PIL import Image
from hospital_client.preprocessing.image.quality import assess_image_quality, compute_laplacian_variance
from hospital_client.preprocessing.image.transforms import BaseTransformer
from hospital_client.preprocessing.config import QualityConfig, ImageConfig
from hospital_client.preprocessing.quarantine.manager import QualityStatus

def test_assess_image_quality_not_exist(tmp_path):
    status, msg, meta = assess_image_quality(str(tmp_path / "nonexistent.jpg"), QualityConfig())
    assert status == QualityStatus.ERROR
    
def test_assess_image_quality_zero_byte(tmp_path):
    p = tmp_path / "zero.jpg"
    p.write_bytes(b"")
    status, msg, meta = assess_image_quality(str(p), QualityConfig())
    assert status == QualityStatus.REJECTED

def test_assess_image_quality_unreadable(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"bad_data")
    status, msg, meta = assess_image_quality(str(p), QualityConfig(reject_unreadable=True))
    assert status == QualityStatus.REJECTED
    
    status, msg, meta = assess_image_quality(str(p), QualityConfig(reject_unreadable=False))
    assert status == QualityStatus.ERROR

def test_assess_image_quality_thresholds(tmp_path):
    p = tmp_path / "good.jpg"
    # Create black image (brightness 0)
    img = Image.new("RGB", (100, 100), color="black")
    img.save(p)
    
    config = QualityConfig(brightness_min=10.0, brightness_max=200.0, blur_threshold=100.0)
    status, msg, meta = assess_image_quality(str(p), config)
    # Brightness is 0, < 10.0
    assert status == QualityStatus.LOW_QUALITY
    assert "low brightness" in msg
    
    # White image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(p)
    status, msg, meta = assess_image_quality(str(p), config)
    assert status == QualityStatus.LOW_QUALITY
    assert "high brightness" in msg

def test_compute_laplacian_variance_small():
    # Array too small
    arr = np.zeros((2, 2))
    assert compute_laplacian_variance(arr) == 0.0

def test_base_transformer_fallback(tmp_path):
    # Transform on a bad image returns False
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"bad")
    out = tmp_path / "out.jpg"
    
    config = ImageConfig()
    transformer = BaseTransformer(config)
    assert not transformer.transform(str(p), str(out))

def test_base_transformer_success(tmp_path):
    p = tmp_path / "img.png"
    out = tmp_path / "out.jpg"
    img = Image.new("RGBA", (100, 100), color="blue")
    img.save(p)
    
    config = ImageConfig(color_mode="RGB", target_size=(50, 50))
    transformer = BaseTransformer(config)
    assert transformer.transform(str(p), str(out))
    
    with Image.open(str(out)) as o:
        assert o.mode == "RGB"
        assert o.size == (50, 50)

