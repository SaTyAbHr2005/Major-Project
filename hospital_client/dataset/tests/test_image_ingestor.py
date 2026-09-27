
import os
from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest

def test_image_formats_and_corruption(comprehensive_image_dataset):
    formats_dir = os.path.join(comprehensive_image_dataset, "formats")
    profile = ingest_image_dataset(formats_dir)
    
    assert profile.dataset_type == "Image"
    assert profile.total_samples == 8 # 7 images + 1 corrupt
    assert profile.valid_samples == 7
    assert profile.invalid_samples == 1
    
    stats = profile.image_statistics
    formats = set(stats["formats_found"])
    assert {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}.issubset(formats)
    
    assert "L" in stats["color_modes"]
    assert "RGB" in stats["color_modes"]
    assert "RGBA" in stats["color_modes"]
    
    assert stats["dimensions"]["min_width"] == 10
    assert stats["dimensions"]["max_height"] == 10
    
    assert any("Unsupported files" in err for err in profile.errors)
    assert any("Corrupted files" in err for err in profile.errors)

def test_splits_and_classes_a(comprehensive_image_dataset):
    dir_a = os.path.join(comprehensive_image_dataset, "split_a")
    profile = ingest_image_dataset(dir_a)
    assert "benign" in profile.classes
    assert "malignant" in profile.classes
    assert not profile.splits["detected_splits"] # No splits

def test_splits_and_classes_b(comprehensive_image_dataset):
    dir_b = os.path.join(comprehensive_image_dataset, "split_b")
    profile = ingest_image_dataset(dir_b)
    assert set(profile.splits["detected_splits"]) == {"train", "test"} # Train and test only
    assert set(profile.classes) == {"benign", "malignant"}

def test_imbalance_warning(comprehensive_image_dataset):
    dir_c = os.path.join(comprehensive_image_dataset, "split_c")
    profile = ingest_image_dataset(dir_c)
    assert "Class distribution is imbalanced." in profile.warnings

