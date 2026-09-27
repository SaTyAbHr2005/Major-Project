
import os
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest
from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash

def test_immutability_tabular(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "valid.csv")
    
    hash_before = compute_file_hash(csv_path)
    detect_and_ingest(csv_path)
    hash_after = compute_file_hash(csv_path)
    
    assert hash_before == hash_after

def test_immutability_image(comprehensive_image_dataset):
    img_path = os.path.join(comprehensive_image_dataset, "formats", "img1.jpg")
    
    hash_before = compute_file_hash(img_path)
    detect_and_ingest(comprehensive_image_dataset)
    hash_after = compute_file_hash(img_path)
    
    assert hash_before == hash_after

