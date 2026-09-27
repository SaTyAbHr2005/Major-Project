
import os
import shutil
from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash

def test_duplicate_image_hashing(comprehensive_image_dataset):
    formats_dir = os.path.join(comprehensive_image_dataset, "formats")
    img1 = os.path.join(formats_dir, "img1.jpg")
    img2 = os.path.join(formats_dir, "img1_dup.jpg")
    img3 = os.path.join(formats_dir, "img2.jpeg")
    
    # Create an exact duplicate
    shutil.copy(img1, img2)
    
    hash1 = compute_file_hash(img1)
    hash2 = compute_file_hash(img2)
    hash3 = compute_file_hash(img3)
    
    assert hash1 == hash2
    assert hash1 != hash3

