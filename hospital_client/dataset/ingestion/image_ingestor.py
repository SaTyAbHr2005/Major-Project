
import os
from pathlib import Path
from collections import Counter
from PIL import Image
from hospital_client.dataset.models.dataset_profile import DatasetProfile
from hospital_client.dataset.ingestion.validation import check_file_readability
from hospital_client.dataset.ingestion.label_detection import infer_labels_and_splits
from hospital_client.dataset.ingestion.duplicate_detection import compute_file_hash

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ingest_image_dataset(dataset_path: str) -> DatasetProfile:
    dataset_path = os.path.abspath(dataset_path)
    
    total = 0
    valid = 0
    invalid = 0
    
    corrupted = []
    unsupported = []
    
    widths = []
    heights = []
    modes = Counter()
    
    file_hashes = {}
    duplicates = 0
    
    discovered_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                unsupported.append(file)
                continue
                
            file_path = os.path.join(root, file)
            discovered_files.append(file_path)
            total += 1
            
            is_readable, reason = check_file_readability(file_path)
            if not is_readable:
                invalid += 1
                corrupted.append({"file": file_path, "reason": reason})
                continue
                
            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    w, h = img.size
                    mode = img.mode
                    widths.append(w)
                    heights.append(h)
                    modes[mode] += 1
                    
                valid += 1
                
                file_hash = compute_file_hash(file_path)
                if file_hash:
                    if file_hash in file_hashes:
                        duplicates += 1
                    else:
                        file_hashes[file_hash] = file_path
                        
            except Exception as e:
                invalid += 1
                corrupted.append({"file": file_path, "reason": f"Corrupted or unreadable image: {str(e)}"})
    
    classes, splits, class_dist = infer_labels_and_splits(discovered_files, dataset_path)
    
    image_stats = {
        "formats_found": list(set(os.path.splitext(f)[1].lower() for f in discovered_files)),
        "dimensions": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "avg_width": sum(widths)/len(widths) if widths else 0,
            "avg_height": sum(heights)/len(heights) if heights else 0
        },
        "color_modes": dict(modes)
    }
    
    warnings = []
    if class_dist:
        vals = list(class_dist.values())
        if max(vals) > min(vals) * 2: # simple heuristic for imbalance
            warnings.append("Class distribution is imbalanced.")
            
    if not splits:
        warnings.append("No standard dataset split detected.")
    
    errors = []
    if corrupted: errors.append(f"Corrupted files: {len(corrupted)}")
    if unsupported: errors.append(f"Unsupported files: {len(unsupported)}")
    
    profile = DatasetProfile(
        dataset_type="Image",
        source_path=dataset_path,
        total_samples=total,
        valid_samples=valid,
        invalid_samples=invalid,
        image_statistics=image_stats,
        classes=classes,
        class_distribution=class_dist,
        splits={"detected_splits": splits},
        duplicate_information={"exact_duplicates": duplicates},
        errors=errors,
        warnings=warnings
    )
    return profile

