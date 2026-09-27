
import os
from typing import List, Tuple, Dict
from collections import Counter

SPLIT_NAMES = {"train", "val", "validation", "test"}

def infer_labels_and_splits(files: List[str], base_dir: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    classes = set()
    splits = set()
    class_counts = Counter()
    
    base_dir = os.path.abspath(base_dir)
    
    for file_path in files:
        rel_path = os.path.relpath(file_path, base_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 2:
            dir1 = parts[0]
            if dir1.lower() in SPLIT_NAMES:
                splits.add(dir1)
                if len(parts) >= 3:
                    c = parts[1]
                    classes.add(c)
                    class_counts[c] += 1
            else:
                c = dir1
                classes.add(c)
                class_counts[c] += 1
                
    return sorted(list(classes)), sorted(list(splits)), dict(class_counts)

