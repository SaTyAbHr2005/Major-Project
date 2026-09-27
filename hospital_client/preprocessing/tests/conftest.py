import pytest
import os
import tempfile
import json
import pandas as pd
import numpy as np
from PIL import Image

@pytest.fixture
def synthetic_image_dataset():
    temp_dir = tempfile.mkdtemp()
    
    # Create classes
    classes = ["normal", "disease/type1", "disease/type2"]
    counts = [10, 5, 5] # Imbalanced
    
    for cls_name, count in zip(classes, counts):
        cls_path = os.path.join(temp_dir, cls_name)
        os.makedirs(cls_path, exist_ok=True)
        
        for i in range(count):
            img_path = os.path.join(cls_path, f"img_{i}.jpg")
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(img_path)
            
    # Create one corrupt image
    with open(os.path.join(temp_dir, "normal", "corrupt.jpg"), "wb") as f:
        f.write(b"NOT_AN_IMAGE_AT_ALL")
        
    # Create profile
    profile_path = os.path.join(temp_dir, "profile.json")
    with open(profile_path, "w") as f:
        json.dump({
            "dataset_type": "image",
            "source_path": temp_dir,
            "classes": ["normal", "disease"]
        }, f)
        
    yield temp_dir, profile_path
    
@pytest.fixture
def synthetic_tabular_dataset():
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, "data.csv")
    
    df = pd.DataFrame({
        "age": [20, 25, np.nan, 30, 1000], # Outlier 1000
        "category": ["A", "B", "A", None, "C"],
        "target": [0, 1, 0, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    
    profile_path = os.path.join(temp_dir, "profile.json")
    with open(profile_path, "w") as f:
        json.dump({
            "dataset_type": "tabular",
            "source_path": csv_path,
            "tabular_statistics": {"target_column": "target"}
        }, f)
        
    yield csv_path, profile_path

