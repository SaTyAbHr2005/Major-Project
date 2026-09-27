import os
import json
import pytest
import pandas as pd
import numpy as np
from PIL import Image
from unittest.mock import patch
from hospital_client.preprocessing.engine import PreprocessingEngine
from hospital_client.preprocessing.config import PreprocessingConfig, OutputMode

@pytest.fixture
def dummy_profile(tmp_path):
    prof = tmp_path / "profile.json"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # create some images
    (data_dir / "classA").mkdir()
    (data_dir / "classB").mkdir()
    imgA = Image.new("RGB", (10, 10))
    imgA.save(data_dir / "classA" / "1.jpg")
    imgB = Image.new("RGB", (10, 10))
    imgB.save(data_dir / "classB" / "2.jpg")
    
    with open(prof, "w") as f:
        json.dump({
            "dataset_type": "image",
            "source_path": str(data_dir),
            "target_column": "label",
            # No real split folders exist in this flat classA/classB layout,
            # so no split is declared here (an inconsistent "splits" claim
            # not matching the actual layout is now caught as AMBIGUOUS by
            # split_validation.py, which this fixture must not trigger).
            "splits": {}
        }, f)
    return prof

def test_lazy_image_pipeline(dummy_profile, tmp_path):
    config = PreprocessingConfig()
    config.dataset.output_mode = OutputMode.LAZY
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    engine.run(str(dummy_profile))
    assert os.path.exists(tmp_path / "out" / "train_manifest.json")

def test_materialized_tabular_pipeline(tmp_path):
    prof = tmp_path / "profile.json"
    data = tmp_path / "data.csv"
    pd.DataFrame({"col1": [1, 2, 3], "label": [0, 1, 0]}).to_csv(data, index=False)
    
    with open(prof, "w") as f:
        json.dump({
            "dataset_type": "tabular",
            "source_path": str(data),
            "target_column": "label",
            "splits": {"train": [0, 1, 2]}
        }, f)
        
    config = PreprocessingConfig()
    config.dataset.output_mode = OutputMode.MATERIALIZED
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    engine.run(str(prof))
    
    assert os.path.exists(tmp_path / "out" / "train.csv")
    assert os.path.exists(tmp_path / "out" / "train_manifest.json")

def test_engine_unknown_type(tmp_path):
    p = tmp_path / "prof.json"
    with open(p, "w") as f:
        json.dump({"dataset_type": "magic"}, f)
    config = PreprocessingConfig()
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    with pytest.raises(ValueError, match="Unknown dataset type"):
        engine.run(str(p))

def test_engine_missing_source(tmp_path):
    p = tmp_path / "prof.json"
    with open(p, "w") as f:
        json.dump({"dataset_type": "image", "source_path": "does_not_exist"}, f)
    config = PreprocessingConfig()
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    with pytest.raises(FileNotFoundError):
        engine.run(str(p))
        
def test_engine_tabular_missing_source(tmp_path):
    p = tmp_path / "prof.json"
    with open(p, "w") as f:
        json.dump({"dataset_type": "tabular", "source_path": "does_not_exist"}, f)
    config = PreprocessingConfig()
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    with pytest.raises(FileNotFoundError):
        engine.run(str(p))

def test_engine_tabular_quality_rejection(tmp_path):
    p = tmp_path / "prof.json"
    csv = tmp_path / "data.csv"
    df = pd.DataFrame({"col": [np.nan, np.nan]})
    df.to_csv(csv, index=False)
    
    with open(p, "w") as f:
        json.dump({"dataset_type": "tabular", "source_path": str(csv), "target_column": "col"}, f)
        
    config = PreprocessingConfig()
    config.dataset.output_path = str(tmp_path / "out")
    engine = PreprocessingEngine(config)
    with pytest.raises(ValueError, match="Tabular dataset rejected"):
        engine.run(str(p))

