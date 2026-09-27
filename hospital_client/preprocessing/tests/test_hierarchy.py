import pytest
import pandas as pd
from hospital_client.preprocessing.labels.hierarchy import LabelResolver
from hospital_client.preprocessing.config import ClassConfig
from hospital_client.preprocessing.quarantine.manager import QuarantineManager, QualityStatus
from hospital_client.preprocessing.splitting.grouping import extract_groups

def test_label_resolver():
    config = ClassConfig(
        hierarchy_mapping={"disease/type1": "disease_mapped", "parent": "parent_mapped"},
        preserve_subtypes=True
    )
    resolver = LabelResolver(config)
    
    # Empty parts
    assert resolver.resolve_label("", []) == (None, None, None)
    
    # Just split
    assert resolver.resolve_label("train/img.jpg", ["train"]) == ("train", None, None)
    
    # Subtype mapping with preserve
    s, p, sub = resolver.resolve_label("train/disease/type1/img.jpg", ["train"])
    assert s == "train"
    assert p == "disease_mapped"
    assert sub == "disease/type1"
    
    # Subtype mapping without preserve
    config.preserve_subtypes = False
    s, p, sub = resolver.resolve_label("train/disease/type1/img.jpg", ["train"])
    assert p == "disease_mapped"
    assert sub is None
    
    # Parent mapping
    s, p, sub = resolver.resolve_label("parent/img.jpg", [])
    assert p == "parent_mapped"
    assert sub is None

def test_quarantine_manager_low_quality():
    qm = QuarantineManager()
    qm.add_sample("a.jpg", QualityStatus.LOW_QUALITY)
    assert qm.summary()["review"] == 1
    assert "a.jpg" not in qm.get_accepted_paths()

def test_extract_groups():
    df = pd.DataFrame({"pid": [1, 1, 2]})
    groups = extract_groups(df, "pid")
    assert list(groups) == [1, 1, 2]
    
    with pytest.raises(ValueError):
        extract_groups(df, "missing")

