import pytest
import numpy as np
import pandas as pd
from hospital_client.preprocessing.splitting.splitter import DatasetSplitter
from hospital_client.preprocessing.config import SplitConfig, SplitStrategy

def test_stratified_split():
    config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, random_seed=42)
    splitter = DatasetSplitter(config)
    y = np.array([0]*50 + [1]*50)
    train_idx, val_idx, test_idx = splitter.split_indices(y)
    
    assert len(train_idx) == 80
    assert len(val_idx) == 10
    assert len(test_idx) == 10
    
    # Check stratification
    assert sum(y[train_idx]) == 40
    assert sum(y[val_idx]) == 5
    assert sum(y[test_idx]) == 5

def test_random_split_fallback():
    # If a class has too few members, it should fallback to random splitting without crashing
    config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, random_seed=42)
    splitter = DatasetSplitter(config)
    
    # 2 classes, but one class has only 1 sample
    y = np.array([0]*99 + [1]*1)
    train_idx, val_idx, test_idx = splitter.split_indices(y)
    
    assert len(train_idx) == 80
    assert len(val_idx) == 10
    assert len(test_idx) == 10

def test_grouped_split():
    config = SplitConfig(train_ratio=0.5, validation_ratio=0.5, test_ratio=0.0, strategy=SplitStrategy.GROUPED, random_seed=42)
    splitter = DatasetSplitter(config)
    y = np.array([0, 0, 1, 1])
    groups = np.array([1, 1, 2, 2])
    
    train, val, test = splitter.split_indices(y, groups=groups)
    assert len(test) == 0
    # GroupShuffleSplit should keep groups intact
    train_groups = groups[train]
    val_groups = groups[val]
    assert len(set(train_groups).intersection(set(val_groups))) == 0

def test_grouped_split_missing_groups():
    config = SplitConfig(strategy=SplitStrategy.GROUPED, train_ratio=0.5, validation_ratio=0.5, test_ratio=0.0)
    splitter = DatasetSplitter(config)
    with pytest.raises(ValueError, match="Grouped split requested"):
        # Make it large enough so it actually tries to split
        splitter.split_indices(np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        
def test_random_split_explicit():
    config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, strategy=SplitStrategy.RANDOM, random_seed=42)
    splitter = DatasetSplitter(config)
    y = np.array([0]*100)
    train, val, test = splitter.split_indices(y)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    
def test_zero_samples():
    config = SplitConfig()
    splitter = DatasetSplitter(config)
    tr, v, te = splitter.split_indices(np.array([]))
    assert len(tr) == 0
    
def test_zero_val_test_ratio():
    config = SplitConfig(train_ratio=1.0, validation_ratio=0.0, test_ratio=0.0)
    splitter = DatasetSplitter(config)
    y = np.array([1]*10)
    tr, v, te = splitter.split_indices(y)
    assert len(tr) == 10
    assert len(v) == 0
    assert len(te) == 0
    
def test_zero_test_ratio():
    config = SplitConfig(train_ratio=0.8, validation_ratio=0.2, test_ratio=0.0)
    splitter = DatasetSplitter(config)
    y = np.array([1]*10)
    tr, v, te = splitter.split_indices(y)
    assert len(tr) == 8
    assert len(v) == 2
    assert len(te) == 0
    
def test_zero_val_ratio():
    config = SplitConfig(train_ratio=0.8, validation_ratio=0.0, test_ratio=0.2)
    splitter = DatasetSplitter(config)
    y = np.array([1]*10)
    tr, v, te = splitter.split_indices(y)
    assert len(tr) == 8
    assert len(v) == 0
    assert len(te) == 2

