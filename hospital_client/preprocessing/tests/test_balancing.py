import numpy as np
from hospital_client.preprocessing.balancing.sampler import ClassBalancer
from hospital_client.preprocessing.config import ClassConfig

def test_balancer_none():
    config = ClassConfig(imbalance_strategy="none")
    balancer = ClassBalancer(config)
    y_train = np.array([0, 0, 1])
    meta = balancer.compute_training_weights(y_train)
    assert meta["strategy"] == "none"
    assert "class_weights" not in meta

def test_balancer_oversample():
    config = ClassConfig(imbalance_strategy="oversample")
    balancer = ClassBalancer(config)
    y_train = np.array([0, 0, 1])
    meta = balancer.compute_training_weights(y_train)
    assert "class_weights" in meta
    assert meta["oversample_ratios"]["0"] == 1.0
    assert meta["oversample_ratios"]["1"] == 2.0
    
def test_balancer_undersample():
    config = ClassConfig(imbalance_strategy="undersample")
    balancer = ClassBalancer(config)
    y_train = np.array([0, 0, 1])
    meta = balancer.compute_training_weights(y_train)
    assert meta["undersample_ratios"]["0"] == 0.5
    assert meta["undersample_ratios"]["1"] == 1.0

def test_balancer_empty():
    config = ClassConfig()
    balancer = ClassBalancer(config)
    meta = balancer.compute_training_weights(np.array([]))
    assert meta == {}

