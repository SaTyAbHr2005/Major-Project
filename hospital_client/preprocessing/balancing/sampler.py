import numpy as np
from typing import Dict, Any, Tuple
from hospital_client.preprocessing.config import ClassConfig

class ClassBalancer:
    def __init__(self, config: ClassConfig):
        self.config = config
        
    def compute_training_weights(self, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Computes sampling weights purely based on the training data.
        Returns a metadata dictionary containing class counts and calculated weights.
        """
        if len(y_train) == 0:
            return {}
            
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))
        
        metadata = {
            "strategy": self.config.imbalance_strategy,
            "training_counts": class_counts
        }
        
        if self.config.imbalance_strategy == "none":
            return metadata
            
        # Compute inverse frequency weights
        total_samples = len(y_train)
        n_classes = len(unique)
        
        # Default sklearn formulation: n_samples / (n_classes * np.bincount(y))
        # Here we just map class -> weight
        weights = {}
        for cls, count in zip(unique, counts):
            if count > 0:
                weights[str(cls)] = total_samples / (n_classes * count)
            else:
                weights[str(cls)] = 0.0
                
        metadata["class_weights"] = weights
        
        if self.config.imbalance_strategy == "oversample":
            max_count = np.max(counts)
            oversample_ratios = {str(k): float(max_count/v) for k, v in class_counts.items()}
            metadata["oversample_ratios"] = oversample_ratios
            
        elif self.config.imbalance_strategy == "undersample":
            min_count = np.min(counts)
            undersample_ratios = {str(k): float(min_count/v) for k, v in class_counts.items()}
            metadata["undersample_ratios"] = undersample_ratios
            
        return metadata

