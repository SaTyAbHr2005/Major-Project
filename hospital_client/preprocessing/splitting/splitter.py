import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from hospital_client.preprocessing.config import SplitConfig, SplitStrategy

class DatasetSplitter:
    def __init__(self, config: SplitConfig):
        self.config = config
        
    def split_indices(self, y: np.ndarray, groups: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (train_idx, val_idx, test_idx).
        Raises ValueError if grouped strategy but no groups provided.
        """
        total = len(y)
        if total == 0:
            return np.array([]), np.array([]), np.array([])
            
        train_ratio = self.config.train_ratio
        val_ratio = self.config.validation_ratio
        test_ratio = self.config.test_ratio
        
        # Normalize just in case
        ratio_sum = train_ratio + val_ratio + test_ratio
        train_ratio /= ratio_sum
        val_ratio /= ratio_sum
        test_ratio /= ratio_sum
        
        val_test_ratio = val_ratio + test_ratio
        if val_test_ratio == 0.0:
            return np.arange(total), np.array([]), np.array([])
            
        val_relative_ratio = val_ratio / val_test_ratio
        
        # First split: Train vs (Val+Test)
        train_idx, temp_idx = self._do_split(
            n_samples=total, 
            test_size=val_test_ratio, 
            y=y, 
            groups=groups
        )
        
        # If no validation or test set is needed
        if len(temp_idx) == 0:
            return train_idx, np.array([]), np.array([])
            
        if val_ratio == 0.0:
            return train_idx, np.array([]), temp_idx
        if test_ratio == 0.0:
            return train_idx, temp_idx, np.array([])
            
        # Second split: Val vs Test
        temp_y = y[temp_idx]
        temp_groups = groups[temp_idx] if groups is not None else None
        
        val_idx_rel, test_idx_rel = self._do_split(
            n_samples=len(temp_idx),
            test_size=(1.0 - val_relative_ratio),
            y=temp_y,
            groups=temp_groups
        )
        
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
        
        return train_idx, val_idx, test_idx
        
    def _do_split(self, n_samples: int, test_size: float, y: np.ndarray, groups: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        # Handle edge case where test_size would result in 0 samples or all samples
        n_test = int(np.round(n_samples * test_size))
        if n_test == 0:
            return np.arange(n_samples), np.array([], dtype=int)
        if n_test == n_samples:
            return np.array([], dtype=int), np.arange(n_samples)
            
        if self.config.strategy == SplitStrategy.GROUPED:
            if groups is None:
                raise ValueError("Grouped split requested but no groups provided.")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.config.random_seed)
            return next(gss.split(np.zeros(n_samples), y, groups))
            
        elif self.config.strategy == SplitStrategy.STRATIFIED:
            # Check if all classes have at least 2 samples for stratification
            unique, counts = np.unique(y, return_counts=True)
            if np.any(counts < 2) or n_test < len(unique) or (n_samples - n_test) < len(unique):
                # Fallback to random split if stratification impossible
                return self._random_split(n_samples, test_size)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.config.random_seed)
            return next(sss.split(np.zeros(n_samples), y))
            
        else: # Random
            return self._random_split(n_samples, test_size)
            
    def _random_split(self, n_samples: int, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(n_samples)
        n_test = int(np.round(n_samples * test_size))
        return indices[n_test:], indices[:n_test]

