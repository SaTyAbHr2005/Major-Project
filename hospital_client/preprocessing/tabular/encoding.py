import pandas as pd
import numpy as np
from typing import Dict, Any, List
from hospital_client.preprocessing.config import TabularConfig

class EncoderFitter:
    def __init__(self, config: TabularConfig):
        self.config = config
        self.scaling_params: Dict[str, Dict[str, float]] = {}
        self.categorical_maps: Dict[str, Dict[str, int]] = {}
        
    def fit(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits scalers and encoders STRICTLY on training data to prevent leakage.
        """
        if df_train.empty:
            return {}
            
        for col in df_train.columns:
            if col in self.config.excluded_columns:
                continue
                
            series = df_train[col]
            if pd.api.types.is_numeric_dtype(series):
                if self.config.numerical_scaling == "standard":
                    mean = float(series.mean())
                    std = float(series.std())
                    if std == 0 or np.isnan(std):
                        std = 1.0
                    if np.isnan(mean):
                        mean = 0.0
                    self.scaling_params[col] = {"mean": mean, "std": std}
                elif self.config.numerical_scaling == "minmax":
                    cmin = float(series.min())
                    cmax = float(series.max())
                    if np.isnan(cmin): cmin = 0.0
                    if np.isnan(cmax): cmax = 1.0
                    if cmax == cmin: cmax = cmin + 1.0
                    self.scaling_params[col] = {"min": cmin, "max": cmax}
            else:
                # Categorical
                if self.config.categorical_encoding == "label":
                    unique_vals = sorted([str(x) for x in series.unique() if pd.notna(x)])
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    self.categorical_maps[col] = mapping
                    
        return {
            "scaling": self.scaling_params,
            "encoding": self.categorical_maps
        }
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted scaling and encoding to a dataframe (train, val, or test).
        """
        df_out = df.copy()
        
        # Apply scaling
        for col, params in self.scaling_params.items():
            if col in df_out.columns:
                if self.config.numerical_scaling == "standard":
                    df_out[col] = (df_out[col] - params["mean"]) / params["std"]
                elif self.config.numerical_scaling == "minmax":
                    df_out[col] = (df_out[col] - params["min"]) / (params["max"] - params["min"])
                    
        # Apply categorical
        for col, mapping in self.categorical_maps.items():
            if col in df_out.columns:
                # Unseen categories will become NaN, which should be handled by missing value strategy later or dropped
                df_out[col] = df_out[col].astype(str).map(mapping)
                
        return df_out

