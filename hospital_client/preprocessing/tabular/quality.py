import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from hospital_client.preprocessing.config import TabularConfig

class TabularQualityAssessor:
    def __init__(self, config: TabularConfig):
        self.config = config
        
    def assess_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Assess tabular quality. 
        Returns (is_valid, rejection_reasons, metadata)
        """
        if df.empty:
            return False, ["Empty dataframe"], {}
            
        reasons = []
        metadata = {}
        
        # Check overall missingness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 1.0
        metadata["missing_ratio"] = float(missing_ratio)
        
        if missing_ratio > 0.9: # 90% missing across the board is practically useless
            reasons.append("Severe missing data (>90%)")
            
        # Analyze columns
        col_stats = {}
        for col in df.columns:
            if col in self.config.excluded_columns:
                continue
                
            series = df[col]
            missing_col = series.isna().mean()
            
            stats = {"missing_ratio": float(missing_col)}
            
            if pd.api.types.is_numeric_dtype(series):
                # Outlier detection using IQR
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_ratio = len(outliers) / len(series) if len(series) > 0 else 0
                stats["outlier_ratio"] = float(outlier_ratio)
                
            col_stats[col] = stats
            
        metadata["column_stats"] = col_stats
        
        is_valid = len(reasons) == 0
        return is_valid, reasons, metadata

