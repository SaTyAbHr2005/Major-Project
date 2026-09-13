import pandas as pd
import numpy as np
from typing import Dict, Any, List
from hospital_client.preprocessing.config import TabularConfig, TabularImputeStrategy

class ImputationFitter:
    def __init__(self, config: TabularConfig):
        self.config = config
        self.imputation_values: Dict[str, Any] = {}
        
    def fit(self, df_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits imputation values STRICTLY on training data to prevent leakage.
        """
        if df_train.empty:
            return {}
            
        for col in df_train.columns:
            if col in self.config.excluded_columns:
                continue
                
            series = df_train[col]
            if pd.api.types.is_numeric_dtype(series):
                strat = self.config.missing_numeric_strategy
                if strat == TabularImputeStrategy.MEAN:
                    self.imputation_values[col] = float(series.mean(skipna=True))
                elif strat == TabularImputeStrategy.MEDIAN:
                    self.imputation_values[col] = float(series.median(skipna=True))
                elif strat == TabularImputeStrategy.MODE:
                    mode_val = series.mode()
                    self.imputation_values[col] = float(mode_val.iloc[0]) if not mode_val.empty else 0.0
                elif strat == TabularImputeStrategy.CONSTANT:
                    self.imputation_values[col] = 0.0
            else:
                # Categorical/Object
                strat = self.config.missing_categorical_strategy
                if strat == TabularImputeStrategy.MODE:
                    mode_val = series.mode()
                    self.imputation_values[col] = str(mode_val.iloc[0]) if not mode_val.empty else self.config.missing_constant_value
                elif strat == TabularImputeStrategy.CONSTANT:
                    self.imputation_values[col] = self.config.missing_constant_value
                    
        return self.imputation_values
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted imputation values to a dataframe (train, val, or test).
        """
        df_out = df.copy()
        if not self.imputation_values:
            return df_out
            
        for col, val in self.imputation_values.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(val)
                
        return df_out

