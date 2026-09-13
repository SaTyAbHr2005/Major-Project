import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np

def extract_groups(df: pd.DataFrame, group_column: str) -> np.ndarray:
    """
    Extracts group identifiers from the dataframe for grouped splitting.
    Raises ValueError if column is not found.
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column {group_column} not found in dataset.")
    return df[group_column].to_numpy()

