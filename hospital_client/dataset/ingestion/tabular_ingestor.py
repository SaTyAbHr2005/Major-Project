
import pandas as pd
import os
from hospital_client.dataset.models.dataset_profile import DatasetProfile
from hospital_client.dataset.ingestion.validation import check_file_readability

def ingest_tabular_dataset(file_path: str) -> DatasetProfile:
    file_path = os.path.abspath(file_path)
    
    is_readable, reason = check_file_readability(file_path)
    if not is_readable:
        return DatasetProfile(
            dataset_type="CSV",
            source_path=file_path,
            errors=[f"File validation failed: {reason}"]
        )
        
    try:
        df = pd.read_csv(file_path)
        
        total_rows = len(df)
        total_columns = len(df.columns)
        
        duplicates = df.duplicated().sum()
        
        missing_counts = df.isnull().sum().to_dict()
        missing_percentages = (df.isnull().sum() / total_rows * 100).to_dict() if total_rows > 0 else {}
        
        dtypes_dict = df.dtypes.astype(str).to_dict()
        
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        candidates = []
        warnings = []
        
        # Check duplicate column names
        if len(df.columns) != len(set(df.columns)):
            warnings.append("Dataset contains duplicate column names.")
            
        for col in df.columns:
            unique_count = df[col].nunique(dropna=True)
            if unique_count == 1 and total_rows > 0:
                warnings.append(f"Column '{col}' is constant (only 1 unique value).")
            elif unique_count >= total_rows * 0.9 and total_rows > 10:
                warnings.append(f"Column '{col}' has very high cardinality (all unique).")
                
            if 1 < unique_count <= 10:
                candidates.append({
                    "column": col,
                    "reason": "Low cardinality (categorical)",
                    "unique_values": unique_count,
                    "type": str(df[col].dtype)
                })
                
        tabular_stats = {
            "columns": df.columns.tolist(),
            "dtypes": dtypes_dict,
            "categorical_columns": categorical_cols,
            "numerical_columns": numerical_cols,
            "candidate_target_columns": candidates
        }
        
        missing_data_info = {
            "missing_counts": missing_counts,
            "missing_percentages": missing_percentages
        }
        
        profile = DatasetProfile(
            dataset_type="CSV",
            source_path=file_path,
            total_samples=total_rows,
            valid_samples=total_rows,
            invalid_samples=0,
            tabular_statistics=tabular_stats,
            duplicate_information={"duplicate_rows": int(duplicates)},
            missing_data_information=missing_data_info,
            warnings=warnings
        )
        return profile
    except pd.errors.EmptyDataError:
        return DatasetProfile(dataset_type="CSV", source_path=file_path, errors=["Empty CSV file"])
    except Exception as e:
        return DatasetProfile(dataset_type="CSV", source_path=file_path, errors=[f"Malformed CSV or error: {str(e)}"])

