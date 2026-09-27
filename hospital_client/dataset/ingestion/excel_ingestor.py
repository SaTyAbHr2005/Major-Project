
import pandas as pd
import os
from hospital_client.dataset.models.dataset_profile import DatasetProfile
from hospital_client.dataset.ingestion.validation import check_file_readability

def ingest_excel_dataset(file_path: str) -> DatasetProfile:
    file_path = os.path.abspath(file_path)
    
    is_readable, reason = check_file_readability(file_path)
    if not is_readable:
        return DatasetProfile(
            dataset_type="Excel",
            source_path=file_path,
            errors=[f"File validation failed: {reason}"]
        )
        
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        tabular_stats = {"sheets": {}}
        total_rows = 0
        total_duplicates = 0
        warnings = []
        
        for sheet in sheet_names:
            df = xl.parse(sheet)
            sheet_rows = len(df)
            total_rows += sheet_rows
            
            duplicates = df.duplicated().sum()
            total_duplicates += duplicates
            
            missing_counts = df.isnull().sum().to_dict()
            missing_percentages = (df.isnull().sum() / sheet_rows * 100).to_dict() if sheet_rows > 0 else {}
            dtypes_dict = df.dtypes.astype(str).to_dict()
            
            candidates = []
            
            if len(df.columns) != len(set(df.columns)):
                warnings.append(f"Sheet '{sheet}' contains duplicate column names.")
                
            for col in df.columns:
                unique_count = df[col].nunique(dropna=True)
                if unique_count == 1 and sheet_rows > 0:
                    warnings.append(f"Sheet '{sheet}', Column '{col}' is constant (only 1 unique value).")
                elif unique_count >= sheet_rows * 0.9 and sheet_rows > 10:
                    warnings.append(f"Sheet '{sheet}', Column '{col}' has very high cardinality (all unique).")
                    
                if 1 < unique_count <= 10:
                    candidates.append({
                        "column": col,
                        "reason": "Low cardinality (categorical)",
                        "unique_values": unique_count,
                        "type": str(df[col].dtype)
                    })
            
            tabular_stats["sheets"][sheet] = {
                "columns": df.columns.tolist(),
                "dtypes": dtypes_dict,
                "rows": sheet_rows,
                "candidate_target_columns": candidates,
                "missing_counts": missing_counts,
                "missing_percentages": missing_percentages
            }
            
        xl.close()
        profile = DatasetProfile(
            dataset_type="Excel",
            source_path=file_path,
            total_samples=total_rows,
            valid_samples=total_rows,
            invalid_samples=0,
            tabular_statistics=tabular_stats,
            duplicate_information={"duplicate_rows_across_sheets": int(total_duplicates)},
            warnings=warnings
        )
        return profile
    except Exception as e:
        return DatasetProfile(dataset_type="Excel", source_path=file_path, errors=[f"Malformed Excel file or error: {str(e)}"])

