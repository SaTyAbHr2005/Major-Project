
import os
from hospital_client.dataset.ingestion.image_ingestor import ingest_image_dataset
from hospital_client.dataset.ingestion.tabular_ingestor import ingest_tabular_dataset
from hospital_client.dataset.ingestion.excel_ingestor import ingest_excel_dataset

def detect_and_ingest(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset path does not exist.")
        
    if os.path.isdir(dataset_path):
        # Assume Image Dataset directory structure
        return ingest_image_dataset(dataset_path)
    elif os.path.isfile(dataset_path):
        ext = os.path.splitext(dataset_path)[1].lower()
        if ext == ".csv":
            return ingest_tabular_dataset(dataset_path)
        elif ext in {".xls", ".xlsx"}:
            return ingest_excel_dataset(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

