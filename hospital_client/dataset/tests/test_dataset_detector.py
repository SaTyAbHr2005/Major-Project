
import os
import pytest
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest

def test_detect_image_dataset(comprehensive_image_dataset):
    profile = detect_and_ingest(comprehensive_image_dataset)
    assert profile.dataset_type == "Image"

def test_detect_csv_dataset(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "valid.csv")
    profile = detect_and_ingest(csv_path)
    assert profile.dataset_type == "CSV"

def test_detect_excel_dataset(comprehensive_tabular_dataset):
    excel_path = os.path.join(comprehensive_tabular_dataset, "data.xlsx")
    profile = detect_and_ingest(excel_path)
    assert profile.dataset_type == "Excel"
    
def test_nonexistent_path():
    with pytest.raises(ValueError, match="does not exist"):
        detect_and_ingest("C:/path/that/does/not/exist.csv")
        
def test_unsupported_path(comprehensive_image_dataset):
    unsupported_path = os.path.join(comprehensive_image_dataset, "formats", "doc.pdf")
    with pytest.raises(ValueError, match="Unsupported file format"):
        detect_and_ingest(unsupported_path)

