
import os
from hospital_client.dataset.ingestion.tabular_ingestor import ingest_tabular_dataset
from hospital_client.dataset.ingestion.excel_ingestor import ingest_excel_dataset
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest

def test_valid_csv(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "valid.csv")
    profile = ingest_tabular_dataset(csv_path)
    
    assert profile.total_samples == 13
    assert profile.duplicate_information["duplicate_rows"] == 1
    
    assert "num.1" in profile.tabular_statistics["columns"] or "num" in profile.tabular_statistics["columns"]
    # duplicate col name inherently resolved by pandas
    assert any("is constant" in w for w in profile.warnings)
    assert any("high cardinality" in w for w in profile.warnings)
    
    candidates = profile.tabular_statistics["candidate_target_columns"]
    cat_candidate = next(c for c in candidates if c["column"] == "cat")
    assert "Low cardinality" in cat_candidate["reason"]
    
def test_empty_csv(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "empty.csv")
    profile = ingest_tabular_dataset(csv_path)
    assert len(profile.errors) > 0
    assert "Zero-byte file" in profile.errors[0]

def test_malformed_csv(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "malformed.csv")
    profile = ingest_tabular_dataset(csv_path)
    assert len(profile.errors) > 0
    
def test_valid_xlsx(comprehensive_tabular_dataset):
    excel_path = os.path.join(comprehensive_tabular_dataset, "data.xlsx")
    profile = ingest_excel_dataset(excel_path)
    
    assert profile.dataset_type == "Excel"
    assert profile.total_samples == 4
    assert profile.duplicate_information["duplicate_rows_across_sheets"] == 2
    assert "S1" in profile.tabular_statistics["sheets"]
    
def test_valid_xls(comprehensive_tabular_dataset):
    xls_path = os.path.join(comprehensive_tabular_dataset, "data.xls")
    profile = detect_and_ingest(xls_path)
    
    assert profile.dataset_type == "Excel"
    assert profile.total_samples == 1
    assert "Sheet1" in profile.tabular_statistics["sheets"]
    assert profile.tabular_statistics["sheets"]["Sheet1"]["rows"] == 1
    assert "col1" in profile.tabular_statistics["sheets"]["Sheet1"]["columns"]

def test_malformed_xlsx(comprehensive_tabular_dataset):
    malformed_excel_path = os.path.join(comprehensive_tabular_dataset, "malformed.xlsx")
    with open(malformed_excel_path, "wb") as f:
        f.write(b"this is not a valid excel file")
    
    profile = ingest_excel_dataset(malformed_excel_path)
    assert len(profile.errors) > 0
    assert "Malformed Excel file" in profile.errors[0]

