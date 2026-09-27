
import json
import os
import tempfile
from hospital_client.dataset.ingestion.dataset_detector import detect_and_ingest
from hospital_client.dataset.ingestion.report_generator import generate_reports

def test_privacy_tabular(comprehensive_tabular_dataset):
    csv_path = os.path.join(comprehensive_tabular_dataset, "valid.csv")
    profile = detect_and_ingest(csv_path)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_reports(profile, tmp_dir)
        
        json_path = os.path.join(tmp_dir, "dataset_profile.json")
        md_path = os.path.join(tmp_dir, "dataset_report.md")
        
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        
        with open(json_path, "r") as f:
            data = json.load(f)
            
        str_data = json.dumps(data).lower()
        assert "hello world" not in str_data
        assert "foo bar" not in str_data

def test_privacy_image(comprehensive_image_dataset):
    dir_b = os.path.join(comprehensive_image_dataset, "split_b")
    profile = detect_and_ingest(dir_b)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_reports(profile, tmp_dir)
        
        json_path = os.path.join(tmp_dir, "dataset_profile.json")
        with open(json_path, "r") as f:
            data = json.load(f)
            
        assert "pixel" not in json.dumps(data).lower()
        assert "benign" in data["classes"]

