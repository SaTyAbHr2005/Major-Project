
import json
import os
from datetime import datetime
from hospital_client.dataset.models.dataset_profile import DatasetProfile

def generate_reports(profile: DatasetProfile, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    profile.generated_at = datetime.utcnow().isoformat() + "Z"
    
    # Generate JSON
    json_path = os.path.join(output_dir, "dataset_profile.json")
    with open(json_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=4)
        
    # Generate Markdown
    md_path = os.path.join(output_dir, "dataset_report.md")
    with open(md_path, "w") as f:
        f.write(f"# Dataset Profile Report\n\n")
        f.write(f"- **Generated At**: {profile.generated_at}\n")
        f.write(f"- **Dataset Type**: {profile.dataset_type}\n")
        f.write(f"- **Source Path**: {profile.source_path}\n\n")
        
        f.write(f"## Dataset Size\n")
        f.write(f"- **Total Samples**: {profile.total_samples}\n")
        f.write(f"- **Valid Samples**: {profile.valid_samples}\n")
        f.write(f"- **Invalid Samples**: {profile.invalid_samples}\n\n")
        
        if profile.dataset_type == "Image":
            f.write(f"## Image Statistics\n")
            f.write(f"```json\n{json.dumps(profile.image_statistics, indent=2)}\n```\n\n")
            
            f.write(f"## Class Information\n")
            f.write(f"- **Classes**: {profile.classes}\n")
            f.write(f"- **Splits Detected**: {profile.splits.get('detected_splits', [])}\n\n")
            
        elif profile.dataset_type in ["CSV", "Excel"]:
            f.write(f"## Tabular Statistics\n")
            # In MD, dump it clearly
            f.write(f"```json\n{json.dumps(profile.tabular_statistics, indent=2)}\n```\n\n")
            f.write(f"## Missing Data Information\n")
            f.write(f"```json\n{json.dumps(profile.missing_data_information, indent=2)}\n```\n\n")
            
        f.write(f"## Duplicate Information\n")
        f.write(f"```json\n{json.dumps(profile.duplicate_information, indent=2)}\n```\n\n")
        
        if profile.warnings:
            f.write(f"## Warnings\n")
            for w in profile.warnings:
                f.write(f"- {w}\n")
            f.write(f"\n")
            
        if profile.errors:
            f.write(f"## Errors\n")
            for e in profile.errors:
                f.write(f"- {e}\n")
            f.write(f"\n")

