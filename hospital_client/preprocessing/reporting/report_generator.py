import json
import os
from datetime import datetime
from typing import Dict, Any, List
from hospital_client.preprocessing.config import PreprocessingConfig

class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(self, quarantine_summary: Dict[str, int], config: PreprocessingConfig, splits_map: Dict[str, List[Any]]):
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "config": {
                "output_mode": config.dataset.output_mode.value,
                "split_strategy": config.split.strategy.value
            },
            "quarantine": quarantine_summary,
            "splits": {k: len(v) for k, v in splits_map.items()}
        }
        
        # Verify reconciliation
        accepted = len(splits_map.get("train", [])) + len(splits_map.get("validation", [])) + len(splits_map.get("test", []))
        report_data["reconciliation"] = {
            "accepted_samples": accepted,
            "quarantine_accepted": quarantine_summary["accepted"],
            "is_balanced": accepted == quarantine_summary["accepted"]
        }
        
        json_path = os.path.join(self.output_dir, "preprocessing_report.json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        md_path = os.path.join(self.output_dir, "preprocessing_report.md")
        with open(md_path, "w") as f:
            f.write(f"# Module 5 Preprocessing Report\n\n")
            f.write(f"Generated at: {report_data['generated_at']}\n\n")
            f.write(f"## Dataset Output Mode: {report_data['config']['output_mode']}\n")
            f.write(f"## Splits Summary\n")
            for k, v in report_data["splits"].items():
                f.write(f"- **{k}**: {v} samples\n")
            f.write(f"\n## Quarantine Summary\n")
            for k, v in report_data["quarantine"].items():
                f.write(f"- **{k}**: {v}\n")
            f.write(f"\n## Reconciliation\n")
            f.write(f"Accepted Output Samples: {accepted}\n")
            f.write(f"Quarantine Accepted Samples: {quarantine_summary['accepted']}\n")
            f.write(f"Reconciled: {report_data['reconciliation']['is_balanced']}\n")

