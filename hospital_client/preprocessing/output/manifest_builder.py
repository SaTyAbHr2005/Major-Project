import json
import os
from typing import Dict, List, Any

class ManifestBuilder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def build_manifest(self, split_name: str, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """
        Builds a JSON manifest for a given split.
        Records should contain: source_path, processed_path, label, subtype, etc.
        """
        manifest = {
            "split": split_name,
            "metadata": metadata,
            "total_records": len(records),
            "records": records
        }
        
        out_path = os.path.join(self.output_dir, f"{split_name}_manifest.json")
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        return out_path

