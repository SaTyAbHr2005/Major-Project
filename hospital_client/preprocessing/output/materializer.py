import os
import shutil
import psutil
from typing import Dict, Any, List, Tuple
import pandas as pd
from hospital_client.preprocessing.config import StorageSafetyConfig
from hospital_client.preprocessing.image.transforms import BaseTransformer

class Materializer:
    def __init__(self, output_dir: str, safety_config: StorageSafetyConfig):
        self.output_dir = output_dir
        self.safety_config = safety_config
        os.makedirs(output_dir, exist_ok=True)
        
    def check_storage_safety(self, estimated_bytes: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Checks if there is enough storage based on safety buffer.
        """
        if not self.safety_config.enforce_safety:
            return True, {"enforced": False}
            
        usage = psutil.disk_usage(self.output_dir)
        total_space = usage.total
        free_space = usage.free
        
        buffer_percent = self.safety_config.safety_buffer_percent / 100.0
        required_buffer = total_space * buffer_percent
        
        available_for_use = free_space - required_buffer
        
        is_safe = available_for_use > estimated_bytes
        
        return is_safe, {
            "estimated_output_size": estimated_bytes,
            "available_storage": free_space,
            "configured_safety_buffer": required_buffer,
            "required_storage": estimated_bytes,
            "result": "safe" if is_safe else "unsafe"
        }
        
    def materialize_image(self, source_path: str, rel_dest_path: str, transformer: BaseTransformer) -> str:
        """
        Transforms and saves image. Returns absolute destination path.
        """
        dest_path = os.path.join(self.output_dir, rel_dest_path)
        success = transformer.transform(source_path, dest_path)
        if not success:
            # Fallback to pure copy if transform fails
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(source_path, dest_path)
        return dest_path
        
    def materialize_tabular(self, df: pd.DataFrame, split_name: str) -> str:
        """
        Saves transformed tabular dataset to a CSV. Returns absolute destination path.
        """
        dest_path = os.path.join(self.output_dir, f"{split_name}.csv")
        df.to_csv(dest_path, index=False)
        return dest_path

