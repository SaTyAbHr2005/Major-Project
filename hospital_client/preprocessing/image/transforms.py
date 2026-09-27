from PIL import Image
import os
from typing import Optional, Tuple
from hospital_client.preprocessing.config import ImageConfig

class BaseTransformer:
    def __init__(self, config: ImageConfig):
        self.config = config
        
    def transform(self, input_path: str, output_path: str) -> bool:
        """
        Applies base deterministic transformations (resizing, mode conversion) 
        and saves to output_path. Returns True on success.
        """
        try:
            with Image.open(input_path) as img:
                if self.config.color_mode and img.mode != self.config.color_mode:
                    img = img.convert(self.config.color_mode)
                    
                if self.config.target_size:
                    # Uses LANCZOS for high-quality downsampling/upsampling
                    img = img.resize(self.config.target_size, Image.Resampling.LANCZOS)
                    
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path)
                return True
        except Exception:
            return False

