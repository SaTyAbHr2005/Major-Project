import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from hospital_client.preprocessing.config import ClassConfig

class LabelResolver:
    def __init__(self, config: ClassConfig):
        self.config = config

    def resolve_label(self, rel_path: str, detected_splits: list) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parses a relative path like 'train/disease/type1/img.jpg'
        Returns: (split, parent_label, subtype_label)
        """
        parts = Path(rel_path).parent.parts
        if not parts:
            return None, None, None
            
        split = None
        # Extract split if it exists as the root folder
        if parts[0] in detected_splits:
            split = parts[0]
            class_parts = parts[1:]
        else:
            class_parts = parts
            
        if not class_parts:
            return split, None, None
            
        parent_label = class_parts[0]
        subtype_label = None
        
        if len(class_parts) > 1:
            # We have a hierarchy
            raw_subtype = "/".join(class_parts) # e.g. "disease/type1"
            
            # Check explicit mapping
            if raw_subtype in self.config.hierarchy_mapping:
                mapped = self.config.hierarchy_mapping[raw_subtype]
                if self.config.preserve_subtypes:
                    parent_label = mapped
                    subtype_label = raw_subtype
                else:
                    parent_label = mapped
            else:
                subtype_label = raw_subtype
                
        # Check mapping for parent label
        if parent_label in self.config.hierarchy_mapping:
            parent_label = self.config.hierarchy_mapping[parent_label]
            
        return split, parent_label, subtype_label

