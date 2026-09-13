from typing import Dict, Any, List
from hospital_client.preprocessing.config import ImageConfig

class AugmentationRegistrar:
    def __init__(self, config: ImageConfig):
        self.config = config
        
    def generate_augmentation_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata describing random training augmentations
        to be applied dynamically in Module 7 during training.
        Module 5 does not permanently bake random augmentations into datasets.
        """
        return {
            "applied_augmentations": self.config.augmentation_flags,
            "instruction": "These augmentations should be applied dynamically to the training split by the ML DataLoader."
        }

