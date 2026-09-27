import os
from PIL import Image, ImageStat
import numpy as np
from typing import Tuple, Dict, Any
from hospital_client.preprocessing.config import QualityConfig
from hospital_client.preprocessing.quarantine.manager import QualityStatus

def assess_image_quality(file_path: str, config: QualityConfig) -> Tuple[QualityStatus, str, Dict[str, Any]]:
    if not os.path.exists(file_path):
        return QualityStatus.ERROR, "File does not exist", {}
        
    if config.reject_zero_byte and os.path.getsize(file_path) == 0:
        return QualityStatus.REJECTED, "Zero-byte file", {}
        
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as e:
        if config.reject_unreadable:
            return QualityStatus.REJECTED, f"Unreadable or corrupted image: {str(e)}", {}
        return QualityStatus.ERROR, f"Unreadable: {str(e)}", {}

    # Actually load image for stats
    try:
        with Image.open(file_path) as img:
            # Convert to grayscale for consistent metric evaluation
            gray = img.convert("L")
            arr = np.array(gray)
            
            stats = ImageStat.Stat(gray)
            brightness = stats.mean[0]
            
            # Laplacian variance for blur
            laplacian_var = np.var(np.lib.stride_tricks.as_strided(
                arr, 
                shape=(arr.shape[0]-2, arr.shape[1]-2, 3, 3), 
                strides=arr.strides + arr.strides
            )) if arr.shape[0] > 2 and arr.shape[1] > 2 else 0.0 # Approximation or we can just use a simple np diff. Wait, numpy doesn't have cv2.Laplacian natively.
            
            # A simpler laplacian approximation in pure numpy:
            # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            # Actually, using scipy or cv2 is better but let's write a clean pure numpy 2D convolution for laplacian.
            
            laplacian_var = compute_laplacian_variance(arr)
            
            metadata = {
                "brightness": brightness,
                "laplacian_variance": laplacian_var,
                "dimensions": img.size,
                "mode": img.mode
            }
            
            if config.brightness_min is not None and brightness < config.brightness_min:
                return QualityStatus.LOW_QUALITY, "Extreme low brightness", metadata
            if config.brightness_max is not None and brightness > config.brightness_max:
                return QualityStatus.LOW_QUALITY, "Extreme high brightness", metadata
            if config.blur_threshold is not None and laplacian_var < config.blur_threshold:
                return QualityStatus.LOW_QUALITY, "Below blur threshold", metadata
                
            return QualityStatus.VALID, "", metadata
    except Exception as e:
        return QualityStatus.ERROR, f"Error processing image stats: {str(e)}", {}

def compute_laplacian_variance(arr: np.ndarray) -> float:
    # Compute laplacian using simple slicing to avoid heavy dependencies if possible
    # L = arr[i-1,j] + arr[i+1,j] + arr[i,j-1] + arr[i,j+1] - 4*arr[i,j]
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    
    # Cast to int16 to avoid underflow/overflow during diff
    arr_int = arr.astype(np.int16)
    
    lap = (
        arr_int[:-2, 1:-1] +
        arr_int[2:, 1:-1] +
        arr_int[1:-1, :-2] +
        arr_int[1:-1, 2:] -
        4 * arr_int[1:-1, 1:-1]
    )
    return float(np.var(lap))

