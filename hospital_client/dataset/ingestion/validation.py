
import os
from typing import Dict, Any, Tuple

def check_file_readability(file_path: str) -> Tuple[bool, str]:
    if not os.path.exists(file_path):
        return False, "File does not exist"
    if not os.access(file_path, os.R_OK):
        return False, "Permission denied"
    if os.path.getsize(file_path) == 0:
        return False, "Zero-byte file"
    return True, ""

