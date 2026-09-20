"""
Module 16: Local Inference. Runs an approved Module 6 model on a NEW local
image entirely on this machine - see module16_readme.md.
"""
from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import (
    DeviceUnavailableError, ImageValidationError, InferenceConfigError, InferenceError,
    ModelCompatibilityError, ModelProvisioningError,
)
from hospital_client.inference.provisioning import (
    ApprovedModelProvider, ApprovedModelSource, LocalModelArtifact, LocalStoreModelProvider,
)
from hospital_client.inference.result import ClassificationOutput, InferenceResult, InferenceStatus, format_summary
from hospital_client.inference.service import LocalInferenceService
