"""
Typed errors for Module 16. Messages never contain the patient image's path
or filename (either can carry patient identifiers) - see module16_readme.md.
"""


class InferenceError(Exception):
    category = "inference_error"


class InferenceConfigError(InferenceError):
    category = "invalid_configuration"


class ModelProvisioningError(InferenceError):
    """The approved model artifact could not be located/validated/loaded."""
    category = "model_provisioning"


class ModelCompatibilityError(InferenceError):
    """Model metadata / preprocessing / class mapping are not usable together."""
    category = "model_compatibility"


class ImageValidationError(InferenceError):
    category = "invalid_image"


class DeviceUnavailableError(InferenceError):
    category = "device_unavailable"
