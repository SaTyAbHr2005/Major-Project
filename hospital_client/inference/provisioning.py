"""
Model provisioning boundary.

Module 16 consumes an already-provisioned LOCAL artifact in Module 6's own
on-disk layout (<root>/<model_id>/v<N>/model.safetensors + metadata.json) -
there is no second checkpoint format. HOW an artifact reaches the hospital
is a future concern, expressed as an `ApprovedModelProvider`:

    Module 9 (global model) -> Module 15 (approval/version) ->
    Module 14 (artifact storage) -> Module 11 (secure provisioning) ->
    an ApprovedModelProvider that materializes the artifact locally -> Module 16

None of those modules exist yet, so nothing here downloads, authenticates,
or approves anything. `LocalStoreModelProvider` (a local directory) is the
only implementation, and its artifacts are recorded as NOT platform-approved.
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from hospital_client.inference.errors import ModelProvisioningError

LOCAL_SOURCE = "local_standalone"


@dataclass(frozen=True)
class LocalModelArtifact:
    store_root: str
    model_id: str
    version: int
    source: str = LOCAL_SOURCE
    # Module 16 cannot establish platform approval (Module 15 owns that). It is
    # False for every artifact this module can currently provision.
    platform_approved: bool = False
    # Extension point for future providers (Module 15 approval record, Module 14
    # artifact reference, Module 11 channel info, ...). Never interpreted here.
    provenance: Dict[str, Any] = field(default_factory=dict)

    def approval_info(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "platform_approved": self.platform_approved,
            "note": (
                "Locally provisioned artifact; platform-level approval (Module 15) is not "
                "available in this deployment." if not self.platform_approved
                else "Approval asserted by the provisioning source."
            ),
            "provenance": self.provenance,
        }


class ApprovedModelProvider(ABC):
    @abstractmethod
    def provision(self) -> LocalModelArtifact:
        """Return a reference to a locally available artifact."""


ApprovedModelSource = ApprovedModelProvider  # spec alias


class LocalStoreModelProvider(ApprovedModelProvider):
    """
    A model version inside a local Module 6 artifact store. The version must
    be explicit - inference never silently picks "latest".
    """

    def __init__(self, store_root: str, model_id: str, version: int):
        self.store_root = store_root
        self.model_id = model_id
        self.version = version

    @classmethod
    def from_training_result(cls, result: Any, training_output_dir: str) -> "LocalStoreModelProvider":
        """Module 7 saves its Module 6 artifacts under <output_dir>/models."""
        if getattr(result, "checkpoint_model_id", None) is None or getattr(result, "checkpoint_version", None) is None:
            raise ModelProvisioningError("The training result has no saved checkpoint to provision.")
        return cls(os.path.join(training_output_dir, "models"), result.checkpoint_model_id, result.checkpoint_version)

    def provision(self) -> LocalModelArtifact:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ModelProvisioningError(f"version must be an explicit positive integer, got {self.version!r}")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ModelProvisioningError("model_id must be a non-empty string.")
        # ModelStore(...) would create a missing root - inference must never create directories.
        if not isinstance(self.store_root, str) or not os.path.isdir(self.store_root):
            raise ModelProvisioningError("Model store directory does not exist.")
        return LocalModelArtifact(store_root=self.store_root, model_id=self.model_id, version=self.version)
