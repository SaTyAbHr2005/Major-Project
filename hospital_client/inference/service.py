"""
LocalInferenceService: the UI-independent facade the Hospital Desktop
(Module 3) - or a CLI, test, or future hospital service - calls. It owns the
engine lifecycle (load once, predict many, unload to free memory) and holds no
UI or network code. Predictions are serialized with a lock so a UI thread and
a worker cannot interleave on the same model.
"""
import threading
from typing import Any, Dict, Optional, Union

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.errors import InferenceError
from hospital_client.inference.provisioning import ApprovedModelProvider, LocalModelArtifact
from hospital_client.inference.result import InferenceResult


class LocalInferenceService:
    def __init__(self):
        self._engine: Optional[LocalInferenceEngine] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def load_model(self, source: Union[ApprovedModelProvider, LocalModelArtifact], config: Optional[InferenceConfig] = None) -> Dict[str, Any]:
        engine = LocalInferenceEngine(source, config)  # raises before touching the currently loaded model
        with self._lock:
            previous, self._engine = self._engine, engine
        if previous is not None:
            previous.close()
        return engine.model_info()

    def model_info(self) -> Dict[str, Any]:
        return self._require_engine().model_info()

    def predict(self, image_path: str, reference: Optional[str] = None) -> InferenceResult:
        engine = self._require_engine()
        with self._lock:
            return engine.predict(image_path, reference)

    def unload(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def _require_engine(self) -> LocalInferenceEngine:
        engine = self._engine
        if engine is None:
            raise InferenceError("No model is loaded; call load_model() first.")
        return engine
