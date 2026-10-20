import enum
from typing import Dict, Any

class FederationEvent(str, enum.Enum):
    FEDERATION_JOB_CREATED = "FEDERATION_JOB_CREATED"
    ROUND_CREATED = "ROUND_CREATED"
    ROUND_OPENED = "ROUND_OPENED"
    UPDATE_RECEIVED = "UPDATE_RECEIVED"
    UPDATE_VALIDATED = "UPDATE_VALIDATED"
    UPDATE_ACCEPTED = "UPDATE_ACCEPTED"
    UPDATE_REJECTED = "UPDATE_REJECTED"
    UPDATE_QUARANTINED = "UPDATE_QUARANTINED"
    UPDATE_FLAGGED = "UPDATE_FLAGGED"
    PARTICIPANT_TIMEOUT = "PARTICIPANT_TIMEOUT"
    ROUND_READY = "ROUND_READY"
    AGGREGATION_STARTED = "AGGREGATION_STARTED"
    AGGREGATION_COMPLETED = "AGGREGATION_COMPLETED"
    AGGREGATION_FAILED = "AGGREGATION_FAILED"
    AGGREGATION_FALLBACK = "AGGREGATION_FALLBACK"
    GLOBAL_MODEL_CREATED = "GLOBAL_MODEL_CREATED"
    EVALUATION_READY = "EVALUATION_READY"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    PROMOTION_DECIDED = "PROMOTION_DECIDED"

class EventEmitter:
    def __init__(self):
        self.listeners = []

    def add_listener(self, listener):
        self.listeners.append(listener)

    def emit(self, event, **kwargs):
        """
        Emit a telemetry event (M13 integration point).
        Accepts both FederationEvent enum values and plain strings.
        """
        event_name = event.value if hasattr(event, 'value') else str(event)
        payload = {"event": event_name, **kwargs}
        for listener in self.listeners:
            listener(payload)
        # Print for simple debugging if no listeners
        if not self.listeners:
            pass # We could print here if desired
