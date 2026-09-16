"""
This is the back half of Module 8's full pipeline:

    Start training -> M8 resource evaluation -> dataset evaluation ->
    model selection -> training configuration -> M7 -> actual training ->
    resource statistics -> M9

The first four steps (resource evaluation, dataset evaluation, model
selection, training configuration) happen upstream in resource_profile.py /
dataset_characteristics.py / recommender.py / resolved_config.py, driven by
cli.py's `train` command in that exact order. `run_resource_aware_training()`
below performs the last four: pre-training re-check -> start monitoring ->
Trainer.run() (M7, "actual training") -> CUDA-OOM adaptation/retry (bounded)
-> stop monitoring -> build ResourceStatistics -> record measured throughput
-> build the Module 7 -> Module 9 FederationHandoff when training reached a
federation-ready status.

Module 7's Trainer is called exactly as-is (no subclassing, no internal
hooks) - this module only wraps the call. The FederationHandoff is built via
Module 7's own existing `build_federation_handoff()` (see
hospital_client.training.trainer) - Module 8 does not implement any part of
Module 9 itself, it only carries the pipeline through to the handoff that
Module 7 already knows how to produce. See module8_readme.md for why
adaptation here is run-level, not intra-epoch.
"""
from typing import List, Optional, Tuple

from hospital_client.resource_training.adaptation import adapt_config_for_oom, detect_cuda_oom, make_event
from hospital_client.resource_training.estimator import HistoricalThroughputStore
from hospital_client.resource_training.evaluator import select_device
from hospital_client.resource_training.monitor import ResourceMonitor
from hospital_client.resource_training.policy import ResourcePolicy
from hospital_client.resource_training.resource_profile import ResourceProfile, build_resource_profile
from hospital_client.resource_training.statistics import ResourceStatistics, build_resource_statistics
from hospital_client.training.config import TrainingConfig
from hospital_client.training.federation import FederationHandoff
from hospital_client.training.result import TrainingResult
from hospital_client.training.trainer import Trainer, build_federation_handoff as _build_federation_handoff


class ResourceCheckFailed(Exception):
    """Raised when hardware conditions changed such that the configured
    device is no longer available. Module 8 never silently substitutes CPU
    for a configuration that explicitly requires CUDA."""


def run_resource_aware_training(
    config: TrainingConfig, policy: Optional[ResourcePolicy] = None,
    history_path: Optional[str] = None, pre_training_profile: Optional[ResourceProfile] = None,
) -> Tuple[TrainingResult, ResourceStatistics, Optional[FederationHandoff]]:
    policy = policy or ResourcePolicy()
    history = HistoricalThroughputStore(history_path) if history_path else None

    profile = pre_training_profile or build_resource_profile(storage_path=config.output_dir or ".", check_network=False, policy=policy)
    device_type, fallback_reason = select_device(profile)
    if config.device == "cuda" and device_type != "cuda":
        raise ResourceCheckFailed(
            f"Configuration requires device='cuda' but a pre-training hardware re-check found it unavailable: "
            f"{fallback_reason} Module 8 does not silently fall back to CPU - choose a new recommendation "
            "for the current hardware instead."
        )

    current_config = config
    adaptations = []
    attempt = 0
    result: Optional[TrainingResult] = None
    monitor_summary = None

    while True:
        monitor = ResourceMonitor(policy, current_config.device)
        monitor.start()
        result = Trainer(current_config).run()
        monitor_summary = monitor.stop()

        if detect_cuda_oom(result) and attempt < policy.max_adaptation_attempts:
            adapted = adapt_config_for_oom(current_config, policy)
            if adapted is None:
                adaptations.append(make_event(
                    f"CUDA out of memory at batch_size={current_config.batch_size}; cannot reduce further.",
                    current_config.batch_size, None, "exhausted",
                ))
                break
            adaptations.append(make_event(
                f"CUDA out of memory at batch_size={current_config.batch_size}; retrying with a smaller batch size.",
                current_config.batch_size, adapted.batch_size, "retrying",
            ))
            current_config = adapted
            attempt += 1
            continue
        break

    measured_sps = None
    if result.training_metrics and result.epochs_run > 0 and result.training_duration_seconds > 0:
        sample_count = result.training_metrics.get("sample_count", 0)
        if sample_count > 0:
            measured_sps = (sample_count * result.epochs_run) / result.training_duration_seconds
            if history is not None:
                device_key = profile.gpu.get("gpu_name") if current_config.device == "cuda" else "cpu"
                history.record(current_config.model_config.architecture, device_key, result.resolved_precision, measured_sps)

    stats = build_resource_statistics(
        profile=profile, training_configuration=current_config.to_dict(), monitor_summary=monitor_summary,
        adaptations=adaptations, actual_duration_seconds=result.training_duration_seconds,
        measured_samples_per_second=measured_sps,
    )

    # -> M9: hand off to Module 7's own (unmodified) FederationHandoff
    # builder whenever training actually reached a federation-ready
    # checkpoint. round_id/client_id are never fabricated by Module 8 -
    # a future federation/identity integration supplies them.
    handoff: Optional[FederationHandoff] = None
    if result.ready_for_federation:
        handoff = _build_federation_handoff(result, current_config.output_dir)

    return result, stats, handoff
