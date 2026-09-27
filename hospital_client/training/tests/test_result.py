from hospital_client.training.result import TrainingResult, TrainingStatus, format_summary


def test_ready_for_federation_only_true_for_awaiting_federation_status():
    completed = TrainingResult(status=TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION, model_id="m", architecture="resnet18")
    failed = TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18")
    interrupted = TrainingResult(status=TrainingStatus.TRAINING_INTERRUPTED, model_id="m", architecture="resnet18")
    plain_completed = TrainingResult(status=TrainingStatus.TRAINING_COMPLETED, model_id="m", architecture="resnet18")

    assert completed.ready_for_federation is True
    assert failed.ready_for_federation is False
    assert interrupted.ready_for_federation is False
    assert plain_completed.ready_for_federation is False


def test_to_dict_contains_status_as_string_and_ready_flag():
    result = TrainingResult(status=TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION, model_id="m", architecture="resnet18")
    d = result.to_dict()
    assert d["status"] == "TRAINING_COMPLETED_AWAITING_FEDERATION"
    assert d["ready_for_federation"] is True


def test_format_summary_never_claims_clinical_validity():
    result = TrainingResult(
        status=TrainingStatus.TRAINING_COMPLETED_AWAITING_FEDERATION, model_id="m", architecture="resnet18",
        validation_metrics={"accuracy": 0.99, "f1": 0.98, "precision": 0.97, "recall": 0.96},
        best_epoch=5, epochs_run=5, checkpoint_model_id="m", checkpoint_version=1,
    )
    summary = format_summary(result, "ResNet-18")
    lowered = summary.lower()
    # The summary must carry an explicit disclaimer, not an unhedged claim.
    assert "not a clinical validation" in lowered
    assert "does not represent guaranteed diagnostic performance" in lowered
    assert "ResNet-18" in summary
    assert "99.00%" in summary


def test_format_summary_shows_failure_status():
    result = TrainingResult(status=TrainingStatus.TRAINING_FAILED, model_id="m", architecture="resnet18", errors=["CUDA out of memory"])
    summary = format_summary(result, "ResNet-18")
    assert "failed" in summary.lower()
    assert "TRAINING_FAILED" in summary
