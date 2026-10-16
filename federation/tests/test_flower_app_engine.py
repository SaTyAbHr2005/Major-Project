"""
Helpers the Flower server uses to drive the Module 9 engine: rebuilding an update from the
weights received over Flower, averaging BatchNorm statistics, replacing a global model's
parameters and keeping filesystem paths out of what the server stores.
"""
import datetime
import json

import numpy as np
import pytest

from federation import participation
from federation.api_adapter import open_round
from federation.base_models import base_model_store
from federation.events import EventEmitter
from federation.flower_app.engine import (average_bn_statistics, handoff_from_parameters,
                                          replace_global_model_parameters, strip_paths)
from federation.intake import IntakeManager
from federation.jobs import JobManager
from federation.models import GlobalModelVersion
from federation.security_hooks import M12SecurityHook
from federation.storage import FederationStorage
from federation.validation import UpdateValidator
from hospital_client.model_management.parameters import get_parameters
from hospital_client.training.federation import FederationHandoff, _compute_parameters_checksum, build_federation_handoff

ARCH = "mobilenet_v3_small"
MAPPING = {"Parasitized": 0, "Uninfected": 1}
ELIGIBLE = {"eligible": True, "train_sample_count": 40}


@pytest.fixture
def setup(tmp_path):
    storage = FederationStorage(str(tmp_path / "fed.db"), str(tmp_path / "artifacts"))
    JobManager(storage).create_job("JOB", "Malaria", "image_classification", ARCH, 2, MAPPING, [], 1, "", {}, {})
    for h in ("H1",):
        participation.request_participation(storage, storage.get_job("JOB"), h, ELIGIBLE)
        participation.decide_participation(storage, storage.get_job("JOB"), h, "APPROVED")
    deadline = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat()
    rnd, _ = open_round(storage, {"job_id": "JOB", "round_id": "JOB_R1", "round_number": 1, "deadline": deadline})
    r = storage.get_round("JOB_R1"); r.expected_participants.append("H1"); storage.save_round(r)
    seed, _ = base_model_store(storage).load_checkpoint(rnd.base_model_id, rnd.base_model_version)
    params = [p + np.asarray(0.01, dtype=p.dtype) if p.dtype.kind == "f" else p for p in get_parameters(seed)]
    handoff = build_federation_handoff(
        params, "FED_JOB_H1", 1, ARCH, rnd.base_model_id, rnd.base_model_version, rnd.base_model_checksum, 40, MAPPING,
        {"loss": 0.5}, {"accuracy": 0.8}, {"epochs": 1, "dataset_dir": "C:\\Users\\op\\hospital\\run", "output_dir": "/home/op/out"},
        "cpu", "fp32", "TRAINING_COMPLETED_AWAITING_FEDERATION")
    return storage, params, handoff.to_dict()


def test_handoff_rebuilt_from_received_weights_is_accepted_by_intake(tmp_path, setup):
    storage, params, meta = setup
    d = handoff_from_parameters(params, meta, str(tmp_path / "in"))
    upd = IntakeManager(storage, UpdateValidator(storage), M12SecurityHook(), EventEmitter()).receive_handoff("JOB", "JOB_R1", "H1", d)
    assert upd.status.value == "ACCEPTED", upd.rejection_reason
    saved = FederationHandoff.load(d)
    assert "dataset_dir" not in saved.training_configuration and saved.training_configuration["epochs"] == 1


def test_weights_changed_in_transit_are_refused(tmp_path, setup):
    _, params, meta = setup
    params = [params[0] + np.asarray(1, dtype=params[0].dtype)] + params[1:]
    with pytest.raises(ValueError, match="checksum"):
        handoff_from_parameters(params, meta, str(tmp_path / "in"))


def test_bn_average_changes_only_running_statistics():
    keys = ["conv.weight", "bn.weight", "bn.running_mean", "bn.running_var", "bn.num_batches_tracked"]
    g = [np.full(2, 5.0, np.float32), np.full(2, 6.0, np.float32), np.zeros(2, np.float32), np.ones(2, np.float32), np.array(7)]
    a = [np.zeros(2, np.float32), np.zeros(2, np.float32), np.full(2, 4.0, np.float32), np.full(2, 2.0, np.float32), np.array(1)]
    b = [np.ones(2, np.float32), np.ones(2, np.float32), np.full(2, 8.0, np.float32), np.full(2, 6.0, np.float32), np.array(2)]
    out = average_bn_statistics(g, keys, [(a, 3), (b, 1)])
    for i, k in enumerate(keys):
        if not k.endswith(("running_mean", "running_var")):
            assert np.array_equal(out[i], g[i])
    assert np.allclose(out[2], 5.0) and np.allclose(out[3], 3.0)
    assert out[2].dtype == np.float32


def test_replacing_global_model_parameters_updates_artifact_and_checksum(tmp_path, setup):
    storage, params, meta = setup
    gm_dir = str(tmp_path / "gm")
    build_federation_handoff(params, "GM-X-V1", 1, ARCH, meta["base_model_id"], 1, meta["base_model_checksum"], 40, MAPPING,
                             {}, {}, {}, "cpu", "fp32", "GLOBAL_MODEL_CREATED").save(gm_dir)
    storage.save_global_model(GlobalModelVersion(
        global_model_id="GM-X-V1", task="Malaria", task_type="image_classification", architecture=ARCH, version=1,
        artifact_location=gm_dir, artifact_checksum=_compute_parameters_checksum(params), artifact_size=0,
        parameter_count=len(params), parameter_metadata={}, federation_job_id="JOB", round_id="JOB_R1",
        source_update_ids=[], aggregation_method="weighted_fedavg", aggregation_metadata={}))
    new = [p * 2 if p.dtype.kind == "f" else p for p in params]
    replace_global_model_parameters(storage, "GM-X-V1", new)
    gm = storage.get_global_model("GM-X-V1")
    assert gm.artifact_checksum == _compute_parameters_checksum(new)
    assert FederationHandoff.load(gm_dir).parameters_checksum == gm.artifact_checksum


def test_paths_are_removed_from_training_configuration():
    assert strip_paths({"epochs": 3, "dataset_dir": "C:\\x", "output_dir": "/y", "note": "C:\\Users\\a",
                        "architecture": "resnet18", "nested": {"run": "/tmp/r", "lr": 0.1}}) == \
        {"epochs": 3, "architecture": "resnet18", "nested": {"lr": 0.1}}
