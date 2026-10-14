"""
Flower simulation bench: partition accounting and skew, CLI argument checks, and one real
end-to-end run (Flower gRPC server + 2 client processes, FedProx, 2 rounds) on synthetic images.
"""
import json
import os
import subprocess
import sys
from collections import Counter

import pytest

from federation.flower_sim import parse_args, partition_records, write_client_datasets
from hospital_client.training.tests.conftest import make_preprocessed_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def records(per_class):
    return [{"label": label, "source_path": f"{label}/{i}.png"} for label, n in per_class.items() for i in range(n)]


@pytest.mark.parametrize("mode", ["iid", "label_skew"])
def test_every_record_goes_to_exactly_one_client(mode):
    recs = records({"A": 300, "B": 100, "C": 50})
    parts = partition_records(recs, 4, mode, 0.5, seed=7)
    assert len(parts) == 4
    flat = [r["source_path"] for p in parts for r in p]
    assert sorted(flat) == sorted(r["source_path"] for r in recs)   # nothing lost, nothing duplicated


def test_partition_is_reproducible_and_seed_dependent():
    recs = records({"A": 200, "B": 200})
    a = partition_records(recs, 3, "label_skew", 0.3, seed=1)
    assert a == partition_records(recs, 3, "label_skew", 0.3, seed=1)
    assert a != partition_records(recs, 3, "label_skew", 0.3, seed=2)


def _majority_share(parts):
    return [max(Counter(r["label"] for r in p).values()) / len(p) for p in parts if p]


def test_iid_keeps_class_balance_and_small_alpha_skews_it():
    recs = records({"A": 500, "B": 500})
    iid = partition_records(recs, 4, "iid", 0.5, seed=0)
    assert max(len(p) for p in iid) - min(len(p) for p in iid) <= 1
    assert max(_majority_share(iid)) < 0.62
    skew = partition_records(recs, 4, "label_skew", 0.1, seed=0)
    assert max(_majority_share(skew)) > 0.8


def test_bad_partition_settings_are_rejected():
    with pytest.raises(ValueError):
        partition_records(records({"A": 5}), 2, "pathological", 0.5, 0)
    with pytest.raises(ValueError):
        partition_records(records({"A": 5}), 2, "label_skew", 0.0, 0)


def test_client_with_no_training_images_is_an_error(tmp_path):
    data = make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12})
    with pytest.raises(ValueError, match="no training images"):
        write_client_datasets(data, str(tmp_path / "clients"), 40, "iid", 0.5, 0)


@pytest.mark.parametrize("argv", [
    ["--strategy", "fedprox"],                    # FedProx without mu
    ["--strategy", "fedavg", "--mu", "0.1"],      # mu without FedProx
    ["--clients", "0"],
])
def test_cli_rejects_inconsistent_settings(argv):
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "d", "--output", "o", "--architecture", "resnet18", *argv])


def test_failed_client_training_fails_the_run(tmp_path):
    data = make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12})
    path = os.path.join(data, "train_manifest.json")
    manifest = json.load(open(path))
    for r in manifest["records"]:                       # training images that cannot be read
        r["source_path"] = r["source_path"] + ".missing"
        r["processed_path"] = None
    json.dump(manifest, open(path, "w"))
    proc = subprocess.run(
        [sys.executable, "-m", "federation.flower_sim", "--dataset", data, "--output", str(tmp_path / "run"),
         "--clients", "2", "--rounds", "1", "--architecture", "mobilenet_v3_small", "--batch-size", "4"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": REPO}, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 1
    assert "Not every client returned an update" in proc.stderr or "exited with code" in proc.stderr


def test_end_to_end_flower_run(tmp_path):
    data = make_preprocessed_dataset(tmp_path, {"cat": 12, "dog": 12})
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, "-m", "federation.flower_sim", "--dataset", data, "--output", str(out), "--clients", "2",
         "--rounds", "2", "--strategy", "fedprox", "--mu", "0.01", "--architecture", "mobilenet_v3_small",
         "--epochs", "1", "--batch-size", "4", "--seed", "3"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": REPO}, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    res = json.loads((out / "results.json").read_text())
    assert res["completed_rounds"] == 2 and [r["round"] for r in res["rounds"]] == [0, 1, 2]
    assert res["configuration"]["strategy"] == "fedprox" and res["configuration"]["seed"] == 3
    for r in res["rounds"][1:]:
        assert sorted(c["client"] for c in r["clients"]) == [0, 1]
        assert all(c["proximal_mu"] == 0.01 for c in r["clients"])   # FedProx mu reached the clients
        assert r["bytes_exchanged"] == 2 * 2 * res["model_bytes"]
        assert 0.0 <= r["test"]["accuracy"] <= 1.0
    # Clients together hold exactly the original training set.
    train = json.load(open(os.path.join(data, "train_manifest.json")))["records"]
    assert sum(sum(c["train"].values()) for c in res["partition"]) == len(train)
    assert (out / "rounds.csv").read_text().count("\n") == 4                # header + rounds 0..2
    assert not (out / "work").exists()                                      # per-round checkpoints cleaned up
