import json
import sys

import pytest

from hospital_client.resource_training.cli import main


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py"] + argv)
    main()


def test_cli_detect_prints_resource_profile(tmp_path, monkeypatch, capsys):
    _run(["detect", "--output", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "cpu" in data and "gpu" in data


def test_cli_recommend_writes_recommendations(real_small_dataset_dir, tmp_path, monkeypatch, capsys):
    _run(["recommend", "--dataset", real_small_dataset_dir, "--output", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "recommendations" in data
    assert "all_model_assessments" in data
    assert len(data["all_model_assessments"]) == 8


def test_cli_train_runs_and_writes_result(real_small_dataset_dir, tmp_path, monkeypatch, capsys):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", real_small_dataset_dir, "--output", output_dir,
        "--model-id", "cli_test_model", "--choice", "fast",
    ], monkeypatch)
    out = capsys.readouterr().out
    assert "Chosen recommendation" in out
    result_path = tmp_path / "out" / "cli_test_model_training_result.json"
    stats_path = tmp_path / "out" / "cli_test_model_resource_statistics.json"
    assert result_path.exists()
    assert stats_path.exists()
    with open(stats_path) as f:
        stats = json.load(f)
    assert "hardware" in stats and "monitor_summary" in stats


def test_cli_no_args_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["__main__.py"])
    main()
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_train_missing_dataset_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "__main__.py", "train", "--dataset", str(tmp_path / "nowhere"), "--output", str(tmp_path / "out"),
        "--model-id", "m",
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_train_batch_size_requires_architecture(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "__main__.py", "train", "--dataset", str(tmp_path), "--output", str(tmp_path / "out"),
        "--model-id", "m", "--batch-size", "8",
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert "--batch-size/--epochs require --architecture" in capsys.readouterr().out


def test_cli_train_manual_architecture_batch_and_epochs(real_small_dataset_dir, tmp_path, monkeypatch, capsys):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", real_small_dataset_dir, "--output", output_dir,
        "--model-id", "manual_model", "--architecture", "mobilenet_v3_small", "--batch-size", "2", "--epochs", "1",
    ], monkeypatch)
    out = capsys.readouterr().out
    assert "Chosen recommendation: Manual Configuration (mobilenet_v3_small)" in out
    with open(tmp_path / "out" / "manual_model_training_result.json") as f:
        result = json.load(f)
    assert result["architecture"] == "mobilenet_v3_small" and result["epochs_run"] == 1
    assert result["training_configuration"]["batch_size"] == 2


def test_cli_recommend_includes_alternatives_and_batch_memory(real_small_dataset_dir, tmp_path, monkeypatch, capsys):
    _run(["recommend", "--dataset", real_small_dataset_dir, "--output", str(tmp_path)], monkeypatch)
    data = json.loads(capsys.readouterr().out)
    assert "alternatives" in data
    for a in data["all_model_assessments"].values():
        if a["feasible"]:
            assert str(a["batch_size"]) in a["batch_memory_mb"] and a["cpu_logical_cores"] >= 1
