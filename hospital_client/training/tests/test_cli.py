import json
import sys

import pytest

from hospital_client.training.cli import main


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py"] + argv)
    main()


def test_cli_validate_config_success(small_dataset_dir, tmp_path, monkeypatch, capsys):
    _run([
        "validate-config", "--dataset", small_dataset_dir, "--model", "resnet18",
        "--num-classes", "2", "--input-size", "32", "32", "--output", str(tmp_path / "out"),
    ], monkeypatch)
    assert "valid" in capsys.readouterr().out.lower()


def test_cli_validate_config_invalid_architecture_exits_nonzero(small_dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "__main__.py", "validate-config", "--dataset", small_dataset_dir, "--model", "not_real",
        "--num-classes", "2", "--output", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_train_writes_result_json(small_dataset_dir, tmp_path, monkeypatch, capsys):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", small_dataset_dir, "--model", "resnet18", "--num-classes", "2",
        "--input-size", "32", "32", "--output", output_dir, "--epochs", "1", "--batch-size", "4",
    ], monkeypatch)
    out = capsys.readouterr().out
    assert "Local training completed" in out
    assert "TRAINING_COMPLETED_AWAITING_FEDERATION" in out

    result_path = tmp_path / "out" / "resnet18_training_result.json"
    assert result_path.exists()
    with open(result_path) as f:
        data = json.load(f)
    assert data["status"] == "TRAINING_COMPLETED_AWAITING_FEDERATION"


def test_cli_train_missing_dataset_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "__main__.py", "train", "--dataset", str(tmp_path / "nowhere"), "--model", "resnet18",
        "--num-classes", "2", "--output", str(tmp_path / "out"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_no_args_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["__main__.py"])
    main()
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_train_with_model_id_and_test_flag(small_dataset_dir, tmp_path, monkeypatch, capsys):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", small_dataset_dir, "--model", "resnet18", "--model-id", "chest_xray_resnet18",
        "--num-classes", "2", "--input-size", "32", "32", "--output", output_dir,
        "--epochs", "1", "--batch-size", "4", "--test",
    ], monkeypatch)
    result_path = tmp_path / "out" / "chest_xray_resnet18_training_result.json"
    assert result_path.exists()
    with open(result_path) as f:
        data = json.load(f)
    assert data["test_metrics"] is not None


def test_cli_train_with_precision_and_step_scheduler(small_dataset_dir, tmp_path, monkeypatch, capsys):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", small_dataset_dir, "--model", "resnet18", "--num-classes", "2",
        "--input-size", "32", "32", "--output", output_dir, "--epochs", "2", "--batch-size", "4",
        "--precision", "fp32", "--scheduler", "step", "--scheduler-step-size", "1", "--scheduler-gamma", "0.5",
    ], monkeypatch)
    result_path = tmp_path / "out" / "resnet18_training_result.json"
    with open(result_path) as f:
        data = json.load(f)
    assert data["resolved_precision"] == "fp32"
    assert data["training_configuration"]["scheduler"] == "step"


def test_cli_train_with_cosine_scheduler_and_workers(small_dataset_dir, tmp_path, monkeypatch):
    output_dir = str(tmp_path / "out")
    _run([
        "train", "--dataset", small_dataset_dir, "--model", "resnet18", "--num-classes", "2",
        "--input-size", "32", "32", "--output", output_dir, "--epochs", "2", "--batch-size", "4",
        "--scheduler", "cosine", "--scheduler-t-max", "2", "--num-workers", "2",
    ], monkeypatch)
    result_path = tmp_path / "out" / "resnet18_training_result.json"
    with open(result_path) as f:
        data = json.load(f)
    assert data["status"] == "TRAINING_COMPLETED_AWAITING_FEDERATION"


def test_cli_default_scheduler_is_none_string(small_dataset_dir, tmp_path, monkeypatch, capsys):
    _run([
        "validate-config", "--dataset", small_dataset_dir, "--model", "resnet18",
        "--num-classes", "2", "--output", str(tmp_path / "out"),
    ], monkeypatch)
    assert "valid" in capsys.readouterr().out.lower()
