import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from hospital_client.model_management.config import ModelConfig
from hospital_client.training.cli import main
from hospital_client.training.config import TrainingConfig
from hospital_client.training.plots import compute_curve_metrics, generate_evaluation_plots
from hospital_client.training.trainer import Trainer


def test_perfect_predictions_give_auc_and_ap_of_one():
    y = [0, 1, 2, 0, 1, 2]
    probs = np.eye(3)[y]
    m = compute_curve_metrics(y, probs, ["a", "b", "c"])
    assert m["macro"]["roc_auc"] == pytest.approx(1.0)
    assert m["macro"]["average_precision"] == pytest.approx(1.0)
    assert m["skipped_classes"] == []


def test_binary_task_reports_only_positive_class():
    y = [0, 0, 1, 1]
    probs = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]
    m = compute_curve_metrics(y, probs, ["neg", "pos"])
    assert list(m["per_class"]) == ["pos"]
    assert m["skipped_classes"] == []


def test_class_missing_from_test_set_is_skipped_not_crashing():
    y = [0, 0, 1, 1]  # class "c" has no samples
    probs = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]]
    m = compute_curve_metrics(y, probs, ["a", "b", "c"])
    assert m["skipped_classes"] == ["c"]
    assert set(m["per_class"]) == {"a", "b"}


def test_generate_writes_all_files_and_summary_json(tmp_path):
    y = [0, 1, 2, 0, 1, 2, 0, 1]
    probs = np.eye(3)[y] * 0.7 + 0.1
    result = generate_evaluation_plots(y, probs, ["a", "b", "c"], str(tmp_path), "m")
    names = {os.path.basename(f) for f in result["files"]}
    assert names == {"confusion_matrix.png", "roc_curves.png", "pr_curves.png", "curve_metrics.json"}
    assert all(os.path.getsize(f) > 0 for f in result["files"])
    with open(os.path.join(result["directory"], "curve_metrics.json")) as f:
        assert json.load(f)["sample_count"] == 8


def test_single_class_test_set_still_writes_confusion_matrix(tmp_path):
    result = generate_evaluation_plots([0, 0, 0], [[0.9, 0.1]] * 3, ["a", "b"], str(tmp_path), "m")
    names = {os.path.basename(f) for f in result["files"]}
    assert "confusion_matrix.png" in names and "roc_curves.png" not in names


def test_empty_test_set_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_evaluation_plots([], np.zeros((0, 2)), ["a", "b"], str(tmp_path), "m")


def test_trainer_captures_test_probabilities_matching_test_metrics(small_dataset_dir, tmp_path):
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32)),
        dataset_dir=small_dataset_dir, output_dir=str(tmp_path / "out"), epochs=1, batch_size=4,
        run_test_evaluation=True,
    )
    trainer = Trainer(config)
    result = trainer.run()
    y_true, probs, class_order = trainer.test_outputs
    probs = np.asarray(probs)
    assert len(y_true) == result.test_metrics["sample_count"]
    assert probs.shape == (len(y_true), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert class_order == result.test_metrics["class_order"]
    # Probabilities come from the same pass as the reported metrics.
    assert (probs.argmax(axis=1) == np.asarray(y_true)).mean() == pytest.approx(result.test_metrics["accuracy"])


def test_trainer_without_test_evaluation_captures_nothing(small_dataset_dir, tmp_path):
    config = TrainingConfig(
        model_id="m", model_config=ModelConfig(architecture="resnet18", num_classes=2, input_size=(32, 32)),
        dataset_dir=small_dataset_dir, output_dir=str(tmp_path / "out"), epochs=1, batch_size=4,
    )
    trainer = Trainer(config)
    trainer.run()
    assert trainer.test_outputs is None


def _cli(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py"] + argv)
    main()


def test_cli_plots_flag_writes_plot_directory(small_dataset_dir, tmp_path, monkeypatch, capsys):
    out = str(tmp_path / "out")
    _cli(["train", "--dataset", small_dataset_dir, "--model", "resnet18", "--num-classes", "2",
          "--input-size", "32", "32", "--output", out, "--epochs", "1", "--batch-size", "4",
          "--test", "--plots"], monkeypatch)
    assert "Evaluation plots written to" in capsys.readouterr().out
    assert os.path.isfile(os.path.join(out, "resnet18_plots", "confusion_matrix.png"))


def test_cli_plots_without_test_is_rejected(small_dataset_dir, tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _cli(["train", "--dataset", small_dataset_dir, "--model", "resnet18", "--num-classes", "2",
              "--input-size", "32", "32", "--output", str(tmp_path / "out"), "--plots"], monkeypatch)
    assert exc.value.code == 1
