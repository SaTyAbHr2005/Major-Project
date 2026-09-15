import sys

import pytest

from hospital_client.model_management.cli import main


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py"] + argv)
    main()


def test_cli_init_creates_version_one(tmp_path, monkeypatch, capsys):
    _run(["init", "--architecture", "resnet18", "--num-classes", "2", "--output", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    assert "version 1" in out
    assert (tmp_path / "resnet18" / "v1" / "model.safetensors").exists()
    assert (tmp_path / "resnet18" / "v1" / "metadata.json").exists()


def test_cli_init_with_explicit_model_id(tmp_path, monkeypatch):
    _run(["init", "--architecture", "mobilenet_v2", "--num-classes", "3", "--output", str(tmp_path), "--model-id", "chest_xray_mobilenet_v2"], monkeypatch)
    assert (tmp_path / "chest_xray_mobilenet_v2" / "v1").exists()


def test_cli_init_invalid_architecture_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["__main__.py", "init", "--architecture", "fake", "--num-classes", "2", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_list_shows_versions(tmp_path, monkeypatch, capsys):
    _run(["init", "--architecture", "resnet18", "--num-classes", "2", "--output", str(tmp_path)], monkeypatch)
    _run(["list", "resnet18", "--store", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    assert "v1" in out
    assert "resnet18" in out


def test_cli_list_no_versions(tmp_path, monkeypatch, capsys):
    _run(["list", "nothing_here", "--store", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    assert "No versions found" in out


def test_cli_inspect_valid_model(tmp_path, monkeypatch, capsys):
    _run(["init", "--architecture", "resnet18", "--num-classes", "2", "--output", str(tmp_path)], monkeypatch)
    _run(["inspect", "resnet18", "--store", str(tmp_path)], monkeypatch)
    out = capsys.readouterr().out
    assert "Valid: True" in out


def test_cli_inspect_invalid_model_exits_nonzero(tmp_path, monkeypatch):
    _run(["init", "--architecture", "resnet18", "--num-classes", "2", "--output", str(tmp_path)], monkeypatch)
    import os
    os.remove(tmp_path / "resnet18" / "v1" / "model.safetensors")

    monkeypatch.setattr(sys, "argv", ["__main__.py", "inspect", "resnet18", "--store", str(tmp_path)])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_cli_no_args_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["__main__.py"])
    main()
    out = capsys.readouterr().out
    assert "usage" in out.lower()
