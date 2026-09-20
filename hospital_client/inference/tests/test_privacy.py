"""
F. The local-only boundary, made testable. The design goal is that the
inference package has NO network capability at all, so the strongest evidence
is structural (no network imports) plus behavioural (it still works when every
network primitive is booby-trapped, and it leaves every file untouched).
"""
import ast
import base64
import glob
import hashlib
import io
import json
import os

import pytest

from hospital_client.inference.engine import InferenceConfig, LocalInferenceEngine
from hospital_client.inference.provisioning import LocalStoreModelProvider
from hospital_client.inference.result import format_summary
from hospital_client.inference.service import LocalInferenceService
from hospital_client.inference.tests.conftest import build_artifact, make_image

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_IMPORTS = {
    "socket", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "websocket", "websockets", "xmlrpc", "socketio", "flwr", "pymongo", "motor", "boto3", "subprocess",
}


def _package_sources():
    return [p for p in glob.glob(os.path.join(PACKAGE_DIR, "*.py"))]


def test_inference_package_imports_no_network_or_central_service_library():
    assert len(_package_sources()) >= 10
    offenders = []
    for path in _package_sources():
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append((os.path.basename(path), name))
    assert offenders == []


@pytest.fixture
def network_trap(monkeypatch):
    """Any attempt to use the network fails the test loudly."""
    import http.client
    import socket
    import urllib.request

    attempts = []

    def trap(name):
        def _raise(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"network primitive used during inference: {name}")
        return _raise

    for owner, attr in [
        (socket, "create_connection"), (socket, "getaddrinfo"), (socket, "gethostbyname"),
        (socket.socket, "connect"), (socket.socket, "connect_ex"), (socket.socket, "sendto"),
        (urllib.request, "urlopen"), (http.client.HTTPConnection, "connect"), (http.client.HTTPSConnection, "connect"),
    ]:
        monkeypatch.setattr(owner, attr, trap(f"{owner.__name__}.{attr}"))
    return attempts


def test_full_inference_works_offline(tmp_path, network_trap):
    artifact = build_artifact(tmp_path / "models")
    engine = LocalInferenceEngine(artifact, InferenceConfig(device="cpu"))
    result = engine.predict(make_image(tmp_path / "x.png"))
    engine.close()
    assert result.succeeded and network_trap == []


def test_service_and_auto_device_selection_also_make_no_network_call(tmp_path, network_trap):
    artifact = build_artifact(tmp_path / "models")
    service = LocalInferenceService()
    service.load_model(LocalStoreModelProvider(artifact.store_root, "m", 1), InferenceConfig(device="auto"))
    assert service.predict(make_image(tmp_path / "x.png")).succeeded
    service.unload()
    assert network_trap == []  # Module 8's network probe is disabled on this path


def test_cli_makes_no_network_call(tmp_path, network_trap, monkeypatch, capsys):
    import sys
    from hospital_client.inference.cli import main
    artifact = build_artifact(tmp_path / "models")
    monkeypatch.setattr(sys, "argv", ["x", "predict", "--models-root", artifact.store_root, "--model-id", "m", "--version", "1",
                                      "--image", make_image(tmp_path / "x.png")])
    main()
    assert "Predicted class" in capsys.readouterr().out and network_trap == []


def _snapshot(*roots):
    state = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for name in files:
                path = os.path.join(dirpath, name)
                state[path] = (hashlib.sha256(open(path, "rb").read()).hexdigest(), os.stat(path).st_mtime_ns)
    return state


def test_inference_leaves_the_image_and_the_filesystem_untouched(tmp_path):
    (tmp_path / "images").mkdir()
    image = make_image(tmp_path / "images" / "patient_scan.png")
    artifact = build_artifact(tmp_path / "models")
    before = _snapshot(tmp_path)
    engine = LocalInferenceEngine(artifact)
    for _ in range(3):
        engine.predict(image)
    engine.close()
    assert _snapshot(tmp_path) == before  # no new files (no copies/caches/temp), no modified bytes or mtimes
    assert os.path.exists(image)  # the raw image remains exactly where it was


def test_result_never_contains_pixels_paths_filenames_or_base64(tmp_path):
    secret_name = "Alex_Kim_DOB19700101_xray"
    image = make_image(tmp_path / f"{secret_name}.png", size=(64, 64))
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    result = engine.predict(image, reference="opaque-ref-1")
    failed = engine.predict(str(tmp_path / f"{secret_name}_missing.png"))
    engine.close()

    raw = open(image, "rb").read()
    for r in (result, failed):
        blob = json.dumps(r.to_dict()) + format_summary(r)
        assert secret_name not in blob and str(tmp_path) not in blob and image not in blob
        assert base64.b64encode(raw).decode()[:64] not in blob
        assert raw[:32].hex() not in blob
        assert len(blob) < 20_000  # a result is metadata, never an embedded image


def test_no_image_data_is_ever_serialized_by_the_engine(tmp_path, monkeypatch):
    import pickle
    import PIL.Image
    engine = LocalInferenceEngine(build_artifact(tmp_path / "models"))
    image = make_image(tmp_path / "x.png")  # created BEFORE the traps (creating it legitimately uses Image.save)
    forbidden = []
    # (Image.tobytes is NOT trapped: torchvision's in-memory ToTensor uses it legitimately.)
    monkeypatch.setattr(PIL.Image.Image, "save", lambda *a, **k: forbidden.append("Image.save"))
    monkeypatch.setattr(pickle, "dumps", lambda *a, **k: forbidden.append("pickle.dumps"))
    monkeypatch.setattr(base64, "b64encode", lambda *a, **k: forbidden.append("base64.b64encode"))
    assert engine.predict(image).succeeded
    assert forbidden == []
    engine.close()
