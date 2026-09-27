import pytest

from hospital_client.model_management.store import ModelStore


@pytest.fixture
def store(tmp_path):
    return ModelStore(str(tmp_path))


@pytest.mark.parametrize("bad_model_id", [
    "../../etc/passwd",
    "..\\..\\windows\\system32",
    "/etc/passwd",
    "C:\\Windows\\System32",
    "a/b",
    "a\\b",
    "..",
    "",
    "a" * 300,
])
def test_invalid_model_ids_rejected(store, bad_model_id):
    with pytest.raises(ValueError):
        store._model_dir(bad_model_id)


@pytest.mark.parametrize("bad_version", [0, -1, 1.5, "1", None, True])
def test_invalid_versions_rejected(store, bad_version):
    with pytest.raises(ValueError):
        store._version_dir("valid_model_id", bad_version)


def test_valid_model_id_and_version_accepted(store):
    path = store._version_dir("chest_xray_resnet18", 1)
    assert path.startswith(store.root)


def test_resolved_path_never_escapes_root(store, tmp_path):
    # Even a "valid-looking" id cannot resolve outside root once joined -
    # the whitelist regex already prevents traversal, this is defense in depth.
    path = store._model_dir("safe_id")
    assert str(tmp_path) in path
