"""Untrusted miner repositories cannot become evaluator code."""

import json
import sys
from types import SimpleNamespace

import pytest

from leoma.infra import model_store as ms
from leoma.infra.model_store import (
    COMPLETION_MARKER,
    ModelRef,
    UnsafeSnapshotError,
    materialize_model,
    validate_snapshot_files,
)
from leoma.app.validator.failures import ErrorClass, classify


DIGEST = "sha256:" + "a" * 64


def _snapshot(root):
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "WanImageToVideoPipeline"}))
    component = root / "transformer"
    component.mkdir()
    (component / "config.json").write_text("{}")
    (component / "weights.safetensors").write_bytes(b"safe")
    return root


def test_snapshot_allowlist_accepts_only_data_and_safetensors(tmp_path):
    snapshot = _snapshot(tmp_path / "model")
    validate_snapshot_files(snapshot)

    (snapshot / "pipeline.py").write_text("raise RuntimeError('executed')")
    with pytest.raises(UnsafeSnapshotError, match="forbidden file 'pipeline.py'"):
        validate_snapshot_files(snapshot)


def test_snapshot_allowlist_rejects_pickle_weights(tmp_path):
    snapshot = _snapshot(tmp_path / "model")
    (snapshot / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(UnsafeSnapshotError, match="pytorch_model.bin"):
        validate_snapshot_files(snapshot)


def test_snapshot_allowlist_rejects_symlinks(tmp_path):
    snapshot = _snapshot(tmp_path / "model")
    (snapshot / "linked.json").symlink_to(snapshot / "model_index.json")
    with pytest.raises(UnsafeSnapshotError, match="non-regular file"):
        validate_snapshot_files(snapshot)


def test_completed_cache_is_revalidated_before_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "MODEL_CACHE_DIR", str(tmp_path))
    ref = ModelRef("u/leoma-model", DIGEST)
    target = _snapshot(tmp_path / "u--leoma-model" / "snapshots" / f"sha256-{'a' * 64}")
    (target / COMPLETION_MARKER).write_text("{}")
    (target / "injected.py").write_text("raise RuntimeError('executed')")

    with pytest.raises(UnsafeSnapshotError, match="injected.py"):
        materialize_model(ref)

    assert classify(UnsafeSnapshotError("forbidden")).kind is ErrorClass.PERMANENT


def test_pipeline_loader_forces_offline_safetensors_and_disables_remote_code(monkeypatch):
    from leoma.eval import video_runner

    captured = {}

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured.update(path=path, **kwargs)
            return cls()

        def to(self, device):
            captured["device"] = device
            return self

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float32="float32",
    )
    fake_diffusers = SimpleNamespace(
        __version__="test",
        WanImageToVideoPipeline=Pipeline,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    gen = SimpleNamespace(dtype="bfloat16", offload="none")
    video_runner.load_video_pipeline("/safe/model", gen=gen, device="cpu")

    assert captured == {
        "path": "/safe/model",
        "torch_dtype": "float32",
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "device": "cpu",
    }
