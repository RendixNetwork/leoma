"""The evaluator runtime is consensus-checked, not merely written to an audit log."""

import pathlib
import tomllib

from leoma.eval import runtime_lock
from leoma.infra.chain_config import SPEC


def test_shipped_uv_lock_matches_the_consensus_pin():
    assert runtime_lock.observed_lock_digest() == SPEC.runtime.eval_lock_digest


def test_expected_numerical_packages_match_uv_lock_exactly():
    root = pathlib.Path(__file__).resolve().parents[2]
    with open(root / "uv.lock", "rb") as stream:
        packages = tomllib.load(stream)["package"]
    locked: dict[str, set[str]] = {}
    for package in packages:
        locked.setdefault(package["name"], set()).add(package["version"])

    for name, expected in runtime_lock.EXPECTED_EVAL_PACKAGES.items():
        assert locked.get(name) == {expected}, f"{name} drifted from the runtime contract"


def test_expected_runtime_digest_is_stable_and_content_addressed():
    digest = runtime_lock.expected_eval_runtime_digest(SPEC.runtime)
    assert digest.startswith("sha256:")
    assert digest == runtime_lock.expected_eval_runtime_digest(SPEC.runtime)


def test_matching_observed_runtime_is_compatible(monkeypatch):
    expected = runtime_lock.expected_runtime_identity(SPEC.runtime)
    runtime_lock.eval_runtime_report.cache_clear()
    monkeypatch.setattr(runtime_lock, "observed_runtime_identity", lambda: expected)
    report = runtime_lock.eval_runtime_report(SPEC.runtime)
    assert report["compatible"] is True
    assert report["digest"] == report["expected_digest"]
    assert report["issues"] == []
    runtime_lock.eval_runtime_report.cache_clear()


def test_one_package_drift_changes_digest_and_fails_closed(monkeypatch):
    observed = runtime_lock.expected_runtime_identity(SPEC.runtime)
    observed = {**observed, "packages": {**observed["packages"], "torch": "9.9.9"}}
    runtime_lock.eval_runtime_report.cache_clear()
    monkeypatch.setattr(runtime_lock, "observed_runtime_identity", lambda: observed)
    report = runtime_lock.eval_runtime_report(SPEC.runtime)
    assert report["compatible"] is False
    assert report["digest"] != report["expected_digest"]
    assert any("package torch" in issue for issue in report["issues"])
    runtime_lock.eval_runtime_report.cache_clear()
