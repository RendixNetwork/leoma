"""Fail-closed identity for the evaluator's numerical runtime.

The scoring-code digest catches stale Leoma source, but the same Python files can
produce different frames under another Torch, Diffusers, CUDA, or GPU runtime.
This module turns those previously audit-only facts into a compatibility contract:

* ``chain.toml [runtime]`` pins Python, CUDA, compute capability, and uv.lock;
* the key generation/scoring package versions below pin what is actually installed;
* the evaluator publishes a canonical digest over the observed identity; and
* the validator derives the expected digest locally and rejects any difference.

The health report is self-reported, so it is an operator-drift guard rather than a
remote-attestation system. Immutable image digests remain the deployment trust root.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
from functools import lru_cache
from importlib import metadata
from typing import Any

from leoma.eval.digests import digest_obj
from leoma.eval.spec import EvalRuntimeSpec


# Packages that directly implement model loading, generation, decoding, and scoring.
# These values deliberately mirror the exact versions resolved in uv.lock. Updating
# one is a consensus/runtime change, not an ordinary unattended dependency bump.
EXPECTED_EVAL_PACKAGES: dict[str, str] = {
    "accelerate": "1.10.1",
    "diffusers": "0.35.2",
    "huggingface-hub": "1.6.0",
    "lpips": "0.1.4",
    "numpy": "2.2.6",
    "open-clip-torch": "3.3.0",
    "opencv-python-headless": "4.12.0.88",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "scipy": "1.18.0",
    "tokenizers": "0.22.2",
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "5.14.1",
}


def _lock_path() -> pathlib.Path | None:
    configured = os.environ.get("LEOMA_UV_LOCK_PATH", "").strip()
    candidates = [pathlib.Path(configured)] if configured else []
    candidates.extend(
        (
            pathlib.Path.cwd() / "uv.lock",
            pathlib.Path(__file__).resolve().parents[2] / "uv.lock",
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def observed_lock_digest() -> str | None:
    """Digest the lock file used to build this evaluator, or return ``None``."""
    path = _lock_path()
    if path is None:
        return None
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def expected_runtime_identity(spec: EvalRuntimeSpec) -> dict[str, Any]:
    """Canonical runtime identity every healthy evaluator must report."""
    return {
        "python": spec.python,
        "cuda": spec.cuda,
        "compute_capabilities": [spec.compute_capability],
        "eval_lock_digest": spec.eval_lock_digest,
        "packages": dict(sorted(EXPECTED_EVAL_PACKAGES.items())),
    }


def expected_eval_runtime_digest(spec: EvalRuntimeSpec) -> str:
    return digest_obj(expected_runtime_identity(spec))


def _installed_packages() -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    for package in EXPECTED_EVAL_PACKAGES:
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = None
    return dict(sorted(installed.items()))


def observed_runtime_identity() -> dict[str, Any]:
    """Read the evaluator environment without silently substituting defaults."""
    cuda: str | None = None
    capabilities: list[str] = []
    try:
        import torch

        cuda = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            capabilities = sorted(
                {
                    f"{major}.{minor}"
                    for major, minor in (
                        torch.cuda.get_device_capability(index)
                        for index in range(torch.cuda.device_count())
                    )
                }
            )
    except Exception:  # noqa: BLE001 — incompatibility is data in the health report
        pass

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "cuda": cuda,
        "compute_capabilities": capabilities,
        "eval_lock_digest": observed_lock_digest(),
        "packages": _installed_packages(),
    }


def _identity_diff(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in ("python", "cuda", "compute_capabilities", "eval_lock_digest"):
        if observed.get(field) != expected.get(field):
            issues.append(
                f"{field}: expected {expected.get(field)!r}, observed {observed.get(field)!r}"
            )
    expected_packages = expected["packages"]
    observed_packages = observed.get("packages") or {}
    for package, version in expected_packages.items():
        if observed_packages.get(package) != version:
            issues.append(
                f"package {package}: expected {version!r}, "
                f"observed {observed_packages.get(package)!r}"
            )
    return issues


@lru_cache(maxsize=1)
def eval_runtime_report(spec: EvalRuntimeSpec) -> dict[str, Any]:
    """Observed identity, expected digest, and a fail-closed compatibility verdict."""
    expected = expected_runtime_identity(spec)
    observed = observed_runtime_identity()
    issues = _identity_diff(expected, observed)
    return {
        "compatible": not issues,
        "digest": digest_obj(observed),
        "expected_digest": digest_obj(expected),
        "identity": observed,
        "issues": issues,
    }


__all__ = [
    "EXPECTED_EVAL_PACKAGES",
    "eval_runtime_report",
    "expected_eval_runtime_digest",
    "expected_runtime_identity",
    "observed_lock_digest",
    "observed_runtime_identity",
]
