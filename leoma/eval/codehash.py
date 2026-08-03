"""A digest of the scoring code itself.

Pinning the *inputs* to a duel is only half the job. If one eval box is running a
build whose LPIPS wrapper resizes differently, or whose bootstrap draws its
resamples in a different order, it will produce distances nobody else can
reproduce — from provably identical inputs.

``eval_code_digest()`` hashes the modules that can change a distance, plus
``chain.toml``. The validator asks a box for it over ``/health`` **before** handing
it an hours-long duel, and it lands in the verdict's audit block. This does not
prevent a modified box from lying about its own hash — nothing self-reported can.
What it prevents is the far more likely failure: an operator who forgot to redeploy
one machine, silently poisoning consensus for weeks.
"""
from __future__ import annotations

import hashlib
import pathlib
from functools import lru_cache

#: Modules whose bytes can change a distance or a verdict. Deliberately *not* the
#: whole package: the validator's own policy code (retries, quarantine) is local by
#: design and must not make two boxes look incompatible.
SCORED_MODULES = (
    "eval/metrics.py",
    "eval/_lpips.py",
    "eval/_flow.py",
    "eval/_clip.py",
    "eval/bootstrap.py",
    "eval/video_runner.py",
    "eval/devices.py",
    "eval/arch_lock.py",
    "eval/dataset.py",
    "eval/manifest.py",
    "eval/spec.py",
    "eval/digests.py",
    "eval/determinism.py",
    "eval/runtime_lock.py",
    "eval_server.py",
    "app/validator/prescreen.py",
    "infra/model_store.py",
    "chain.toml",
)


@lru_cache(maxsize=1)
def eval_code_digest() -> str:
    """``sha256:`` over the scoring modules + the pinned chain config."""
    pkg = pathlib.Path(__file__).resolve().parent.parent  # .../leoma/
    h = hashlib.sha256()
    for rel in SCORED_MODULES:  # fixed order — never os.listdir()
        path = pkg / rel
        h.update(rel.encode())
        h.update(b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            # A module that is *absent* must hash differently from one that is
            # empty, or a botched deploy could collide with a healthy one.
            h.update(b"<missing>")
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


__all__ = ["SCORED_MODULES", "eval_code_digest"]
