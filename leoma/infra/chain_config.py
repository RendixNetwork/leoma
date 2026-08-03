"""Single source of truth for the pinned architecture + genesis king.

``chain.toml`` ships **inside the package** (``leoma/chain.toml``) because it is
consensus-critical: the pin must travel with the code, in a wheel as well as in a
git checkout. It is read at import and exposes the constants used by the miner
(default challenger namespace), the validator (repo naming + config-lock), and the
eval server (arch loading + seed king). To swap the pinned base model or the
genesis king, edit ``chain.toml`` — no code edits should be necessary.

Override knob: ``LEOMA_CHAIN_OVERRIDE`` env var points at an alternate TOML
(absolute, or relative to the current working directory). Used by local testing and
archived alternate configs so the shipped ``chain.toml`` can stay pointed at the
live chain.
"""
from __future__ import annotations

import importlib
import importlib.resources
import os
import pathlib
import re
import tomllib
from types import ModuleType

from leoma.eval.spec import (
    ArchSpec,
    ConsensusSpec,
    CorpusSpec,
    DeterminismSpec,
    DuelSpec,
    EvalRuntimeSpec,
    GenSpec,
)


def _resolve_toml_path() -> pathlib.Path:
    """Locate chain.toml in a wheel, an editable install, or the repo.

    Order: LEOMA_CHAIN_OVERRIDE -> packaged (leoma/chain.toml) -> legacy repo root.
    """
    override = os.environ.get("LEOMA_CHAIN_OVERRIDE", "").strip()
    if override:
        candidate = pathlib.Path(override)
        if not candidate.is_absolute():
            candidate = pathlib.Path.cwd() / candidate
        if not candidate.is_file():
            raise RuntimeError(f"LEOMA_CHAIN_OVERRIDE points at a missing file: {candidate}")
        return candidate

    # Packaged location — works for both a wheel and an editable install.
    try:
        packaged = importlib.resources.files("leoma") / "chain.toml"
        if packaged.is_file():
            return pathlib.Path(str(packaged))
    except (ModuleNotFoundError, AttributeError, TypeError):
        pass

    pkg_dir = pathlib.Path(__file__).resolve().parents[1]  # .../leoma/
    for candidate in (pkg_dir / "chain.toml", pkg_dir.parent / "chain.toml"):
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "chain.toml not found. It must ship inside the package (leoma/chain.toml); "
        "ensure [tool.setuptools.package-data] includes it, or set LEOMA_CHAIN_OVERRIDE."
    )


_TOML_PATH = _resolve_toml_path()

with open(_TOML_PATH, "rb") as _f:
    _doc = tomllib.load(_f)

_chain = _doc.get("chain", {})
_arch = _doc.get("arch", {})
_seed = _doc.get("seed", {})
_corpus = _doc.get("corpus", {})
_gen = _doc.get("gen", {})
_duel = _doc.get("duel", {})
_determinism = _doc.get("determinism", {})
_runtime = _doc.get("runtime", {})

_VALID_SEED_REPO_BACKENDS = {"hf", "hippius"}


def _default_seed_repo_backend(seed_digest: str) -> str:
    digest = (seed_digest or "").strip()
    if digest.startswith("sha256:"):
        return "hippius"
    if digest.startswith("hf:"):
        return "hf"
    return "hf"


NAME: str = _chain["name"]
SEED_REPO: str = _chain["seed_repo"]
REPO_PATTERN: str = _chain.get("repo_pattern") or rf"^[^/]+/{re.escape(NAME)}-.+$"

# Optional custom-code module whose import side effect registers config/model
# classes (empty for stock diffusers pipelines resolved by class name).
ARCH_MODULE: str = _arch.get("module", "")
# The pinned video-gen base: every challenger is weights for THIS architecture.
ARCH_BASE_REPO: str = _arch.get("base_repo", "")
# Immutable config revision for the base architecture. Pinning only the repo name
# would let a mutable tag silently change the prescreen decision across validators.
ARCH_BASE_DIGEST: str = _arch.get("base_digest", "")
# diffusers pipeline class used to load king + challenger (e.g. an I2V pipeline).
ARCH_PIPELINE: str = _arch.get("pipeline", "")
# Config keys locked to the base arch's values (challenger config must match).
EXTRA_LOCK_KEYS: tuple[str, ...] = tuple(_arch.get("extra_lock_keys", []))

SEED_DIGEST: str = _seed.get("seed_digest", "")
SEED_REPO_BACKEND: str = (_seed.get("repo_backend") or _default_seed_repo_backend(SEED_DIGEST)).strip().lower()
if SEED_REPO_BACKEND not in _VALID_SEED_REPO_BACKENDS:
    raise RuntimeError(
        f"chain.toml [seed].repo_backend must be one of "
        f"{sorted(_VALID_SEED_REPO_BACKENDS)}, got {SEED_REPO_BACKEND!r}"
    )

# Namespace inferred from the seed repo. Miners default their challenger repo to
# "<namespace>/<NAME>-<suffix>" though they can override to publish under their
# own account.
SEED_NAMESPACE: str = SEED_REPO.split("/", 1)[0] if "/" in SEED_REPO else ""


def _build_spec() -> ConsensusSpec:
    """Validate the pinned consensus surface — loudly, at import.

    A missing or malformed field here is a **startup** failure, not a runtime one.
    That is the whole point: a validator that silently defaults ``num_frames`` or
    ``metric`` produces a plausible verdict that quietly disagrees with the rest of
    the subnet, and nobody notices for weeks. Better to refuse to boot with a
    message naming the field.

    (An *unpinned* corpus digest is the one exception — see
    ``ConsensusSpec.require_duel_ready``. That degrades to "run, but burn", because
    a crash-looping validator can't even tell you why it's unhappy.)
    """
    try:
        return ConsensusSpec(
            corpus=CorpusSpec(**_corpus),
            gen=GenSpec(**_gen),
            duel=DuelSpec(**_duel),
            arch=ArchSpec(
                base_repo=ARCH_BASE_REPO,
                base_digest=ARCH_BASE_DIGEST,
                pipeline=ARCH_PIPELINE,
            ),
            determinism=DeterminismSpec(**_determinism),
            runtime=EvalRuntimeSpec(**_runtime),
        )
    except Exception as e:  # pydantic ValidationError / TypeError
        raise RuntimeError(
            f"chain.toml does not define a valid consensus surface ({_TOML_PATH}):\n{e}\n\n"
            "Every field of [corpus], [gen], [duel], [determinism] and [runtime] is "
            "REQUIRED and has "
            "no default. A field with a default is a field a validator can silently forget, "
            "and a forgotten field means two validators grade the same challenger "
            "differently."
        ) from e


#: The pinned consensus surface. Sent with every eval request, echoed in every verdict.
SPEC: ConsensusSpec = _build_spec()

#: The one hash that says "we are running the same exam".
CONSENSUS_DIGEST: str = SPEC.digest()


def load_arch() -> ModuleType:
    """Import the configured custom-arch module (registers HF/diffusers classes).

    Only needed for architectures that ship custom modeling code; stock
    diffusers pipelines are resolved by ``ARCH_PIPELINE`` class name instead.
    """
    if not ARCH_MODULE:
        raise RuntimeError("chain.toml [arch].module is not set (stock pipeline?)")
    return importlib.import_module(ARCH_MODULE)


__all__ = [
    "NAME",
    "SEED_REPO",
    "REPO_PATTERN",
    "ARCH_MODULE",
    "ARCH_BASE_REPO",
    "ARCH_BASE_DIGEST",
    "ARCH_PIPELINE",
    "EXTRA_LOCK_KEYS",
    "SEED_DIGEST",
    "SEED_REPO_BACKEND",
    "SEED_NAMESPACE",
    "SPEC",
    "CONSENSUS_DIGEST",
    "load_arch",
]
