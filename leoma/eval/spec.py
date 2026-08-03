"""The consensus surface: every input that can change a verdict, pinned in one place.

A duel must be a **pure function of the chain**. Today it is not: the prompt, the
frame count, the resolution, the metric — all of them come from per-box env vars
and dataclass defaults, so two honest validators can hand the same challenger two
different exams and disagree about who won.

This module is the fix. :class:`ConsensusSpec` is the complete set of duel inputs;
it is built once from ``chain.toml``, hashed into a :meth:`ConsensusSpec.digest`,
sent *explicitly* to the eval server with every request, and **echoed back in the
verdict** so the validator can prove the box actually used it
(:func:`verify_echo`). Anything not in here cannot influence a distance.

Two design rules do the real work:

* **No defaults.** Every field is required. A field with a default is a field a
  validator can silently forget — and forgetting it produces a *plausible* verdict
  that quietly disagrees with everyone else's. Missing field ⇒ loud 422.
* **``extra="forbid"``.** A newer validator sending a field an older box ignores
  would otherwise diverge in silence. Here it is a hard error at the door.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from leoma.eval.digests import digest_obj
from leoma.eval.errors import ConsensusConfigError


class _Pinned(BaseModel):
    """Strict base: no extra fields, no mutation after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CorpusSpec(_Pinned):
    """Which clips exist at all. Pinned by digest, never by a live listing."""

    bucket: str = Field(min_length=1)
    manifest_key: str = Field(min_length=1)
    #: sha256 of the manifest *bytes*. The whole corpus hangs off this one hash.
    #: Empty means **unpinned** — see :meth:`ConsensusSpec.require_duel_ready`.
    manifest_digest: str

    @model_validator(mode="after")
    def _digest_is_a_digest(self):
        if self.manifest_digest and not self.manifest_digest.startswith("sha256:"):
            raise ValueError(
                f"corpus.manifest_digest must be a 'sha256:...' value or empty, got "
                f"{self.manifest_digest!r}"
            )
        return self

    @property
    def pinned(self) -> bool:
        return bool(self.manifest_digest)


class GenSpec(_Pinned):
    """How king and challenger generate. Identical for both, by construction."""

    num_frames: int = Field(ge=2)
    fps: int = Field(ge=1)
    num_inference_steps: int = Field(ge=1)
    guidance_scale: float
    width: int = Field(ge=8)
    height: int = Field(ge=8)
    negative_prompt: str
    #: ``manifest`` = use the clip's own caption; ``fixed`` = use ``prompt`` for all.
    prompt_mode: Literal["manifest", "fixed"]
    prompt: str
    #: bf16 on GPU is the production path; fp32 exists for CPU test runs.
    dtype: Literal["bfloat16", "float16", "float32"]
    #: Placement is consensus-pinned because changing CPU offload can change frames,
    #: distances, and therefore the crown. Wan2.2 production uses ``model`` so its
    #: two transformer experts leave enough H100 VRAM for video activations.
    offload: Literal["none", "model", "sequential"]

    @model_validator(mode="after")
    def _fixed_prompt_is_present(self):
        if self.prompt_mode == "fixed" and not self.prompt:
            raise ValueError("gen.prompt_mode='fixed' requires a non-empty gen.prompt")
        return self


class DuelSpec(_Pinned):
    """How the generations are scored and how the verdict is decided."""

    metric: str = Field(min_length=1)
    #: **cpu** in production. Generation is already the one irreducibly fuzzy step
    #: across GPU architectures; running LPIPS/CLIP on the GPU too would add a
    #: second source of cross-validator noise for no benefit. Scoring on CPU in
    #: fp32 costs minutes against *hours* of generation, and makes the metric an
    #: exactly reproducible function of the frames.
    metric_device: Literal["cpu", "cuda"]
    n_clips: int = Field(ge=1)
    delta_threshold: float = Field(gt=0)
    alpha: float = Field(gt=0, lt=1)
    n_bootstrap: int = Field(ge=1)
    base_seed: int
    #: Early stopping is consensus-visible because enabling it can change how many
    #: clips are scored. It must remain disabled unless ``early_stop_factor`` is
    #: backed by a proven global upper bound for the pinned metric's per-clip
    #: advantage. An empirical quantile or a convenient multiple is not such a bound.
    early_stop_enabled: bool
    #: When enabled, each remaining clip is assumed able to contribute at most
    #: ``early_stop_factor × delta_threshold`` of challenger advantage.
    early_stop_factor: float = Field(ge=0)
    #: The freeze-baseline gate's margin, as a fraction of the cheat's own mean score.
    #: Dimensionless on purpose — see chain.toml.
    freeze_margin_fraction: float = Field(ge=0)


class ArchSpec(_Pinned):
    """The pinned architecture both duelists must be."""

    base_repo: str = Field(min_length=1)
    #: Immutable revision whose configs define the architecture lock. A repo name
    #: alone is mutable and could silently change which challengers validators accept.
    base_digest: str = Field(pattern=r"^(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")
    pipeline: str = Field(min_length=1)


class DeterminismSpec(_Pinned):
    """Torch knobs that make a run reproducible on a given GPU model.

    These do **not** buy bit-exactness across GPU architectures for a 14B bf16
    diffusion model, and no flag will. What they buy is run-to-run determinism on
    one box — which is what makes a disputed verdict *investigable*.
    """

    torch_deterministic: bool
    cudnn_benchmark: bool
    allow_tf32: bool


class EvalRuntimeSpec(_Pinned):
    """The execution environment allowed to produce consensus verdicts.

    Source-code equality is insufficient for diffusion inference: changing Torch,
    Diffusers, CUDA, Python, or the GPU architecture can change generated frames.
    The complete dependency resolution is anchored by ``eval_lock_digest`` while
    the remaining fields pin the hardware/runtime axes that do not live in uv.lock.
    """

    python: str = Field(pattern=r"^\d+\.\d+$")
    cuda: str = Field(pattern=r"^\d+\.\d+$")
    compute_capability: str = Field(pattern=r"^\d+\.\d+$")
    eval_lock_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class ConsensusSpec(_Pinned):
    """Everything that can change a verdict — and nothing that can't."""

    corpus: CorpusSpec
    gen: GenSpec
    duel: DuelSpec
    arch: ArchSpec
    determinism: DeterminismSpec
    runtime: EvalRuntimeSpec

    def digest(self) -> str:
        """The one hash that says "we are running the same exam"."""
        return digest_obj(self.model_dump(mode="json"))

    @property
    def early_stop_max_advantage(self) -> Optional[float]:
        """Return the pinned bound only when early stopping is explicitly enabled."""
        if not self.duel.early_stop_enabled:
            return None
        return self.duel.early_stop_factor * self.duel.delta_threshold

    def require_duel_ready(self) -> None:
        """Refuse to duel on an unpinned corpus — but let the process *start*.

        An unpinned ``manifest_digest`` means the corpus is whatever the bucket
        happens to hold today, which is not a consensus surface. It must not
        produce a verdict.

        It is deliberately **not** an import-time error, though. A validator that
        crash-loops cannot burn emissions, cannot publish a dashboard, and cannot
        tell its operator *why* it is unhappy — it just dies. So the process comes
        up, refuses to duel, burns to UID 0, and says so. Same reasoning as an
        unpinned ``seed_digest``: degrade loudly and safely, never crown blindly.
        """
        if not self.corpus.pinned:
            raise ConsensusConfigError(
                "chain.toml [corpus].manifest_digest is not pinned. The duel corpus "
                "would be whatever the bucket currently holds, which is not "
                "reproducible by other validators. Publish a manifest "
                "(`leoma corpus publish-manifest`) and pin its digest. Refusing to duel."
            )


def verify_echo(sent: ConsensusSpec, echoed: dict | None) -> None:
    """Fail closed unless the eval box echoed back *exactly* the spec we sent.

    This is the check that catches the entire "field not sent, default silently
    used" bug class — including the version of it where a stale eval box quietly
    ignores a field a newer validator added. Called **before crowning**, so a
    verdict produced under different parameters can never take the crown.
    """
    if not echoed:
        raise ConsensusConfigError(
            "eval server returned no consensus echo — it is running a build that "
            "predates the pinned consensus surface and may have used its own "
            "generation parameters. Refusing the verdict."
        )
    try:
        parsed = ConsensusSpec.model_validate(echoed)
    except Exception as e:  # pydantic ValidationError
        raise ConsensusConfigError(f"eval server echoed a malformed spec: {e}") from e

    if parsed.digest() != sent.digest():
        raise ConsensusConfigError(
            "eval server ran a DIFFERENT consensus spec than the one it was sent "
            f"(sent {sent.digest()[:19]}…, echoed {parsed.digest()[:19]}…). "
            f"Differences: {_diff(sent, parsed)}"
        )


def _diff(a: ConsensusSpec, b: ConsensusSpec) -> str:
    """Human-readable field-level diff — the operator needs to know *what* drifted."""
    da, db = a.model_dump(mode="json"), b.model_dump(mode="json")
    out: list[str] = []
    for section in sorted(set(da) | set(db)):
        sa, sb = da.get(section, {}), db.get(section, {})
        for key in sorted(set(sa) | set(sb)):
            if sa.get(key) != sb.get(key):
                out.append(f"{section}.{key}: sent={sa.get(key)!r} echoed={sb.get(key)!r}")
    return "; ".join(out) or "(digest differs but no field does — check float encoding)"


__all__ = [
    "ArchSpec",
    "ConsensusSpec",
    "CorpusSpec",
    "DeterminismSpec",
    "DuelSpec",
    "EvalRuntimeSpec",
    "GenSpec",
    "verify_echo",
]
