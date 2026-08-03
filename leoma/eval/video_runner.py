"""Video-generation king-of-the-hill duel runner.

On the same held-out clips, king and challenger each generate a continuation from
the clip's first frame + prompt using the SAME per-clip seed; each generation is
scored against the real continuation with a reference metric, and the per-clip
distances feed the paired bootstrap verdict.

``run_duel`` is the pure orchestration — it takes *generate* callables and a
*distance* function, so the whole scoring→verdict flow is unit-testable with numpy
fakes. The torch/diffusers loading + generation used in production is imported
lazily (``load_video_pipeline`` / ``generate``); ``duel`` wires those into
``run_duel``.

Two things ``run_duel`` now records that it did not before, both so a disagreement
between validators can be *localized* instead of argued about:

* **per-clip generated-frame digests** — if two validators' distances differ, the
  digests say immediately whether the generations differed (GPU noise) or only the
  scoring did (a broken box);
* **the clip's manifest id**, alongside the index the seed was derived from.

``run_duel`` can also generate king and challenger **concurrently** (``concurrent=
True``): each clip's two generations are independent (same seed, different pipeline),
so on a multi-GPU box they can run on separate devices at the same time instead of
one after the other. This is a throughput setting, not a consensus one — see
``eval/devices.py`` for why, and for the one thing an operator must not forget when
turning it on.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from leoma.eval.bootstrap import can_still_win, paired_bootstrap_verdict
from leoma.eval.digests import digest_frames
from leoma.eval.errors import DuelCancelled, is_cuda_fatal
from leoma.eval.guards import validate_generation
from leoma.eval.metrics import Metric, get_metric
from leoma.app.validator.seeds import clip_generation_seed


@dataclass(frozen=True)
class GenParams:
    """Generation knobs shared by king and challenger (identical for both, by construction).

    Built from the pinned :class:`~leoma.eval.spec.GenSpec` — the defaults here are
    only for tests. In production nothing constructs a ``GenParams`` out of thin
    air: ``from_spec`` is the only path, because a default that silently differs
    from the chain's pin is the exact bug this whole tier exists to kill.
    """

    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    width: int = 832
    height: int = 480
    negative_prompt: str = "low quality, blurry, distorted"
    dtype: str = "bfloat16"
    offload: str = "none"

    @classmethod
    def from_spec(cls, gen) -> "GenParams":
        return cls(
            num_frames=gen.num_frames,
            fps=gen.fps,
            num_inference_steps=gen.num_inference_steps,
            guidance_scale=gen.guidance_scale,
            width=gen.width,
            height=gen.height,
            negative_prompt=gen.negative_prompt,
            dtype=gen.dtype,
            offload=gen.offload,
        )


@dataclass
class Clip:
    """One held-out duel item."""

    clip_index: int                    # index into the corpus manifest; the seed hangs off it
    first_frame: np.ndarray            # (H, W, C) — the conditioning image
    prompt: str
    truth_frames: np.ndarray           # (T, H, W, C) — the real continuation (ground truth)
    params: GenParams = field(default_factory=GenParams)
    clip_id: str = ""                  # the manifest's stable id, for the audit block


# A generate callable: (clip, seed) -> generated frames (T, H, W, C).
GenerateFn = Callable[[Clip, int], np.ndarray]


def _generate_pair(
    generate_king: GenerateFn,
    generate_challenger: GenerateFn,
    clip: Clip,
    gseed: int,
    executor: Optional[ThreadPoolExecutor],
) -> tuple[np.ndarray, np.ndarray]:
    """Run both generators for one clip — concurrently if an executor is given.

    Only the raw pipeline call is concurrent; validation, digesting and scoring stay
    on the caller's thread afterward exactly as in the sequential path, so this is the
    *only* place threading touches the duel at all.

    Both futures are always retrieved before either exception is raised. Calling
    ``king_future.result()`` first and letting it raise immediately would leave
    ``chall_future``'s outcome never retrieved — and an exception nobody ever calls
    ``.result()``/``.exception()`` on is simply dropped, not logged, not re-raised,
    gone. That is exactly how a CUDA-fatal error on the CHALLENGER side could vanish
    behind a merely-benign KING-side error: the eval server's self-kill logic only
    ever inspects the one exception that actually propagates out of ``run_duel``, so a
    dropped fatal exception means a poisoned CUDA context is never detected, the lock
    is released, and the *next* challenger inherits a dead GPU.
    """
    if executor is None:
        return generate_king(clip, gseed), generate_challenger(clip, gseed)
    king_future = executor.submit(generate_king, clip, gseed)
    chall_future = executor.submit(generate_challenger, clip, gseed)

    king_exc = king_future.exception()   # blocks until done; never raises itself
    chall_exc = chall_future.exception()

    if king_exc is None and chall_exc is None:
        return king_future.result(), chall_future.result()

    # At least one side failed. A CUDA-fatal exception must be what propagates,
    # whichever side it came from — the context is poisoned regardless of which
    # duelist's generate() call happened to surface it. Otherwise, king-first,
    # matching the order generation always ran in before concurrency existed.
    chall_is_the_fatal_one = chall_exc is not None and is_cuda_fatal(chall_exc) and (
        king_exc is None or not is_cuda_fatal(king_exc)
    )
    if chall_is_the_fatal_one:
        raise chall_exc
    if king_exc is not None:
        raise king_exc
    raise chall_exc


def run_duel(
    clips: Sequence[Clip],
    generate_king: GenerateFn,
    generate_challenger: GenerateFn,
    distance_fn: Metric,
    *,
    master_seed: int,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    on_phase: Optional[Callable[[dict], None]] = None,
    early_stop_max_advantage: Optional[float] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    freeze_margin_fraction: Optional[float] = None,
    concurrent: bool = False,
) -> dict:
    """Score every clip and return the verdict plus per-clip distances.

    ``master_seed`` drives the per-clip generation seed (identical for king and
    challenger on a given clip). If ``early_stop_max_advantage`` is set, the duel
    abandons early once the challenger provably can't clear the threshold — the
    verdict stays "king" in that case, so the outcome is unchanged and a validator
    running with early stop still agrees with one running without it.

    ``should_cancel`` is checked **between clips** — the only honest granularity. A
    single generation is one uninterruptible call into diffusers; pretending we can
    stop mid-clip would just be a lie that leaves the GPU busy anyway.

    ``freeze_margin_fraction`` enables the freeze-baseline gate: the challenger must
    also beat a frozen conditioning frame, confidently. See ``eval/baselines.py``.

    ``concurrent=True`` runs each clip's king and challenger generation at the same
    time (on separate devices, if the pipelines were loaded onto separate devices —
    see ``eval/devices.py``). It changes only *when* the two generations happen, never
    *what* they compute: the seed, the pipeline, and every downstream step (validation,
    digesting, scoring) are identical either way. A pinned test asserts the two modes
    produce byte-identical verdicts.
    """
    if not clips:
        raise ValueError("no clips to duel on")

    king_scores: list[float] = []
    challenger_scores: list[float] = []
    per_clip: list[dict] = []

    executor = ThreadPoolExecutor(max_workers=2) if concurrent else None
    try:
        for pos, clip in enumerate(clips):
            if should_cancel and should_cancel():
                raise DuelCancelled(f"duel cancelled after {pos} of {len(clips)} clips")

            gseed = clip_generation_seed(master_seed, clip.clip_index)
            p = clip.params
            expected = int(np.asarray(clip.truth_frames).shape[0])

            raw_king, raw_chall = _generate_pair(generate_king, generate_challenger, clip, gseed, executor)

            king_frames = validate_generation(
                raw_king, expected_frames=expected, width=p.width, height=p.height, who="king",
            )
            chall_frames = validate_generation(
                raw_chall, expected_frames=expected, width=p.width, height=p.height, who="challenger",
            )

            kd = float(distance_fn(king_frames, clip.truth_frames))
            cd = float(distance_fn(chall_frames, clip.truth_frames))
            king_scores.append(kd)
            challenger_scores.append(cd)
            per_clip.append({
                "clip_index": clip.clip_index,
                "clip_id": clip.clip_id,
                "gen_seed": gseed,
                "king_distance": kd,
                "challenger_distance": cd,
                "king_frames_digest": digest_frames(king_frames),
                "challenger_frames_digest": digest_frames(chall_frames),
            })

            if on_phase:
                on_phase({
                    "phase": "scored_clip",
                    "position": pos + 1,
                    "total": len(clips),
                    "clip_id": clip.clip_id,
                    "king_distance": round(kd, 6),
                    "challenger_distance": round(cd, 6),
                })

            if early_stop_max_advantage is not None and not can_still_win(
                king_scores, challenger_scores, remaining=len(clips) - (pos + 1),
                delta_threshold=delta_threshold, best_possible_advantage=early_stop_max_advantage,
            ):
                if on_phase:
                    on_phase({"phase": "early_stop", "reason": "challenger cannot clear threshold"})
                verdict = paired_bootstrap_verdict(
                    king_scores, challenger_scores,
                    delta_threshold=delta_threshold, alpha=alpha, n_bootstrap=n_bootstrap, seed=master_seed,
                )
                verdict["early_stopped"] = True
                verdict["per_clip"] = per_clip
                return verdict
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    verdict = paired_bootstrap_verdict(
        king_scores, challenger_scores,
        delta_threshold=delta_threshold, alpha=alpha, n_bootstrap=n_bootstrap, seed=master_seed,
    )
    verdict["early_stopped"] = False
    verdict["per_clip"] = per_clip

    # Copy-of-king, caught at duel time — and free.
    #
    # The duel is deterministic: the same weights, the same seed and the same clip
    # produce bit-identical frames. So if the challenger's generations are identical
    # to the king's on EVERY clip, the challenger IS the king — whatever repo it was
    # uploaded under and whatever digest it carries. No amount of repackaging survives
    # this, because it compares what the models *do*, not what they claim to be.
    #
    # It also cannot false-positive on an honest model: two independently trained 14B
    # diffusion models do not produce bit-identical video on 32 clips.
    if per_clip and all(
        row["king_frames_digest"] == row["challenger_frames_digest"] for row in per_clip
    ):
        verdict["accepted"] = False
        verdict["verdict"] = "king"
        verdict["rejected_by"] = "copy_of_king"
        verdict["reason"] = (
            "the challenger generated bit-identical video to the king on every clip — "
            "it is the king's own weights, repackaged. Copying the incumbent is not an "
            "improvement on it."
        )
        if on_phase:
            on_phase({"phase": "copy_of_king", "clips": len(per_clip)})
        return verdict

    if freeze_margin_fraction is not None:
        _apply_freeze_gate(
            verdict, clips, king_scores, challenger_scores, distance_fn,
            margin_fraction=freeze_margin_fraction, alpha=alpha,
            n_bootstrap=n_bootstrap, seed=master_seed, on_phase=on_phase,
        )

    return verdict


def _apply_freeze_gate(
    verdict: dict,
    clips: Sequence[Clip],
    king_scores: Sequence[float],
    challenger_scores: Sequence[float],
    distance_fn: Metric,
    *,
    margin_fraction: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
    on_phase: Optional[Callable[[dict], None]] = None,
) -> None:
    """A challenger must beat the king AND the freeze cheat. Mutates ``verdict``.

    Deliberately *after* the main verdict, and only ever able to take the crown away —
    never to grant it. A challenger that loses to the king is already rejected; the
    gate exists to stop one that *beat* a mediocre king by doing nothing but holding
    the conditioning frame.
    """
    from leoma.eval.baselines import evaluate_freeze_gates

    gates = evaluate_freeze_gates(
        clips, king_scores, challenger_scores, distance_fn,
        margin_fraction=margin_fraction, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed,
    )
    verdict["gates"] = gates

    if verdict["accepted"] and not gates["challenger_passed"]:
        verdict["accepted"] = False
        verdict["verdict"] = "king"
        verdict["rejected_by"] = "freeze_gate"
        verdict["reason"] = (
            "the challenger beat the king but not a frozen conditioning frame "
            f"(freeze lcb={gates['challenger']['lcb']}, margin={gates['challenger']['margin']}). "
            "A model that beats a mediocre king by holding still has not learned anything, "
            "and must not inherit the crown."
        )
        if on_phase:
            on_phase({"phase": "freeze_gate_rejected", "lcb": gates["challenger"]["lcb"]})

    if gates["king_failed"] and on_phase:
        # An alarm, never an auto-dethrone: see baselines.king_gate_is_advisory().
        on_phase({
            "phase": "king_alarm",
            "reason": "the reigning king is no better than a frozen frame",
            "avg_king_distance": verdict["avg_king_distance"],
            "avg_freeze_distance": gates["avg_freeze_distance"],
        })


# ---------------------------------------------------------------------------
# Production generation path (torch + diffusers) — imported lazily.
# ---------------------------------------------------------------------------

_TORCH_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


#: The king's loaded pipeline, cached across duels: {cache_key: pipeline}.
#:
#: The king is the SAME model for every challenger in the queue, and loading a 14B
#: pipeline off disk into VRAM costs minutes. Re-loading it per duel is pure waste.
#: Exactly one entry — this is a "keep the king warm" cache, not an LRU; two 14B
#: pipelines is already most of an H100.
_king_cache: dict[str, object] = {}


def _place_pipeline(pipeline, *, offload: str, device: str):
    """Apply the consensus-pinned placement policy to a loaded pipeline.

    Wan2.2 A14B contains two transformer experts. Keeping the entire pipeline on a
    single 80 GB H100 leaves too little room for 480p/81-frame activations, so the
    production spec uses Diffusers' model-level CPU offload. This helper is kept
    separate from loading so all three pinned modes can be tested without importing
    the GPU stack.
    """
    if offload == "none":
        return pipeline.to(device)
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            f"pinned gen.offload={offload!r} requires a CUDA target, got {device!r}"
        )
    if offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
        return pipeline
    if offload == "sequential":
        pipeline.enable_sequential_cpu_offload(device=device)
        return pipeline
    raise RuntimeError(f"unsupported pinned gen.offload mode: {offload!r}")


def load_video_pipeline(snapshot_dir: str, *, gen, device: Optional[str] = None, cache_key: str = ""):
    """Load the pinned diffusers I2V pipeline for a materialized model snapshot.

    The pipeline class comes from the pinned :class:`~leoma.eval.spec.ArchSpec`, and
    an unresolvable class is a **hard error**. The old code fell back to
    ``AutoPipelineForImage2Video`` when the pinned class couldn't be resolved — which
    meant the "pinned" pipeline was not actually pinned: a box with the wrong
    diffusers version would quietly load a *different* pipeline and generate
    different video from the same weights.

    ``cache_key`` (the king's ``repo@digest``) keeps the reigning king's pipeline
    warm between duels. It is safe precisely because the ref is immutable: the same
    digest can only ever produce the same weights.
    """
    if cache_key and cache_key in _king_cache:
        return _king_cache[cache_key]

    import torch
    import diffusers

    from leoma.infra.chain_config import SPEC

    pipeline_name = SPEC.arch.pipeline
    pipeline_cls = getattr(diffusers, pipeline_name, None)
    if pipeline_cls is None:
        raise RuntimeError(
            f"pinned pipeline {pipeline_name!r} does not exist in diffusers "
            f"{diffusers.__version__}. This box cannot run the duel the chain specifies; "
            "upgrade diffusers rather than falling back to a different pipeline."
        )

    if torch.cuda.is_available():
        dtype = getattr(torch, _TORCH_DTYPES[gen.dtype])
    else:
        dtype = torch.float32  # CPU has no usable bf16 path; tests only

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Miner snapshots are untrusted. model_store enforces a safetensors/config-only
    # filesystem allowlist, and these explicit flags prevent a future loader default
    # from re-enabling network fetches, pickle weights, or repository code.
    pipe = pipeline_cls.from_pretrained(
        snapshot_dir,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    pipe = _place_pipeline(pipe, offload=gen.offload, device=target_device)

    if cache_key:
        # A new king deposed the old one: drop the stale pipeline before holding two.
        for stale in [k for k in _king_cache if k != cache_key]:
            release_pipeline(_king_cache.pop(stale))
        _king_cache[cache_key] = pipe

    return pipe


def release_pipeline(pipeline) -> None:
    """Free a pipeline's VRAM. The first teardown in the codebase.

    ``grep empty_cache`` across this repo returned **zero hits**. Both 14B pipelines
    were held co-resident for the whole duel and then simply dropped on the floor —
    torch frees the tensors when the refcount hits zero, but the *caching allocator*
    keeps the VRAM reserved, so the next duel's load could OOM against memory that
    nothing was actually using.

    Safe to call on anything, including ``None``: this runs in ``finally`` blocks on
    the error and cancellation paths, where the pipeline may not exist at all, and a
    teardown that can itself raise is worse than no teardown.
    """
    if pipeline is None:
        return
    try:
        # Offloaded pipelines carry Accelerate hooks that own device placement.
        # Remove them before the best-effort CPU move so teardown really releases
        # their target GPU instead of asking a stale hook to move tensors again.
        pipeline.remove_all_hooks()
    except Exception:  # noqa: BLE001 — not every test/fake pipeline has hooks
        pass
    try:
        pipeline.to("cpu")
    except Exception:  # noqa: BLE001 — a failed move must not mask the real error
        pass
    del pipeline
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def release_king_cache() -> None:
    """Drop the cached king pipeline (used when the box is shutting a duel down hard)."""
    for key in list(_king_cache):
        release_pipeline(_king_cache.pop(key))


def generate(pipeline, clip: Clip, seed: int) -> np.ndarray:
    """Run the I2V pipeline for one clip; returns frames (T, H, W, C) uint8."""
    import torch
    from PIL import Image

    from leoma.app.validator.seeds import torch_seed

    p = clip.params
    # Under CPU offload ``pipeline.device`` is CPU by design, while Diffusers'
    # execution-device property is the CUDA device selected for its hooks. Preserve
    # the existing CUDA-generator semantics instead of silently changing RNG streams
    # just because the weights rest on CPU between component calls.
    device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None)
    gen = torch.Generator(device=str(device) if device is not None else "cpu")
    gen.manual_seed(torch_seed(seed))

    first = clip.first_frame
    image = first if isinstance(first, Image.Image) else Image.fromarray(np.asarray(first).astype("uint8"))
    image = image.convert("RGB").resize((p.width, p.height))

    with torch.inference_mode():
        result = pipeline(
            image=image,
            prompt=clip.prompt,
            negative_prompt=p.negative_prompt,
            height=p.height,
            width=p.width,
            num_frames=p.num_frames,
            num_inference_steps=p.num_inference_steps,
            guidance_scale=p.guidance_scale,
            generator=gen,
            output_type="np",
        )
    frames = result.frames[0]
    arr = np.asarray(frames)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255)
    return arr.astype("uint8")


def duel(
    king_pipeline,
    challenger_pipeline,
    clips: Sequence[Clip],
    *,
    master_seed: int,
    metric: str,
    metric_device: str,
    delta_threshold: float,
    alpha: float,
    n_bootstrap: int,
    generate_fn: GenerateFn = None,  # type: ignore[assignment]
    on_phase: Optional[Callable[[dict], None]] = None,
    early_stop_max_advantage: Optional[float] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    freeze_margin_fraction: Optional[float] = None,
    concurrent: bool = False,
) -> dict:
    """Production wrapper: bind loaded pipelines + the named metric into ``run_duel``.

    ``concurrent=True`` only helps if ``king_pipeline`` and ``challenger_pipeline``
    were loaded onto *different* devices (``eval/devices.py``) — generating
    concurrently on the same device just serializes on that device's queue anyway,
    so the caller is responsible for both being true together.
    """
    gen = generate_fn or generate
    distance_fn = get_metric(metric, device=metric_device)
    return run_duel(
        clips,
        generate_king=lambda c, s: gen(king_pipeline, c, s),
        generate_challenger=lambda c, s: gen(challenger_pipeline, c, s),
        distance_fn=distance_fn,
        master_seed=master_seed,
        delta_threshold=delta_threshold,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        on_phase=on_phase,
        early_stop_max_advantage=early_stop_max_advantage,
        should_cancel=should_cancel,
        freeze_margin_fraction=freeze_margin_fraction,
        concurrent=concurrent,
    )
