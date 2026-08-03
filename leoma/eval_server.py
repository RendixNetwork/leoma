"""FastAPI eval server for the video king-of-the-hill duel.

The validator POSTs a duel (king vs challenger, both by immutable ``repo@digest``)
and streams progress + the final verdict. One duel runs at a time (global lock):
the GPU box downloads both models, generates on the deterministic held-out clips,
scores against the ground-truth continuations, and returns the paired-bootstrap
verdict.

The heavy work is a pluggable *runner* (``create_app(runner=...)``) so the HTTP
contract, the single-flight lock, the event log and the watchdog are unit-testable
with a fake runner and no GPU.

**Why this file is defensive out of proportion to its size:** it holds the only GPU
in the subnet behind a single global lock, and a duel takes hours. Every way that
lock could be held forever is a way to halt the subnet — no crowns, no progress, a
permanent 409 — and the failure looks *exactly* like "we're busy". Three invariants
carry the weight:

* **The lock is always released.** Once, and only once, on every path — success,
  exception, cancellation, watchdog kill, or a failure to even start the thread.
* **Exactly one terminal event, always.** Every job ends in exactly one of
  ``verdict`` / ``error`` / ``cancelled``. A stream that ends without one used to be
  indistinguishable from a busy server, so a *broken* duel read as "try again later"
  — forever.
* **Progress is watched, not wall-clock.** A flat timeout has to guess how long a
  14B video duel takes; guess low and you kill healthy work, guess high and a hung
  socket wedges the box for an hour. The watchdog instead asks "is it still making
  forward progress?", which the duel already tells us, clip by clip.

The hard-kill fallback (``os._exit``) is only defensible because **the chain is the
queue**: the validator re-derives its work list from on-chain reveals every tick, so
a restart costs exactly one duel, which comes back for free.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from leoma.eval.errors import DuelCancelled, is_cuda_fatal
from leoma.eval.spec import ConsensusSpec


class EvalRequest(BaseModel):
    """A duel job posted by a validator.

    **The request carries the entire consensus surface.** It used to carry a handful
    of loose knobs (``metric``, ``n_clips``, ``num_frames``…) each with a *default* —
    so a validator that forgot one, or an eval box running an older build that
    ignored one, silently ran a different exam and produced a verdict nobody else
    could reproduce. Worse, the parameters it *didn't* carry (prompt, resolution,
    negative prompt, fps) came from the eval box's own environment.

    Now the validator sends the pinned :class:`~leoma.eval.spec.ConsensusSpec` and the
    server executes exactly that. The server reads **nothing** about how to duel from
    its own environment, and echoes the spec back in the verdict so the validator can
    verify what it actually ran before crowning anyone.
    """

    model_config = ConfigDict(extra="forbid")

    king_repo: str
    king_digest: str
    challenger_repo: str
    challenger_digest: str
    block_hash: str = ""
    hotkey: str = ""
    #: The pinned exam. No defaults — a missing field is a 422, not a guess.
    spec: ConsensusSpec
    #: The validator's own digest of ``spec``. Guards against a spec that was mangled
    #: in transit into something that still parses.
    consensus_digest: str


# A runner turns a request into a verdict, emitting progress events along the way.
Emit = Callable[[dict], None]
ShouldCancel = Callable[[], bool]
Runner = Callable[["EvalRequest", Emit, ShouldCancel], dict]

TERMINAL_PHASES = ("verdict", "error", "cancelled")

#: How long a finished job stays readable. A validator that restarts mid-duel must
#: still be able to collect the verdict it is owed; without a TTL the jobs dict grew
#: forever, each entry holding a full per_clip payload.
JOB_TTL_SECONDS = float(os.environ.get("LEOMA_JOB_TTL", "3600"))

#: Seconds of **no forward progress** tolerated, per phase. Not a wall-clock cap on
#: the phase — a 70 GB download over a slow link is perfectly healthy for an hour, as
#: long as bytes keep arriving. What is *not* healthy is a socket that has produced
#: nothing for 15 minutes. `load` has no natural progress signal, so it is the one
#: genuine flat cap here.
PHASE_BUDGETS: dict[str, float] = {
    "materialize": float(os.environ.get("LEOMA_BUDGET_MATERIALIZE", "1800")),
    "load": float(os.environ.get("LEOMA_BUDGET_LOAD", "1800")),
    "sample_clips": float(os.environ.get("LEOMA_BUDGET_CLIPS", "900")),
    "clip_ready": float(os.environ.get("LEOMA_BUDGET_CLIPS", "900")),
    "duel": float(os.environ.get("LEOMA_BUDGET_CLIP_DUEL", "5400")),
    "scored_clip": float(os.environ.get("LEOMA_BUDGET_CLIP_DUEL", "5400")),
}
DEFAULT_BUDGET = float(os.environ.get("LEOMA_BUDGET_DEFAULT", "1800"))

WATCHDOG_INTERVAL = float(os.environ.get("LEOMA_WATCHDOG_INTERVAL", "10"))
#: After a watchdog cancel, how long the worker gets to notice before we take the
#: process down. A CUDA call wedged in the driver cannot be interrupted from Python;
#: the only honest way out is to die and let the supervisor restart us.
CANCEL_GRACE_SECONDS = float(os.environ.get("LEOMA_CANCEL_GRACE", "300"))

#: SSE poll interval. The event log is a list; checking its length is free.
STREAM_POLL_SECONDS = 0.05

# ---------------------------------------------------------------------------
# Fatal-CUDA self-kill
# ---------------------------------------------------------------------------
# Once a CUDA context is corrupted — an illegal memory access, a device-side assert,
# a cuBLAS execution failure — the VRAM allocator and every stream on this process
# are poisoned. Every subsequent `.from_pretrained` / generate / `empty_cache` will
# keep raising against the same dead context. The watchdog's `os._exit` only fires
# after a duel *stalls*; a duel that FAILS FAST on a CUDA fault would release the lock
# and cheerfully hand the next challenger a poisoned box, mis-rejecting an honest model
# as broken. For Leoma this is expensive in a way it is not for a text subnet: a
# poisoned box wastes *hours* of the next duel, not minutes.
#
# The only recovery is to exit so the supervisor restarts with a fresh CUDA context.
# We delay briefly so the in-flight error event reaches the validator (which classifies
# `cuda_fatal` as TRANSIENT and retries against a — now freshly restarted — box), then
# `os._exit` to skip atexit hooks that would touch the corrupted GPU and hang.
# Ported from Teutonic's eval_server self-kill, which cites a real 2.5h box degradation.
#
# is_cuda_fatal() itself lives in eval/errors.py, not here: with concurrent king/
# challenger generation, the duel runner ALSO needs it (to decide which of two
# simultaneous exceptions must be the one that propagates), and it must be the same
# token list on both sides rather than two definitions that can drift apart.
CUDA_FATAL_EXIT_DELAY_S = float(os.environ.get("LEOMA_CUDA_FATAL_DELAY", "3"))
CUDA_FATAL_EXIT_CODE = int(os.environ.get("LEOMA_CUDA_FATAL_EXIT_CODE", "75"))
_self_kill_scheduled = threading.Event()


def schedule_self_kill(reason: str, *, delay_s: Optional[float] = None) -> None:
    """Exit the process because CUDA state is poisoned. Idempotent; delayed.

    Safe only because *the chain is the queue*: the restart loses at most the current
    duel, which the validator re-dispatches next tick. Holding a poisoned box would
    instead corrupt every future verdict.
    """
    if _self_kill_scheduled.is_set():
        return
    _self_kill_scheduled.set()
    delay = CUDA_FATAL_EXIT_DELAY_S if delay_s is None else float(delay_s)

    def _die():
        try:
            time.sleep(delay)  # let the SSE error event reach the validator first
        except Exception:
            pass
        os._exit(CUDA_FATAL_EXIT_CODE)

    threading.Thread(target=_die, daemon=True, name="leoma-cuda-fatal-self-kill").start()


def _install_cuda_excepthook() -> None:
    """Self-kill on a CUDA-fatal error that escapes a daemon thread (e.g. a diffusers
    worker). Without this such an error prints a traceback and the process limps on
    with a corrupted context, poisoning every later duel."""
    prior = threading.excepthook

    def _hook(args):
        try:
            if is_cuda_fatal(f"{args.exc_type.__name__}: {args.exc_value}"):
                schedule_self_kill(f"uncaught in thread {getattr(args.thread, 'name', '?')}")
        finally:
            prior(args)

    threading.excepthook = _hook


_install_cuda_excepthook()


@dataclass
class _Job:
    """One duel, its append-only event log, and its lifecycle."""

    eval_id: str
    status: str = "running"                       # running | done | error | cancelled
    events: list[dict] = field(default_factory=list)
    verdict: Optional[dict] = None
    error: Optional[str] = None
    reason: str = ""
    phase: str = "queued"
    started_at: float = field(default_factory=time.monotonic)
    last_progress: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str = ""
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _terminal: bool = False

    # -- event log ---------------------------------------------------------
    def append(self, event: dict) -> None:
        """Append to the log. Never blocks, never drops, never has one consumer.

        The log used to be a ``queue.Queue`` consumed by the SSE handler, which meant
        a subscriber that arrived late (or reconnected) blocked forever waiting for
        events that had already been consumed, and two subscribers *split* the
        stream between them. A list plus a cursor has neither problem, and it lets a
        restarted validator replay the duel it missed from the beginning.
        """
        with self._guard:
            self.events.append(event)
            self.last_progress = time.monotonic()
            phase = event.get("phase")
            if phase and phase not in TERMINAL_PHASES:
                self.phase = phase

    def since(self, cursor: int) -> tuple[list[dict], bool]:
        with self._guard:
            return self.events[cursor:], self._terminal

    def stalled_for(self) -> float:
        with self._guard:
            return time.monotonic() - self.last_progress

    def budget(self) -> float:
        with self._guard:
            return PHASE_BUDGETS.get(self.phase, DEFAULT_BUDGET)

    # -- lifecycle ---------------------------------------------------------
    def finish(self, phase: str, payload: dict) -> bool:
        """Record the ONE terminal event. First writer wins; later ones are dropped.

        This is what makes "exactly one terminal event" true even when the watchdog
        fires at the same moment the duel finishes — otherwise a job could emit both
        ``cancelled`` and ``verdict``, and a validator reading the stream would see
        whichever raced first.
        """
        with self._guard:
            if self._terminal:
                return False
            self._terminal = True
            self.finished_at = time.monotonic()
            self.status = {"verdict": "done"}.get(phase, phase)
            if phase == "verdict":
                self.verdict = payload
            else:
                self.error = str(payload.get("error") or phase)
                self.reason = str(payload.get("reason") or "")
            self.events.append({"phase": phase, **payload})
            return True

    @property
    def terminal(self) -> bool:
        with self._guard:
            return self._terminal

    def expired(self, now: float) -> bool:
        return self.finished_at is not None and (now - self.finished_at) > JOB_TTL_SECONDS


def create_app(runner: Optional[Runner] = None) -> FastAPI:
    app = FastAPI(title="leoma-eval-server")
    # Supplying a runner is the explicit unit-test/in-process injection seam. Only
    # the real production runner requires CUDA and the pinned eval dependency graph.
    enforce_runtime = runner is None
    run_job: Runner = runner or run_eval_job

    @app.middleware("http")
    async def require_eval_token(request: Request, call_next):
        """Authenticate every route when ``LEOMA_EVAL_TOKEN`` is configured.

        Health, poll, stream and cancel expose operational state; POST starts costly
        model execution. Protecting only POST would still leak job data and leave an
        unauthenticated cancellation surface.
        """
        expected = os.environ.get("LEOMA_EVAL_TOKEN", "")
        if expected:
            supplied = request.headers.get("authorization", "")
            wanted = f"Bearer {expected}"
            if not hmac.compare_digest(supplied, wanted):
                return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)

    lock = threading.Lock()
    lock_guard = threading.Lock()
    lock_holder: dict[str, Optional[str]] = {"eval_id": None}
    jobs: dict[str, _Job] = {}
    jobs_guard = threading.Lock()

    def _release(eval_id: str) -> None:
        """Release the duel lock — idempotently, and only if this job holds it.

        The old code released in the worker's ``finally``, which never ran if
        ``Thread.start()`` itself raised: the lock leaked and the box answered 409 to
        every future duel, forever, while looking perfectly healthy.
        """
        with lock_guard:
            if lock_holder["eval_id"] != eval_id:
                return
            lock_holder["eval_id"] = None
            lock.release()

    def _reap(now: float) -> None:
        with jobs_guard:
            for eval_id in [k for k, j in jobs.items() if j.expired(now)]:
                del jobs[eval_id]

    def _execute(job: _Job, req: EvalRequest) -> None:
        try:
            verdict = run_job(req, job.append, job.cancel.is_set)
            job.finish("verdict", verdict)
        except DuelCancelled as e:
            job.finish("cancelled", {"error": str(e), "reason": job.cancel_reason or "cancelled"})
        except BaseException as e:  # noqa: BLE001 — see below
            # BaseException, not Exception: a KeyboardInterrupt or a SystemExit inside
            # the worker would otherwise leave the job with NO terminal event, the
            # stream hanging, and the lock held. Every exit is a terminal event.
            job.finish("error", {"error": str(e), "reason": getattr(e, "reason", "")})
            # A CUDA-fatal error is reported cleanly here AND self-kills: the context
            # is poisoned, so releasing the lock and serving the next duel would just
            # hand it a dead GPU. The delay lets this error event reach the validator
            # (which retries the challenger against the restarted box) before we exit.
            if is_cuda_fatal(e):
                schedule_self_kill(f"duel {job.eval_id}: {type(e).__name__}: {e}")
        finally:
            _release(job.eval_id)

    def _watch(job: _Job, worker: threading.Thread) -> None:
        """Kill a duel that has stopped making forward progress.

        A flat wall-clock cap would have to guess how long a 14B video duel takes:
        guess low and healthy work dies, guess high and a hung socket wedges the only
        GPU in the subnet for an hour. Progress is the honest signal, and the duel
        already reports it — one event per clip.
        """
        while not job.terminal:
            if job.stalled_for() > job.budget():
                job.cancel_reason = f"no progress in phase '{job.phase}' for {job.budget():.0f}s"
                job.cancel.set()
                job.finish("error", {
                    "error": f"watchdog: {job.cancel_reason}",
                    # TRANSIENT on the validator's side: a hung box is not the
                    # challenger's fault, and quarantining a miner for our stall would
                    # be the worst possible outcome.
                    "reason": "watchdog_stalled",
                })
                _release(job.eval_id)

                # Cooperative cancel is checked between clips. If the worker is wedged
                # *inside* a CUDA call it will never see it — nothing in Python can
                # interrupt that. Then the only honest move is to die: the chain is the
                # queue, so the supervisor restarts us and the validator re-dispatches
                # the duel next tick. Losing one duel beats holding the GPU forever.
                worker.join(timeout=CANCEL_GRACE_SECONDS)
                if worker.is_alive():
                    os._exit(1)
                return
            if job.cancel.wait(timeout=WATCHDOG_INTERVAL):
                # Cancelled from outside (DELETE). The worker owns the terminal event.
                return

    @app.get("/health")
    def health() -> dict:
        """Enough for a validator to reject a stale box *before* handing it a duel.

        Source/config digests are necessary but not sufficient: the numerical
        runtime (Torch, Diffusers, CUDA, GPU capability, and uv.lock) is enforced too.
        """
        from leoma.infra.chain_config import CONSENSUS_DIGEST, SPEC
        from leoma.eval.codehash import eval_code_digest
        from leoma.eval.runtime_lock import eval_runtime_report

        with lock_guard:
            busy_with = lock_holder["eval_id"]
        job = jobs.get(busy_with) if busy_with else None
        runtime = eval_runtime_report(SPEC.runtime)
        runtime_ok = bool(runtime["compatible"] or not enforce_runtime)

        return {
            "status": "ok" if runtime_ok else "runtime_mismatch",
            "busy": busy_with is not None,
            "eval_id": busy_with,
            "phase": job.phase if job else None,
            "stalled_for": round(job.stalled_for(), 1) if job else None,
            "consensus_digest": CONSENSUS_DIGEST,
            "eval_code_digest": eval_code_digest(),
            "eval_runtime_enforced": enforce_runtime,
            "eval_runtime_compatible": runtime["compatible"],
            "eval_runtime_digest": runtime["digest"],
            "expected_eval_runtime_digest": runtime["expected_digest"],
            "eval_runtime": runtime["identity"],
            "eval_runtime_issues": runtime["issues"],
        }

    @app.post("/eval")
    def start_eval(req: EvalRequest):
        if enforce_runtime:
            from leoma.infra.chain_config import SPEC
            from leoma.eval.runtime_lock import eval_runtime_report

            runtime = eval_runtime_report(SPEC.runtime)
            if not runtime["compatible"]:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "evaluator runtime does not match chain.toml [runtime]",
                        "reason": "runtime_mismatch",
                        "issues": runtime["issues"],
                    },
                )
        _reap(time.monotonic())

        if not lock.acquire(blocking=False):
            return JSONResponse(status_code=409, content={"error": "an eval is already running"})

        eval_id = f"eval-{uuid.uuid4().hex[:12]}"
        with lock_guard:
            lock_holder["eval_id"] = eval_id
        job = _Job(eval_id=eval_id)
        with jobs_guard:
            jobs[eval_id] = job

        try:
            worker = threading.Thread(target=_execute, args=(job, req), daemon=True)
            worker.start()
            threading.Thread(target=_watch, args=(job, worker), daemon=True).start()
        except BaseException as e:  # noqa: BLE001 — thread creation itself can fail
            job.finish("error", {"error": f"could not start the duel: {e}", "reason": "thread_start"})
            _release(eval_id)
            return JSONResponse(status_code=500, content={"error": str(e)})

        return {"eval_id": eval_id}

    @app.get("/eval/{eval_id}")
    def get_eval(eval_id: str):
        with jobs_guard:
            job = jobs.get(eval_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "unknown eval_id"})
        return {
            "eval_id": eval_id,
            "status": job.status,
            "phase": job.phase,
            "verdict": job.verdict,
            "error": job.error,
            "reason": job.reason,
        }

    @app.delete("/eval/{eval_id}")
    def cancel_eval(eval_id: str):
        """Ask a duel to stop. Checked between clips — the only honest granularity."""
        with jobs_guard:
            job = jobs.get(eval_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "unknown eval_id"})
        if job.terminal:
            return {"eval_id": eval_id, "status": job.status, "cancelled": False}
        job.cancel_reason = "cancelled by request"
        job.cancel.set()
        return {"eval_id": eval_id, "status": "cancelling", "cancelled": True}

    @app.get("/eval/{eval_id}/stream")
    async def stream_eval(eval_id: str, request: Request):
        """Replay the whole event log from the start, then follow it live.

        ``async def``, deliberately. As a sync route it ran in anyio's threadpool and
        *blocked a worker slot for the entire duel* — a handful of subscribers could
        starve the server of threads while the GPU was still warming up.
        """
        with jobs_guard:
            job = jobs.get(eval_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "unknown eval_id"})

        async def gen():
            cursor = 0
            while True:
                events, terminal = job.since(cursor)
                for event in events:
                    cursor += 1
                    yield f"data: {json.dumps(event)}\n\n"
                if terminal and not events:
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(STREAM_POLL_SECONDS)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def check_request(req: EvalRequest) -> None:
    """Refuse a duel this box cannot run reproducibly. Called before any GPU work.

    Three ways a duel is dead on arrival, all cheap to detect and catastrophic to
    miss:

    * the validator's ``consensus_digest`` doesn't match the spec it sent (mangled in
      transit, or a validator computing digests differently);
    * this box's ``chain.toml`` pins a *different* exam than the validator's (an
      operator forgot to redeploy one machine — the most likely failure by far);
    * the corpus isn't pinned at all.

    Any of them and we stop here, rather than spending an hour of GPU producing a
    verdict nobody else can reproduce.
    """
    from leoma.infra.chain_config import CONSENSUS_DIGEST
    from leoma.eval.errors import ConsensusConfigError

    if req.spec.digest() != req.consensus_digest:
        raise ConsensusConfigError(
            f"request consensus_digest {req.consensus_digest} does not match the spec it "
            f"carries ({req.spec.digest()}) — the request was altered in transit"
        )
    if req.consensus_digest != CONSENSUS_DIGEST:
        raise ConsensusConfigError(
            f"this eval box pins a different consensus surface than the validator "
            f"(box {CONSENSUS_DIGEST}, validator {req.consensus_digest}). One of the two is "
            "running a stale chain.toml. Refusing the duel rather than producing a verdict "
            "the rest of the subnet cannot reproduce."
        )
    req.spec.require_duel_ready()


def _resolve_devices():
    """Glue: read the concurrency env vars + the real CUDA device count.

    The decision itself (``leoma.eval.devices.resolve_duel_devices``) is pure and
    unit-tested without a GPU; this function exists only because *something* has to
    call ``torch.cuda.device_count()``, and that something needs torch installed.
    """
    import os

    import torch

    from leoma.eval.devices import resolve_duel_devices

    return resolve_duel_devices(
        concurrent_enabled=os.environ.get("LEOMA_CONCURRENT_GENERATION", "0") == "1",
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        king_device_override=os.environ.get("LEOMA_KING_DEVICE") or None,
        challenger_device_override=os.environ.get("LEOMA_CHALLENGER_DEVICE") or None,
    )


def _download_watcher(emit: Emit, which: str, path, stop: threading.Event) -> None:
    """Report a download's forward progress so the watchdog can tell slow from hung.

    ``materialize`` is otherwise completely silent: one event at the start, then
    nothing for however long a ~30-70 GB pull takes. Without this, the watchdog can
    only choose between killing healthy downloads and tolerating dead sockets. Polling
    the snapshot's size on disk turns "is it hung?" into a question with an answer.
    """
    from leoma.infra.model_store import snapshot_size

    last = -1
    while not stop.wait(timeout=30.0):
        try:
            size = snapshot_size(path)
        except OSError:
            continue
        if size > last:
            last = size
            emit({"phase": "materialize", "which": which, "bytes": size})


def run_eval_job(req: EvalRequest, emit: Emit, should_cancel: ShouldCancel = lambda: False) -> dict:
    """Production duel: download both models, generate on held-out clips, score.

    A **pure executor of the request**. Every parameter comes from ``req.spec``; this
    function reads nothing about how to duel from its own environment, which is what
    makes the verdict a function of the chain rather than of the box.

    Heavy deps (torch/diffusers, the corpus) are imported here, lazily, so the server
    module stays import-safe.
    """
    from leoma.infra.model_store import ModelRef, cache_path, materialize_model, snapshot_size
    from leoma.eval.arch_lock import check_size
    from leoma.infra.storage_backend import create_source_read_client
    from leoma.eval.video_runner import GenParams, load_video_pipeline, duel, release_pipeline
    from leoma.eval.codehash import eval_code_digest
    from leoma.eval.dataset import build_duel_clips, corpus_audit, fetch_manifest
    from leoma.eval.determinism import apply_determinism, runtime_env
    from leoma.eval.digests import digest_obj
    from leoma.eval.runtime_lock import eval_runtime_report
    from leoma.app.validator.seeds import eval_seed

    check_request(req)
    spec = req.spec
    apply_determinism(spec.determinism)

    king_ref = ModelRef(req.king_repo, req.king_digest)
    chall_ref = ModelRef(req.challenger_repo, req.challenger_digest)

    # The working set: never evict the two models this duel is about, however cold
    # the king's snapshot looks after a long reign.
    keep = [cache_path(king_ref), cache_path(chall_ref)]

    def _materialize(ref, which: str) -> str:
        emit({"phase": "materialize", "which": which, "ref": ref.immutable_ref})
        stop = threading.Event()
        watcher = threading.Thread(
            target=_download_watcher, args=(emit, which, cache_path(ref), stop), daemon=True
        )
        watcher.start()
        try:
            return materialize_model(ref, keep=keep)
        finally:
            stop.set()

    king_dir = _materialize(king_ref, "king")
    chall_dir = _materialize(chall_ref, "challenger")
    # Prescreen intentionally fetches configs only, so the authoritative full-size
    # bound can only be applied after materialization. A tiny stub or a disk-filling
    # artifact must never reach pipeline loading merely because its configs looked
    # plausible.
    check_size(snapshot_size(chall_dir))

    # A throughput setting, not a consensus one — see eval/devices.py. Resolved from
    # this box's own environment, never from the request: if it changed per-request
    # the cached king pipeline (below) could silently end up on the wrong device.
    devices = _resolve_devices()
    emit({"phase": "devices", "concurrent": devices.concurrent, "note": devices.note})

    chall_pipe = None
    try:
        emit({"phase": "load", "which": "king", "device": devices.king_device})
        # Keyed by the king's immutable ref: the same king faces every challenger in
        # the queue, and re-loading 28 GB into VRAM for each one is pure waste.
        king_pipe = load_video_pipeline(
            king_dir, gen=spec.gen, device=devices.king_device, cache_key=king_ref.immutable_ref
        )
        emit({"phase": "load", "which": "challenger", "device": devices.challenger_device})
        chall_pipe = load_video_pipeline(chall_dir, gen=spec.gen, device=devices.challenger_device)

        master_seed = eval_seed(req.block_hash, req.hotkey, spec.duel.base_seed)
        params = GenParams.from_spec(spec.gen)

        emit({"phase": "sample_clips", "n_clips": spec.duel.n_clips, "seed": master_seed})
        client = create_source_read_client()
        manifest = fetch_manifest(client, spec.corpus)
        clips, entries = build_duel_clips(
            manifest,
            client=client,
            bucket=spec.corpus.bucket,
            master_seed=master_seed,
            n_clips=spec.duel.n_clips,
            gen=params,
            prompt_mode=spec.gen.prompt_mode,
            fixed_prompt=spec.gen.prompt,
            on_progress=lambda done, total, entry: emit({
                "phase": "clip_ready", "done": done, "total": total, "clip_id": entry.clip_id,
            }),
        )

        emit({"phase": "duel", "n_clips": len(clips), "metric": spec.duel.metric})
        verdict = duel(
            king_pipe, chall_pipe, clips,
            master_seed=master_seed,
            metric=spec.duel.metric,
            metric_device=spec.duel.metric_device,
            delta_threshold=spec.duel.delta_threshold,
            alpha=spec.duel.alpha,
            n_bootstrap=spec.duel.n_bootstrap,
            on_phase=emit,
            early_stop_max_advantage=spec.early_stop_max_advantage,
            should_cancel=should_cancel,
            freeze_margin_fraction=spec.duel.freeze_margin_fraction,
            concurrent=devices.concurrent,
        )
    finally:
        # The challenger is done with either way — success, error, or cancellation.
        # Without this its VRAM stayed reserved by torch's caching allocator, and the
        # NEXT duel could OOM against memory nothing was using. The king stays warm.
        release_pipeline(chall_pipe)

    verdict["king_digest"] = req.king_digest
    verdict["challenger_digest"] = req.challenger_digest
    verdict["challenger_repo"] = req.challenger_repo

    # The spec, verbatim. The validator re-parses this and refuses to crown unless it
    # matches what it sent — which is what makes "field silently defaulted" a caught
    # error instead of an invisible fork.
    verdict["echo"] = spec.model_dump(mode="json")

    # Everything a third party needs to replay this duel and get the same numbers.
    verdict["audit"] = {
        "master_seed": master_seed,
        "block_hash": req.block_hash,
        "hotkey": req.hotkey,
        "consensus_digest": req.consensus_digest,
        "eval_code_digest": eval_code_digest(),
        "corpus": corpus_audit(manifest, entries),
        "env": runtime_env(),
        "eval_runtime": eval_runtime_report(spec.runtime),
        # Not part of the consensus surface (see eval/devices.py) — recorded here so a
        # distance disagreement between validators can be checked against "did the two
        # duelists even run on the same kind of device" before assuming a real bug.
        "generation": {
            "concurrent": devices.concurrent,
            "king_device": devices.king_device,
            "challenger_device": devices.challenger_device,
        },
    }
    # Hashes ONLY the consensus surface — the decision, the exam, the parameters.
    # Deliberately excludes env and produced_at: two validators on different GPUs
    # SHOULD produce the same verdict_digest when they agree, and a wall-clock or a GPU
    # name inside it would guarantee they never do.
    verdict["verdict_digest"] = digest_obj({
        "accepted": verdict["accepted"],
        "consensus_digest": req.consensus_digest,
        "master_seed": master_seed,
        "king_digest": req.king_digest,
        "challenger_digest": req.challenger_digest,
        "clip_keys_digest": verdict["audit"]["corpus"]["clip_keys_digest"],
        "per_clip": [
            {
                "clip_id": c["clip_id"],
                "king_distance": c["king_distance"],
                "challenger_distance": c["challenger_distance"],
            }
            for c in verdict["per_clip"]
        ],
    })
    verdict["produced_at"] = datetime.now(timezone.utc).isoformat()
    return verdict


def _check_bind_safety(host: str) -> None:
    """Refuse to expose an unauthenticated GPU box to the network.

    Without a token, ``POST /eval`` would let a remote caller consume the GPU with
    arbitrary model refs. The validator normally reaches the box over an SSH tunnel,
    so loopback is both safe and sufficient. A public bind is allowed only when the
    Bearer-token middleware above is active.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        return
    if os.environ.get("LEOMA_EVAL_TOKEN"):
        return
    raise SystemExit(
        f"refusing to bind an unauthenticated eval server to {host}. POST /eval makes this "
        "box download and run an arbitrary model. Either bind to 127.0.0.1 (the validator "
        "reaches it over an SSH tunnel) or set LEOMA_EVAL_TOKEN."
    )


def main() -> None:  # pragma: no cover - entrypoint
    import uvicorn

    host = os.environ.get("EVAL_SERVER_HOST", "127.0.0.1")
    _check_bind_safety(host)
    uvicorn.run(
        create_app(),
        host=host,
        port=int(os.environ.get("EVAL_SERVER_PORT", "9000")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
