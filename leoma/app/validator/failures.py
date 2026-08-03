"""Duel failure classification.

Leoma has no persistent queue — the chain *is* the queue, re-derived from
``get_all_revealed_commitments`` every tick. So we don't need Teutonic's durable
queue with ``requeue_front``/``retry_count``; we need a *decision function* over a
stateless work list. This module is that decision.

Four classes, and the distinctions all matter:

* ``BUSY``      — a property of the SERVER (the eval box is running someone else's
                  duel). Not a failure; consumes no attempt. The caller must
                  ``break`` (continuing would just 409 N more times).
* ``TRANSIENT`` — a property of the ENVIRONMENT (network, disk, GPU). Retry with
                  block-based backoff; quarantine only after the attempt budget is
                  exhausted.
* ``PERMANENT`` — a property of the ARTIFACT (the repo 404s, the weights won't
                  load, the arch is wrong). ``repo@digest`` is immutable, so this
                  can never succeed; quarantine it.
* ``LOCAL``     — a property of **THIS VALIDATOR** (our corpus doesn't match the
                  pinned manifest, our eval box runs a stale chain.toml). Costs the
                  challenger **nothing**: no attempt, no backoff, no quarantine. The
                  caller must ``break`` — every challenger would hit the identical
                  wall, including the king.

**Design rule: when in doubt, TRANSIENT.** A misclassified transient costs four
retries. A misclassified permanent locks a legitimate miner out of an artifact. That
asymmetry is why auth errors are deliberately *not* permanent: a validator's own
token misconfiguration would otherwise quarantine every miner on the subnet.

**LOCAL exists because TRANSIENT is not actually harmless.** The attempt ledger
quarantines an artifact once its attempts are ``exhausted``, whatever the class. So a
validator whose *own* corpus was broken would fail every duel transiently, four times
each, and then quarantine **every miner on the subnet** — permanently locking out
honest models because of its own misconfiguration. Faults that are ours must therefore
not touch the challenger's ledger at all. They are a reason to stop dueling, not a
reason to blame whoever happened to be next in the queue.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorClass(str, Enum):
    BUSY = "busy"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    LOCAL = "local"


@dataclass(frozen=True)
class DuelFailure:
    kind: ErrorClass
    reason: str   # machine token; shown on the dashboard
    detail: str   # full message (truncated when stored)

    @property
    def is_permanent(self) -> bool:
        return self.kind is ErrorClass.PERMANENT

    @property
    def is_transient(self) -> bool:
        return self.kind is ErrorClass.TRANSIENT

    @property
    def is_local(self) -> bool:
        return self.kind is ErrorClass.LOCAL


# OUR fault, not the challenger's. These would fail identically for every model on
# the subnet — including the reigning king — so they can never be evidence about a
# particular challenger. Checked FIRST, before anything else can claim them.
_LOCAL_SIGNS: tuple[tuple[str, str], ...] = (
    ("corpus_integrity", "corpus_integrity"),
    ("consensus_config", "consensus_config"),
    ("consensus_mismatch", "consensus_mismatch"),
    ("consensus_echo_mismatch", "consensus_echo_mismatch"),
    ("code_mismatch", "code_mismatch"),
    ("runtime_mismatch", "runtime_mismatch"),
    ("does not match the manifest", "corpus_integrity"),
    ("decoded ground truth does not match", "corpus_integrity"),
    ("corpus manifest digest mismatch", "corpus_integrity"),
    ("manifest_digest is not pinned", "consensus_config"),
    ("different consensus surface", "consensus_mismatch"),
)

# Substring -> (class, reason). Order matters: the first match wins, so the
# PERMANENT artifact signatures are checked before the generic transient ones.
_PERMANENT_SIGNS: tuple[tuple[str, str], ...] = (
    # repo / revision does not exist
    ("repositorynotfound", "model_not_found"),
    ("revisionnotfound", "model_not_found"),
    ("entrynotfound", "model_not_found"),
    ("does not exist", "model_not_found"),
    ("not found on", "model_not_found"),
    # the artifact exists but is not a loadable model
    ("does not appear to have a file named", "model_invalid"),
    ("error no file named", "model_invalid"),
    ("is not a valid", "model_invalid"),
    ("headertoolarge", "model_invalid"),
    ("metadataincompletebuffer", "model_invalid"),
    ("size mismatch", "model_invalid"),
    ("unexpected key", "model_invalid"),
    ("invalid hippius repo id", "model_invalid"),
    ("invalid hippius oci digest", "model_invalid"),
    ("invalid oci digest", "model_invalid"),
    # pinned-architecture mismatch (the config lock)
    ("arch_mismatch", "arch_mismatch"),
    ("config_rejected", "arch_mismatch"),
)

_TRANSIENT_SIGNS: tuple[tuple[str, str], ...] = (
    ("out of memory", "oom"),
    ("illegal memory access", "cuda_fatal"),
    ("device-side assert", "cuda_fatal"),
    ("cublas_status_", "cuda_fatal"),
    ("no space left on device", "disk_full"),
    ("no clips to duel on", "corpus_unavailable"),
    # Auth is NOT permanent: it is almost always OUR misconfiguration, and
    # treating it as permanent would quarantine every miner on the subnet.
    ("hippiushubautherror", "hub_auth"),
    ("401", "hub_auth"),
    ("403", "hub_auth"),
)

# Exception type names that are transient regardless of message.
_TRANSIENT_TYPES = frozenset(
    {
        "connecterror",
        "connecttimeout",
        "readtimeout",
        "writetimeout",
        "pooltimeout",
        "remoteprotocolerror",
        "networkerror",
        "timeouterror",
        "storeunavailable",
        "storecorrupt",
        "outofmemoryerror",
    }
)


class EvalBusy(RuntimeError):
    """The eval server is already running a duel (HTTP 409)."""


class EvalJobFailed(RuntimeError):
    """The eval server reported a terminal error for this duel."""

    def __init__(self, detail: str, reason: str = ""):
        super().__init__(detail)
        self.detail = detail
        self.reason = reason


def _match(text: str, table: tuple[tuple[str, str], ...]) -> str:
    for needle, reason in table:
        if needle in text:
            return reason
    return ""


def classify_remote(message: str, reason: str = "") -> DuelFailure:
    """Classify a terminal error reported BY the eval server.

    A structured ``reason`` token from the server wins; otherwise fall back to
    substring matching on the message.
    """
    text = f"{reason} {message}".lower()

    token = reason.strip().lower()
    if token:
        for _, r in _LOCAL_SIGNS:
            if token == r:
                return DuelFailure(ErrorClass.LOCAL, r, message)
        for _, r in _PERMANENT_SIGNS:
            if token == r:
                return DuelFailure(ErrorClass.PERMANENT, r, message)
        for _, r in _TRANSIENT_SIGNS:
            if token == r:
                return DuelFailure(ErrorClass.TRANSIENT, r, message)
        if token.startswith("watchdog_stall"):
            return DuelFailure(ErrorClass.TRANSIENT, token, message)

    hit = _match(text, _LOCAL_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.LOCAL, hit, message)

    hit = _match(text, _PERMANENT_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.PERMANENT, hit, message)

    hit = _match(text, _TRANSIENT_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.TRANSIENT, hit, message)

    # Unknown remote failure: fail OPEN into retry, never into punishment.
    return DuelFailure(ErrorClass.TRANSIENT, reason or "eval_error", message)


def classify(exc: BaseException) -> DuelFailure:
    """Classify an exception raised while dispatching or settling a duel."""
    if isinstance(exc, EvalBusy):
        return DuelFailure(ErrorClass.BUSY, "eval_busy", str(exc) or "eval server busy")

    if isinstance(exc, EvalJobFailed):
        return classify_remote(exc.detail, exc.reason)

    # The typed eval errors say who is at fault directly — no substring guessing.
    from leoma.eval.errors import (
        ChallengerFault,
        ConsensusConfigError,
        CorpusIntegrityError,
        DuelCancelled,
    )

    if isinstance(exc, (CorpusIntegrityError, ConsensusConfigError)):
        return DuelFailure(ErrorClass.LOCAL, exc.reason, str(exc))
    if isinstance(exc, DuelCancelled):
        # Checked BEFORE ChallengerFault would be, and deliberately transient: a
        # watchdog stall or an operator's DELETE says nothing about the model.
        return DuelFailure(ErrorClass.TRANSIENT, exc.reason, str(exc))
    if isinstance(exc, ChallengerFault):
        return DuelFailure(ErrorClass.PERMANENT, exc.reason, str(exc))

    type_name = type(exc).__name__.lower()
    message = str(exc)
    text = f"{type_name} {message}".lower()

    if type_name in _TRANSIENT_TYPES:
        reason = _match(text, _TRANSIENT_SIGNS) or "eval_unreachable"
        return DuelFailure(ErrorClass.TRANSIENT, reason, message)

    hit = _match(text, _LOCAL_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.LOCAL, hit, message)

    hit = _match(text, _PERMANENT_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.PERMANENT, hit, message)

    hit = _match(text, _TRANSIENT_SIGNS)
    if hit:
        return DuelFailure(ErrorClass.TRANSIENT, hit, message)

    # HTTP status hints from httpx.HTTPStatusError et al.
    if "httpstatuserror" in type_name or "status" in text:
        if any(code in text for code in ("502", "503", "504", "500")):
            return DuelFailure(ErrorClass.TRANSIENT, "eval_unreachable", message)

    # Default: TRANSIENT. See the module docstring — this asymmetry is deliberate.
    return DuelFailure(ErrorClass.TRANSIENT, "eval_error", message)
