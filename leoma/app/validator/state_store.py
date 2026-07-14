"""Persisted king-of-the-hill state (this validator's own bucket).

Each validator keeps its king state in its own object bucket so it survives
restarts. The state is deterministic given the chain + reveals, so validators
converge on the same king without a shared store; the bucket is durable *local
memory*, never a source of truth.

That framing is what makes read integrity critical: losing the cache must never
silently corrupt the chain-derived answer. So:

* ``get`` distinguishes a genuine **miss** (key absent -> ``None``) from an
  **error** (raises ``StoreUnavailable`` / ``StoreCorrupt``). It previously
  swallowed every exception and returned ``None``, so a transient bucket outage
  produced a *blank* state -- re-seeding genesis, re-dueling every past
  challenger, wiping history -- and the next flush **overwrote the good state
  with the blank one**.
* ``KingState.load`` **refuses to start** on a partial or failed read rather than
  falling back to blank state.
* State is written as **one canonical object** (``state/state.json``). An S3
  object PUT is atomic; five separate PUTs are not, and a failure midway used to
  leave king updated but seen/history stale. The old five keys are still written
  as best-effort mirrors for compatibility/auditing.
* Every Minio call runs in a worker thread -- they are blocking, and the caller
  is the validator's async loop.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from leoma.bootstrap import emit_log as log

SCHEMA_VERSION = 2

# ── attempt-ledger policy ───────────────────────────────────────────────────
MAX_DUEL_ATTEMPTS = int(os.environ.get("LEOMA_MAX_DUEL_ATTEMPTS", "4"))
# Block-based backoff, ~12s/block: 1m, 5m, 20m, 80m.
BACKOFF_BLOCKS = (5, 25, 100, 400)
# A PERMANENT failure must be seen this many times before quarantine — cheap
# insurance against a flaky 404 locking out a legitimate miner.
PERMANENT_SIGHTINGS = int(os.environ.get("LEOMA_PERMANENT_SIGHTINGS", "2"))
# Hotkeys with this many DISTINCT quarantined digests are dropped at scan time.
BAN_AFTER_QUARANTINED = int(os.environ.get("LEOMA_BAN_AFTER_QUARANTINED", "3"))

# Canonical, atomically-written state object.
KEY_STATE = "state/state.json"

# Legacy per-concern keys. Still written (best-effort mirrors) and still read on
# first load so an existing validator bucket migrates transparently.
KEY_KING = "king/current.json"
KEY_KING_CHAIN = "state/king_chain.json"
KEY_VALIDATOR_STATE = "state/validator_state.json"
KEY_SEEN = "state/seen_hotkeys.json"
KEY_HISTORY = "state/history.json"
LEGACY_KEYS = (KEY_KING, KEY_KING_CHAIN, KEY_VALIDATOR_STATE, KEY_SEEN, KEY_HISTORY)

# Recent duel verdicts kept for the dashboard (newest first).
HISTORY_LIMIT = 200


# ── errors ──────────────────────────────────────────────────────────────────
class StoreUnavailable(RuntimeError):
    """A bucket read/write failed for a reason other than 'key absent'."""


class StoreCorrupt(StoreUnavailable):
    """The object exists but is not valid JSON."""


class StateInconsistent(RuntimeError):
    """A partial read: some state keys present, others absent. Refuse to start."""


# minio raises S3Error carrying a `.code`; treat only these as a genuine miss.
_MISSING_CODES = frozenset({"NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound"})


def _is_missing(exc: BaseException) -> bool:
    if getattr(exc, "code", None) in _MISSING_CODES:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status", None) == 404


def _close_quietly(resp) -> None:
    try:
        resp.close()
        resp.release_conn()
    except Exception:
        pass


class JsonBucketStore:
    """JSON KV over a Minio/S3 bucket.

    ``get`` returns None ONLY when the key genuinely does not exist. Every other
    failure raises -- that distinction is the whole point of this class.
    """

    def __init__(self, client, bucket: str, *, retries: int = 3, backoff: float = 0.5):
        self.client = client
        self.bucket = bucket
        self.retries = max(1, retries)
        self.backoff = backoff

    # ---- blocking core (called via to_thread) ----------------------------
    def get_sync(self, key: str) -> Optional[dict]:
        last: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                resp = self.client.get_object(self.bucket, key)
            except Exception as e:
                if _is_missing(e):
                    return None
                last = e
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise StoreUnavailable(f"get {key}: {e}") from e

            try:
                data = resp.read()
            finally:
                _close_quietly(resp)

            try:
                return json.loads(data)
            except (ValueError, TypeError) as e:
                # A corrupt object is NOT a miss. Surfacing it prevents the
                # caller from treating it as "fresh bucket" and overwriting.
                raise StoreCorrupt(f"{key} is not valid JSON: {e}") from e

        raise StoreUnavailable(f"get {key}: {last}")  # pragma: no cover - defensive

    def put_sync(self, key: str, obj: Any) -> None:
        payload = json.dumps(obj, default=str, sort_keys=True).encode("utf-8")
        for attempt in range(self.retries):
            try:
                self.client.put_object(
                    self.bucket,
                    key,
                    io.BytesIO(payload),
                    length=len(payload),
                    content_type="application/json",
                )
                return
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (2**attempt))
                    continue
                raise StoreUnavailable(f"put {key}: {e}") from e

    # ---- async wrappers (Minio blocks; the validator loop must not) -------
    async def get(self, key: str) -> Optional[dict]:
        return await asyncio.to_thread(self.get_sync, key)

    async def put(self, key: str, obj: Any) -> None:
        await asyncio.to_thread(self.put_sync, key, obj)


def _normalize_inflight(raw: Any) -> list:
    """Migrate the persisted ``inflight`` shape: a single dict-or-None (every bucket
    written before multi-eval-server dispatch) into the list every bucket uses now.

    A live validator's existing bucket has ``inflight`` as ``null`` (no duel running)
    or a bare dict (exactly one). Wrapping the dict in a 1-item list is exactly
    "one slot, the same slot it always had" — nothing about an in-flight duel is
    lost or reinterpreted, just represented honestly for N>1 servers.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw] if raw else []
    if isinstance(raw, list):
        return raw
    return []  # pragma: no cover - defensive; not a shape this field has ever had


@dataclass
class KingState:
    """In-memory view of the persisted king state."""

    king: dict = field(default_factory=dict)
    king_chain: list = field(default_factory=list)
    last_weight_block: int = 0
    last_winner_hotkey: Optional[str] = None
    counter: int = 0
    stats: dict = field(
        default_factory=lambda: {"accepted": 0, "rejected": 0, "failed": 0, "transient_errors": 0}
    )
    seen_hotkeys: set = field(default_factory=set)
    history: list = field(default_factory=list)  # recent duel verdicts, newest first

    # ── schema v2: liveness fields (behavior wired in the head-of-line change) ──
    # attempts: "<hotkey>|<digest>" -> {attempts, first_block, last_block,
    #                                   next_retry_block, last_class, last_reason,
    #                                   last_error, quarantined, quarantine_reason}
    attempts: dict = field(default_factory=dict)
    # Per-hotkey duel budget (cooldown, per-reign cap, strikes). Keyed by HOTKEY, not
    # by artifact: the seen-set already stops the same artifact being dueled twice, but
    # a hotkey can mint unlimited fresh digests and each one buys a free multi-hour duel.
    duels: dict = field(default_factory=dict)
    # Dispatched duels currently in flight, one per busy eval-server URL (0..N — see
    # EVAL_SERVER_URLS). Was a single dict-or-None when there was only ever one eval
    # server; a list is the honest shape once a validator can dispatch to several.
    inflight: list = field(default_factory=list)
    weight_failures: int = 0                 # consecutive genuine set_weights failures
    next_weight_block: int = 0               # block-based backoff after a failure

    # Not persisted: a human-readable reason the validator is degraded.
    degraded: Optional[str] = field(default=None, compare=False)
    _dirty: bool = field(default=False, compare=False, repr=False)

    # ── serialisation ─────────────────────────────────────────────────────
    def to_doc(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "king": self.king,
            "king_chain": self.king_chain,
            "last_weight_block": self.last_weight_block,
            "last_winner_hotkey": self.last_winner_hotkey,
            "counter": self.counter,
            "stats": self.stats,
            "seen": sorted(self.seen_hotkeys),
            "history": self.history,
            "attempts": self.attempts,
            "duels": self.duels,
            "inflight": self.inflight,
            "weight_failures": self.weight_failures,
            "next_weight_block": self.next_weight_block,
        }

    @classmethod
    def _from_doc(cls, doc: dict) -> "KingState":
        self = cls()
        self.king = doc.get("king") or {}
        self.king_chain = doc.get("king_chain") or []
        self.last_weight_block = doc.get("last_weight_block", 0)
        self.last_winner_hotkey = doc.get("last_winner_hotkey")
        self.counter = doc.get("counter", 0)
        self.stats = {**cls().stats, **(doc.get("stats") or {})}
        self.seen_hotkeys = set(doc.get("seen") or [])
        self.history = doc.get("history") or []
        self.attempts = doc.get("attempts") or {}
        # Default-on-missing: live validators have existing buckets whose state.json
        # predates this field, and they must load rather than crash.
        self.duels = doc.get("duels") or {}
        self.inflight = _normalize_inflight(doc.get("inflight"))
        self.weight_failures = doc.get("weight_failures", 0)
        self.next_weight_block = doc.get("next_weight_block", 0)
        return self

    @classmethod
    def _from_legacy(cls, legacy: dict) -> "KingState":
        self = cls()
        self.king = legacy.get(KEY_KING) or {}
        self.king_chain = (legacy.get(KEY_KING_CHAIN) or {}).get("chain", [])
        vs = legacy.get(KEY_VALIDATOR_STATE) or {}
        self.last_weight_block = vs.get("last_weight_block", 0)
        self.last_winner_hotkey = vs.get("last_winner_hotkey")
        self.counter = vs.get("counter", 0)
        self.stats = {**cls().stats, **(vs.get("stats") or {})}
        self.seen_hotkeys = set((legacy.get(KEY_SEEN) or {}).get("hotkeys", []))
        self.history = (legacy.get(KEY_HISTORY) or {}).get("history", [])
        return self

    def _legacy_payloads(self) -> list[tuple[str, Any]]:
        return [
            (KEY_KING, self.king),
            (KEY_KING_CHAIN, {"chain": self.king_chain}),
            (
                KEY_VALIDATOR_STATE,
                {
                    "last_weight_block": self.last_weight_block,
                    "last_winner_hotkey": self.last_winner_hotkey,
                    "counter": self.counter,
                    "stats": self.stats,
                },
            ),
            (KEY_SEEN, {"hotkeys": sorted(self.seen_hotkeys)}),
            (KEY_HISTORY, {"history": self.history}),
        ]

    # ── persistence ───────────────────────────────────────────────────────
    @classmethod
    async def load(cls, store: JsonBucketStore) -> "KingState":
        """Load state, or RAISE.

        Never returns blank state because of an I/O problem: a validator that
        cannot read its state must not run, since on a chain-derived system that
        means re-dueling every past challenger and re-seeding genesis over the
        reigning king.
        """
        doc = await store.get(KEY_STATE)  # raises on outage / corruption
        if doc is not None:
            return cls._from_doc(doc)

        legacy = {key: await store.get(key) for key in LEGACY_KEYS}
        present = [key for key, value in legacy.items() if value]
        if not present:
            return cls()  # genuine fresh start

        if KEY_KING not in present and (legacy.get(KEY_SEEN) or legacy.get(KEY_HISTORY)):
            raise StateInconsistent(
                "seen/history are present but king/current.json is absent — the bucket is "
                "partially written or partially deleted. Refusing to re-seed genesis over a "
                "live chain. Restore the bucket, or set LEOMA_FORCE_FRESH_STATE=1 to override."
            )

        log(f"Migrating legacy state ({len(present)} keys) to {KEY_STATE}", "info")
        state = cls._from_legacy(legacy)
        state._dirty = True
        return state

    async def flush(self, store: JsonBucketStore, *, mirror: bool = True, force: bool = False) -> None:
        """Write state. The canonical object is a single atomic PUT."""
        if not (self._dirty or force):
            return

        await store.put(KEY_STATE, self.to_doc())  # the only write that matters
        self._dirty = False

        if not mirror:
            return
        for key, payload in self._legacy_payloads():
            try:
                await store.put(key, payload)
            except StoreUnavailable as e:
                # Mirrors are decorative; the canonical state is already durable.
                log(f"legacy mirror {key} failed (canonical state is durable): {e}", "warn")

    # ── helpers ───────────────────────────────────────────────────────────
    def touch(self) -> None:
        self._dirty = True

    def next_eval_id(self) -> str:
        self.counter += 1
        self._dirty = True
        return f"eval-{self.counter:04d}"

    def mark_seen(self, key: str) -> None:
        """`key` is the `_seen_key(hotkey, digest)` idempotency key."""
        self.seen_hotkeys.add(key)
        self._dirty = True

    def record_duel(self, entry: dict) -> None:
        """Prepend a duel verdict to the bounded history (newest first)."""
        self.history.insert(0, entry)
        del self.history[HISTORY_LIMIT:]
        self._dirty = True

    # ── attempt ledger ────────────────────────────────────────────────────
    # The chain re-supplies the work item every tick; the ledger supplies the
    # memory. Keyed by "<hotkey>|<digest>": repo@digest is immutable, so "this
    # artifact cannot be evaluated" is a permanent, ARTIFACT-scoped fact — a
    # miner who fixes the model gets a brand-new key and a clean slate.
    # Quarantine is never a ban on a person.

    def attempt_row(self, key: str) -> dict:
        row = self.attempts.get(key)
        if row is None:
            row = {
                "attempts": 0,
                "first_block": 0,
                "last_block": 0,
                "next_retry_block": 0,
                "last_class": "",
                "last_reason": "",
                "last_error": "",
                "quarantined": False,
                "quarantine_reason": None,
            }
            self.attempts[key] = row
            self._dirty = True
        return row

    def next_retry_block(self, key: str) -> int:
        row = self.attempts.get(key)
        return int(row.get("next_retry_block", 0)) if row else 0

    def is_quarantined(self, key: str) -> bool:
        row = self.attempts.get(key)
        return bool(row and row.get("quarantined"))

    def record_failure(
        self,
        key: str,
        *,
        block: int,
        failure,
        max_attempts: int = MAX_DUEL_ATTEMPTS,
        permanent_after: int = PERMANENT_SIGHTINGS,
    ) -> dict:
        """Record a failed attempt; quarantine when appropriate. Returns the row."""
        row = self.attempt_row(key)
        row["attempts"] = int(row.get("attempts", 0)) + 1
        if not row["first_block"]:
            row["first_block"] = block
        row["last_block"] = block
        row["last_class"] = getattr(failure.kind, "value", str(failure.kind))
        row["last_reason"] = failure.reason
        row["last_error"] = (failure.detail or "")[:500]

        backoff = BACKOFF_BLOCKS[min(row["attempts"] - 1, len(BACKOFF_BLOCKS) - 1)]
        row["next_retry_block"] = block + backoff

        if failure.is_permanent and row["attempts"] >= permanent_after:
            # Require two sightings: cheap insurance against a flaky 404.
            row["quarantined"] = True
            row["quarantine_reason"] = "permanent"
        elif row["attempts"] >= max_attempts:
            row["quarantined"] = True
            row["quarantine_reason"] = "exhausted"

        self._dirty = True
        return row

    def clear_attempts(self, key: str) -> None:
        if self.attempts.pop(key, None) is not None:
            self._dirty = True

    def banned_hotkeys(self, *, ban_after: int = BAN_AFTER_QUARANTINED) -> set:
        """Hotkeys with >= ban_after DISTINCT quarantined digests.

        This — not the per-digest quarantine — is what feeds
        ``scan_reveals(blacklist=)``, which is hotkey-keyed by construction.
        It is a spam guard, not a punishment for one bad artifact.
        """
        counts: dict[str, int] = {}
        for key, row in self.attempts.items():
            if not row.get("quarantined"):
                continue
            hotkey = key.split("|", 1)[0]
            counts[hotkey] = counts.get(hotkey, 0) + 1
        return {hk for hk, n in counts.items() if n >= ban_after}
