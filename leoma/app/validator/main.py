"""
Validator service for Leoma — king of the hill.

Each permissioned validator independently:
  1. scans the chain's revealed commitments for miner model submissions
     (``reveal_scan``) — no owner-api, no rotation,
  2. duels each new challenger against the reigning king on a GPU eval server
     (deterministic, block-hash-seeded), and
  3. crowns a challenger that wins by a confident margin, then sets equal weights
     across the king + recent prior kings (else burns to UID 0).

Because the duel is deterministic given the chain + reveals, every validator
converges on the same king with no cross-validator coordination. The king state
persists to this validator's own bucket (``state_store``).
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Optional

import bittensor as bt

from leoma.bootstrap import (
    NETUID,
    NETWORK,
    WALLET_NAME,
    HOTKEY_NAME,
    R2_OWN_BUCKET,
)
from leoma.bootstrap import emit_log as log, emit_header as log_header, log_exception
from leoma.infra.storage_backend import create_own_write_client, ensure_bucket_exists
from leoma.infra.chain_config import (
    CONSENSUS_DIGEST,
    NAME as CHAIN_NAME,
    SEED_REPO,
    SEED_DIGEST,
    SPEC,
)
from leoma.eval.codehash import eval_code_digest
from leoma.eval.errors import ConsensusConfigError
from leoma.eval.spec import verify_echo
from leoma.app.validator.reveal_scan import scan_reveals, ChallengerEntry
from leoma.app.validator.state_store import (
    JsonBucketStore,
    KingState,
    StateInconsistent,
    StoreUnavailable,
)
from leoma.app.validator.dashboard import build_dashboard, publish_dashboard
from leoma.app.validator.failures import (
    DuelFailure,
    ErrorClass,
    EvalBusy,
    EvalJobFailed,
    classify,
    classify_remote,
)
from leoma.app.validator import rate_limit as RL
from leoma.app.validator.prescreen import prescreen
from leoma.app.validator.copy_check import check_model_copy
from leoma.app.validator.state_store import MAX_DUEL_ATTEMPTS
from leoma.app.validator import king as K

EVAL_SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:9000")

# One eval server means one duel at a time — the next bottleneck once the prescreen
# is doing its job (a bad model is rejected in seconds, but a QUEUE of good ones still
# drains one multi-hour duel at a time). EVAL_SERVER_URLS lets an operator point at
# several independently-run eval-server processes (e.g. one per GPU pair on an 8xH100
# box, each pinned via LEOMA_KING_DEVICE/LEOMA_CHALLENGER_DEVICE) and the validator
# fills whichever is free. Falls back to the single EVAL_SERVER_URL when unset, so a
# single-box operator's config — and every existing single-server behavior — is
# unchanged.
def _parse_eval_server_urls(raw: str, fallback: str) -> list[str]:
    """Pure so it's directly testable — no env-var/reload dance required.

    A blank/whitespace-only entry between commas is dropped rather than kept as an
    empty string, which would otherwise become a URL nothing can ever dispatch to
    (permanently "busy" from the moment it's counted, since it can never resolve).
    """
    raw = raw.strip()
    if not raw:
        return [fallback]
    return [u.strip() for u in raw.split(",") if u.strip()]


EVAL_SERVER_URLS: list[str] = _parse_eval_server_urls(
    os.environ.get("EVAL_SERVER_URLS", ""), EVAL_SERVER_URL
)

CHALLENGE_POLL_INTERVAL = int(os.environ.get("LEOMA_CHALLENGE_POLL_INTERVAL", "60"))

# The pre-dispatch architecture check. On by default: it is the difference between a
# bad model costing ~5 seconds and costing hours of GPU with the lock held. Off only
# as an operator escape hatch if the Hub's config endpoint is misbehaving.
PRESCREEN_ENABLED = os.environ.get("LEOMA_PRESCREEN", "1") != "0"

# The OCI-layer copy check + earliest-author displacement. On by default: one manifest
# fetch catches a repackaged copy of the king that the free exact-digest check misses,
# for the cost of a few KB instead of a multi-hour duel. Fails open if it cannot run.
COPY_CHECK_ENABLED = os.environ.get("LEOMA_COPY_CHECK", "1") != "0"

# The duel parameters (metric, n_clips, delta, alpha, bootstrap, generation knobs)
# are NOT read from the environment any more. They are the consensus surface, and
# they live in chain.toml as `SPEC`. An env var is per-box; a per-box exam is not
# consensus. LEOMA_DUEL_METRIC / _N_CLIPS / _DELTA_THRESHOLD / _ALPHA / _N_BOOTSTRAP
# are deliberately gone — setting one no longer does anything, which is the point.

# Every call to the eval server is now short: dispatch, or poll. Neither waits for a
# duel. LEOMA_EVAL_TIMEOUT is gone — the validator no longer has to guess how long a
# 14B video duel takes, which is a guess it could only ever get wrong. The wall-clock
# bound lives on the eval server, where the phase information actually is.
_EVAL_CONNECT_TIMEOUT = 30.0
_EVAL_POLL_TIMEOUT = 60.0

# Validator-side backstop for a duel the eval box never finishes reporting. Generous
# on purpose (~18h at 12s blocks): a real duel of two 14B models can run hours, so this
# only fires on a box that is genuinely wedged, not on slow-but-alive work.
MAX_INFLIGHT_BLOCKS = int(os.environ.get("LEOMA_MAX_INFLIGHT_BLOCKS", "5400"))


def _seen_key(hotkey: str, digest: str) -> str:
    return f"{hotkey}|{digest}"


def _is_current_king(state: KingState, entry: ChallengerEntry) -> bool:
    k = state.king or {}
    return k.get("hotkey") == entry.hotkey and k.get("model_digest") == entry.model_digest


def _copies_a_king(state: KingState, entry: ChallengerEntry) -> Optional[str]:
    """Is this a *different* hotkey submitting a king's exact weights? Returns whose.

    ``_is_current_king`` requires the hotkey to match as well as the digest — which is
    right for "don't re-duel the incumbent against itself", but it left a hole: a
    **different** hotkey re-committing the king's exact digest was treated as a novel
    challenger and handed a full multi-hour duel on the subnet's only GPU. It would
    then tie the king exactly and lose (the threshold requires strictly better), so it
    could never actually win — but it cost hours of GPU every time, for free, and
    could be repeated indefinitely.

    The digest is content-addressed, so identical digest means identical bytes. There
    is nothing to evaluate. Reject before dispatch.

    Prior kings count too: submitting a *previous* king's weights is the same
    plagiarism, and it has already been beaten by the current king.
    """
    for who in ([state.king] if state.king else []) + list(state.king_chain or []):
        if not who:
            continue
        if who.get("model_digest") == entry.model_digest and who.get("hotkey") != entry.hotkey:
            return str(who.get("hotkey", ""))
    return None


async def refresh_uid_map(subtensor: bt.AsyncSubtensor) -> dict[str, int]:
    meta = await subtensor.metagraph(NETUID)
    hotkeys = list(getattr(meta, "hotkeys", []) or [])
    return {hk: uid for uid, hk in enumerate(hotkeys)}


async def preflight_eval_server(client, eval_server_url: str = EVAL_SERVER_URL) -> None:
    """Refuse a box whose pinned exam or scoring code has drifted from ours.

    A stale eval box is the most *likely* consensus failure in the whole system —
    not an attack, just an operator who redeployed three machines out of four. It
    produces confident, plausible verdicts that no other validator can reproduce.
    One cheap GET catches it; the alternative is finding out after an hour of GPU.
    """
    resp = await client.get(f"{eval_server_url}/health")
    resp.raise_for_status()
    health = resp.json()

    theirs = health.get("consensus_digest")
    if theirs != CONSENSUS_DIGEST:
        raise EvalJobFailed(
            f"eval server {eval_server_url} pins a different consensus surface (box "
            f"{theirs}, validator {CONSENSUS_DIGEST}). One of us is running a stale chain.toml.",
            reason="consensus_mismatch",
        )
    their_code = health.get("eval_code_digest")
    if their_code and their_code != eval_code_digest():
        raise EvalJobFailed(
            f"eval server {eval_server_url} runs different scoring code (box {their_code}, "
            f"validator {eval_code_digest()}). Its distances would not be reproducible.",
            reason="code_mismatch",
        )


async def start_duel(
    entry: ChallengerEntry, king: dict, block_hash: str, *, eval_server_url: str = EVAL_SERVER_URL,
) -> str:
    """POST a duel to ``eval_server_url`` and return its ``eval_id``. Seconds, not hours.

    The validator used to dispatch a duel and then *stream it to completion* — an
    ``await`` that blocked the entire tick loop for as long as the duel took. For a
    32-clip duel of two 14B video models, that is hours in which the validator sets no
    weights, publishes no dashboard, and looks dead to the chain. The duel was not the
    only thing that stopped; the validator did.

    Now the tick is bounded: dispatch, persist the slot, return. The verdict is
    collected on a later tick by :func:`settle_inflight`.

    Sends the **whole pinned consensus surface** (``SPEC``) rather than a handful of
    loose knobs. Raises ``EvalBusy`` on 409 (a property of the *server*) and
    ``EvalJobFailed`` otherwise (a property of the *challenger*, or of us).
    """
    import httpx

    payload = {
        "king_repo": king["model_repo"],
        "king_digest": king["model_digest"],
        "challenger_repo": entry.model_repo,
        "challenger_digest": entry.model_digest,
        "block_hash": block_hash,
        "hotkey": entry.hotkey,
        "spec": SPEC.model_dump(mode="json"),
        "consensus_digest": CONSENSUS_DIGEST,
    }
    timeout = httpx.Timeout(_EVAL_POLL_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await preflight_eval_server(client, eval_server_url)

        resp = await client.post(f"{eval_server_url}/eval", json=payload)
        if resp.status_code == 409:
            raise EvalBusy(f"eval server {eval_server_url} is already running a duel")
        resp.raise_for_status()
        return resp.json()["eval_id"]


async def cancel_duel(eval_id: str, *, eval_server_url: str = EVAL_SERVER_URL) -> None:
    """Best-effort ``DELETE /eval/{id}`` — ask the box to stop burning GPU. Never raises."""
    import httpx

    timeout = httpx.Timeout(_EVAL_POLL_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            await client.delete(f"{eval_server_url}/eval/{eval_id}")
    except Exception:  # noqa: BLE001 — the abandon must proceed whether or not this lands
        pass


async def poll_duel(eval_id: str, *, eval_server_url: str = EVAL_SERVER_URL) -> dict:
    """Ask ``eval_server_url`` how a dispatched duel is going.

    Returns ``{status, phase, verdict, error, reason}``. ``status`` is one of
    ``running`` / ``done`` / ``error`` / ``cancelled``.

    A **404** means the eval box no longer knows about this duel — it restarted, or
    the job aged out. That is transient and ours to retry; it must never be read as a
    verdict, and it must never be charged to the miner.
    """
    import httpx

    timeout = httpx.Timeout(_EVAL_POLL_TIMEOUT, connect=_EVAL_CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{eval_server_url}/eval/{eval_id}")
        if resp.status_code == 404:
            raise EvalJobFailed(
                f"eval server {eval_server_url} no longer knows about {eval_id} (it restarted, "
                "or the job aged out). The duel is lost; it will be re-dispatched.",
                reason="eval_job_lost",
            )
        resp.raise_for_status()
        return resp.json()


async def _dispatch_to_first_free_server(
    entry: ChallengerEntry, king: dict, block_hash: str, busy_urls: set[str],
) -> tuple[str, str]:
    """Try every configured eval-server URL not already busy, in order, and return
    ``(url, eval_id)`` from the first that accepts the duel.

    A LOCAL fault (a stale consensus surface, mismatched scoring code) on one URL is
    skipped in favor of trying another — **unless there is only one configured URL**,
    in which case it propagates immediately, exactly as it always has for a
    single-box validator. Only when every free URL is locally faulty does the fault
    propagate, because at that point there genuinely is no eval capacity anywhere.
    """
    free = [u for u in EVAL_SERVER_URLS if u not in busy_urls]
    last_local: Optional[Exception] = None
    for url in free:
        try:
            eval_id = await start_duel(entry, king, block_hash, eval_server_url=url)
            return url, eval_id
        except EvalBusy:
            continue  # this url raced us onto another duel; try the next free one
        except Exception as e:  # noqa: BLE001 — classified below, and by the caller
            failure = classify(e)
            if failure.is_local and len(EVAL_SERVER_URLS) > 1:
                log(f"Eval server {url} is locally faulty ({failure.reason}); trying another "
                    "configured server", "warn")
                last_local = e
                continue
            raise  # a single configured URL, or a non-local error: behave exactly as before
    if last_local is not None:
        raise last_local  # every free URL was locally faulty — genuinely no capacity
    raise EvalBusy("no free eval server accepted the duel")


def _verified_verdict(verdict: Optional[dict]) -> dict:
    """Fail closed unless the box ran EXACTLY the exam we set.

    This is the check that catches "the field wasn't sent, so the box used its own
    default" — the entire bug class that made two honest validators disagree. It runs
    **before crowning**, so a verdict produced under different parameters can never
    take the crown.
    """
    if not verdict:
        raise EvalJobFailed(
            "eval server reported success but returned no verdict", reason="stream_no_terminal"
        )
    try:
        verify_echo(SPEC, verdict.get("echo"))
    except ConsensusConfigError as e:
        raise EvalJobFailed(str(e), reason="consensus_echo_mismatch") from e
    return verdict


# bittensor 9.12.2's set_weights returns (success: bool, message: str). Its
# rate-limit path returns this literal WITHOUT ever submitting an extrinsic.
_RATE_LIMIT_TOKENS = (
    "no attempt made",
    "too soon",
    "rate limit",
    "settingweightstoofast",
)


def _unpack_set_weights(result) -> tuple[bool, str]:
    """Tolerate (bool, str), a bare bool, or an ExtrinsicResponse-like object."""
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1])
    if hasattr(result, "success"):
        return bool(result.success), str(getattr(result, "message", "") or "")
    return bool(result), ""


def _is_rate_limited(message: str) -> bool:
    lowered = (message or "").lower()
    return any(token in lowered for token in _RATE_LIMIT_TOKENS)


async def _chain_says_too_soon(subtensor: bt.AsyncSubtensor, wallet: bt.Wallet) -> bool:
    """Ask the CHAIN whether it is too soon to set weights.

    The chain is the queue, so the chain is also the weight clock. Trusting our
    own ``last_weight_block`` is what made "state says the weights landed but they
    didn't" possible; asking the chain makes that structurally impossible.
    Degrades to False (i.e. attempt the set) if the node doesn't expose these.
    """
    try:
        uid = await subtensor.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, NETUID)
        if uid is None:
            return False
        since = await subtensor.blocks_since_last_update(NETUID, uid)
        limit = await subtensor.weights_rate_limit(NETUID)
    except Exception:
        return False
    if since is None or limit is None:
        return False
    return since <= limit


def _weight_failed(state: KingState, block: int, message: str) -> bool:
    state.weight_failures += 1
    # Exponential block backoff, capped — a genuine failure should be retried
    # soon, NOT after a full WEIGHT_INTERVAL.
    state.next_weight_block = block + min(2**state.weight_failures, 20)
    state.touch()
    log(f"[{block}] set_weights FAILED ({state.weight_failures}x): {message}", "error")
    return False


async def maybe_set_weights(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    *,
    force: bool = False,
) -> bool:
    """Set equal weights over the king chain (else burn UID 0), rate-limited.

    The return value of ``set_weights`` used to be discarded. bittensor returns
    ``(success, message)`` and its rate-limit path returns ``False`` *without
    submitting anything* — so a no-op advanced ``last_weight_block`` as if the
    weights had landed, blocking any retry for a full WEIGHT_INTERVAL (~1h) and
    misreporting the state. We now only advance on a real success.
    """
    current_block = await subtensor.get_current_block()

    if not force:
        if current_block < state.next_weight_block:
            return False  # backing off after a genuine failure
        if current_block - state.last_weight_block < K.WEIGHT_INTERVAL:
            return False
        if await _chain_says_too_soon(subtensor, wallet):
            return False

    uids, weights, label = K.weight_targets(
        state.king, state.king_chain, uid_map, burn_uid=K.BURN_UID
    )
    log(
        f"[{current_block}] set_weights -> {label}: uids={uids} "
        f"weights={[round(w, 4) for w in weights]}",
        "info",
    )
    try:
        result = await subtensor.set_weights(
            wallet=wallet, netuid=NETUID, uids=uids, weights=weights, wait_for_inclusion=True
        )
    except Exception as e:
        return _weight_failed(state, current_block, f"exception: {e}")

    success, message = _unpack_set_weights(result)
    if not success:
        if _is_rate_limited(message):
            # A no-op, not a failure: do NOT advance last_weight_block.
            log(f"[{current_block}] set_weights rate-limited ({message}); retrying next tick", "warn")
            return False
        return _weight_failed(state, current_block, message)

    state.last_weight_block = current_block
    state.last_winner_hotkey = label
    state.weight_failures = 0
    state.next_weight_block = 0
    state.touch()
    await state.flush(store)
    return True


def ensure_genesis_king(state: KingState, block: int) -> None:
    """Seed the genesis king from chain.toml on first run (empty hotkey ⇒ burns
    emission until a miner's challenger dethrones the base model)."""
    if state.king:
        return
    if not SEED_DIGEST:
        log("No seed_digest in chain.toml; king unset (burning) until first crown", "warn")
        return
    king, chain = K.crown(
        None, [], hotkey="", model_repo=SEED_REPO, model_digest=SEED_DIGEST,
        block=block, challenge_id=K.SEED_CHALLENGE_ID,
    )
    state.king, state.king_chain = king, chain
    state.touch()
    log(f"Seeded genesis king from {SEED_REPO}@{SEED_DIGEST[:16]}...", "success")


def _duel_history_entry(entry: ChallengerEntry, verdict: dict, uid_map: dict[str, int]) -> dict:
    """A dashboard history row from a completed duel verdict."""
    return {
        "challenge_id": verdict.get("challenge_id") or f"block-{entry.block}",
        "hotkey": entry.hotkey,
        "uid": uid_map.get(entry.hotkey),
        "model_repo": entry.model_repo,
        "model_digest": entry.model_digest,
        "accepted": bool(verdict.get("accepted")),
        "verdict": verdict.get("verdict", "unknown"),
        "mu_hat": verdict.get("mu_hat"),
        "lcb": verdict.get("lcb"),
        "delta": verdict.get("delta_threshold"),
        "avg_king_distance": verdict.get("avg_king_distance"),
        "avg_challenger_distance": verdict.get("avg_challenger_distance"),
        "metric": SPEC.duel.metric,
        "n_clips": verdict.get("n_clips"),
        "block": entry.block,
        # Why a challenger that beat the king still didn't take the crown: it copied
        # the king, or it beat a mediocre king only by holding the conditioning frame.
        "rejected_by": verdict.get("rejected_by"),
        "gates": verdict.get("gates"),
        # The eval server stamps produced_at OUTSIDE the digested surface, so two
        # validators that agree still produce identical verdict_digests.
        "timestamp": verdict.get("produced_at"),
        "early_stopped": bool(verdict.get("early_stopped")),
        # The audit anchors: anyone can now check that two validators graded the
        # same exam (clip_keys_digest) and reached the same call (verdict_digest).
        "verdict_digest": verdict.get("verdict_digest"),
        "consensus_digest": (verdict.get("audit") or {}).get("consensus_digest"),
        "clip_keys_digest": ((verdict.get("audit") or {}).get("corpus") or {}).get("clip_keys_digest"),
    }


def _build_queue(state: KingState, entries: list[ChallengerEntry], uid_map: dict[str, int]) -> list[dict]:
    """Pending-challenger view for the dashboard (excludes the reigning king)."""
    queue = []
    for e in entries:
        if _is_current_king(state, e):
            continue
        seen = _seen_key(e.hotkey, e.model_digest) in state.seen_hotkeys
        queue.append({
            "hotkey": e.hotkey,
            "uid": uid_map.get(e.hotkey),
            "model_repo": e.model_repo,
            "model_digest": e.model_digest,
            "block": e.block,
            "status": "seen" if seen else "unseen",
        })
    return queue


async def _publish_dashboard(
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    queue: list[dict],
) -> None:
    """Build + publish the public dashboard.json snapshot (best-effort)."""
    try:
        payload = build_dashboard(
            state,
            uid_map,
            chain_meta={
                "name": CHAIN_NAME,
                "seed_repo": SEED_REPO,
                "seed_digest": SEED_DIGEST,
                "netuid": NETUID,
            },
            duel_params={
                "metric": SPEC.duel.metric,
                "metric_device": SPEC.duel.metric_device,
                "n_clips": SPEC.duel.n_clips,
                "delta_threshold": SPEC.duel.delta_threshold,
                "alpha": SPEC.duel.alpha,
                "n_bootstrap": SPEC.duel.n_bootstrap,
                "consensus_digest": CONSENSUS_DIGEST,
                "corpus_pinned": SPEC.corpus.pinned,
            },
            updated_at=datetime.now(timezone.utc).isoformat(),
            queue=queue,
        )
        await publish_dashboard(store, payload)
    except Exception as e:
        log(f"Dashboard publish failed: {e}", "warn")


def _error_history_entry(
    entry: ChallengerEntry,
    failure: DuelFailure,
    uid_map: dict[str, int],
    row: dict,
) -> dict:
    """A dashboard history row for a QUARANTINED challenger.

    The frontend has typed `DuelHistoryEntry.error` since day one and it never
    had a producer: dispatch failures were only logged, so the dashboard could
    never explain why a miner's model was ignored.
    """
    return {
        "challenge_id": f"block-{entry.block}",
        "hotkey": entry.hotkey,
        "uid": uid_map.get(entry.hotkey),
        "model_repo": entry.model_repo,
        "model_digest": entry.model_digest,
        "accepted": False,
        "verdict": "error",
        "error": (failure.detail or "")[:500],
        "error_reason": failure.reason,
        "attempts": row.get("attempts"),
        "metric": SPEC.duel.metric,
        "block": entry.block,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _note_failure(
    state: KingState,
    store: JsonBucketStore,
    uid_map: dict[str, int],
    entry: ChallengerEntry,
    key: str,
    failure: DuelFailure,
    block: int,
) -> None:
    """Record a failed attempt; quarantine + surface it when the budget is spent."""
    row = state.record_failure(key, block=block, failure=failure)

    if not row["quarantined"]:
        state.stats["transient_errors"] = state.stats.get("transient_errors", 0) + 1
        state.touch()
        log(
            f"Duel failed ({failure.reason}) for {entry.hotkey[:12]}...: "
            f"attempt {row['attempts']}/{MAX_DUEL_ATTEMPTS}, retry at block {row['next_retry_block']}",
            "warn",
        )
    else:
        # stats.failed counts QUARANTINED challengers, once each. Retries go to
        # transient_errors so the dashboard number doesn't inflate with backoff.
        state.stats["failed"] = state.stats.get("failed", 0) + 1
        state.touch()
        state.record_duel(_error_history_entry(entry, failure, uid_map, row))
        state.mark_seen(key)  # never retried
        log(
            f"Challenger {entry.hotkey[:12]}... QUARANTINED "
            f"({row['quarantine_reason']}: {failure.reason})",
            "error",
        )

    await state.flush(store)


def _slot_entry(slot: dict) -> ChallengerEntry:
    """Rebuild the challenger from the persisted slot.

    Deliberately reconstructed from the slot rather than looked up in this tick's
    reveals: the slot is the authoritative record of *what we dispatched*. A reveal
    that has since scrolled out of the commitment window must still be settled — the
    duel ran, and the miner is owed its verdict.
    """
    return ChallengerEntry(
        hotkey=slot["hotkey"],
        model_repo=slot["model_repo"],
        model_digest=slot["model_digest"],
        block=int(slot.get("block", 0)),
    )


async def _local_fault(state: KingState, failure, *, url: Optional[str] = None) -> None:
    """OUR fault: stop dueling, say so loudly, and burn until an operator fixes it.

    A corpus that doesn't match the pinned manifest, an eval box on a stale
    chain.toml. It would fail identically for *every* challenger — including the
    reigning king — so it is not evidence about any one of them. And charging it to
    the ledger would, after four attempts, quarantine every honest miner on the
    subnet for our own misconfiguration.

    Reached in two situations: a fault discovered while POLLING an already-dispatched
    duel (rare — there is no per-poll consensus check to trip), or every configured
    eval server having failed its DISPATCH-time preflight (``_dispatch_to_first_free_server``
    already tried the others). Either way there is, at this point, no working eval
    capacity at all, so degrading the whole validator is correct regardless of how
    many servers are configured — the per-server "try another one" nuance lives
    entirely in ``_dispatch_to_first_free_server``, upstream of here.
    """
    where = f" ({url})" if url else ""
    log(f"LOCAL FAULT{where} ({failure.reason}) — this validator cannot duel: {failure.detail}", "error")
    state.degraded = failure.reason


async def _note_server_fault(state: KingState, failure, *, url: str) -> None:
    """A LOCAL fault discovered while POLLING an already-dispatched duel.

    Scoped exactly like the dispatch-time equivalent (see
    ``_dispatch_to_first_free_server``): with only one configured server, its failure
    IS total capacity loss, so the whole validator degrades. With several, one bad
    server does not mean the fleet is down — the other configured servers keep
    dueling and crowning normally, so unconditionally setting the validator-wide
    ``state.degraded`` flag here would be a false "everything is down" alarm (or
    worse, train an operator to start ignoring it). The affected challenger is never
    marked seen either way, so it is simply retried later — most likely on a
    different, healthy server.
    """
    if len(EVAL_SERVER_URLS) == 1:
        await _local_fault(state, failure, url=url)
        return
    log(
        f"LOCAL FAULT on {url} ({failure.reason}): {failure.detail} — other configured "
        "servers are unaffected; this duel will be retried, likely elsewhere",
        "error",
    )


async def _settle_one_slot(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    block: int,
    slot: dict,
) -> bool:
    """Try to resolve ONE in-flight slot. Returns True iff it is now resolved
    (settled, abandoned, or discarded) and has been removed from ``state.inflight``;
    False if it is still running and was left in place.

    Removal happens at the exact point the original single-slot code used to set
    ``state.inflight = None`` — i.e. *before* calling anything that flushes (
    ``_note_failure``, the crown/reject paths). A crash must never leave a resolved
    duel's outcome persisted while its slot still looks in-flight: on restart that
    would re-settle — and re-record, re-crown, re-charge — the same verdict twice.
    """
    entry = _slot_entry(slot)
    key = _seen_key(entry.hotkey, entry.model_digest)
    url = slot.get("eval_server_url") or EVAL_SERVER_URL

    def _remove() -> None:
        state.inflight = [s for s in state.inflight if s is not slot]

    try:
        result = await poll_duel(slot["eval_id"], eval_server_url=url)
    except Exception as e:  # noqa: BLE001 — classify, never crash the tick
        failure = classify(e)
        if failure.is_local:
            await _note_server_fault(state, failure, url=url)
            return False        # keep the slot; the duel may still be fine once we are
        _remove()
        state.touch()
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    status = result.get("status")
    if status == "running":
        # The eval server has its own forward-progress watchdog, and normally it fires
        # first. This is the validator-side backstop for the case that watchdog can't
        # catch: a box that keeps reporting "running" without ever tripping a phase
        # budget (a disabled watchdog, a lying box, a partition where poll succeeds but
        # the box is wedged). Without it, one stuck duel holds its slot forever and
        # that server never picks up another challenger. The bound is deliberately
        # generous — a real 32-clip duel of two 14B models can legitimately run hours —
        # so this only ever fires on a genuinely hung box.
        age = block - int(slot.get("dispatched_block", block))
        if age > MAX_INFLIGHT_BLOCKS:
            log(f"Duel {slot['eval_id']} on {url} has been in flight {age} blocks "
                f"(> {MAX_INFLIGHT_BLOCKS}) — abandoning as a stuck box and re-dispatching "
                "next tick", "warn")
            await cancel_duel(slot["eval_id"], eval_server_url=url)
            _remove()
            state.touch()
            # TRANSIENT, never LOCAL or a strike: a hung box is not the challenger's
            # fault. Backoff + the 4-attempt budget means a genuinely pathological model
            # that hangs every time still quarantines, while a one-off box wedge retries.
            failure = DuelFailure(ErrorClass.TRANSIENT, "inflight_timeout",
                                  f"duel exceeded {MAX_INFLIGHT_BLOCKS} blocks in flight")
            await _note_failure(state, store, uid_map, entry, key, failure, block)
            return True
        log(f"Duel {slot['eval_id']} on {url} still running (phase={result.get('phase')}, "
            f"age={age} blocks)", "info")
        return False

    # Terminal, one way or another: the slot is free from here on.
    _remove()
    state.touch()

    if status != "done":
        failure = classify_remote(str(result.get("error") or status), str(result.get("reason") or ""))
        if failure.is_local:
            await _note_server_fault(state, failure, url=url)
            await state.flush(store)
            return True
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    try:
        verdict = _verified_verdict(result.get("verdict"))
    except Exception as e:  # noqa: BLE001
        failure = classify(e)
        if failure.is_local:
            await _note_server_fault(state, failure, url=url)
            await state.flush(store)
            return True
        await _note_failure(state, store, uid_map, entry, key, failure, block)
        return True

    # The king may have changed while this duel was in flight (a crown from a
    # different challenger, or an operator re-seed). The verdict measured the
    # challenger against a king that no longer reigns, so it cannot be acted on —
    # and it must NOT be marked seen, or the challenger would never get a fair duel
    # against the king it actually has to beat.
    current_king_digest = (state.king or {}).get("model_digest")
    if slot.get("king_digest") != current_king_digest:
        log(
            f"King changed while {entry.hotkey[:12]}...'s duel was in flight — discarding "
            "the stale verdict; it will be re-dueled against the current king.",
            "warn",
        )
        await state.flush(store)
        return True

    # A completed duel clears the artifact's failure history and charges the hotkey's
    # GPU budget — whatever the outcome. The duel cost hours either way.
    state.clear_attempts(key)
    state.mark_seen(key)
    RL.record_verdict(state.duels, entry.hotkey, king=state.king, block=block)

    # A gate rejection is a strike; LOSING a duel fairly is not. A miner whose honest
    # model simply isn't good enough has done nothing wrong, and penalizing them for
    # trying would deter exactly the people the subnet exists to attract.
    rejected_by = str(verdict.get("rejected_by") or "")
    if rejected_by:
        RL.record_strike(state.duels, entry.hotkey, rejected_by)

    state.record_duel(_duel_history_entry(entry, verdict, uid_map))

    if verdict.get("accepted"):
        state.king, state.king_chain = K.crown(
            state.king, state.king_chain, hotkey=entry.hotkey, model_repo=entry.model_repo,
            model_digest=entry.model_digest, block=entry.block,
            challenge_id=verdict.get("challenge_id") or f"block-{entry.block}",
        )
        state.stats["accepted"] = state.stats.get("accepted", 0) + 1
        state.touch()
        await state.flush(store)
        log(f"Challenger {entry.hotkey[:12]}... CROWNED (lcb={verdict.get('lcb')})", "success")
        await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
    else:
        state.stats["rejected"] = state.stats.get("rejected", 0) + 1
        state.touch()
        await state.flush(store)
        log(f"Challenger {entry.hotkey[:12]}... rejected (lcb={verdict.get('lcb')})", "info")

    return True


async def settle_inflight(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    block: int,
) -> bool:
    """Collect verdicts for every dispatched duel. Returns True iff at least one
    eval-server slot is free once settling is done.

    **Restart-safe.** Each slot is persisted, so a validator that restarts mid-duel
    re-attaches to the jobs it left running instead of orphaning them. Before this, a
    restart meant the eval box spent hours on a duel nobody would ever read, and
    answered 409 to everyone else the entire time.

    Settled **sequentially against live state**, not a snapshot: crowning a challenger
    from one slot changes ``state.king``, and the next slot's "did the king change
    under me" check must see that change, or a challenger dueled against a
    since-deposed king could be crowned a second time in the same tick.
    """
    for slot in list(state.inflight):
        await _settle_one_slot(subtensor, wallet, state, uid_map, store, block, slot)
    return len(state.inflight) < len(EVAL_SERVER_URLS)


async def _crown_earlier(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    entry: ChallengerEntry,
    key: str,
    copy: dict,
    block: int,
) -> None:
    """Displace the king with a byte-identical model that was pushed to the registry earlier.

    No duel: the weights are provably identical to the reigning king's, so there is
    nothing to evaluate — the only question was authorship, and the registry's own push
    time answered it. The challenger is the original, front-run by whoever got crowned
    first, and it takes the crown as a synthetic accepted verdict.
    """
    log(f"{entry.hotkey[:12]}... has the king's exact weights but an EARLIER registry "
        f"push — crowning the original author, no duel. {copy['reason']}", "warn")

    verdict = {
        "accepted": True,
        "verdict": "crown_earlier",
        "challenge_id": f"block-{entry.block}",
        "reason": copy["reason"],
        "challenger_committed_at": copy.get("challenger_committed_at"),
        "king_committed_at": copy.get("king_committed_at"),
        "produced_at": datetime.now(timezone.utc).isoformat(),
    }
    state.clear_attempts(key)
    state.mark_seen(key)
    RL.record_verdict(state.duels, entry.hotkey, king=state.king, block=block)
    state.record_duel(_duel_history_entry(entry, verdict, uid_map))
    state.king, state.king_chain = K.crown(
        state.king, state.king_chain, hotkey=entry.hotkey, model_repo=entry.model_repo,
        model_digest=entry.model_digest, block=entry.block, challenge_id=verdict["challenge_id"],
    )
    state.stats["accepted"] = state.stats.get("accepted", 0) + 1
    state.touch()
    await state.flush(store)
    await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)


async def process_challengers(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    uid_map: dict[str, int],
    store: JsonBucketStore,
    entries: list[ChallengerEntry],
    block: int,
) -> None:
    """Settle every in-flight duel, then dispatch at most one more. Returns in seconds.

    This used to duel every challenger *inline*, streaming each one to completion —
    so a single tick could take hours, during which the validator set no weights and
    published no dashboard. The duel wasn't the only thing blocked; the validator was.

    Now the tick is bounded: settle whatever is in flight, dispatch the next one to
    whichever configured eval server is free, come back. Everything else in ``tick``
    (weights, dashboard) therefore runs *every* tick, even mid-duel. With several
    ``EVAL_SERVER_URLS`` configured, several duels accumulate in flight across
    successive ticks — one new dispatch per tick is enough to reach full utilization
    within a handful of ticks, since a tick is ~60s against duels that run hours.

    A failing challenger must never block the ones behind it, so the dispatch loop
    still distinguishes:

      * BUSY  -> ``break``    (a property of the SERVER(S): every configured one is busy)
      * LOCAL -> ``break``    (a property of US: no configured server has working capacity)
      * error -> ``continue`` (a property of the CHALLENGER: everyone else proceeds)
    """
    # An unpinned corpus is a subnet-wide condition, not a per-challenger one: the exam
    # itself does not exist yet. Refuse every duel and burn, exactly as with a missing
    # seed_digest — and say so once, not once per challenger.
    try:
        SPEC.require_duel_ready()
    except ConsensusConfigError as e:
        log(str(e), "error")
        state.degraded = "corpus_unpinned"
        return

    if not await settle_inflight(subtensor, wallet, state, uid_map, store, block):
        return  # every configured server is still busy; the tick moves on to weights + dashboard

    # At most one in-flight duel per HOTKEY, regardless of digest. This closes TWO
    # holes at once:
    #
    # 1. With a single eval server this was structurally impossible: settle_inflight
    #    returning "free" meant the ONE slot had just emptied, so the loop below could
    #    never see the challenger it belonged to still pending. With several servers,
    #    one busy slot no longer blocks the whole loop, so the same artifact could be
    #    dispatched a second time if this weren't checked.
    # 2. RL.check (the cooldown/per-reign-cap rate limiter) only ever sees a hotkey
    #    once one of its duels has SETTLED — record_verdict is called from
    #    _settle_one_slot, not at dispatch time — so with a single server this was
    #    ALSO never a gap: only one duel could be in flight for anyone, ever. With
    #    several servers, a hotkey minting a fresh digest every tick would sail
    #    through RL.check every time (it has no settled row yet) and could occupy
    #    every configured server simultaneously before its first duel ever resolves —
    #    exactly the GPU-fleet monopolization the rate limiter exists to prevent,
    #    reachable by an honest miner's automated resubmission, not just an adversary.
    inflight_hotkeys = {s["hotkey"] for s in state.inflight}

    for entry in entries:
        key = _seen_key(entry.hotkey, entry.model_digest)

        if key in state.seen_hotkeys or _is_current_king(state, entry):
            continue
        if entry.hotkey in inflight_hotkeys:
            continue  # this hotkey already has a duel in flight somewhere, under any digest
        if state.is_quarantined(key):
            continue  # this artifact can never be evaluated
        if block < state.next_retry_block(key):
            continue  # backoff has not elapsed

        # A different hotkey submitting a king's exact weights. Content-addressed
        # digests mean identical digest = identical bytes: there is nothing to
        # evaluate, and dispatching it would burn hours of GPU on a model that
        # already reigns. It could never win anyway (the threshold demands strictly
        # better, and a copy ties exactly) — it was simply free to repeat.
        plagiarized = _copies_a_king(state, entry)
        if plagiarized:
            failure = DuelFailure(
                ErrorClass.PERMANENT, "copy_of_king",
                f"{entry.model_digest} is {plagiarized[:12]}...'s model, repackaged. "
                "Copying a king is not an improvement on it.",
            )
            RL.record_strike(state.duels, entry.hotkey, failure.reason)
            await _note_failure(state, store, uid_map, entry, key, failure, block)
            continue

        # The exact-digest check above misses a copy that changed only its README or
        # tokenizer: identical WEIGHTS, new top-level digest. This catches it from OCI
        # layer digests — one manifest fetch, no weight download — and also displaces
        # the king when the challenger is the byte-identical ORIGINAL, front-run by
        # whoever got crowned first. Both decisions rest only on registry-observed
        # push time, so validators agree; a metadata hiccup fails OPEN (returns None).
        if COPY_CHECK_ENABLED and state.king:
            try:
                copy = await asyncio.to_thread(
                    check_model_copy, entry.model_repo, entry.model_digest,
                    state.king.get("model_repo", ""), state.king.get("model_digest", ""),
                )
            except Exception:  # noqa: BLE001 — the check itself must never crash the tick
                copy = None
            if copy and copy["action"] == "reject":
                failure = DuelFailure(ErrorClass.PERMANENT, "copy_of_king", copy["reason"])
                RL.record_strike(state.duels, entry.hotkey, failure.reason)
                await _note_failure(state, store, uid_map, entry, key, failure, block)
                continue
            if copy and copy["action"] == "crown_earlier":
                await _crown_earlier(subtensor, wallet, state, uid_map, store, entry, key, copy, block)
                return  # the king changed with no duel; re-scan next tick

        # The GPU is the scarcest thing in the subnet. A hotkey that has just been
        # dueled, or has already spent its allowance against this king, waits.
        limited = RL.check(state.duels, entry.hotkey, king=state.king, block=block)
        if limited:
            log(f"Rate-limited {entry.hotkey[:12]}...: {limited}", "info")
            continue

        if not state.king:
            # No king AND no chain.toml seed_digest. We refuse to crown an unevaluated
            # model — the burn path already exists and is correct (king_hotkeys -> [] ->
            # weight_targets -> burn UID 0). The old code short-circuited it by crowning
            # the first reveal it happened to see.
            log(
                "No king and no chain.toml seed_digest — refusing to crown an "
                "unevaluated challenger. Burning to UID 0 until seed_digest is pinned.",
                "error",
            )
            state.degraded = "no_seed_digest"
            return

        # Configs only (~200 KB), on this box, before the GPU is touched. A
        # wrong-architecture model costs seconds here instead of hours of download +
        # load, during which the eval server's lock is held and every honest
        # challenger waits behind it.
        if PRESCREEN_ENABLED:
            try:
                await asyncio.to_thread(prescreen, entry.model_repo, entry.model_digest)
            except Exception as e:  # noqa: BLE001
                failure = classify(e)
                if failure.is_local:
                    await _local_fault(state, failure)
                    return
                if failure.is_permanent:
                    RL.record_strike(state.duels, entry.hotkey, failure.reason)
                await _note_failure(state, store, uid_map, entry, key, failure, block)
                continue

        # .get() with a fallback, not direct indexing: a slot migrated from a
        # pre-multi-server bucket (_normalize_inflight) has no eval_server_url key at
        # all, and a legacy duel can still be in flight on the very restart that adds
        # a second EVAL_SERVER_URLS entry. Every other read site already falls back
        # the same way (see _settle_one_slot) — this one must match, or that upgrade
        # path raises KeyError every tick until the legacy duel finally resolves,
        # silently skipping maybe_set_weights and _publish_dashboard for as long as
        # MAX_INFLIGHT_BLOCKS allows (hours).
        busy_urls = {slot.get("eval_server_url") or EVAL_SERVER_URL for slot in state.inflight}
        try:
            block_hash = await subtensor.get_block_hash(entry.block)
            log(
                f"Dispatching duel: {entry.hotkey[:12]}... vs king "
                f"{state.king.get('hotkey','')[:12] or 'base'}...",
                "info",
            )
            free_url, eval_id = await _dispatch_to_first_free_server(
                entry, state.king, block_hash, busy_urls
            )
        except EvalBusy:
            log("Eval server(s) busy; challengers deferred to next tick", "warn")
            return
        except Exception as e:  # noqa: BLE001 — deliberate: classify, never crash the tick
            failure = classify(e)
            if failure.is_local:
                await _local_fault(state, failure)
                return
            await _note_failure(state, store, uid_map, entry, key, failure, block)
            continue  # one bad challenger no longer blocks the rest

        # Persist the slot BEFORE returning: a crash between the POST and the flush
        # would otherwise orphan a duel that is already burning GPU hours, and that
        # eval box would 409 everyone until it finished a job nobody was waiting for.
        state.inflight.append({
            "eval_id": eval_id,
            "eval_server_url": free_url,
            "hotkey": entry.hotkey,
            "model_repo": entry.model_repo,
            "model_digest": entry.model_digest,
            "block": entry.block,
            "king_digest": (state.king or {}).get("model_digest"),
            "dispatched_block": block,
        })
        state.touch()
        await state.flush(store)
        log(f"Duel {eval_id} dispatched to {free_url}; the tick continues "
            "(weights + dashboard stay live)", "info")
        return  # at most one NEW dispatch per tick — any freed slot is picked up next tick


async def tick(
    subtensor: bt.AsyncSubtensor,
    wallet: bt.Wallet,
    state: KingState,
    store: JsonBucketStore,
) -> None:
    block = await subtensor.get_current_block()
    uid_map = await refresh_uid_map(subtensor)
    ensure_genesis_king(state, block)

    commits = await subtensor.get_all_revealed_commitments(NETUID, block=block)
    # Hotkeys with several quarantined artifacts are dropped at scan time — this
    # finally wires reveal_scan's long-dead `blacklist=` hook.
    # Two distinct reasons a hotkey stops being scanned: several unevaluable artifacts
    # (it keeps uploading broken models), or several GATE rejections (it keeps uploading
    # models that should never have been dispatched — copies, wrong architectures).
    # Neither counts a fair loss.
    entries = scan_reveals(
        commits, blacklist=state.banned_hotkeys() | RL.struck_out(state.duels)
    )

    # Bounded: settles the in-flight duel and dispatches at most one more. Always
    # returns in seconds, even when a multi-hour duel is running on the GPU box.
    await process_challengers(subtensor, wallet, state, uid_map, store, entries, block)

    # These two now run on EVERY tick, including mid-duel. They used to be starved for
    # hours behind an inline duel: no weights (the chain concludes the validator is
    # dead) and a dashboard frozen on whatever was true when the duel started.
    await maybe_set_weights(subtensor, wallet, state, uid_map, store)
    await _publish_dashboard(state, uid_map, store, _build_queue(state, entries, uid_map))


async def main() -> None:
    """Run the king-of-the-hill validator loop."""
    log_header("Leoma Validator Starting (king of the hill)")

    if not R2_OWN_BUCKET:
        log("R2_OWN_BUCKET not set; validator disabled (cannot persist king state)", "error")
        return

    subtensor = bt.AsyncSubtensor(network=NETWORK)
    wallet = bt.Wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    log(f"Wallet: {WALLET_NAME}/{HOTKEY_NAME}  Network: {NETWORK}  NetUID: {NETUID}", "info")
    log(
        f"Eval server(s): {', '.join(EVAL_SERVER_URLS)}  metric={SPEC.duel.metric}@"
        f"{SPEC.duel.metric_device} n_clips={SPEC.duel.n_clips} delta={SPEC.duel.delta_threshold}",
        "info",
    )
    log(f"Consensus digest: {CONSENSUS_DIGEST}  (eval code: {eval_code_digest()})", "info")

    own_client = create_own_write_client()
    await ensure_bucket_exists(own_client, R2_OWN_BUCKET)
    store = JsonBucketStore(own_client, R2_OWN_BUCKET)

    # A validator that cannot read its state must NOT run: on a chain-derived
    # system, running on blank state re-duels every past challenger and re-seeds
    # genesis over the reigning king — and the first flush would then overwrite
    # the good bucket state with the blank one.
    try:
        state = await KingState.load(store)
    except (StoreUnavailable, StateInconsistent) as e:
        if os.environ.get("LEOMA_FORCE_FRESH_STATE") == "1":
            log(f"State unreadable ({e}); LEOMA_FORCE_FRESH_STATE=1 — starting BLANK", "warn")
            state = KingState()
        else:
            log(f"Cannot read validator state: {e}", "critical")
            log("Refusing to start on unknown state. Restore the bucket, or set "
                "LEOMA_FORCE_FRESH_STATE=1 to deliberately start blank.", "critical")
            raise SystemExit(1)

    log(f"Loaded king: {(state.king or {}).get('model_repo', 'none')} chain={len(state.king_chain)}", "info")

    # Re-assert weights immediately on startup. Without this, a restarted
    # validator with a recently-persisted last_weight_block sits idle for up to a
    # full WEIGHT_INTERVAL (~1h) before it sets weights again.
    try:
        block = await subtensor.get_current_block()
        ensure_genesis_king(state, block)
        uid_map = await refresh_uid_map(subtensor)
        await maybe_set_weights(subtensor, wallet, state, uid_map, store, force=True)
    except Exception as e:
        log(f"Startup weight-set failed (will retry on the first tick): {e}", "warn")

    while True:
        try:
            await tick(subtensor, wallet, state, store)
        except (StoreUnavailable, StateInconsistent) as e:
            # Losing the store mid-run means we can no longer persist crowns.
            log(f"State store unavailable: {e}", "critical")
            log_exception("State store unavailable — exiting for supervisor restart", e)
            raise SystemExit(1)
        except Exception as e:
            log(f"Validator tick error: {e}", "error")
            log_exception("Validator tick error", e)
        await asyncio.sleep(CHALLENGE_POLL_INTERVAL)


def main_sync() -> None:
    """Synchronous entry point for CLI."""
    asyncio.run(main())
