"""Build + publish the public ``dashboard.json`` (Teutonic-style).

The validator has no API, so — like Teutonic — it publishes a single JSON
snapshot to its own bucket that the website polls. This module builds that
payload from the in-memory king state (pure, testable) and writes it to the
bucket. The bucket object must be public-read for the site to fetch it.

Shape (consumed by the leoma-app dashboard):
  updated_at, chain{name,seed_repo,seed_digest,netuid}, duel_params,
  king{hotkey,uid,model_repo,model_digest,reign_number,crowned_at,crowned_block,...},
  king_chain[{...,uid,weight}]   (current king first; weight = equal share, 1/n),
  stats{accepted,rejected,failed},
  queue[{hotkey,uid,model_repo,model_digest,block,status}],
  history[{hotkey,uid,model_repo,verdict,accepted,mu_hat,lcb,...}]  (newest first)
  live_duels[{eval_id,hotkey,uid,model_repo,model_digest,dispatched_block,eval_server_url}]
    (one entry per duel currently in flight — usually 0 or 1, more with several eval servers)
  live{...}  (back-compat: the first live_duels entry, or null — the pre-multi-server
    shape the deployed frontend still reads; drop once it reads live_duels instead)
"""
from __future__ import annotations

from typing import Optional

from leoma.app.validator import king as K
from leoma.app.validator.state_store import JsonBucketStore, KingState

KEY_DASHBOARD = "dashboard.json"


def _king_entry(entry: dict, uid_map: dict[str, int]) -> dict:
    hk = entry.get("hotkey", "")
    return {
        "hotkey": hk,
        "uid": uid_map.get(hk),
        "model_repo": entry.get("model_repo", ""),
        "model_digest": entry.get("model_digest", ""),
        "reign_number": entry.get("reign_number"),
        "crowned_at": entry.get("crowned_at"),
        "crowned_block": entry.get("crowned_block"),
        "challenge_id": entry.get("challenge_id"),
        "previous_repo": entry.get("previous_repo", ""),
    }


def build_dashboard(
    state: KingState,
    uid_map: dict[str, int],
    *,
    chain_meta: dict,
    duel_params: dict,
    updated_at: str,
    queue: Optional[list[dict]] = None,
) -> dict:
    """Assemble the dashboard payload (pure — no I/O, no wall clock)."""
    # Distinct king hotkeys sharing emission (current king first), and the equal
    # share among those actually registered on the metagraph.
    hks = K.king_hotkeys(state.king, state.king_chain)
    registered = [hk for hk in hks if hk in uid_map]
    weight = round(1.0 / len(registered), 9) if registered else None

    chain = ([state.king] if state.king else []) + list(state.king_chain or [])
    king_chain = []
    for entry in chain:
        row = _king_entry(entry, uid_map)
        row["weight"] = weight if entry.get("hotkey", "") in uid_map else None
        king_chain.append(row)

    king = _king_entry(state.king, uid_map) if state.king else {}

    # The duel(s) currently on the GPU(s). The dashboard used to go dark for the
    # entire length of a duel — hours in which the most interesting thing in the
    # subnet was happening and the site showed nothing at all. A list, not a single
    # object, because a validator with several eval servers can have several duels
    # running at once.
    live_duels = [
        {
            "eval_id": slot.get("eval_id"),
            "hotkey": slot.get("hotkey"),
            "uid": uid_map.get(slot.get("hotkey", "")),
            "model_repo": slot.get("model_repo"),
            "model_digest": slot.get("model_digest"),
            "dispatched_block": slot.get("dispatched_block"),
            "eval_server_url": slot.get("eval_server_url"),
        }
        for slot in (state.inflight or [])
    ]

    return {
        "updated_at": updated_at,
        "chain": chain_meta,
        "duel_params": duel_params,
        "king": king,
        "king_chain": king_chain,
        "stats": dict(state.stats),
        "queue": list(queue or []),
        "history": list(state.history),
        "live_duels": live_duels,
        # Back-compat for the already-deployed leoma-app frontend, which reads
        # ``data.live`` (a single dict-or-null) and has no knowledge of
        # ``live_duels`` yet. First in-flight duel, or None when idle — exactly
        # the old single-server shape. Drop once the frontend reads live_duels.
        "live": live_duels[0] if live_duels else None,
        # Why the validator is not crowning anyone, if it isn't: an unpinned corpus, a
        # missing seed digest, a stale eval box. Without this the operator sees a
        # subnet burning 100% to UID 0 and no reason anywhere.
        "degraded": state.degraded,
    }


async def publish_dashboard(store: JsonBucketStore, payload: dict) -> None:
    """Write the dashboard snapshot to the bucket (public-read object)."""
    await store.put(KEY_DASHBOARD, payload)
