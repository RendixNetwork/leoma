"""Validator-side discovery of miner model submissions from on-chain reveals.

The validator reads the revealed commitments from chain, parses each into an
immutable ``(repo, digest)`` model reference, and queues the challenger for
evaluation. It downloads the weights itself (``model_store``) — no miner-hosted
endpoint is ever contacted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from leoma.infra.model_store import parse_reveal_v4
from leoma.infra.commit_parser import validate_repo_name


@dataclass(frozen=True)
class ChallengerEntry:
    """A discovered miner model submission ready for evaluation."""

    hotkey: str
    block: int
    model_repo: str
    model_digest: str


def decode_scale_revealed_message(encoded: object) -> str:
    """Decode one SCALE ``Bytes`` commitment from either SDK representation.

    Bittensor 10.5's commitment helper expects the async substrate client to return
    a hex string.  Newer clients can instead return the same SCALE bytes as a
    latin-1 ``str`` (one character per byte).  Feeding that value to
    ``bytes.fromhex`` aborts the *entire* subnet scan when any commitment exists.

    Accept both representations, validate the compact length exactly, and decode
    UTF-8 strictly.  Callers treat a failure as an invalid reveal; corrupted chain
    data must never be massaged into a miner submission.
    """
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        raw = bytes(encoded)
    elif isinstance(encoded, str):
        body = encoded.removeprefix("0x")
        looks_hex = bool(body) and len(body) % 2 == 0 and all(
            char in "0123456789abcdefABCDEF" for char in body
        )
        if encoded.startswith("0x") or looks_hex:
            try:
                raw = bytes.fromhex(body)
            except ValueError as exc:
                raise ValueError("invalid hex commitment") from exc
        else:
            try:
                raw = encoded.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise ValueError("commitment is not a byte-preserving string") from exc
    else:
        raise ValueError(f"unsupported commitment type: {type(encoded).__name__}")

    if not raw:
        raise ValueError("empty SCALE commitment")

    mode = raw[0] & 0b11
    if mode == 0:
        size = raw[0] >> 2
        offset = 1
    elif mode == 1:
        if len(raw) < 2:
            raise ValueError("truncated two-byte SCALE length")
        size = int.from_bytes(raw[:2], "little") >> 2
        offset = 2
    elif mode == 2:
        if len(raw) < 4:
            raise ValueError("truncated four-byte SCALE length")
        size = int.from_bytes(raw[:4], "little") >> 2
        offset = 4
    else:
        size_bytes = (raw[0] >> 2) + 4
        offset = 1 + size_bytes
        if len(raw) < offset:
            raise ValueError("truncated big-integer SCALE length")
        size = int.from_bytes(raw[1:offset], "little")

    end = offset + size
    if end != len(raw):
        raise ValueError(
            f"SCALE commitment length mismatch: declared {size}, "
            f"available {len(raw) - offset}"
        )
    try:
        return raw[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("commitment payload is not valid UTF-8") from exc


def scan_reveals(
    commits: Optional[Dict[str, Sequence[Tuple[int, str]]]],
    *,
    blacklist: Optional[set] = None,
) -> List[ChallengerEntry]:
    """Parse revealed commitments into challenger entries (latest valid per hotkey).

    ``commits`` is the shape returned by
    ``AsyncSubtensor.get_all_revealed_commitments``: ``{hotkey: [(block, payload), ...]}``.
    For each hotkey we take the latest payload, parse the ``v4`` reveal, and:
      - drop blacklisted hotkeys,
      - require the payload's author hotkey to match the chain signer (chain wins),
      - enforce the repo naming rule (starts "leoma", ends with the hotkey),
      - skip legacy/malformed/non-v4 payloads (e.g. the old JSON commit format).

    Entries are returned in ascending commit-block order (older submissions
    first), so a stable, deterministic queue across validators.
    """
    blk = blacklist or set()
    out: List[ChallengerEntry] = []

    for hotkey, history in (commits or {}).items():
        if not history or hotkey in blk:
            continue

        block, payload = history[-1]
        try:
            ref, author_hotkey = parse_reveal_v4(payload)
        except ValueError:
            continue  # legacy JSON / malformed / non-v4 reveal

        if author_hotkey != hotkey:
            continue  # payload author must match the chain commitment signer

        ok, _reason = validate_repo_name(ref.repo, hotkey=hotkey)
        if not ok:
            continue

        out.append(
            ChallengerEntry(
                hotkey=hotkey,
                block=int(block),
                model_repo=ref.repo,
                model_digest=ref.digest,
            )
        )

    out.sort(key=lambda e: (e.block, e.hotkey))
    return out
