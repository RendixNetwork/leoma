"""
Unit tests for the validator's on-chain reveal scan.

``scan_reveals`` turns the raw ``{hotkey: [(block, payload), ...]}`` map from
``get_all_revealed_commitments`` into validated challenger entries.
"""

import pytest

from leoma.app.validator.reveal_scan import (
    ChallengerEntry,
    decode_scale_revealed_message,
    scan_reveals,
)
from leoma.infra.model_store import ModelRef, build_reveal_v4

HK1 = "5C7LM2i42XgL2oB4x3rcmB7KDiof4B92KZzUpg5miZ6DogjU"
HK2 = "5DJ76XJdWvU7PcmKmBjzoAKYC3i4YjhdR92uVYGA7FthyCv2"
HK3 = "5GW8VcE7gLU8pJFWvYYC378RyDWxN8rTCm1fNYX6AxzDV1de"

SHA = "sha256:" + "a" * 64
SHB = "sha256:" + "b" * 64


def _reveal(repo: str, digest: str, author: str) -> str:
    return build_reveal_v4(ModelRef(repo, digest), author)


def _valid_reveal(hotkey: str, digest: str = SHA) -> str:
    """A well-formed reveal whose repo obeys the leoma-prefix + hotkey-suffix rule."""
    return _reveal(f"user/leoma-m-{hotkey}", digest, hotkey)


def _scale_bytes(payload: str) -> bytes:
    body = payload.encode()
    size = len(body)
    if size < 1 << 6:
        prefix = bytes([size << 2])
    elif size < 1 << 14:
        prefix = ((size << 2) | 1).to_bytes(2, "little")
    else:
        prefix = ((size << 2) | 2).to_bytes(4, "little")
    return prefix + body


class TestScanReveals:
    def test_latest_per_hotkey_wins(self):
        commits = {HK1: [(100, "garbage"), (250, _valid_reveal(HK1, SHB))]}
        entries = scan_reveals(commits)
        assert len(entries) == 1
        assert entries[0] == ChallengerEntry(
            hotkey=HK1, block=250, model_repo=f"user/leoma-m-{HK1}", model_digest=SHB
        )

    def test_latest_invalid_is_not_backfilled(self):
        # Latest reveal is malformed; we do NOT fall back to an older valid one.
        commits = {HK1: [(100, _valid_reveal(HK1)), (250, "garbage")]}
        assert scan_reveals(commits) == []

    def test_author_mismatch_dropped(self):
        # Payload claims HK1 authored it, but the chain signer key is HK2.
        commits = {HK2: [(300, _reveal(f"user/leoma-m-{HK1}", SHA, HK1))]}
        assert scan_reveals(commits) == []

    def test_legacy_json_skipped(self):
        legacy = '{"model_name": "user/leoma-old", "model_revision": "abc", "endpoint_id": "x"}'
        assert scan_reveals({HK1: [(120, legacy)]}) == []

    def test_bad_repo_name_dropped(self):
        # repo does not end with the author's hotkey
        commits = {HK1: [(90, _reveal("user/leoma-wrongsuffix", SHA, HK1))]}
        assert scan_reveals(commits) == []

        # repo does not start with "leoma"
        commits2 = {HK1: [(90, _reveal(f"user/notleoma-{HK1}", SHA, HK1))]}
        assert scan_reveals(commits2) == []

    def test_blacklist_excluded(self):
        commits = {HK1: [(250, _valid_reveal(HK1))]}
        assert scan_reveals(commits, blacklist={HK1}) == []
        assert len(scan_reveals(commits, blacklist={HK2})) == 1

    def test_empty_and_none(self):
        assert scan_reveals(None) == []
        assert scan_reveals({}) == []
        assert scan_reveals({HK1: []}) == []

    def test_sorted_by_block_then_hotkey(self):
        commits = {
            HK2: [(300, _valid_reveal(HK2))],
            HK1: [(100, _valid_reveal(HK1))],
            HK3: [(100, _valid_reveal(HK3))],
        }
        entries = scan_reveals(commits)
        assert [(e.block, e.hotkey) for e in entries] == [
            (100, HK1),
            (100, HK3),
            (300, HK2),
        ]

    def test_multiple_valid_miners(self):
        commits = {
            HK1: [(100, _valid_reveal(HK1, SHA))],
            HK2: [(200, _valid_reveal(HK2, SHB))],
        }
        entries = scan_reveals(commits)
        assert {e.hotkey for e in entries} == {HK1, HK2}
        assert {e.model_digest for e in entries} == {SHA, SHB}


class TestCommitmentWireCompatibility:
    def test_decodes_hex_sdk_representation(self):
        payload = _valid_reveal(HK1)
        assert decode_scale_revealed_message("0x" + _scale_bytes(payload).hex()) == payload

    def test_decodes_latin1_async_substrate_representation(self):
        payload = _valid_reveal(HK1)
        wire = _scale_bytes(payload).decode("latin-1")
        assert decode_scale_revealed_message(wire) == payload

    def test_rejects_truncated_or_non_utf8_payloads(self):
        with pytest.raises(ValueError, match="length mismatch"):
            decode_scale_revealed_message(bytes([12]) + b"ab")
        with pytest.raises(ValueError, match="UTF-8"):
            decode_scale_revealed_message(bytes([4]) + b"\xff")
