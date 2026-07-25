"""Compatibility coverage for Bittensor's bulk commitment decoder fallback."""

import leoma.app.validator.main as vmain

from leoma.app.validator.reveal_scan import scan_reveals
from leoma.infra.model_store import ModelRef, build_reveal_v4


HK1 = "5C7LM2i42XgL2oB4x3rcmB7KDiof4B92KZzUpg5miZ6DogjU"
HK2 = "5DJ76XJdWvU7PcmKmBjzoAKYC3i4YjhdR92uVYGA7FthyCv2"
DIGEST = "sha256:" + "a" * 64


def _payload(hotkey: str) -> str:
    return build_reveal_v4(
        ModelRef(f"user/leoma-m-{hotkey}", DIGEST),
        hotkey,
    )


def _wire(payload: str) -> str:
    body = payload.encode()
    size = len(body)
    prefix = ((size << 2) | 1).to_bytes(2, "little")
    return (prefix + body).decode("latin-1")


class _BrokenSdk:
    def __init__(self):
        self.query_args = None

    async def get_all_revealed_commitments(self, netuid, block):
        raise ValueError("SDK expected a hex string")

    async def query_map(self, **kwargs):
        self.query_args = kwargs

        async def rows():
            # The malformed latest item must suppress HK1's older valid reveal.
            yield HK1, [(_wire(_payload(HK1)), 100), ("not-scale", 200)]
            yield HK2, [(_wire(_payload(HK2)), 150)]

        return rows()


async def test_fallback_keeps_valid_rows_without_backfilling_malformed_latest():
    subtensor = _BrokenSdk()

    commits = await vmain._load_revealed_commitments(subtensor, block=1234)
    entries = scan_reveals(commits)

    assert [(entry.hotkey, entry.block) for entry in entries] == [(HK2, 150)]
    assert subtensor.query_args == {
        "module": "Commitments",
        "name": "RevealedCommitments",
        "params": [vmain.NETUID],
        "block": 1234,
    }
