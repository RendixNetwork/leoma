"""
Unit tests for weight setting.

The headline case pins a real bug: bittensor's `set_weights` returns
`(success, message)` and its rate-limit path returns `(False, "No attempt made.
Perhaps it is too soon to set weights!")` WITHOUT ever submitting an extrinsic.
The return value was discarded, so that no-op advanced `last_weight_block` as if
the weights had landed — blocking any retry for a full WEIGHT_INTERVAL (~1h) and
misreporting the persisted state.
"""

import pytest

from leoma.app.validator import king as K
from leoma.app.validator import main as validator_main
from leoma.app.validator.main import (
    _is_rate_limited,
    _unpack_set_weights,
    maybe_set_weights,
)
from leoma.app.validator.state_store import JsonBucketStore, KingState

from tests.unit.conftest import FakeMinio

# The literal bittensor 9.12.2 rate-limit sentinel.
RATE_LIMIT_MSG = "No attempt made. Perhaps it is too soon to set weights!"


def _store() -> JsonBucketStore:
    return JsonBucketStore(FakeMinio(), "own", backoff=0)


class _Wallet:
    class hotkey:  # noqa: N801
        ss58_address = "5KING"


class _FakeSubtensor:
    """Chain stub. `result` is whatever set_weights should return."""

    def __init__(self, result, block: int = 5000, too_soon: bool = False):
        self.result = result
        self.block = block
        self.too_soon = too_soon
        self.calls: list[dict] = []

    async def get_current_block(self):
        return self.block

    async def set_weights(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    # the "chain is the weight clock" probe
    async def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        return 3

    async def blocks_since_last_update(self, netuid, uid):
        return 1 if self.too_soon else 10_000

    async def weights_rate_limit(self, netuid):
        return 100


def _state_with_king() -> KingState:
    st = KingState()
    st.king = {"hotkey": "5A", "model_repo": "u/leoma-A", "model_digest": "sha256:a"}
    return st


class TestUnpack:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ((True, "ok"), (True, "ok")),
            ((False, "boom"), (False, "boom")),
            (True, (True, "")),
            (False, (False, "")),
        ],
    )
    def test_shapes(self, raw, expected):
        assert _unpack_set_weights(raw) == expected

    def test_extrinsic_response_like(self):
        class Resp:
            success = True
            message = "included"

        assert _unpack_set_weights(Resp()) == (True, "included")

    @pytest.mark.parametrize(
        "msg,expected",
        [
            (RATE_LIMIT_MSG, True),
            ("Too soon to set weights", True),
            ("SettingWeightsTooFast", True),
            ("Subtensor returned an error", False),
            ("", False),
        ],
    )
    def test_is_rate_limited(self, msg, expected):
        assert _is_rate_limited(msg) is expected


class TestMaybeSetWeights:
    async def test_rehearsal_suppresses_even_forced_extrinsic(self):
        sub, st, store = _FakeSubtensor((True, "ok")), _state_with_king(), _store()
        token = validator_main._WEIGHT_WRITES_ALLOWED.set(False)
        try:
            ok = await maybe_set_weights(
                sub, _Wallet(), st, {"5A": 7}, store, force=True
            )
        finally:
            validator_main._WEIGHT_WRITES_ALLOWED.reset(token)

        assert ok is False
        assert sub.calls == []
        assert st.last_weight_block == 0
        assert st.weight_failures == 0

    async def test_rehearsal_context_is_restored_after_bad_bounds(self):
        with pytest.raises(ValueError, match="max_ticks"):
            await validator_main.main(
                rehearsal=True, max_ticks=None, state_bucket="rehearsal-state"
            )

        assert validator_main._WEIGHT_WRITES_ALLOWED.get() is True

    async def test_tick_persists_dirty_state_when_weight_write_is_a_noop(self, monkeypatch):
        state, store = KingState(), _store()

        class Subtensor:
            async def get_current_block(self):
                return 5000

        async def empty_async(*args, **kwargs):
            return []

        async def no_weight(*args, **kwargs):
            return False

        async def no_publish(*args, **kwargs):
            return None

        def seed(state, block):
            state.king = {"hotkey": "", "model_repo": "leoma/base", "model_digest": "sha256:a"}
            state.touch()

        monkeypatch.setattr(validator_main, "refresh_uid_map", empty_async)
        monkeypatch.setattr(validator_main, "ensure_genesis_king", seed)
        monkeypatch.setattr(validator_main, "_load_revealed_commitments", empty_async)
        monkeypatch.setattr(validator_main, "scan_reveals", lambda *args, **kwargs: [])
        monkeypatch.setattr(validator_main, "process_challengers", no_publish)
        monkeypatch.setattr(validator_main, "maybe_set_weights", no_weight)
        monkeypatch.setattr(validator_main, "_publish_dashboard", no_publish)
        monkeypatch.setattr(validator_main, "_build_queue", lambda *args, **kwargs: [])

        await validator_main.tick(Subtensor(), _Wallet(), state, store)

        restored = await KingState.load(store)
        assert restored.king["model_repo"] == "leoma/base"

    async def test_success_advances_last_weight_block(self):
        sub, st, store = _FakeSubtensor((True, "ok")), _state_with_king(), _store()
        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)
        assert ok is True
        assert st.last_weight_block == 5000
        assert st.weight_failures == 0
        assert sub.calls[0]["uids"] == [7]

    # ── the headline ──────────────────────────────────────────────────────
    async def test_rate_limited_no_op_does_not_advance(self):
        """A no-op must NOT look like a landed weight-set."""
        sub, st, store = _FakeSubtensor((False, RATE_LIMIT_MSG)), _state_with_king(), _store()
        st.last_weight_block = 0

        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)

        assert ok is False
        assert st.last_weight_block == 0      # unchanged -> retried next tick
        assert st.weight_failures == 0        # a no-op is not a failure
        assert st.next_weight_block == 0      # and does not trigger backoff

    async def test_sdk_10_blank_no_op_uses_the_chain_weight_clock(self):
        """10.5 returns ``ExtrinsicResponse(False)`` with no message when its
        internal clock says too soon. The public chain clock is authoritative."""
        sub, st, store = _FakeSubtensor(False, too_soon=True), _state_with_king(), _store()

        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)

        assert ok is False
        assert st.last_weight_block == 0
        assert st.weight_failures == 0
        assert st.next_weight_block == 0

    async def test_blank_failure_is_genuine_when_the_chain_is_not_rate_limited(self):
        sub, st, store = _FakeSubtensor(False, too_soon=False), _state_with_king(), _store()

        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)

        assert ok is False
        assert st.weight_failures == 1
        assert st.next_weight_block > sub.block

    async def test_genuine_failure_does_not_advance_and_backs_off(self):
        sub, st, store = _FakeSubtensor((False, "Subtensor error")), _state_with_king(), _store()
        st.last_weight_block = 0

        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)

        assert ok is False
        assert st.last_weight_block == 0
        assert st.weight_failures == 1
        assert st.next_weight_block > 5000    # retried soon, not after WEIGHT_INTERVAL

    async def test_exception_does_not_advance(self):
        sub = _FakeSubtensor(RuntimeError("connection reset"))
        st, store = _state_with_king(), _store()
        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)
        assert ok is False
        assert st.last_weight_block == 0
        assert st.weight_failures == 1

    async def test_success_clears_backoff(self):
        st, store = _state_with_king(), _store()
        st.weight_failures, st.next_weight_block = 3, 4000
        sub = _FakeSubtensor((True, "ok"))
        assert await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True) is True
        assert st.weight_failures == 0
        assert st.next_weight_block == 0

    async def test_chain_is_the_weight_clock(self):
        """If the CHAIN says it's too soon, don't even attempt the extrinsic."""
        sub = _FakeSubtensor((True, "ok"), too_soon=True)
        st, store = _state_with_king(), _store()
        st.last_weight_block = 0  # our own clock says we're due

        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store)  # not forced

        assert ok is False
        assert sub.calls == []          # never submitted
        assert st.last_weight_block == 0

    async def test_backoff_blocks_a_retry_until_next_weight_block(self):
        sub = _FakeSubtensor((True, "ok"), block=100)
        st, store = _state_with_king(), _store()
        st.next_weight_block = 500

        assert await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store) is False
        assert sub.calls == []

    async def test_burns_to_uid0_when_no_king(self):
        sub, store = _FakeSubtensor((True, "ok")), _store()
        st = KingState()  # no king
        ok = await maybe_set_weights(sub, _Wallet(), st, {"5A": 7}, store, force=True)
        assert ok is True
        assert sub.calls[0]["uids"] == [K.BURN_UID]
        assert sub.calls[0]["weights"] == [1.0]
