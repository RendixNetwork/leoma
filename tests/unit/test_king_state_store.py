"""
Unit tests for the persisted king state.

The headline cases here pin the highest-severity bug in the subnet: a transient
bucket failure used to be indistinguishable from an empty bucket, so
`KingState.load` returned BLANK state — re-seeding genesis, re-dueling every past
challenger, wiping history — and the next flush overwrote the good bucket state
with the blank one.
"""

import json

import pytest

from tests.unit.conftest import FakeMinio

from leoma.app.validator.state_store import (
    HISTORY_LIMIT,
    KEY_HISTORY,
    KEY_KING,
    KEY_KING_CHAIN,
    KEY_SEEN,
    KEY_STATE,
    KEY_VALIDATOR_STATE,
    SCHEMA_VERSION,
    JsonBucketStore,
    KingState,
    StateInconsistent,
    StoreCorrupt,
    StoreUnavailable,
)

BUCKET = "own-bucket"


def _store(**kwargs) -> JsonBucketStore:
    # backoff=0 keeps retry tests instant
    return JsonBucketStore(FakeMinio(**kwargs), BUCKET, backoff=0)


class TestJsonBucketStore:
    async def test_put_get_roundtrip(self):
        s = _store()
        await s.put("k.json", {"a": 1, "b": [2, 3]})
        assert await s.get("k.json") == {"a": 1, "b": [2, 3]}

    async def test_missing_key_returns_none(self):
        assert await _store().get("absent.json") is None

    # ── the headline: a transport error is NOT a miss ──────────────────────
    async def test_transport_error_raises_not_none(self):
        """An outage must RAISE. Returning None here is what blanked the state."""
        s = _store(fail_get={"k.json": "InternalError"})
        with pytest.raises(StoreUnavailable):
            await s.get("k.json")

    async def test_corrupt_json_raises_store_corrupt(self):
        s = _store()
        s.client.seed_raw(BUCKET, "k.json", b"{not json")
        with pytest.raises(StoreCorrupt):
            await s.get("k.json")

    async def test_get_retries_then_succeeds(self):
        s = _store(flaky_get={"k.json": 2})  # fail twice, succeed on the 3rd
        await s.put("k.json", {"ok": True})
        assert await s.get("k.json") == {"ok": True}
        assert s.client.get_calls.count("k.json") == 3

    async def test_get_gives_up_after_retries(self):
        s = _store(flaky_get={"k.json": 99})
        with pytest.raises(StoreUnavailable):
            await s.get("k.json")

    async def test_put_error_raises(self):
        s = _store(fail_put={"k.json": "AccessDenied"})
        with pytest.raises(StoreUnavailable):
            await s.put("k.json", {"a": 1})


class TestLoad:
    async def test_empty_bucket_is_a_true_fresh_start(self):
        state = await KingState.load(_store())
        assert state.king == {}
        assert state.king_chain == []
        assert state.seen_hotkeys == set()

    async def test_load_propagates_store_error_and_writes_nothing(self):
        """The catastrophic path: outage -> must NOT return blank state."""
        s = _store(fail_get={KEY_STATE: "InternalError"})
        with pytest.raises(StoreUnavailable):
            await KingState.load(s)
        assert s.client.put_calls == []  # nothing was overwritten

    async def test_load_refuses_partial_state(self):
        """seen/history present but the king is gone => the bucket is damaged."""
        s = _store()
        await s.put(KEY_SEEN, {"hotkeys": ["A|d1"]})
        await s.put(KEY_HISTORY, {"history": [{"hotkey": "A"}]})
        with pytest.raises(StateInconsistent):
            await KingState.load(s)

    async def test_canonical_roundtrip(self):
        s = _store()
        st = KingState()
        st.king = {"hotkey": "A", "model_repo": "u/leoma-A", "reign_number": 3}
        st.king_chain = [{"hotkey": "B"}, {"hotkey": "C"}]
        st.last_weight_block = 4321
        st.last_winner_hotkey = "A"
        st.counter = 9
        st.stats = {"accepted": 2, "rejected": 1, "failed": 0, "transient_errors": 0}
        st.mark_seen("A|sha256:aaa")
        await st.flush(s)

        loaded = await KingState.load(s)
        assert loaded.king == st.king
        assert [e["hotkey"] for e in loaded.king_chain] == ["B", "C"]
        assert loaded.last_weight_block == 4321
        assert loaded.counter == 9
        assert loaded.seen_hotkeys == {"A|sha256:aaa"}

    async def test_migrates_from_legacy_five_keys(self):
        s = _store()
        await s.put(KEY_KING, {"hotkey": "A", "model_repo": "u/leoma-A"})
        await s.put(KEY_KING_CHAIN, {"chain": [{"hotkey": "B"}]})
        await s.put(KEY_VALIDATOR_STATE, {"last_weight_block": 7, "counter": 2,
                                          "stats": {"accepted": 1, "rejected": 0, "failed": 0}})
        await s.put(KEY_SEEN, {"hotkeys": ["A|d1"]})
        await s.put(KEY_HISTORY, {"history": [{"hotkey": "A"}]})

        loaded = await KingState.load(s)
        assert loaded.king["hotkey"] == "A"
        assert [e["hotkey"] for e in loaded.king_chain] == ["B"]
        assert loaded.last_weight_block == 7
        assert loaded.seen_hotkeys == {"A|d1"}
        assert loaded.stats["accepted"] == 1
        # new v2 fields default safely on an old bucket
        assert loaded.attempts == {}
        assert loaded.inflight == []


class TestFlush:
    async def test_flush_writes_one_canonical_object(self):
        s = _store()
        st = KingState(king={"hotkey": "A"})
        st.touch()
        await st.flush(s, mirror=False)
        assert s.client.put_calls == [KEY_STATE]  # ONE atomic PUT

        doc = json.loads(s.client.blobs[(BUCKET, KEY_STATE)])
        assert doc["schema_version"] == SCHEMA_VERSION

    async def test_flush_mirrors_legacy_keys(self):
        s = _store()
        st = KingState(king={"hotkey": "A"})
        st.touch()
        await st.flush(s)
        for key in (KEY_STATE, KEY_KING, KEY_KING_CHAIN, KEY_VALIDATOR_STATE, KEY_SEEN, KEY_HISTORY):
            assert (BUCKET, key) in s.client.blobs

    async def test_flush_tolerates_legacy_mirror_failure(self):
        """A mirror failure must not lose a crown — canonical state is durable."""
        s = _store(fail_put={KEY_KING: "InternalError"})
        st = KingState(king={"hotkey": "A"})
        st.touch()
        await st.flush(s)  # must not raise
        assert (BUCKET, KEY_STATE) in s.client.blobs

    async def test_flush_propagates_canonical_failure(self):
        s = _store(fail_put={KEY_STATE: "InternalError"})
        st = KingState(king={"hotkey": "A"})
        st.touch()
        with pytest.raises(StoreUnavailable):
            await st.flush(s)

    async def test_clean_state_is_not_rewritten(self):
        s = _store()
        st = KingState(king={"hotkey": "A"})  # not dirty
        await st.flush(s)
        assert s.client.put_calls == []


class TestHelpers:
    def test_mark_seen_and_record_duel_set_dirty(self):
        st = KingState()
        assert st._dirty is False
        st.mark_seen("A|d")
        assert st._dirty is True

    def test_history_newest_first_and_bounded(self):
        st = KingState()
        for i in range(HISTORY_LIMIT + 25):
            st.record_duel({"hotkey": f"h{i}"})
        assert len(st.history) == HISTORY_LIMIT
        assert st.history[0]["hotkey"] == f"h{HISTORY_LIMIT + 24}"

    async def test_history_roundtrip_newest_first(self):
        s = _store()
        st = KingState()
        st.record_duel({"hotkey": "A"})
        st.record_duel({"hotkey": "B"})
        await st.flush(s)
        loaded = await KingState.load(s)
        assert [h["hotkey"] for h in loaded.history] == ["B", "A"]

    async def test_next_eval_id_increments_and_persists(self):
        s = _store()
        st = KingState()
        assert st.next_eval_id() == "eval-0001"
        assert st.next_eval_id() == "eval-0002"
        await st.flush(s)
        assert (await KingState.load(s)).counter == 2


class TestDirtyFlagCannotSwallowACrown:
    """The _dirty optimisation must never make a real state change a no-op."""

    async def test_crown_like_mutation_is_persisted(self):
        s = _store()
        st = KingState()
        # simulate the crown path: mutate king + stats, then flush
        st.king = {"hotkey": "B", "model_repo": "u/leoma-B", "model_digest": "sha256:b"}
        st.stats["accepted"] = 1
        st.touch()                      # <- what main.py must do at every mutation site
        await st.flush(s, mirror=False)

        loaded = await KingState.load(s)
        assert loaded.king["hotkey"] == "B"
        assert loaded.stats["accepted"] == 1

    async def test_force_flush_ignores_dirty(self):
        s = _store()
        st = KingState(king={"hotkey": "A"})   # deliberately not dirty
        await st.flush(s, mirror=False, force=True)
        assert (await KingState.load(s)).king["hotkey"] == "A"
