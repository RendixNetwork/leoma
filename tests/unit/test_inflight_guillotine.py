"""The validator-side stuck-duel backstop.

The eval box has its own forward-progress watchdog, and normally it fires first. But
Leoma moved the wall-clock bound entirely onto the box, so if the box keeps reporting
"running" without ever tripping a phase budget — a disabled watchdog, a lying box, a
partition where poll succeeds but the box is wedged — one stuck duel would hold the
single in-flight slot forever and no other challenger would ever be dispatched.

This is the guillotine: abandon a duel that has been in flight far past any plausible
wall clock, and free the slot. It is deliberately generous (a real duel of two 14B
models runs hours), so it only fires on a genuinely hung box.
"""

import leoma.app.validator.main as vmain
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState

from tests.unit.conftest import FakeEvalBox, FakeMinio

KING_DIGEST = "sha256:" + "k" * 64


def _store():
    return JsonBucketStore(FakeMinio(), "own", backoff=0)


def _state():
    st = KingState()
    st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king",
               "model_digest": KING_DIGEST, "reign_number": 1}
    return st


def _entry(hotkey="5a", digest="sha256:" + "a" * 64, block=100):
    return ChallengerEntry(hotkey=hotkey, model_repo="u/leoma-a", model_digest=digest, block=block)


class TestGuillotine:
    async def test_a_running_duel_within_the_bound_is_left_alone(self, monkeypatch, duel_ready):
        monkeypatch.setattr(vmain, "MAX_INFLIGHT_BLOCKS", 1000)
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state(), _store()
        e = _entry()

        await box.drive(st, store, [e], block=200, ticks=1)   # dispatched at 200
        # Poll 500 blocks later — well inside the 1000-block bound.
        await vmain.settle_inflight(_Sub(), None, st, {}, store, block=700)

        assert st.inflight != [], "a duel within the bound was abandoned"
        assert box.cancelled == []

    async def test_a_duel_past_the_bound_is_abandoned_and_the_slot_freed(self, monkeypatch, duel_ready):
        monkeypatch.setattr(vmain, "MAX_INFLIGHT_BLOCKS", 1000)
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state(), _store()
        e = _entry()

        await box.drive(st, store, [e], block=200, ticks=1)   # dispatched at 200
        eval_id = st.inflight[0]["eval_id"]

        # Poll 1500 blocks later — past the 1000-block bound.
        free = await vmain.settle_inflight(_Sub(), None, st, {}, store, block=1700)

        assert free is True                       # the slot is free for the next dispatch
        assert st.inflight == []
        assert eval_id in box.cancelled           # we asked the box to stop burning GPU
        key = vmain._seen_key(e.hotkey, e.model_digest)
        assert st.attempts[key]["last_reason"] == "inflight_timeout"

    async def test_the_stuck_box_is_never_blamed_on_the_miner(self, monkeypatch, duel_ready):
        """A hung box is infrastructure, not the challenger. TRANSIENT + backoff, never
        a strike — so a one-off wedge retries, but a model that hangs EVERY time still
        exhausts its attempt budget and quarantines."""
        monkeypatch.setattr(vmain, "MAX_INFLIGHT_BLOCKS", 100)
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state(), _store()
        e = _entry()

        await box.drive(st, store, [e], block=200, ticks=1)
        key = vmain._seen_key(e.hotkey, e.model_digest)

        await vmain.settle_inflight(_Sub(), None, st, {}, store, block=1000)
        assert st.attempts[key]["last_class"] == "transient"
        assert st.duels.get(e.hotkey, {}).get("strikes", 0) == 0

    async def test_after_abandon_the_next_challenger_is_dispatched(self, monkeypatch, duel_ready):
        """The whole point: freeing the slot lets someone else run."""
        monkeypatch.setattr(vmain, "MAX_INFLIGHT_BLOCKS", 100)
        outcomes = {"5stuck": {"status": "running"}, "5next": {"status": "running"}}
        box = FakeEvalBox(monkeypatch, lambda e: outcomes[e.hotkey], duel_ready)
        st, store = _state(), _store()
        stuck = _entry("5stuck", "sha256:" + "a" * 64)
        nxt = _entry("5next", "sha256:" + "b" * 64)

        # Tick 1 at block 200: dispatch 5stuck.
        await vmain.process_challengers(_Sub(), None, st, {}, store, [stuck, nxt], 200)
        assert st.inflight[0]["hotkey"] == "5stuck"

        # Tick 2 far later: 5stuck is guillotined, then 5next dispatched in the same tick.
        await vmain.process_challengers(_Sub(), None, st, {}, store, [stuck, nxt], 2000)
        assert len(st.inflight) == 1 and st.inflight[0]["hotkey"] == "5next"

    async def test_a_slot_with_no_dispatched_block_does_not_crash(self, monkeypatch, duel_ready):
        """A slot persisted by an older build has no dispatched_block; treat it as
        just-dispatched (age 0) rather than tripping the guillotine or raising."""
        monkeypatch.setattr(vmain, "MAX_INFLIGHT_BLOCKS", 100)
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state(), _store()
        st.inflight = [{
            "eval_id": "eval-old", "hotkey": "5a", "model_repo": "u/leoma-a",
            "model_digest": "sha256:" + "a" * 64, "block": 100, "king_digest": KING_DIGEST,
            # no dispatched_block, no eval_server_url — an older build's persisted slot
        }]
        box.jobs["eval-old"] = {"status": "running"}

        free = await vmain.settle_inflight(_Sub(), None, st, {}, store, block=999999)
        assert free is False              # age defaults to 0 -> not stuck
        assert st.inflight != []


class _Sub:
    async def get_block_hash(self, block):
        return f"0x{block:064x}"

    async def get_current_block(self):
        return 1000

    async def blocks_since_last_update(self, netuid, uid):
        return 10_000

    async def weights_rate_limit(self, netuid):
        return 100

    async def set_weights(self, **kwargs):
        return True, "ok"

    async def metagraph(self, netuid):
        class M:
            hotkeys: list = []
        return M()
