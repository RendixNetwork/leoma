"""Dispatching to several independently-run eval-server processes.

One eval server means one duel at a time — the next bottleneck once the prescreen is
doing its job: a bad model is rejected in seconds, but a QUEUE of good challengers
still drains one multi-hour duel at a time. ``EVAL_SERVER_URLS`` lets the validator
fill several configured servers (e.g. one per GPU pair on an 8xH100 box) instead of
being pinned to exactly one.

This is deliberately built as **validator-side dispatch across independent server
processes**, not a rewrite of the eval server's single-flight lock — that keeps the
riskiest file in the system (the one whose own docstring says it is "defensive out of
proportion to its size") completely untouched. Each server still only ever runs one
duel; the validator just knows about more than one.

The dispatch model is intentionally simple: settle everything in flight, then dispatch
AT MOST ONE new duel per tick — to whichever configured URL is free. With a ~60s tick
against multi-hour duels, a handful of ticks is enough to fill every server, so this
buys nearly all the throughput of a "fill every free slot in one tick" scheduler with
far less code to get right.
"""

import leoma.app.validator.main as vmain
from leoma.app.validator.failures import EvalJobFailed
from leoma.app.validator.main import _parse_eval_server_urls
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import JsonBucketStore, KingState

from tests.unit.conftest import FakeMinio, make_verdict

KING_DIGEST = "sha256:" + "k" * 64


def _store():
    return JsonBucketStore(FakeMinio(), "own", backoff=0)


def _state():
    st = KingState()
    st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king",
               "model_digest": KING_DIGEST, "reign_number": 1}
    return st


def _entry(hotkey, digest, block=100):
    return ChallengerEntry(hotkey=hotkey, model_repo=f"u/leoma-{hotkey}",
                           model_digest=digest, block=block)


class _FakeSubtensor:
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


class FakeEvalFleet:
    """Per-URL-aware fake: each configured server has its OWN outcome function.

    Unlike the shared ``FakeEvalBox`` (one global outcome for one implicit server),
    this is exactly what testing "server A is stale, server B is fine" needs: distinct,
    independently-controllable behavior per URL.
    """

    def __init__(self, monkeypatch, urls, outcomes):
        """``outcomes``: {url: (entry) -> dict | BaseException}."""
        monkeypatch.setattr(vmain, "EVAL_SERVER_URLS", list(urls))
        self._outcomes = outcomes
        self.jobs: dict[str, dict] = {}
        self.job_url: dict[str, str] = {}
        self.dispatched: list[tuple[str, str]] = []   # (hotkey, url), in order
        self.polled: list[tuple[str, str]] = []
        self.cancelled: list[tuple[str, str]] = []
        self._counter = 0
        monkeypatch.setattr(vmain, "start_duel", self.start_duel)
        monkeypatch.setattr(vmain, "poll_duel", self.poll_duel)
        monkeypatch.setattr(vmain, "cancel_duel", self.cancel_duel)

    async def start_duel(self, entry, king, block_hash, *, eval_server_url):
        outcome = self._outcomes[eval_server_url](entry)
        if isinstance(outcome, BaseException):
            raise outcome
        self._counter += 1
        eval_id = f"eval-{eval_server_url}-{self._counter:04d}"
        self.jobs[eval_id] = outcome
        self.job_url[eval_id] = eval_server_url
        self.dispatched.append((entry.hotkey, eval_server_url))
        return eval_id

    async def poll_duel(self, eval_id, *, eval_server_url):
        self.polled.append((eval_id, eval_server_url))
        outcome = self.jobs[eval_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def cancel_duel(self, eval_id, *, eval_server_url):
        self.cancelled.append((eval_id, eval_server_url))

    async def tick(self, state, store, entries, *, block):
        await vmain.process_challengers(_FakeSubtensor(), object(), state, {}, store, entries, block)


class TestUrlParsing:
    """Pure, so it needs neither an env var nor a module reload to test."""

    def test_unset_falls_back_to_the_single_url(self):
        assert _parse_eval_server_urls("", "http://localhost:9000") == ["http://localhost:9000"]

    def test_a_comma_list_is_split_and_trimmed(self):
        assert _parse_eval_server_urls(
            " http://a:9000 , http://b:9000,http://c:9000 ", "fallback"
        ) == ["http://a:9000", "http://b:9000", "http://c:9000"]

    def test_a_single_url_with_no_comma_is_one_item(self):
        assert _parse_eval_server_urls("http://only:9000", "fallback") == ["http://only:9000"]

    def test_blank_entries_between_commas_are_dropped(self):
        """An empty-string URL could never resolve, so it would occupy a permanent
        'free' slot that dispatch keeps trying and failing against."""
        assert _parse_eval_server_urls("http://a:9000,,http://b:9000,", "fallback") == [
            "http://a:9000", "http://b:9000",
        ]

    def test_whitespace_only_falls_back(self):
        assert _parse_eval_server_urls("   ", "http://localhost:9000") == ["http://localhost:9000"]


class TestFillingMultipleServers:
    async def test_two_free_servers_get_two_different_challengers_over_two_ticks(self, monkeypatch, duel_ready):
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1, e2 = _entry("5one", "sha256:" + "1" * 64), _entry("5two", "sha256:" + "2" * 64)

        await fleet.tick(st, store, [e1, e2], block=200)   # settle (nothing yet) + dispatch #1
        await fleet.tick(st, store, [e1, e2], block=200)   # settle (still running) + dispatch #2

        assert fleet.dispatched == [("5one", "http://a:9000"), ("5two", "http://b:9000")]
        assert {s["eval_server_url"] for s in st.inflight} == {"http://a:9000", "http://b:9000"}
        assert len(st.inflight) == 2

    async def test_the_same_challenger_is_never_dispatched_twice_while_in_flight(self, monkeypatch, duel_ready):
        """The bug this class exists to prevent: with a single server, settle_inflight
        returning 'busy' meant the dispatch loop was never even reached. With several
        servers, one busy slot no longer blocks the loop, so an entry already running
        somewhere must be skipped explicitly."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)

        for _ in range(4):
            await fleet.tick(st, store, [e1], block=200)

        assert fleet.dispatched == [("5one", "http://a:9000")]   # never sent to b as well
        assert len(st.inflight) == 1

    async def test_settling_one_slot_frees_its_server_for_the_next_challenger(self, monkeypatch, duel_ready):
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1, e2, e3 = (_entry("5one", "sha256:" + "1" * 64), _entry("5two", "sha256:" + "2" * 64),
                     _entry("5three", "sha256:" + "3" * 64))

        await fleet.tick(st, store, [e1, e2, e3], block=200)   # dispatch e1 -> a
        await fleet.tick(st, store, [e1, e2, e3], block=200)   # dispatch e2 -> b
        assert {s["hotkey"] for s in st.inflight} == {"5one", "5two"}

        # The eval box on `a` finishes: mutate its stored job, exactly as the real
        # box's own state would change between one poll and the next.
        a_slot = next(s for s in st.inflight if s["eval_server_url"] == "http://a:9000")
        fleet.jobs[a_slot["eval_id"]] = {"status": "done", "verdict": make_verdict(duel_ready, accepted=False)}

        # This tick settles `a` (freeing it) and dispatches e3 there in the same call.
        await fleet.tick(st, store, [e1, e2, e3], block=200)

        assert ("5three", "http://a:9000") in fleet.dispatched
        assert {s["hotkey"] for s in st.inflight} == {"5two", "5three"}


class TestPerServerLocalFault:
    async def test_a_stale_server_is_skipped_in_favor_of_a_healthy_one(self, monkeypatch, duel_ready):
        """Server A pins a different consensus surface than this validator (an
        operator redeployed B but forgot A). B still works, so a challenger must
        still get dueled — not blocked by A's misconfiguration."""
        def stale(entry):
            return EvalJobFailed("box pins a different surface", reason="consensus_mismatch")

        fleet = FakeEvalFleet(
            monkeypatch, ["http://stale:9000", "http://healthy:9000"],
            {"http://stale:9000": stale, "http://healthy:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)

        await fleet.tick(st, store, [e1], block=200)

        assert fleet.dispatched == [("5one", "http://healthy:9000")]
        assert st.degraded is None                 # NOT globally degraded — b covers for a
        assert st.attempts == {}                   # and the challenger was not blamed either

    async def test_every_server_locally_faulty_degrades_the_validator(self, monkeypatch, duel_ready):
        """When NO configured server has working capacity, that genuinely is a
        subnet-wide condition — burning is correct here, exactly as with one server."""
        def stale(entry):
            return EvalJobFailed("box pins a different surface", reason="consensus_mismatch")

        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": stale, "http://b:9000": stale},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)

        await fleet.tick(st, store, [e1], block=200)

        assert fleet.dispatched == []
        assert st.degraded == "consensus_mismatch"

    async def test_a_single_configured_server_still_degrades_immediately(self, monkeypatch, duel_ready):
        """The exact pre-existing single-server behavior, unchanged: with nothing to
        fail over to, a LOCAL fault degrades right away rather than trying a
        nonexistent 'other' server first."""
        def stale(entry):
            return EvalJobFailed("box pins a different surface", reason="consensus_mismatch")

        fleet = FakeEvalFleet(monkeypatch, ["http://only:9000"], {"http://only:9000": stale})
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)

        await fleet.tick(st, store, [e1], block=200)

        assert st.degraded == "consensus_mismatch"


class TestCrownDuringMultiSettle:
    async def test_a_stale_verdict_on_slot_B_is_discarded_after_slot_A_crowns_in_the_SAME_tick(
        self, monkeypatch, duel_ready,
    ):
        """Slots are settled SEQUENTIALLY against live state, not a frozen snapshot:
        if settling slot A crowns a new king, slot B's own "did the king change under
        me" check must see that change immediately, in the same tick — or a
        challenger that only tied the OLD king could be crowned a second time."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)},
             "http://b:9000": lambda e: {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)}},
        )
        st, store = _state(), _store()
        e1, e2 = _entry("5one", "sha256:" + "1" * 64), _entry("5two", "sha256:" + "2" * 64)

        # Dispatch both first, each outcome function stubbed to "running" for the
        # dispatch ticks so neither resolves until we're ready.
        fleet._outcomes["http://a:9000"] = lambda e: {"status": "running"}
        fleet._outcomes["http://b:9000"] = lambda e: {"status": "running"}
        await fleet.tick(st, store, [e1, e2], block=200)
        await fleet.tick(st, store, [e1, e2], block=200)
        assert len(st.inflight) == 2

        # Now make BOTH resolve as accepted verdicts against the SAME (original) king.
        a_slot = next(s for s in st.inflight if s["eval_server_url"] == "http://a:9000")
        b_slot = next(s for s in st.inflight if s["eval_server_url"] == "http://b:9000")
        fleet.jobs[a_slot["eval_id"]] = {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)}
        fleet.jobs[b_slot["eval_id"]] = {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)}

        await vmain.settle_inflight(_FakeSubtensor(), object(), st, {}, store, 200)

        # Exactly one of them took the crown; the other's stale verdict was discarded
        # (never marked seen, so it will be re-dueled fairly against the new king).
        assert st.king["hotkey"] in ("5one", "5two")
        winner, loser = (e1, e2) if st.king["hotkey"] == "5one" else (e2, e1)
        loser_key = vmain._seen_key(loser.hotkey, loser.model_digest)
        assert loser_key not in st.seen_hotkeys
        assert st.inflight == []                      # both slots resolved either way


class TestRateLimiterBypassAcrossServers:
    async def test_one_hotkey_cannot_occupy_two_servers_with_two_digests(self, monkeypatch, duel_ready):
        """RL.record_verdict only ever runs at SETTLE time (from _settle_one_slot), so
        with a single server this was never reachable: only one duel could ever be in
        flight for anyone. With several servers, a hotkey minting a fresh digest every
        tick could otherwise occupy every configured server before its first duel ever
        resolves -- the GPU-fleet monopolization the rate limiter exists to prevent."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)
        e2 = _entry("5one", "sha256:" + "2" * 64)   # same hotkey, different digest

        for _ in range(4):
            await fleet.tick(st, store, [e1, e2], block=200)

        assert len(st.inflight) == 1               # never both, despite two free servers
        assert fleet.dispatched == [("5one", "http://a:9000")]

    async def test_a_different_hotkey_still_fills_the_other_server(self, monkeypatch, duel_ready):
        """The per-hotkey cap must not spill over onto unrelated challengers."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)
        e2 = _entry("5one", "sha256:" + "2" * 64)
        e3 = _entry("5two", "sha256:" + "3" * 64)

        for _ in range(4):
            await fleet.tick(st, store, [e1, e2, e3], block=200)

        assert {s["hotkey"] for s in st.inflight} == {"5one", "5two"}
        assert len(st.inflight) == 2


class TestLegacySlotMissingEvalServerUrl:
    async def test_dispatch_time_busy_urls_tolerates_a_legacy_slot(self, monkeypatch, duel_ready):
        """A slot migrated from a pre-multi-server bucket (_normalize_inflight) has no
        eval_server_url key at all -- a legacy duel can still be in flight on the very
        restart that adds a second EVAL_SERVER_URLS entry. Computing busy_urls must not
        raise KeyError, and the legacy slot must still count as occupying the single
        implicit server (EVAL_SERVER_URL) it was actually dispatched to."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://localhost:9000", "http://b:9000"],
            {"http://localhost:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        st.inflight = [{
            "eval_id": "eval-legacy", "hotkey": "5legacy", "model_repo": "u/leoma-legacy",
            "model_digest": "sha256:" + "9" * 64, "dispatched_block": 199,
            # no "eval_server_url" key -- exactly what a pre-migration slot looks like
        }]
        fleet.jobs["eval-legacy"] = {"status": "running"}
        e1 = _entry("5one", "sha256:" + "1" * 64)

        await fleet.tick(st, store, [e1], block=200)   # must not raise KeyError

        assert fleet.dispatched == [("5one", "http://b:9000")]   # localhost:9000 correctly busy
        assert len(st.inflight) == 2


class TestSettleTimeLocalFaultScoping:
    async def test_a_local_fault_while_polling_does_not_degrade_when_another_server_is_healthy(
        self, monkeypatch, duel_ready,
    ):
        """A LOCAL fault surfacing while POLLING an already-dispatched duel (not at
        dispatch-time preflight) must be scoped exactly like the dispatch-time case:
        with a healthy alternative configured, this is not a subnet-wide condition."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000"],
            {"http://a:9000": lambda e: {"status": "running"},
             "http://b:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1, e2 = _entry("5one", "sha256:" + "1" * 64), _entry("5two", "sha256:" + "2" * 64)

        await fleet.tick(st, store, [e1, e2], block=200)   # dispatch e1 -> a
        await fleet.tick(st, store, [e1, e2], block=200)   # dispatch e2 -> b
        assert len(st.inflight) == 2

        a_slot = next(s for s in st.inflight if s["eval_server_url"] == "http://a:9000")
        fleet.jobs[a_slot["eval_id"]] = EvalJobFailed(
            "box pins a different surface", reason="consensus_mismatch",
        )

        await vmain.settle_inflight(_FakeSubtensor(), object(), st, {}, store, 200)

        assert st.degraded is None                  # b is still healthy -- no false alarm
        # A LOCAL fault mid-poll keeps its slot rather than discarding it (the duel may
        # still be fine once the box recovers) -- both slots remain in flight.
        assert {s["eval_server_url"] for s in st.inflight} == {"http://a:9000", "http://b:9000"}

    async def test_a_local_fault_while_polling_degrades_with_only_one_configured_server(
        self, monkeypatch, duel_ready,
    ):
        """The exact pre-existing single-server behavior, unchanged: with nothing to
        fail over to, a LOCAL fault discovered mid-poll still degrades the validator."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://only:9000"],
            {"http://only:9000": lambda e: {"status": "running"}},
        )
        st, store = _state(), _store()
        e1 = _entry("5one", "sha256:" + "1" * 64)

        await fleet.tick(st, store, [e1], block=200)   # dispatch e1
        assert len(st.inflight) == 1

        slot = st.inflight[0]
        fleet.jobs[slot["eval_id"]] = EvalJobFailed(
            "box pins a different surface", reason="consensus_mismatch",
        )

        await vmain.settle_inflight(_FakeSubtensor(), object(), st, {}, store, 200)

        assert st.degraded == "consensus_mismatch"


class TestBusyUrlsExcludeDispatch:
    async def test_busy_urls_are_computed_from_inflight_not_guessed(self, monkeypatch, duel_ready):
        """A slot's OWN url must never be offered again while it's still running —
        this is the thing that makes 'one duel per server' actually hold."""
        fleet = FakeEvalFleet(
            monkeypatch, ["http://a:9000", "http://b:9000", "http://c:9000"],
            {u: (lambda e: {"status": "running"}) for u in
             ("http://a:9000", "http://b:9000", "http://c:9000")},
        )
        st, store = _state(), _store()
        entries = [_entry(f"5e{i}", f"sha256:{i:064x}") for i in range(3)]

        for _ in range(4):
            await fleet.tick(st, store, entries, block=200)

        assert len(st.inflight) == 3
        assert {s["eval_server_url"] for s in st.inflight} == {
            "http://a:9000", "http://b:9000", "http://c:9000",
        }
