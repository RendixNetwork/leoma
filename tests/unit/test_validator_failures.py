"""
Unit tests for duel failure handling.

The headline test here — `test_one_failing_challenger_does_not_block_the_rest` —
pins the worst practical bug in the subnet: `process_challengers` used to
`return` on any duel error, so a single challenger whose repo 404s (or whose
weights crashed the pipeline) permanently blocked EVERY later challenger. That
was a free griefing vector.
"""

import pytest

from leoma.app.validator import main as vmain
from leoma.app.validator.failures import (
    DuelFailure,
    ErrorClass,
    EvalBusy,
    EvalJobFailed,
    classify,
    classify_remote,
)
from leoma.app.validator.reveal_scan import ChallengerEntry
from leoma.app.validator.state_store import (
    BACKOFF_BLOCKS,
    MAX_DUEL_ATTEMPTS,
    JsonBucketStore,
    KingState,
)

from tests.unit.conftest import FakeEvalBox, FakeMinio, make_verdict

BUCKET = "own"
KING_DIGEST = "sha256:" + "k" * 64


def _store() -> JsonBucketStore:
    return JsonBucketStore(FakeMinio(), BUCKET, backoff=0)


def _entry(name: str, block: int = 100) -> ChallengerEntry:
    return ChallengerEntry(
        hotkey=f"5{name}", block=block,
        model_repo=f"u/leoma-{name}", model_digest="sha256:" + (name[0] * 64)[:64],
    )


def _state_with_king() -> KingState:
    st = KingState()
    st.king = {"hotkey": "5KING", "model_repo": "u/leoma-king", "model_digest": KING_DIGEST,
               "reign_number": 1}
    return st


class _FakeSubtensor:
    async def get_block_hash(self, block):
        return "0x" + f"{block:064x}"

    async def get_current_block(self):
        return 1000


class TestClassify:
    @pytest.mark.parametrize(
        "message,expected_kind,expected_reason",
        [
            # PERMANENT — a property of the artifact; repo@digest is immutable.
            ("RepositoryNotFoundError: repo missing", ErrorClass.PERMANENT, "model_not_found"),
            ("Revision does not exist", ErrorClass.PERMANENT, "model_not_found"),
            ("does not appear to have a file named model_index.json", ErrorClass.PERMANENT, "model_invalid"),
            ("size mismatch for transformer.weight", ErrorClass.PERMANENT, "model_invalid"),
            ("HeaderTooLarge", ErrorClass.PERMANENT, "model_invalid"),
            # TRANSIENT — a property of the environment.
            ("CUDA out of memory", ErrorClass.TRANSIENT, "oom"),
            ("an illegal memory access was encountered", ErrorClass.TRANSIENT, "cuda_fatal"),
            ("No space left on device", ErrorClass.TRANSIENT, "disk_full"),
            ("no clips to duel on", ErrorClass.TRANSIENT, "corpus_unavailable"),
            # Unknown -> TRANSIENT (fail open into retry, never into punishment).
            ("some brand new error nobody has seen", ErrorClass.TRANSIENT, "eval_error"),
        ],
    )
    def test_taxonomy(self, message, expected_kind, expected_reason):
        f = classify(RuntimeError(message))
        assert f.kind is expected_kind
        assert f.reason == expected_reason

    def test_auth_error_is_transient_not_permanent(self):
        """A validator's own token misconfig must not quarantine every miner."""
        f = classify(RuntimeError("HippiusHubAuthError: missing token"))
        assert f.kind is ErrorClass.TRANSIENT
        assert f.reason == "hub_auth"

    def test_busy_is_its_own_class(self):
        assert classify(EvalBusy("busy")).kind is ErrorClass.BUSY

    def test_verdictless_stream_is_transient_not_busy(self):
        """The `None`-conflation fix: a broken stream != a busy server."""
        f = classify(EvalJobFailed("stream ended", reason="stream_no_terminal"))
        assert f.kind is ErrorClass.TRANSIENT
        assert f.reason == "stream_no_terminal"

    def test_classify_remote_prefers_the_structured_reason(self):
        f = classify_remote("something went wrong", reason="model_not_found")
        assert f.kind is ErrorClass.PERMANENT


class TestAttemptLedger:
    def test_transient_backs_off_then_quarantines_when_exhausted(self):
        st = KingState()
        fail = DuelFailure(ErrorClass.TRANSIENT, "oom", "CUDA out of memory")
        for i in range(1, MAX_DUEL_ATTEMPTS):
            row = st.record_failure("A|d", block=100, failure=fail)
            assert row["quarantined"] is False
            assert row["next_retry_block"] == 100 + BACKOFF_BLOCKS[i - 1]
        row = st.record_failure("A|d", block=100, failure=fail)
        assert row["quarantined"] is True
        assert row["quarantine_reason"] == "exhausted"

    def test_permanent_quarantines_after_two_sightings(self):
        st = KingState()
        fail = DuelFailure(ErrorClass.PERMANENT, "model_not_found", "404")
        assert st.record_failure("A|d", block=10, failure=fail)["quarantined"] is False
        assert st.record_failure("A|d", block=20, failure=fail)["quarantined"] is True

    def test_clear_attempts_on_success(self):
        st = KingState()
        st.record_failure("A|d", block=10, failure=DuelFailure(ErrorClass.TRANSIENT, "oom", "x"))
        st.clear_attempts("A|d")
        assert st.is_quarantined("A|d") is False
        assert st.next_retry_block("A|d") == 0

    def test_quarantine_is_artifact_scoped_not_a_person_ban(self):
        """A miner who fixes the model gets a NEW key and a clean slate."""
        st = KingState()
        fail = DuelFailure(ErrorClass.PERMANENT, "model_invalid", "bad")
        st.record_failure("A|bad", block=1, failure=fail)
        st.record_failure("A|bad", block=2, failure=fail)
        assert st.is_quarantined("A|bad") is True
        assert st.is_quarantined("A|fixed") is False

    def test_banned_hotkeys_after_n_distinct_quarantined_digests(self):
        st = KingState()
        fail = DuelFailure(ErrorClass.PERMANENT, "model_invalid", "bad")
        for d in ("d1", "d2", "d3"):
            st.record_failure(f"SPAM|{d}", block=1, failure=fail)
            st.record_failure(f"SPAM|{d}", block=2, failure=fail)
        assert "SPAM" in st.banned_hotkeys()
        assert "HONEST" not in st.banned_hotkeys()


class TestProcessChallengers:
    """The duel is now dispatched, not awaited: the validator posts the job, persists
    an in-flight slot, and collects the verdict on a LATER tick. So these drive ticks.
    """

    async def test_one_failing_challenger_does_not_block_the_rest(self, monkeypatch, duel_ready):
        """THE headline: `return` -> `continue`. A bad model must not wedge the queue."""
        def outcome(entry):
            if entry.hotkey == "5bad":
                return RuntimeError("RepositoryNotFoundError: repo missing")
            return {"status": "done", "verdict": make_verdict(duel_ready, accepted=False)}

        box = FakeEvalBox(monkeypatch, outcome, duel_ready)
        st, store = _state_with_king(), _store()
        entries = [_entry("bad", 100), _entry("good1", 101), _entry("good2", 102)]

        await box.drive(st, store, entries, block=200)

        # The two challengers BEHIND the failing one were still evaluated.
        assert box.dispatched == ["5good1", "5good2"]
        assert st.stats["rejected"] == 2
        # The bad one recorded a failure but is not yet quarantined (1 sighting).
        assert st.attempts["5bad|" + entries[0].model_digest]["attempts"] == 1

    async def test_busy_breaks_and_consumes_no_attempt(self, monkeypatch, duel_ready):
        """BUSY is a property of the SERVER: stop, but don't punish anyone."""
        box = FakeEvalBox(monkeypatch, lambda e: EvalBusy("busy"), duel_ready)
        st, store = _state_with_king(), _store()
        entries = [_entry("a", 100), _entry("b", 101)]

        await box.drive(st, store, entries, block=200)

        assert st.attempts == {}          # no attempt consumed
        assert st.seen_hotkeys == set()   # nobody marked seen
        assert st.inflight == []

    async def test_permanent_failure_quarantines_and_records_an_error_row(self, monkeypatch, duel_ready):
        box = FakeEvalBox(
            monkeypatch,
            lambda e: RuntimeError("does not appear to have a file named model_index.json"),
            duel_ready,
        )
        st, store = _state_with_king(), _store()
        e = _entry("bad")
        key = vmain._seen_key(e.hotkey, e.model_digest)

        for block in (200, 900):          # two sightings -> quarantine
            await box.drive(st, store, [e], block=block, ticks=1)

        assert st.is_quarantined(key) is True
        assert st.stats["failed"] == 1              # counted once, on quarantine
        assert key in st.seen_hotkeys               # never retried

        row = st.history[0]                         # the error row the frontend needs
        assert row["verdict"] == "error"
        assert row["error_reason"] == "model_invalid"
        assert row["error"]

    async def test_backoff_defers_until_the_retry_block(self, monkeypatch, duel_ready):
        attempted: list[str] = []

        def outcome(entry):
            attempted.append(entry.hotkey)
            return RuntimeError("CUDA out of memory")

        box = FakeEvalBox(monkeypatch, outcome, duel_ready)
        st, store = _state_with_king(), _store()
        e = _entry("a")

        await box.drive(st, store, [e], block=100, ticks=1)
        assert len(attempted) == 1

        # still inside the backoff window -> skipped entirely
        await box.drive(st, store, [e], block=101, ticks=1)
        assert len(attempted) == 1

        # past it -> retried
        key = vmain._seen_key(e.hotkey, e.model_digest)
        await box.drive(st, store, [e], block=st.next_retry_block(key), ticks=1)
        assert len(attempted) == 2

    async def test_quarantined_is_skipped_forever(self, monkeypatch, duel_ready):
        box = FakeEvalBox(
            monkeypatch,
            lambda e: AssertionError("must not dispatch a quarantined artifact"),
            duel_ready,
        )
        st, store = _state_with_king(), _store()
        e = _entry("a")
        key = vmain._seen_key(e.hotkey, e.model_digest)
        st.attempt_row(key)["quarantined"] = True

        await box.drive(st, store, [e], block=10_000)
        assert box.dispatched == []

    async def test_successful_duel_clears_the_failure_history(self, monkeypatch, duel_ready):
        box = FakeEvalBox(
            monkeypatch,
            lambda e: {"status": "done", "verdict": make_verdict(duel_ready, accepted=False)},
            duel_ready,
        )
        st, store = _state_with_king(), _store()
        e = _entry("a")
        key = vmain._seen_key(e.hotkey, e.model_digest)
        st.record_failure(key, block=1, failure=DuelFailure(ErrorClass.TRANSIENT, "oom", "x"))

        await box.drive(st, store, [e], block=10_000)
        assert key not in st.attempts

    async def test_no_king_no_seed_burns_and_crowns_nobody(self, monkeypatch, duel_ready):
        """The unopposed-crown path is gone: never crown an unevaluated model."""
        box = FakeEvalBox(
            monkeypatch, lambda e: AssertionError("must not dispatch without a king"), duel_ready
        )
        st, store = KingState(), _store()      # NO king, NO seed
        e = _entry("first")

        await box.drive(st, store, [e], block=100)

        assert st.king == {}                    # nobody crowned
        assert st.degraded == "no_seed_digest"
        assert st.stats["accepted"] == 0
        assert box.dispatched == []
        # and the burn path is what weight_targets will produce:
        from leoma.app.validator import king as K
        uids, weights, label = K.weight_targets(st.king, st.king_chain, {"5first": 1})
        assert uids == [K.BURN_UID] and weights == [1.0] and label.startswith("burn:")


class TestTheTickIsBounded:
    """A duel used to be awaited INLINE — a tick could take hours, during which the
    validator set no weights (the chain concludes it is dead) and published no
    dashboard. Now it dispatches, persists a slot, and returns in seconds."""

    async def test_dispatch_returns_immediately_with_the_duel_still_running(self, monkeypatch, duel_ready):
        box = FakeEvalBox(
            monkeypatch, lambda e: {"status": "running", "phase": "materialize"}, duel_ready
        )
        st, store = _state_with_king(), _store()
        e = _entry("a")

        await box.drive(st, store, [e], block=200, ticks=1)

        assert len(st.inflight) == 1
        assert st.inflight[0]["hotkey"] == "5a"
        assert st.inflight[0]["king_digest"] == KING_DIGEST
        assert st.seen_hotkeys == set()      # not settled yet — no verdict, no crown

    async def test_a_running_duel_is_not_dispatched_twice(self, monkeypatch, duel_ready):
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state_with_king(), _store()
        entries = [_entry("a", 100), _entry("b", 101)]

        await box.drive(st, store, entries, block=200, ticks=5)

        assert box.dispatched == ["5a"], "the GPU can only run one duel at a time"

    async def test_the_slot_survives_a_restart_and_is_settled_later(self, monkeypatch, duel_ready):
        """Restart-safe: a validator that restarts mid-duel RE-ATTACHES. Before, the
        eval box would finish a duel nobody would ever read, and 409 everyone else
        for the entire time it took."""
        outcomes = iter([
            {"status": "running", "phase": "duel"},
            {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)},
        ])
        box = FakeEvalBox(monkeypatch, lambda e: next(outcomes), duel_ready)
        st, store = _state_with_king(), _store()
        e = _entry("a")

        await box.drive(st, store, [e], block=200, ticks=1)
        slot = dict(st.inflight[0])
        await st.flush(store)

        # ... the validator restarts here. The slot comes back from the bucket.
        revived = await KingState.load(store)
        assert revived.inflight == [slot]

        box.jobs[slot["eval_id"]] = {"status": "done", "verdict": make_verdict(duel_ready, accepted=True)}
        await box.drive(revived, store, [e], block=300, ticks=1)

        assert revived.inflight == []
        assert revived.king["hotkey"] == "5a"      # the duel we left running still counted

    async def test_a_lost_job_is_transient_and_never_blamed_on_the_miner(self, monkeypatch, duel_ready):
        """The eval box restarted and forgot the job. That is ours, not the miner's."""
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state_with_king(), _store()
        e = _entry("a")
        await box.drive(st, store, [e], block=200, ticks=1)

        eval_id = st.inflight[0]["eval_id"]
        box.jobs[eval_id] = EvalJobFailed("gone", reason="eval_job_lost")

        await box.drive(st, store, [e], block=201, ticks=1)

        key = vmain._seen_key(e.hotkey, e.model_digest)
        assert st.inflight == []
        assert not st.is_quarantined(key)                 # retried, not punished
        assert st.attempts[key]["last_class"] == "transient"

    async def test_a_verdict_against_a_DEPOSED_king_is_discarded(self, monkeypatch, duel_ready):
        """The king changed while the duel was in flight. The verdict measured the
        challenger against a king that no longer reigns — acting on it would crown
        someone who never beat the incumbent."""
        box = FakeEvalBox(monkeypatch, lambda e: {"status": "running"}, duel_ready)
        st, store = _state_with_king(), _store()
        e = _entry("a")
        await box.drive(st, store, [e], block=200, ticks=1)

        # A different challenger takes the crown mid-duel.
        st.king = {"hotkey": "5NEW", "model_repo": "u/leoma-new",
                   "model_digest": "sha256:" + "n" * 64, "reign_number": 2}

        box.jobs[st.inflight[0]["eval_id"]] = {
            "status": "done", "verdict": make_verdict(duel_ready, accepted=True)
        }
        await box.drive(st, store, [e], block=201, ticks=1)

        assert st.king["hotkey"] == "5NEW"           # the stale verdict did NOT crown 5a
        key = vmain._seen_key(e.hotkey, e.model_digest)
        assert key not in st.seen_hotkeys            # and 5a will be re-dueled, fairly

    async def test_a_verdict_that_ran_a_DIFFERENT_exam_is_refused(self, monkeypatch, duel_ready):
        """verify_echo, at the crowning gate. A box that quietly used its own metric
        must never take the crown."""
        tampered = make_verdict(duel_ready, accepted=True)
        tampered["echo"]["duel"]["metric"] = "mse"

        box = FakeEvalBox(monkeypatch, lambda e: {"status": "done", "verdict": tampered}, duel_ready)
        st, store = _state_with_king(), _store()
        e = _entry("a")

        await box.drive(st, store, [e], block=200, ticks=2)

        assert st.king["hotkey"] == "5KING"          # not crowned
        key = vmain._seen_key(e.hotkey, e.model_digest)
        assert st.degraded == "consensus_echo_mismatch"
        assert key not in st.seen_hotkeys
