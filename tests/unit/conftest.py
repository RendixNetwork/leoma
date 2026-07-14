"""Shared unit-test fixtures.

The in-memory Minio stub lives here (it used to be local to
test_king_state_store.py) because several suites now need it — and because the
state-integrity tests need it to **inject faults**. That is the whole point of
that work: a transport error must no longer be indistinguishable from an empty
bucket.
"""

import pytest


class FakeS3Error(Exception):
    """Mirrors minio.error.S3Error's `.code` contract."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class FakeMinio:
    """In-memory stand-in for the Minio client's put/get object API.

    Fault injection:
      fail_get / fail_put: {key: error_code} — raise FakeS3Error(code) for that key.
                           Use code="NoSuchKey" to simulate a genuine miss.
      flaky_get:           {key: n} — fail the first n GETs for that key, then succeed.
    """

    def __init__(self, *, fail_get=None, fail_put=None, flaky_get=None):
        self.blobs: dict[tuple, bytes] = {}
        self.fail_get = dict(fail_get or {})
        self.fail_put = dict(fail_put or {})
        self.flaky_get = dict(flaky_get or {})
        self.get_calls: list[str] = []
        self.put_calls: list[str] = []

    # ---- minio API surface -------------------------------------------------
    def put_object(self, bucket, key, data, length, content_type=None):
        self.put_calls.append(key)
        if key in self.fail_put:
            raise FakeS3Error(self.fail_put[key])
        payload = data.read()
        assert len(payload) == length
        self.blobs[(bucket, key)] = payload

    def get_object(self, bucket, key):
        self.get_calls.append(key)

        remaining = self.flaky_get.get(key, 0)
        if remaining:
            self.flaky_get[key] = remaining - 1
            raise FakeS3Error("InternalError", "transient")

        if key in self.fail_get:
            raise FakeS3Error(self.fail_get[key])

        if (bucket, key) not in self.blobs:
            raise FakeS3Error("NoSuchKey")

        blob = self.blobs[(bucket, key)]

        class _Resp:
            def read(self_inner):
                return blob

            def close(self_inner):
                pass

            def release_conn(self_inner):
                pass

        return _Resp()

    # ---- test helpers ------------------------------------------------------
    def seed_raw(self, bucket: str, key: str, raw: bytes) -> None:
        """Put raw bytes directly (e.g. to seed corrupt JSON)."""
        self.blobs[(bucket, key)] = raw


@pytest.fixture
def fake_minio() -> FakeMinio:
    return FakeMinio()


# ---------------------------------------------------------------------------
# The pinned consensus surface
# ---------------------------------------------------------------------------

#: The digest a test corpus manifest pretends to have.
TEST_MANIFEST_DIGEST = "sha256:" + "c" * 64


def pinned_spec():
    """The shipped SPEC, but with the corpus pinned.

    The chain.toml in the repo ships with ``manifest_digest = ""`` — deliberately,
    because the corpus has not been published yet and a validator must refuse to
    duel on an unpinned corpus. Tests that exercise the duel path need a spec that
    is duel-ready, so they pin a fake digest here rather than weakening the shipped
    default (which is exactly the safety property we want to keep testing).
    """
    from leoma.infra.chain_config import SPEC

    return SPEC.model_copy(
        update={"corpus": SPEC.corpus.model_copy(update={"manifest_digest": TEST_MANIFEST_DIGEST})}
    )


@pytest.fixture
def duel_ready(monkeypatch):
    """Give the validator a pinned corpus so the duel path can run.

    The architecture prescreen is switched off here: it fetches real configs from the
    Hub, and these tests are about the dispatch/settle policy, not about the lock. The
    lock has its own tests (``test_arch_lock.py``), and one test below deliberately
    turns the prescreen back on to prove it is wired in.
    """
    import leoma.app.validator.main as vmain

    spec = pinned_spec()
    monkeypatch.setattr(vmain, "SPEC", spec)
    monkeypatch.setattr(vmain, "CONSENSUS_DIGEST", spec.digest())
    monkeypatch.setattr(vmain, "PRESCREEN_ENABLED", False)
    # The OCI copy check reaches the Hippius registry; off by default in these tests,
    # which are about dispatch/settle policy. It has its own tests in test_copy_check.py,
    # and one test in test_antiabuse.py turns it back on to prove it is wired.
    monkeypatch.setattr(vmain, "COPY_CHECK_ENABLED", False)
    return spec


def make_verdict(spec, *, accepted: bool, **extra) -> dict:
    """A verdict shaped like the real thing — including the consensus echo.

    Tests that skip the echo would sail past ``verify_echo``, which is precisely the
    guard standing between the subnet and a verdict produced under someone else's
    parameters. So every fake verdict carries one.
    """
    return {
        "accepted": accepted,
        "verdict": "challenger" if accepted else "king",
        "lcb": 0.01 if accepted else -0.01,
        "mu_hat": 0.02 if accepted else -0.02,
        "n_clips": spec.duel.n_clips,
        "echo": spec.model_dump(mode="json"),
        "audit": {"consensus_digest": spec.digest(), "corpus": {"clip_keys_digest": "sha256:x"}},
        "verdict_digest": "sha256:v",
        "produced_at": "2026-07-13T00:00:00+00:00",
        **extra,
    }


class FakeEvalBox:
    """In-memory eval server: dispatch returns an id, poll returns the outcome.

    The validator no longer blocks on a duel — it dispatches, persists a slot, and
    collects the verdict on a later tick. So a test has to drive *ticks*, not calls.
    :meth:`drive` does that.

    Simulates exactly ONE eval server (the default single-``EVAL_SERVER_URL`` setup
    every existing test exercises). ``eval_server_url`` is accepted on every method
    purely so this stays a drop-in replacement for the real, now url-aware,
    ``start_duel``/``poll_duel``/``cancel_duel`` — single-server tests never need to
    look at it. Multi-server dispatch has its own fixture in
    ``test_multi_eval_server.py``, since faking N independent servers with distinct
    busy/faulty states needs per-url state this class doesn't track.
    """

    def __init__(self, monkeypatch, outcome, spec):
        import leoma.app.validator.main as vmain

        self._outcome = outcome        # (entry) -> dict | BaseException (raised at dispatch)
        self.spec = spec
        self.dispatched: list[str] = []
        self.polled: list[str] = []
        self.jobs: dict[str, dict] = {}
        self.cancelled: list[str] = []
        monkeypatch.setattr(vmain, "start_duel", self.start_duel)
        monkeypatch.setattr(vmain, "poll_duel", self.poll_duel)
        monkeypatch.setattr(vmain, "cancel_duel", self.cancel_duel)

    async def start_duel(self, entry, king, block_hash, *, eval_server_url=""):
        outcome = self._outcome(entry)
        if isinstance(outcome, BaseException):
            raise outcome
        eval_id = f"eval-{len(self.jobs):04d}"
        self.jobs[eval_id] = outcome
        self.dispatched.append(entry.hotkey)
        return eval_id

    async def poll_duel(self, eval_id, *, eval_server_url=""):
        self.polled.append(eval_id)
        outcome = self.jobs[eval_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def cancel_duel(self, eval_id, *, eval_server_url=""):
        self.cancelled.append(eval_id)

    async def drive(self, state, store, entries, *, block, ticks=None):
        """Run enough ticks to settle + dispatch every challenger."""
        import leoma.app.validator.main as vmain

        for _ in range(ticks or (len(entries) * 2 + 2)):
            await vmain.process_challengers(
                _FakeSubtensor(), object(), state, {}, store, entries, block
            )


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
