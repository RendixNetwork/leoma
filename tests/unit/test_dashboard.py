"""
Unit tests for the public dashboard.json payload builder.
"""

from leoma.app.validator.state_store import KingState
from leoma.app.validator.dashboard import build_dashboard
from leoma.app.validator import king as K


def _digest(c):
    return "sha256:" + c * 64


def _two_king_state():
    st = KingState()
    st.king, st.king_chain = K.crown(
        None, [], hotkey="A", model_repo="u/leoma-A", model_digest=_digest("a"),
        block=100, challenge_id="seed", crowned_at="t0",
    )
    st.king, st.king_chain = K.crown(
        st.king, st.king_chain, hotkey="B", model_repo="u/leoma-B", model_digest=_digest("b"),
        block=200, challenge_id="eval-1", crowned_at="t1",
    )
    return st


class TestBuildDashboard:
    def test_king_carries_uid_and_reign(self):
        st = _two_king_state()
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={"name": "leoma"},
                            duel_params={}, updated_at="now")
        assert d["king"]["hotkey"] == "B"
        assert d["king"]["uid"] == 20
        assert d["king"]["reign_number"] == 1
        assert d["king"]["previous_repo"] == "u/leoma-A"

    def test_equal_weight_across_registered_kings(self):
        st = _two_king_state()
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={}, duel_params={}, updated_at="now")
        rows = {r["hotkey"]: r for r in d["king_chain"]}
        assert [r["hotkey"] for r in d["king_chain"]] == ["B", "A"]  # current king first
        assert rows["A"]["weight"] == 0.5
        assert rows["B"]["weight"] == 0.5
        assert rows["A"]["uid"] == 10

    def test_unregistered_king_gets_null_weight(self):
        st = _two_king_state()
        d = build_dashboard(st, {"B": 20}, chain_meta={}, duel_params={}, updated_at="now")
        rows = {r["hotkey"]: r for r in d["king_chain"]}
        assert rows["B"]["weight"] == 1.0   # only B registered -> 100%
        assert rows["A"]["weight"] is None

    def test_no_king_burns_empty(self):
        st = KingState()
        d = build_dashboard(st, {}, chain_meta={}, duel_params={}, updated_at="now")
        assert d["king"] == {}
        assert d["king_chain"] == []

    def test_history_and_queue_passthrough(self):
        st = _two_king_state()
        st.record_duel({"hotkey": "B", "verdict": "challenger", "accepted": True})
        st.record_duel({"hotkey": "C", "verdict": "king", "accepted": False})
        q = [{"hotkey": "C", "status": "unseen"}]
        d = build_dashboard(st, {"A": 10, "B": 20}, chain_meta={}, duel_params={}, updated_at="now", queue=q)
        assert [h["hotkey"] for h in d["history"]] == ["C", "B"]  # newest first
        assert d["queue"] == q

    def test_carries_meta_and_params(self):
        st = _two_king_state()
        d = build_dashboard(
            st, {"A": 10, "B": 20},
            chain_meta={"name": "leoma", "netuid": 99},
            duel_params={"metric": "lpips", "delta_threshold": 0.0025},
            updated_at="2026-01-01T00:00:00Z",
        )
        assert d["chain"]["name"] == "leoma"
        assert d["duel_params"]["metric"] == "lpips"
        assert d["updated_at"] == "2026-01-01T00:00:00Z"
        # `transient_errors` is tracked separately from `failed` so that retries
        # of a flaky duel don't inflate the failure count on the dashboard.
        assert d["stats"] == {"accepted": 0, "rejected": 0, "failed": 0, "transient_errors": 0}


class TestLivePanel:
    """The dashboard used to go dark for the whole length of a duel — hours in which
    the most interesting thing in the subnet was happening and the site showed
    nothing at all. The in-flight slot is exactly the data it needed."""

    def _dash(self, state):
        return build_dashboard(
            state, {"5a": 7},
            chain_meta={"name": "leoma"}, duel_params={"metric": "lpips"},
            updated_at="2026-07-13T00:00:00Z",
        )

    def test_the_running_duel_is_published(self):
        st = KingState()
        st.inflight = [{
            "eval_id": "eval-abc", "hotkey": "5a", "model_repo": "u/leoma-a",
            "model_digest": "sha256:" + "a" * 64, "dispatched_block": 900,
            "eval_server_url": "http://box-a:9000",
        }]
        live = self._dash(st)["live_duels"]
        assert len(live) == 1
        assert live[0]["eval_id"] == "eval-abc"
        assert live[0]["hotkey"] == "5a"
        assert live[0]["uid"] == 7
        assert live[0]["dispatched_block"] == 900
        assert live[0]["eval_server_url"] == "http://box-a:9000"

    def test_several_duels_in_flight_are_all_published(self):
        """A validator with several eval servers can have several duels running at
        once — the whole point of not being limited to one in-flight slot."""
        st = KingState()
        st.inflight = [
            {"eval_id": "eval-a", "hotkey": "5a", "model_repo": "u/leoma-a",
             "model_digest": "sha256:" + "a" * 64, "dispatched_block": 900},
            {"eval_id": "eval-b", "hotkey": "5b", "model_repo": "u/leoma-b",
             "model_digest": "sha256:" + "b" * 64, "dispatched_block": 901},
        ]
        live = self._dash(st)["live_duels"]
        assert [d["eval_id"] for d in live] == ["eval-a", "eval-b"]

    def test_no_duel_means_an_empty_list_not_null(self):
        assert self._dash(KingState())["live_duels"] == []

    def test_live_back_compat_mirrors_first_live_duel(self):
        """The deployed leoma-app frontend reads `data.live` (a single dict-or-null)
        and has no knowledge of `live_duels` yet — it must keep working unmodified."""
        st = KingState()
        st.inflight = [{
            "eval_id": "eval-abc", "hotkey": "5a", "model_repo": "u/leoma-a",
            "model_digest": "sha256:" + "a" * 64, "dispatched_block": 900,
            "eval_server_url": "http://box-a:9000",
        }]
        d = self._dash(st)
        assert d["live"] == d["live_duels"][0]

    def test_live_back_compat_is_null_when_idle(self):
        assert self._dash(KingState())["live"] is None

    def test_the_reason_the_subnet_is_burning_is_published(self):
        """Without this an operator sees 100% burning to UID 0 and no reason anywhere."""
        st = KingState()
        st.degraded = "corpus_unpinned"
        assert self._dash(st)["degraded"] == "corpus_unpinned"
