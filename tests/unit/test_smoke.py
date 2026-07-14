"""Testnet scenario matchers.

A dress rehearsal is only worth running if the outcomes can be asserted. These
matchers read a validator's dashboard.json and confirm each expected behavior
happened; the tests pin what counts as "observed" for each scenario so a green smoke
run genuinely means the subnet did the right thing.
"""

from leoma.app.smoke import (
    run_smoke,
    saw_copy_rejected,
    saw_crown,
    saw_error_quarantine,
    saw_freeze_cheat_rejected,
    saw_healthy,
    saw_live_duel,
    saw_rejection,
)


def _dash(history=None, live=None, degraded=None):
    return {"history": history or [], "live": live, "degraded": degraded}


def _crown_row():
    return {"accepted": True, "verdict": "challenger", "hotkey": "5win", "model_repo": "u/leoma-win", "block": 100}


def _held_row():
    return {"accepted": False, "verdict": "king", "lcb": -0.01, "hotkey": "5lose",
            "model_repo": "u/leoma-lose", "block": 101}


class TestIndividualScenarios:
    def test_crown_needs_an_accepted_challenger(self):
        assert saw_crown(_dash([_crown_row()])).observed
        assert not saw_crown(_dash([_held_row()])).observed

    def test_fair_rejection_needs_a_scored_loss_not_a_gate_rejection(self):
        assert saw_rejection(_dash([_held_row()])).observed
        # a gate rejection (rejected_by set) is NOT a fair scored loss
        gate = {"verdict": "king", "rejected_by": "freeze_gate", "lcb": 0.004}
        assert not saw_rejection(_dash([gate])).observed

    def test_error_quarantine_needs_an_error_row(self):
        row = {"verdict": "error", "error_reason": "model_not_found", "hotkey": "5bad", "model_repo": "u/x"}
        s = saw_error_quarantine(_dash([row]))
        assert s.observed and "model_not_found" in s.evidence

    def test_freeze_cheat_needs_a_freeze_gate_rejection(self):
        row = {"verdict": "king", "rejected_by": "freeze_gate"}
        assert saw_freeze_cheat_rejected(_dash([row])).observed
        assert not saw_freeze_cheat_rejected(_dash([_held_row()])).observed

    def test_copy_needs_a_copy_of_king_rejection(self):
        row = {"verdict": "king", "rejected_by": "copy_of_king"}
        assert saw_copy_rejected(_dash([row])).observed

    def test_live_duel_needs_a_live_block(self):
        assert saw_live_duel(_dash(live={"model_repo": "u/x", "dispatched_block": 9})).observed
        assert not saw_live_duel(_dash()).observed

    def test_live_duel_prefers_the_live_duels_list(self):
        """A multi-eval-server validator publishes live_duels (a list) instead of the
        older single live dict — the smoke tool must read the current key, not just
        the back-compat one."""
        dash = {"history": [], "live_duels": [{"model_repo": "u/multi", "dispatched_block": 42}],
                "degraded": None}
        s = saw_live_duel(dash)
        assert s.observed
        assert "u/multi" in s.evidence

    def test_live_duel_reports_how_many_more_are_in_flight(self):
        dash = {"history": [], "live_duels": [
            {"model_repo": "u/a", "dispatched_block": 1},
            {"model_repo": "u/b", "dispatched_block": 2},
        ], "degraded": None}
        s = saw_live_duel(dash)
        assert s.observed
        assert "+1 more" in s.evidence

    def test_live_duel_falls_back_to_live_when_live_duels_key_is_absent(self):
        """A dashboard snapshot published before live_duels existed still works."""
        dash = {"history": [], "live": {"model_repo": "u/old", "dispatched_block": 5}, "degraded": None}
        assert saw_live_duel(dash).observed

    def test_live_duel_is_not_observed_when_live_duels_is_an_empty_list(self):
        dash = {"history": [], "live_duels": [], "degraded": None}
        assert not saw_live_duel(dash).observed

    def test_healthy_is_the_absence_of_a_degraded_reason(self):
        assert saw_healthy(_dash()).observed
        assert not saw_healthy(_dash(degraded="corpus_unpinned")).observed


class TestFullReport:
    def _rich(self):
        return _dash(
            history=[
                _crown_row(),
                _held_row(),
                {"verdict": "error", "error_reason": "model_not_found", "model_repo": "u/e"},
                {"verdict": "king", "rejected_by": "copy_of_king", "model_repo": "u/c"},
                {"verdict": "king", "rejected_by": "freeze_gate", "model_repo": "u/f"},
            ],
            live={"model_repo": "u/live", "dispatched_block": 200},
        )

    def test_a_full_rehearsal_is_complete(self):
        r = run_smoke(self._rich(), require_live=True)
        assert r.complete
        assert r.passed == r.total

    def test_a_partial_run_reports_exactly_what_is_missing(self):
        # only a crown so far
        r = run_smoke(_dash([_crown_row()]))
        assert not r.complete
        missing = {s.key for s in r.missing}
        assert "copy_of_king" in missing
        assert "freeze_cheat" in missing
        assert "error_quarantine" in missing
        assert "crown" not in missing

    def test_live_is_excluded_from_completeness_by_default(self):
        """Whether a duel is running at snapshot time is timing-dependent, so it is
        informational unless explicitly required."""
        r = run_smoke(self._rich())  # require_live=False
        assert all(s.key != "live_duel" for s in r.scenarios)
        assert r.complete

    def test_a_degraded_validator_fails_the_healthy_scenario(self):
        r = run_smoke(_dash([_crown_row()], degraded="no_seed_digest"))
        assert not r.complete
        assert any(s.key == "healthy" and not s.observed for s in r.scenarios)
