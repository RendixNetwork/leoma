"""
Unit tests for the eval-server HTTP contract (fake runner, no GPU).

Covers: POST /eval -> eval_id, SSE stream of phases + verdict, status poll,
single-flight 409 while a duel is running, 404 for unknown ids — and the v2
request contract, in which the request carries the ENTIRE consensus surface
instead of a handful of individually-defaulted knobs.
"""

import json
import threading

import pytest
from starlette.testclient import TestClient

from leoma.eval_server import create_app

from .conftest import pinned_spec


SPEC = pinned_spec()

REQ = dict(
    king_repo="u/leoma-k", king_digest="sha256:" + "a" * 64,
    challenger_repo="u/leoma-c", challenger_digest="sha256:" + "b" * 64,
    block_hash="0xabc", hotkey="5C7L",
    spec=SPEC.model_dump(mode="json"),
    consensus_digest=SPEC.digest(),
)


def _stream_events(client, eval_id):
    events = []
    with client.stream("GET", f"/eval/{eval_id}/stream") as s:
        for line in s.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_health():
    c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["busy"] is False


def test_health_publishes_the_digests_a_validator_preflights_on():
    """A stale eval box is the likeliest consensus failure there is — an operator
    who redeployed three machines out of four. The validator checks these two
    digests before handing over an hours-long duel."""
    c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
    body = c.get("/health").json()
    assert body["consensus_digest"].startswith("sha256:")
    assert body["eval_code_digest"].startswith("sha256:")
    assert body["eval_runtime_digest"].startswith("sha256:")
    assert body["expected_eval_runtime_digest"].startswith("sha256:")
    assert body["eval_runtime_enforced"] is False
    assert isinstance(body["eval_runtime_issues"], list)


def test_production_server_rejects_an_incompatible_runtime(monkeypatch):
    from leoma.eval import runtime_lock

    runtime_lock.eval_runtime_report.cache_clear()
    monkeypatch.setattr(
        runtime_lock,
        "eval_runtime_report",
        lambda spec: {
            "compatible": False,
            "digest": "sha256:" + "a" * 64,
            "expected_digest": "sha256:" + "b" * 64,
            "identity": {},
            "issues": ["torch: expected '2.6.0', observed '9.9.9'"],
        },
    )
    c = TestClient(create_app())
    health = c.get("/health").json()
    assert health["status"] == "runtime_mismatch"
    response = c.post("/eval", json=REQ)
    assert response.status_code == 503
    assert response.json()["reason"] == "runtime_mismatch"


def test_configured_token_protects_every_route(monkeypatch):
    monkeypatch.setenv("LEOMA_EVAL_TOKEN", "correct-horse")
    c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))

    assert c.get("/health").status_code == 401
    assert c.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    auth = {"Authorization": "Bearer correct-horse"}
    assert c.get("/health", headers=auth).status_code == 200
    assert c.post("/eval", json=REQ).status_code == 401
    assert c.post("/eval", json=REQ, headers=auth).status_code == 200


def test_full_duel_flow():
    def runner(req, emit, cancel):
        emit({"phase": "materialize", "which": "king"})
        emit({"phase": "duel", "n_clips": req.spec.duel.n_clips})
        return {"accepted": True, "verdict": "challenger", "lcb": 0.09,
                "n_clips": req.spec.duel.n_clips}

    c = TestClient(create_app(runner=runner))
    r = c.post("/eval", json=REQ)
    assert r.status_code == 200
    eval_id = r.json()["eval_id"]

    events = _stream_events(c, eval_id)
    phases = [e["phase"] for e in events]
    assert phases == ["materialize", "duel", "verdict"]
    assert events[-1]["accepted"] is True
    assert events[-1]["verdict"] == "challenger"

    poll = c.get(f"/eval/{eval_id}").json()
    assert poll["status"] == "done"
    assert poll["verdict"]["accepted"] is True


def test_error_in_runner_becomes_error_event():
    def runner(req, emit, cancel):
        raise RuntimeError("boom")

    c = TestClient(create_app(runner=runner))
    eval_id = c.post("/eval", json=REQ).json()["eval_id"]
    events = _stream_events(c, eval_id)
    assert events[-1]["phase"] == "error"
    assert "boom" in events[-1]["error"]
    assert c.get(f"/eval/{eval_id}").json()["status"] == "error"


def test_single_flight_409_while_busy():
    release = threading.Event()

    def blocking_runner(req, emit, cancel):
        release.wait(timeout=5)
        return {"accepted": False, "verdict": "king"}

    c = TestClient(create_app(runner=blocking_runner))
    first = c.post("/eval", json=REQ)
    assert first.status_code == 200

    # Second request while the first still holds the lock -> 409.
    second = c.post("/eval", json=REQ)
    assert second.status_code == 409
    assert "already running" in second.json()["error"]

    # Let the first finish, drain its stream, then a new eval is accepted.
    release.set()
    _stream_events(c, first.json()["eval_id"])
    third = c.post("/eval", json=REQ)
    assert third.status_code == 200


def test_unknown_eval_id_404():
    c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
    assert c.get("/eval/does-not-exist").status_code == 404
    assert c.get("/eval/does-not-exist/stream").status_code == 404


class TestRequestCarriesTheWholeExam:
    """The v2 contract. The old EvalRequest had a DEFAULT for every duel knob
    (metric="lpips", n_clips=32, num_frames=81...), and said nothing at all about
    the prompt, the resolution or the negative prompt — those came from the eval
    box's own environment. So a validator that forgot a field, or a box running an
    older build that ignored one, silently ran a different exam and produced a
    verdict nobody else could reproduce. Now every field is required."""

    def test_a_request_without_a_spec_is_rejected(self):
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        legacy = {k: v for k, v in REQ.items() if k not in ("spec", "consensus_digest")}
        legacy.update(metric="lpips", n_clips=8)   # the old shape, verbatim
        assert c.post("/eval", json=legacy).status_code == 422

    def test_a_spec_missing_one_field_is_rejected_not_defaulted(self):
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        broken = dict(REQ)
        broken["spec"] = dict(broken["spec"])
        broken["spec"]["duel"] = {
            k: v for k, v in broken["spec"]["duel"].items() if k != "metric"
        }
        # The whole point: a missing field is a 422, never a silent "lpips".
        assert c.post("/eval", json=broken).status_code == 422

    def test_an_unknown_field_is_rejected(self):
        # A newer validator sending a field this box does not understand must not
        # be silently ignored — that is a divergence the box would never report.
        c = TestClient(create_app(runner=lambda req, emit, cancel: {"accepted": False}))
        extra = dict(REQ, some_future_knob=1)
        assert c.post("/eval", json=extra).status_code == 422

    def test_the_runner_sees_exactly_what_was_sent(self):
        seen = {}

        def runner(req, emit, cancel):
            seen["digest"] = req.spec.digest()
            seen["metric"] = req.spec.duel.metric
            seen["frames"] = req.spec.gen.num_frames
            return {"accepted": False, "verdict": "king"}

        c = TestClient(create_app(runner=runner))
        eval_id = c.post("/eval", json=REQ).json()["eval_id"]
        _stream_events(c, eval_id)

        assert seen["digest"] == SPEC.digest()
        assert seen["metric"] == SPEC.duel.metric
        assert seen["frames"] == SPEC.gen.num_frames
