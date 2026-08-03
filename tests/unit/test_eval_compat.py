"""Validator/evaluator compatibility is checked before and after every duel."""

import pytest

import leoma.app.validator.main as vmain
from leoma.app.validator.failures import EvalJobFailed
from leoma.eval.codehash import eval_code_digest
from leoma.eval.runtime_lock import expected_eval_runtime_digest

from .conftest import make_verdict, pinned_spec


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Client:
    def __init__(self, body):
        self.body = body

    async def get(self, url):
        return _Response(self.body)


def _health(spec):
    return {
        "consensus_digest": spec.digest(),
        "eval_code_digest": eval_code_digest(),
        "eval_runtime_compatible": True,
        "eval_runtime_digest": expected_eval_runtime_digest(spec.runtime),
    }


@pytest.mark.asyncio
async def test_dispatch_preflight_requires_code_and_runtime_digests(monkeypatch):
    spec = pinned_spec()
    monkeypatch.setattr(vmain, "SPEC", spec)
    monkeypatch.setattr(vmain, "CONSENSUS_DIGEST", spec.digest())

    await vmain.preflight_eval_server(_Client(_health(spec)), "http://eval")

    missing_code = _health(spec)
    missing_code.pop("eval_code_digest")
    with pytest.raises(EvalJobFailed) as code_error:
        await vmain.preflight_eval_server(_Client(missing_code), "http://eval")
    assert code_error.value.reason == "code_mismatch"

    missing_runtime = _health(spec)
    missing_runtime.pop("eval_runtime_digest")
    with pytest.raises(EvalJobFailed) as runtime_error:
        await vmain.preflight_eval_server(_Client(missing_runtime), "http://eval")
    assert runtime_error.value.reason == "runtime_mismatch"


def test_terminal_verdict_must_repeat_code_and_runtime_proof(monkeypatch):
    spec = pinned_spec()
    monkeypatch.setattr(vmain, "SPEC", spec)

    valid = make_verdict(spec, accepted=False)
    assert vmain._verified_verdict(valid) is valid

    missing_code = make_verdict(spec, accepted=False)
    missing_code["audit"].pop("eval_code_digest")
    with pytest.raises(EvalJobFailed) as code_error:
        vmain._verified_verdict(missing_code)
    assert code_error.value.reason == "code_mismatch"

    missing_runtime = make_verdict(spec, accepted=False)
    missing_runtime["audit"].pop("eval_runtime")
    with pytest.raises(EvalJobFailed) as runtime_error:
        vmain._verified_verdict(missing_runtime)
    assert runtime_error.value.reason == "runtime_mismatch"
