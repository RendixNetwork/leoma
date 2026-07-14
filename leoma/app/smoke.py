"""Did the subnet actually do the right thing? Scenario matchers over dashboard.json.

A testnet dress rehearsal is only worth running if you can *assert* the outcomes, not
eyeball them. During the rehearsal an operator drives a fixed set of scenarios — submit
a good model, a broken repo, a freeze-cheat, a copy of the king — and this module reads
the validator's published ``dashboard.json`` and confirms each one was handled the way
the design says it should be.

Every matcher is a pure function of the dashboard dict, so the assertion logic is fully
unit-testable here and then pointed at a live validator during the rehearsal. Each
returns a :class:`Scenario` with ``observed`` True/False and the evidence, so a partial
run tells you exactly which behaviors have and haven't been exercised yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    observed: bool
    evidence: str = ""


def _history(dash: dict) -> list[dict]:
    return list(dash.get("history") or [])


def _find(dash: dict, pred: Callable[[dict], bool]) -> Optional[dict]:
    for row in _history(dash):
        if pred(row):
            return row
    return None


def _row_evidence(row: dict) -> str:
    hk = str(row.get("hotkey") or "")
    who = (hk[:10] + "…") if hk else "base"
    return f"{row.get('model_repo', '?')} ({who}) at block {row.get('block', '?')}"


def saw_crown(dash: dict) -> Scenario:
    """A challenger genuinely beat the king and took the crown."""
    row = _find(dash, lambda r: r.get("accepted") and r.get("verdict") == "challenger")
    return Scenario("crown", "a challenger beat the king and was crowned",
                    row is not None, _row_evidence(row) if row else "no crowning recorded")


def saw_rejection(dash: dict) -> Scenario:
    """A challenger was scored and lost fairly (lcb below delta) — the king held."""
    row = _find(dash, lambda r: r.get("verdict") == "king" and not r.get("rejected_by")
                and r.get("lcb") is not None)
    return Scenario("rejection", "a challenger was scored and lost fairly (king held)",
                    row is not None, _row_evidence(row) if row else "no fair rejection recorded")


def saw_error_quarantine(dash: dict) -> Scenario:
    """A broken/unloadable model produced an error row (and was quarantined)."""
    row = _find(dash, lambda r: r.get("verdict") == "error")
    ev = f"{_row_evidence(row)} — {row.get('error_reason', '?')}" if row else "no error row recorded"
    return Scenario("error_quarantine", "a broken model was recorded as an error (not silently dropped)",
                    row is not None, ev)


def saw_freeze_cheat_rejected(dash: dict) -> Scenario:
    """A model that beat the king only by holding the conditioning frame was rejected."""
    row = _find(dash, lambda r: r.get("rejected_by") == "freeze_gate")
    return Scenario("freeze_cheat", "a freeze-frame cheat was rejected by the gate",
                    row is not None, _row_evidence(row) if row else "no freeze-gate rejection recorded")


def saw_copy_rejected(dash: dict) -> Scenario:
    """A copy of the king (exact or repackaged) was rejected without a full duel."""
    row = _find(dash, lambda r: r.get("rejected_by") == "copy_of_king")
    return Scenario("copy_of_king", "a copy of the king was rejected pre-duel",
                    row is not None, _row_evidence(row) if row else "no copy-of-king rejection recorded")


def saw_live_duel(dash: dict) -> Scenario:
    """The dashboard surfaced a duel running on the GPU (the site moves mid-duel)."""
    live = dash.get("live")
    ev = f"{live.get('model_repo')} dispatched at block {live.get('dispatched_block')}" if live else "no live duel at snapshot time"
    return Scenario("live_duel", "the dashboard shows a live in-flight duel", bool(live), ev)


def saw_healthy(dash: dict) -> Scenario:
    """The validator is NOT degraded (it is actually dueling, not burning)."""
    degraded = dash.get("degraded")
    return Scenario("healthy", "the validator is dueling, not degraded/burning",
                    not degraded, f"degraded={degraded}" if degraded else "not degraded")


#: The full rehearsal set, in the order an operator would naturally exercise them.
ALL_SCENARIOS: tuple[Callable[[dict], Scenario], ...] = (
    saw_healthy,
    saw_crown,
    saw_rejection,
    saw_error_quarantine,
    saw_copy_rejected,
    saw_freeze_cheat_rejected,
    saw_live_duel,
)


@dataclass
class SmokeReport:
    scenarios: list[Scenario] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scenarios if s.observed)

    @property
    def total(self) -> int:
        return len(self.scenarios)

    @property
    def complete(self) -> bool:
        return all(s.observed for s in self.scenarios)

    @property
    def missing(self) -> list[Scenario]:
        return [s for s in self.scenarios if not s.observed]


def run_smoke(dash: dict, *, require_live: bool = False) -> SmokeReport:
    """Evaluate every rehearsal scenario against one dashboard snapshot.

    ``require_live`` includes the live-duel scenario in completeness — off by default
    because whether a duel is running *at snapshot time* is timing-dependent, so it is
    informational unless the operator is specifically checking it.
    """
    matchers = ALL_SCENARIOS if require_live else tuple(m for m in ALL_SCENARIOS if m is not saw_live_duel)
    return SmokeReport([m(dash) for m in matchers])


__all__ = ["Scenario", "SmokeReport", "run_smoke", "ALL_SCENARIOS"]
