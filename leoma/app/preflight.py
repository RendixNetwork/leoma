"""Is this validator actually ready to launch? A hard gate, not a hope.

The subnet has several pins that must be set before it does anything useful, and each
one fails *safe but silent*: an unpinned seed digest, an unpinned corpus, or an eval
box on a stale config all make the validator burn to UID 0 rather than crown. That is
the correct behavior, but an operator who flips the switch and walks away would see a
dead subnet with no single place that says *why*.

``run_preflight`` is that single place. It runs every readiness check, classifies each
as pass / warn / fail, and returns an overall verdict. ``fail`` means "the validator
cannot safely launch"; ``warn`` is reserved for non-consensus observability such as
an optional public dashboard bucket.

The checks are **pure functions of already-fetched inputs** — the CLI does the I/O
(HTTP to the eval box, a HEAD on the corpus) and hands the results here — so the
decision logic is unit-testable with no network, GPU, or chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

from leoma.infra.model_store import DIGEST_RE

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != FAIL


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[CheckResult, ...]

    @property
    def ready(self) -> bool:
        """True when nothing is FAIL. Warnings do not block launch."""
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status == FAIL)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.status == WARN)


def check_seed(seed_digest: str) -> CheckResult:
    digest = (seed_digest or "").strip()
    if not digest:
        return CheckResult(
            "seed_digest", FAIL,
            "chain.toml [seed].seed_digest is empty — with no genesis king the subnet burns "
            "100% to UID 0. Pin the exact base-model revision before launch.",
        )
    if not DIGEST_RE.match(digest):
        return CheckResult(
            "seed_digest", FAIL,
            f"chain.toml [seed].seed_digest ({digest[:23]}…) is not a recognized digest — "
            "expected 'sha256:<64hex>' (Hippius) or 'hf:<40hex>' (HuggingFace commit SHA). "
            "A malformed pin means the genesis king can never resolve.",
        )
    return CheckResult("seed_digest", PASS, f"genesis king pinned ({digest[:23]}…)")


def check_corpus_pin(corpus_pinned: bool, manifest_digest: str) -> CheckResult:
    if corpus_pinned:
        return CheckResult("corpus_pin", PASS, f"corpus manifest pinned ({manifest_digest[:23]}…)")
    return CheckResult(
        "corpus_pin", FAIL,
        "chain.toml [corpus].manifest_digest is empty — the duel exam is not reproducible "
        "and the validator refuses to duel. Publish a manifest (`leoma corpus publish-manifest`) "
        "and pin its digest.",
    )


def check_consensus_digest(consensus_digest: str) -> CheckResult:
    # If chain_config imported at all, SPEC validated and this exists — so this is really
    # a "surface the digest so the operator can compare it across boxes" check.
    if consensus_digest and consensus_digest.startswith("sha256:"):
        return CheckResult("consensus_digest", PASS, consensus_digest)
    return CheckResult("consensus_digest", FAIL, "consensus surface did not produce a digest")


def check_corpus_reachable(fetched_digest: Optional[str], pinned_digest: str, error: Optional[str]) -> CheckResult:
    """Given the digest of the manifest actually fetched from the bucket, does it match?"""
    if error:
        return CheckResult("corpus_fetch", FAIL, f"could not fetch the pinned corpus manifest: {error}")
    if not fetched_digest:
        return CheckResult(
            "corpus_fetch", FAIL,
            "corpus manifest was not fetched (missing or unusable source-bucket credentials)",
        )
    if fetched_digest == pinned_digest:
        return CheckResult("corpus_fetch", PASS, "published manifest matches the pinned digest")
    return CheckResult(
        "corpus_fetch", FAIL,
        f"the bucket's manifest ({fetched_digest[:19]}…) does NOT match the pinned digest "
        f"({pinned_digest[:19]}…). Republish, or fix the pin.",
    )


def check_eval_server(
    health: Optional[dict],
    our_consensus_digest: str,
    our_eval_code_digest: str,
    our_eval_runtime_digest: str,
    error: Optional[str] = None,
    *,
    name: str = "eval_server",
) -> CheckResult:
    """Given one eval box's /health, do its consensus + code digests match ours?

    ``name`` lets the caller disambiguate several servers (e.g. ``eval_server[url]``)
    when checking a whole ``EVAL_SERVER_URLS`` fleet instead of a single box.
    """
    if error:
        return CheckResult(name, FAIL, f"eval server not reachable ({error})")
    if health is None:
        return CheckResult(
            name, FAIL, "no eval server configured (set EVAL_SERVER_URL or EVAL_SERVER_URLS)"
        )

    theirs_consensus = health.get("consensus_digest")
    theirs_code = health.get("eval_code_digest")
    if theirs_consensus != our_consensus_digest:
        return CheckResult(
            name, FAIL,
            f"eval box pins a DIFFERENT consensus surface (box {str(theirs_consensus)[:19]}…, "
            f"validator {our_consensus_digest[:19]}…) — one of you is on a stale chain.toml.",
        )
    if theirs_code is None:
        # A current box always reports this field (see eval_server.py's /health). A box
        # missing it entirely is running a build old enough to predate the field — we
        # have literally no evidence its scoring code matches, so this must not read as
        # a silent PASS.
        return CheckResult(
            name, FAIL,
            "eval box's /health did not report eval_code_digest (stale build?) — its "
            "scoring code could not be verified against this validator's.",
        )
    if theirs_code != our_eval_code_digest:
        return CheckResult(
            name, FAIL,
            f"eval box runs DIFFERENT scoring code (box {str(theirs_code)[:19]}…, "
            f"validator {our_eval_code_digest[:19]}…) — its distances would not be reproducible.",
        )
    theirs_runtime = health.get("eval_runtime_digest")
    if health.get("eval_runtime_compatible") is not True:
        issues = health.get("eval_runtime_issues") or []
        detail = f": {'; '.join(str(item) for item in issues[:5])}" if issues else ""
        return CheckResult(name, FAIL, f"eval box reports an incompatible numerical runtime{detail}")
    if theirs_runtime != our_eval_runtime_digest:
        return CheckResult(
            name, FAIL,
            f"eval box runs a DIFFERENT numerical runtime (box "
            f"{str(theirs_runtime)[:19]}…, validator {our_eval_runtime_digest[:19]}…).",
        )
    return CheckResult(
        name, PASS, "eval box matches this validator's consensus, code, and runtime"
    )


class EvalServerProbe(NamedTuple):
    """One configured server's raw /health result, ready for ``check_eval_servers``."""
    url: str
    health: Optional[dict]
    error: Optional[str] = None


def check_eval_servers(
    probes: tuple[EvalServerProbe, ...],
    our_consensus_digest: str,
    our_eval_code_digest: str,
    our_eval_runtime_digest: str,
) -> tuple[CheckResult, ...]:
    """One :func:`check_eval_server` result per configured ``EVAL_SERVER_URLS`` entry.

    A single-server validator gets exactly one ``eval_server`` check, unchanged. A
    multi-server validator gets one check per URL — silently checking only the first
    configured server (or none at all) would leave the rest of the fleet unverified.
    """
    if not probes:
        return (
            check_eval_server(
                None, our_consensus_digest, our_eval_code_digest, our_eval_runtime_digest
            ),
        )
    if len(probes) == 1:
        p = probes[0]
        return (
            check_eval_server(
                p.health,
                our_consensus_digest,
                our_eval_code_digest,
                our_eval_runtime_digest,
                p.error,
            ),
        )
    return tuple(
        check_eval_server(
            p.health,
            our_consensus_digest,
            our_eval_code_digest,
            our_eval_runtime_digest,
            p.error,
            name=f"eval_server[{p.url}]",
        )
        for p in probes
    )


def check_state_bucket(own_bucket: Optional[str]) -> CheckResult:
    if own_bucket and own_bucket.strip():
        return CheckResult("state_bucket", PASS, f"king state persists to {own_bucket}")
    return CheckResult(
        "state_bucket", FAIL,
        "R2_OWN_BUCKET is not set — the validator cannot persist king state and will refuse to run.",
    )


def check_dashboard_bucket(
    dashboard_bucket: Optional[str],
    own_bucket: Optional[str],
    error: Optional[str] = None,
) -> CheckResult:
    """Require operators to notice when public delivery still shares private state.

    Dashboard publication is observability, not consensus, so a missing dedicated
    bucket is a warning rather than a launch-blocking failure. The validator retains
    the legacy fallback to ``R2_OWN_BUCKET`` but preflight makes that fallback visible.
    """
    dashboard = (dashboard_bucket or "").strip()
    state = (own_bucket or "").strip()
    if not dashboard:
        return CheckResult(
            "dashboard_bucket",
            WARN,
            "LEOMA_DASHBOARD_BUCKET is not set — dashboard.json falls back to the "
            "private state bucket and cannot be exposed independently.",
        )
    if dashboard == state:
        return CheckResult(
            "dashboard_bucket",
            WARN,
            f"dashboard and canonical state both use {dashboard}; use a separate "
            "public-read bucket before exposing dashboard.json.",
        )
    if error:
        return CheckResult(
            "dashboard_bucket",
            WARN,
            f"public dashboard bucket {dashboard} is configured but not reachable "
            f"with its write credentials ({error}); the validator will keep running "
            "and publish only to its private state bucket.",
        )
    return CheckResult(
        "dashboard_bucket",
        PASS,
        f"public dashboard publishes separately to {dashboard}",
    )


class WalletProbe(NamedTuple):
    wallet_name: str
    hotkey_name: str
    hotkey_exists: bool
    hotkey_ss58: Optional[str]
    error: Optional[str] = None


class ChainProbe(NamedTuple):
    network: str
    netuid: int
    subnet_name: Optional[str]
    block: Optional[int]
    hotkey_ss58: Optional[str]
    uid: Optional[int]
    validator_permit: bool
    error: Optional[str] = None


def check_wallet(probe: WalletProbe) -> CheckResult:
    if probe.error:
        return CheckResult(
            "wallet", FAIL,
            f"cannot load {probe.wallet_name}/{probe.hotkey_name}: {probe.error}",
        )
    if not probe.hotkey_exists or not probe.hotkey_ss58:
        return CheckResult(
            "wallet", FAIL,
            f"hotkey file does not exist or has no address for "
            f"{probe.wallet_name}/{probe.hotkey_name}",
        )
    return CheckResult(
        "wallet", PASS,
        f"{probe.wallet_name}/{probe.hotkey_name} ({probe.hotkey_ss58})",
    )


def check_chain(probe: ChainProbe, expected_subnet_name: str) -> CheckResult:
    if probe.error:
        return CheckResult(
            "chain", FAIL,
            f"cannot verify {probe.network} NetUID {probe.netuid}: {probe.error}",
        )
    if not probe.subnet_name:
        return CheckResult(
            "chain", FAIL,
            f"NetUID {probe.netuid} does not exist on {probe.network}",
        )
    if probe.subnet_name.casefold() != expected_subnet_name.casefold():
        return CheckResult(
            "chain", FAIL,
            f"configured {probe.network} NetUID {probe.netuid} is {probe.subnet_name!r}, "
            f"not {expected_subnet_name!r}",
        )
    if probe.uid is None or not probe.hotkey_ss58:
        return CheckResult(
            "chain", FAIL,
            f"validator hotkey is not registered on {probe.network} NetUID {probe.netuid}",
        )
    if not probe.validator_permit:
        return CheckResult(
            "chain", FAIL,
            f"hotkey {probe.hotkey_ss58} is UID {probe.uid} but has no validator permit "
            f"on NetUID {probe.netuid}",
        )
    return CheckResult(
        "chain", PASS,
        f"{probe.subnet_name} on {probe.network} NetUID {probe.netuid}, UID {probe.uid}, "
        f"validator permit confirmed at block {probe.block}",
    )


def run_preflight(
    *,
    seed_digest: str,
    corpus_pinned: bool,
    manifest_digest: str,
    consensus_digest: str,
    eval_code_digest: str,
    eval_runtime_digest: str,
    own_bucket: Optional[str],
    wallet_probe: WalletProbe,
    chain_probe: ChainProbe,
    expected_subnet_name: str,
    dashboard_bucket: Optional[str] = None,
    dashboard_error: Optional[str] = None,
    corpus_fetched_digest: Optional[str] = None,
    corpus_error: Optional[str] = None,
    eval_servers: tuple[EvalServerProbe, ...] = (),
) -> PreflightReport:
    """Assemble every readiness check into one verdict. Pure — the caller does the I/O.

    ``eval_servers`` is one probe per configured ``EVAL_SERVER_URLS`` entry (empty when
    none are configured) — a single-server validator still gets exactly one
    ``eval_server`` check; a multi-server one gets one per URL, so a stale box can't
    hide behind a healthy sibling.
    """
    checks = [
        check_seed(seed_digest),
        check_corpus_pin(corpus_pinned, manifest_digest),
        check_consensus_digest(consensus_digest),
        check_corpus_reachable(corpus_fetched_digest, manifest_digest, corpus_error),
        *check_eval_servers(
            eval_servers, consensus_digest, eval_code_digest, eval_runtime_digest
        ),
        check_state_bucket(own_bucket),
        check_dashboard_bucket(dashboard_bucket, own_bucket, dashboard_error),
        check_wallet(wallet_probe),
        check_chain(chain_probe, expected_subnet_name),
    ]
    return PreflightReport(tuple(checks))


__all__ = [
    "PASS", "WARN", "FAIL",
    "CheckResult", "PreflightReport", "EvalServerProbe", "WalletProbe", "ChainProbe",
    "run_preflight",
    "check_seed", "check_corpus_pin", "check_consensus_digest",
    "check_corpus_reachable", "check_eval_server", "check_eval_servers",
    "check_state_bucket", "check_dashboard_bucket", "check_wallet", "check_chain",
]
