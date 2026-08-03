"""The launch readiness gate.

Every pin the subnet needs fails safe-but-silent: an unpinned seed, an unpinned
corpus, or a stale eval box all make the validator burn to UID 0 rather than crown.
Preflight is the one place that says WHY before an operator flips the switch. These
tests pin the classification of each check and the overall ready/not-ready verdict.
"""

import pytest

from leoma.app.preflight import (
    FAIL,
    PASS,
    WARN,
    ChainProbe,
    EvalServerProbe,
    WalletProbe,
    check_chain,
    check_corpus_reachable,
    check_eval_server,
    check_eval_servers,
    check_seed,
    check_wallet,
    expected_subnet_name,
    run_preflight,
)

RUNTIME_DIGEST = "sha256:" + "r" * 64
WALLET = WalletProbe("val", "hk", True, "5ValidatorHotkey", None)
CHAIN = ChainProbe(
    "finney", 36, "Leoma", 8753870, "5ValidatorHotkey", 12, True, None
)

GOOD = dict(
    seed_digest="sha256:" + "5" * 64,
    corpus_pinned=True,
    manifest_digest="sha256:" + "a" * 64,
    consensus_digest="sha256:" + "c" * 64,
    eval_code_digest="sha256:" + "e" * 64,
    eval_runtime_digest=RUNTIME_DIGEST,
    own_bucket="leoma-state",
    dashboard_bucket="leoma-dashboard",
    wallet_probe=WALLET,
    chain_probe=CHAIN,
    expected_subnet_name="leoma",
)


def _health(*, consensus=None, code=None, runtime=None, compatible=True):
    return {
        "consensus_digest": consensus or GOOD["consensus_digest"],
        "eval_code_digest": code or GOOD["eval_code_digest"],
        "eval_runtime_digest": runtime or GOOD["eval_runtime_digest"],
        "eval_runtime_compatible": compatible,
        "eval_runtime_issues": [] if compatible else ["torch drifted"],
    }


def _good_probe():
    return EvalServerProbe("http://eval:9000", _health(), None)


def _run(**overrides):
    return run_preflight(
        **{**GOOD, **overrides},
        corpus_fetched_digest=overrides.get(
            "manifest_digest", GOOD["manifest_digest"]
        ),
        eval_servers=(_good_probe(),),
    )


class TestOverallVerdict:
    def test_a_fully_configured_validator_is_ready(self):
        r = _run()
        assert r.ready
        assert r.failures == ()

    def test_an_unpinned_seed_blocks_launch(self):
        r = _run(seed_digest="")
        assert not r.ready
        assert any(c.name == "seed_digest" and c.status == FAIL for c in r.checks)

    def test_an_unpinned_corpus_blocks_launch(self):
        r = _run(corpus_pinned=False, manifest_digest="")
        assert not r.ready
        assert any(c.name == "corpus_pin" and c.status == FAIL for c in r.checks)

    def test_a_missing_state_bucket_blocks_launch(self):
        r = _run(own_bucket=None)
        assert not r.ready

    def test_an_unreachable_state_bucket_blocks_launch(self):
        r = _run(state_error="AccessDenied")
        check = next(c for c in r.checks if c.name == "state_bucket")
        assert check.status == FAIL
        assert "AccessDenied" in check.detail
        assert not r.ready

    def test_a_separate_dashboard_bucket_passes(self):
        r = _run()
        check = next(c for c in r.checks if c.name == "dashboard_bucket")
        assert check.status == PASS

    def test_dashboard_falling_back_to_state_warns(self):
        r = _run(dashboard_bucket=None)
        check = next(c for c in r.checks if c.name == "dashboard_bucket")
        assert check.status == WARN
        assert r.ready

    def test_dashboard_sharing_state_warns(self):
        r = _run(dashboard_bucket=GOOD["own_bucket"])
        check = next(c for c in r.checks if c.name == "dashboard_bucket")
        assert check.status == WARN

    def test_an_unreachable_dashboard_bucket_warns_but_does_not_block_consensus(self):
        r = _run(dashboard_error="AccessDenied")
        check = next(c for c in r.checks if c.name == "dashboard_bucket")
        assert check.status == WARN
        assert "AccessDenied" in check.detail
        assert r.ready

    def test_only_non_consensus_warnings_do_not_block_launch(self):
        r = _run(dashboard_bucket=None)
        assert r.ready
        assert [warning.name for warning in r.warnings] == ["dashboard_bucket"]

    def test_missing_eval_and_corpus_access_now_block_launch(self):
        r = run_preflight(**GOOD)
        assert not r.ready
        assert {failure.name for failure in r.failures} >= {"eval_server", "corpus_fetch"}


class TestSubnetNameOverride:
    def test_non_mainnet_can_explicitly_name_a_borrowed_test_subnet(self):
        assert expected_subnet_name("test", "leoma", "fish") == "fish"

    def test_no_override_keeps_the_chain_toml_identity(self):
        assert expected_subnet_name("test", "leoma", "") == "leoma"

    @pytest.mark.parametrize("network", ["finney", "mainnet", "FINNEY"])
    def test_mainnet_identity_can_never_be_overridden(self, network):
        with pytest.raises(ValueError, match="cannot override"):
            expected_subnet_name(network, "leoma", "fish")

class TestCorpusReachable:
    def test_matching_digest_passes(self):
        c = check_corpus_reachable("sha256:aaa", "sha256:aaa", None)
        assert c.status == PASS

    def test_a_drifted_manifest_is_a_hard_fail(self):
        """The bucket serves a manifest that isn't the pinned one — validators would
        grade different exams. Must block."""
        c = check_corpus_reachable("sha256:bbb", "sha256:aaa", None)
        assert c.status == FAIL
        assert "does NOT match" in c.detail

    def test_a_fetch_error_is_a_hard_fail(self):
        c = check_corpus_reachable(None, "sha256:aaa", "connection refused")
        assert c.status == FAIL

    def test_no_credentials_is_a_hard_fail(self):
        c = check_corpus_reachable(None, "sha256:aaa", None)
        assert c.status == FAIL


class TestEvalServer:
    def test_matching_box_passes(self):
        c = check_eval_server(_health(), GOOD["consensus_digest"],
                              GOOD["eval_code_digest"], RUNTIME_DIGEST)
        assert c.status == PASS

    def test_a_stale_consensus_surface_is_a_hard_fail(self):
        """The single most likely consensus failure — an operator who redeployed some
        boxes but not others. Must block."""
        c = check_eval_server(_health(consensus="sha256:OLD"),
                              "sha256:NEW", GOOD["eval_code_digest"], RUNTIME_DIGEST)
        assert c.status == FAIL
        assert "DIFFERENT consensus surface" in c.detail

    def test_stale_scoring_code_is_a_hard_fail(self):
        c = check_eval_server(_health(code="sha256:OLD"), GOOD["consensus_digest"],
                              "sha256:NEW", RUNTIME_DIGEST)
        assert c.status == FAIL
        assert "DIFFERENT scoring code" in c.detail

    def test_no_eval_server_configured_is_a_hard_fail(self):
        c = check_eval_server(None, "sha256:c", "sha256:e", RUNTIME_DIGEST)
        assert c.status == FAIL

    def test_unreachable_eval_server_is_a_hard_fail(self):
        c = check_eval_server(None, "sha256:c", "sha256:e", RUNTIME_DIGEST,
                              error="timeout")
        assert c.status == FAIL

    def test_missing_eval_code_digest_is_a_hard_fail(self):
        """A box whose /health doesn't report eval_code_digest at all (a build old
        enough to predate the field) gives us zero evidence its scoring code matches
        — this must surface, not silently PASS."""
        c = check_eval_server({"consensus_digest": "sha256:c"}, "sha256:c", "sha256:e",
                              RUNTIME_DIGEST)
        assert c.status == FAIL
        assert "could not be verified" in c.detail

    def test_missing_or_drifted_runtime_is_a_hard_fail(self):
        missing = check_eval_server(
            {"consensus_digest": "sha256:c", "eval_code_digest": "sha256:e"},
            "sha256:c", "sha256:e", RUNTIME_DIGEST,
        )
        drifted = check_eval_server(
            _health(runtime="sha256:OLD"), GOOD["consensus_digest"],
            GOOD["eval_code_digest"], RUNTIME_DIGEST,
        )
        incompatible = check_eval_server(
            _health(compatible=False), GOOD["consensus_digest"],
            GOOD["eval_code_digest"], RUNTIME_DIGEST,
        )
        assert missing.status == FAIL
        assert drifted.status == FAIL
        assert incompatible.status == FAIL

    def test_a_custom_name_labels_the_check(self):
        c = check_eval_server(_health(), GOOD["consensus_digest"],
                              GOOD["eval_code_digest"], RUNTIME_DIGEST,
                              name="eval_server[http://a:9000]")
        assert c.name == "eval_server[http://a:9000]"


class TestSeedDigestFormat:
    def test_a_valid_hippius_digest_passes(self):
        assert check_seed("sha256:" + "a" * 64).status == PASS

    def test_a_valid_hf_commit_sha_passes(self):
        assert check_seed("hf:" + "a" * 40).status == PASS

    def test_blank_fails(self):
        c = check_seed("")
        assert c.status == FAIL
        assert "is empty" in c.detail

    def test_malformed_digest_fails(self):
        """A truncated/mistyped pin would otherwise resolve to nothing at genesis
        time — catch it here, not mid-launch."""
        c = check_seed("sha256:tooshort")
        assert c.status == FAIL
        assert "not a recognized digest" in c.detail

    def test_wrong_prefix_fails(self):
        c = check_seed("md5:" + "a" * 32)
        assert c.status == FAIL


class TestWalletAndChain:
    def test_real_wallet_and_permitted_registration_pass(self):
        assert check_wallet(WALLET).status == PASS
        assert check_chain(CHAIN, "leoma").status == PASS

    def test_missing_hotkey_file_blocks_launch(self):
        probe = WalletProbe("val", "missing", False, None, None)
        assert check_wallet(probe).status == FAIL

    def test_wrong_subnet_identity_blocks_launch(self):
        probe = CHAIN._replace(subnet_name="Thirty Spokes")
        assert check_chain(probe, "leoma").status == FAIL

    def test_unregistered_or_unpermitted_hotkey_blocks_launch(self):
        unregistered = CHAIN._replace(uid=None, validator_permit=False)
        unpermitted = CHAIN._replace(validator_permit=False)
        assert check_chain(unregistered, "leoma").status == FAIL
        assert check_chain(unpermitted, "leoma").status == FAIL


class TestEvalServersFleet:
    def test_no_probes_gives_one_failing_check(self):
        checks = check_eval_servers((), "sha256:c", "sha256:e", RUNTIME_DIGEST)
        assert len(checks) == 1
        assert checks[0].name == "eval_server"
        assert checks[0].status == FAIL

    def test_a_single_probe_keeps_the_unlabeled_name(self):
        """Single-server operators shouldn't see a URL-qualified name change under them."""
        probe = EvalServerProbe("http://only:9000", _health(), None)
        checks = check_eval_servers(
            (probe,), GOOD["consensus_digest"], GOOD["eval_code_digest"], RUNTIME_DIGEST
        )
        assert len(checks) == 1
        assert checks[0].name == "eval_server"
        assert checks[0].status == PASS

    def test_several_probes_are_each_checked_and_labeled_by_url(self):
        """A stale box among several configured servers must not hide behind a
        healthy sibling — every configured URL gets checked independently."""
        healthy = EvalServerProbe("http://a:9000", _health(), None)
        stale = EvalServerProbe("http://b:9000", _health(consensus="sha256:OLD"), None)
        checks = check_eval_servers(
            (healthy, stale), GOOD["consensus_digest"], GOOD["eval_code_digest"],
            RUNTIME_DIGEST,
        )
        assert len(checks) == 2
        by_name = {c.name: c for c in checks}
        assert by_name["eval_server[http://a:9000]"].status == PASS
        assert by_name["eval_server[http://b:9000]"].status == FAIL

    def test_every_server_stale_fails_the_whole_fleet_check(self):
        stale_a = EvalServerProbe("http://a:9000", {"consensus_digest": "sha256:OLD"}, None)
        stale_b = EvalServerProbe("http://b:9000", None, "connection refused")
        checks = check_eval_servers(
            (stale_a, stale_b), "sha256:c", "sha256:e", RUNTIME_DIGEST
        )
        assert all(c.status != PASS for c in checks)
