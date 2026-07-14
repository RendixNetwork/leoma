"""The launch readiness gate.

Every pin the subnet needs fails safe-but-silent: an unpinned seed, an unpinned
corpus, or a stale eval box all make the validator burn to UID 0 rather than crown.
Preflight is the one place that says WHY before an operator flips the switch. These
tests pin the classification of each check and the overall ready/not-ready verdict.
"""

from leoma.app.preflight import (
    FAIL,
    PASS,
    WARN,
    EvalServerProbe,
    check_corpus_reachable,
    check_eval_server,
    check_eval_servers,
    check_seed,
    run_preflight,
)

GOOD = dict(
    seed_digest="sha256:" + "5" * 64,
    corpus_pinned=True,
    manifest_digest="sha256:" + "a" * 64,
    consensus_digest="sha256:" + "c" * 64,
    eval_code_digest="sha256:" + "e" * 64,
    own_bucket="leoma-state",
    wallet_name="val",
    hotkey_name="hk",
)


def _good_probe():
    return EvalServerProbe(
        "http://eval:9000",
        {"consensus_digest": GOOD["consensus_digest"], "eval_code_digest": GOOD["eval_code_digest"]},
        None,
    )


class TestOverallVerdict:
    def test_a_fully_configured_validator_is_ready(self):
        r = run_preflight(**GOOD, corpus_fetched_digest=GOOD["manifest_digest"],
                          eval_servers=(_good_probe(),))
        assert r.ready
        assert r.failures == ()

    def test_an_unpinned_seed_blocks_launch(self):
        r = run_preflight(**{**GOOD, "seed_digest": ""})
        assert not r.ready
        assert any(c.name == "seed_digest" and c.status == FAIL for c in r.checks)

    def test_an_unpinned_corpus_blocks_launch(self):
        r = run_preflight(**{**GOOD, "corpus_pinned": False, "manifest_digest": ""})
        assert not r.ready
        assert any(c.name == "corpus_pin" and c.status == FAIL for c in r.checks)

    def test_a_missing_state_bucket_blocks_launch(self):
        r = run_preflight(**{**GOOD, "own_bucket": None})
        assert not r.ready

    def test_warnings_do_not_block_launch(self):
        # No eval server configured and no corpus fetch -> warnings only, still ready.
        r = run_preflight(**GOOD)
        assert r.ready
        assert len(r.warnings) >= 1  # eval_server + corpus_fetch are warns here


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

    def test_a_fetch_error_is_only_a_warning(self):
        c = check_corpus_reachable(None, "sha256:aaa", "connection refused")
        assert c.status == WARN

    def test_no_credentials_is_a_warning_not_a_block(self):
        c = check_corpus_reachable(None, "sha256:aaa", None)
        assert c.status == WARN


class TestEvalServer:
    def test_matching_box_passes(self):
        c = check_eval_server({"consensus_digest": "sha256:c", "eval_code_digest": "sha256:e"},
                              "sha256:c", "sha256:e")
        assert c.status == PASS

    def test_a_stale_consensus_surface_is_a_hard_fail(self):
        """The single most likely consensus failure — an operator who redeployed some
        boxes but not others. Must block."""
        c = check_eval_server({"consensus_digest": "sha256:OLD", "eval_code_digest": "sha256:e"},
                              "sha256:NEW", "sha256:e")
        assert c.status == FAIL
        assert "DIFFERENT consensus surface" in c.detail

    def test_stale_scoring_code_is_a_hard_fail(self):
        c = check_eval_server({"consensus_digest": "sha256:c", "eval_code_digest": "sha256:OLD"},
                              "sha256:c", "sha256:NEW")
        assert c.status == FAIL
        assert "DIFFERENT scoring code" in c.detail

    def test_no_eval_server_configured_is_a_warning(self):
        c = check_eval_server(None, "sha256:c", "sha256:e")
        assert c.status == WARN

    def test_unreachable_eval_server_is_a_warning(self):
        c = check_eval_server(None, "sha256:c", "sha256:e", error="timeout")
        assert c.status == WARN

    def test_missing_eval_code_digest_is_a_warning_not_a_silent_pass(self):
        """A box whose /health doesn't report eval_code_digest at all (a build old
        enough to predate the field) gives us zero evidence its scoring code matches
        — this must surface, not silently PASS."""
        c = check_eval_server({"consensus_digest": "sha256:c"}, "sha256:c", "sha256:e")
        assert c.status == WARN
        assert "could not be verified" in c.detail

    def test_a_custom_name_labels_the_check(self):
        c = check_eval_server({"consensus_digest": "sha256:c", "eval_code_digest": "sha256:e"},
                              "sha256:c", "sha256:e", name="eval_server[http://a:9000]")
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


class TestEvalServersFleet:
    def test_no_probes_gives_one_warning_check(self):
        checks = check_eval_servers((), "sha256:c", "sha256:e")
        assert len(checks) == 1
        assert checks[0].name == "eval_server"
        assert checks[0].status == WARN

    def test_a_single_probe_keeps_the_unlabeled_name(self):
        """Single-server operators shouldn't see a URL-qualified name change under them."""
        probe = EvalServerProbe("http://only:9000", {"consensus_digest": "sha256:c",
                                                     "eval_code_digest": "sha256:e"}, None)
        checks = check_eval_servers((probe,), "sha256:c", "sha256:e")
        assert len(checks) == 1
        assert checks[0].name == "eval_server"
        assert checks[0].status == PASS

    def test_several_probes_are_each_checked_and_labeled_by_url(self):
        """A stale box among several configured servers must not hide behind a
        healthy sibling — every configured URL gets checked independently."""
        healthy = EvalServerProbe("http://a:9000", {"consensus_digest": "sha256:c",
                                                     "eval_code_digest": "sha256:e"}, None)
        stale = EvalServerProbe("http://b:9000", {"consensus_digest": "sha256:OLD",
                                                   "eval_code_digest": "sha256:e"}, None)
        checks = check_eval_servers((healthy, stale), "sha256:c", "sha256:e")
        assert len(checks) == 2
        by_name = {c.name: c for c in checks}
        assert by_name["eval_server[http://a:9000]"].status == PASS
        assert by_name["eval_server[http://b:9000]"].status == FAIL

    def test_every_server_stale_fails_the_whole_fleet_check(self):
        stale_a = EvalServerProbe("http://a:9000", {"consensus_digest": "sha256:OLD"}, None)
        stale_b = EvalServerProbe("http://b:9000", None, "connection refused")
        checks = check_eval_servers((stale_a, stale_b), "sha256:c", "sha256:e")
        assert all(c.status != PASS for c in checks)
