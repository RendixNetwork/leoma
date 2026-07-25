"""
Unit tests for the Hippius Hub model store.

Covers the content-addressed model reference (``ModelRef``), the on-chain reveal
serialisation/parsing (``build_reveal_v4`` / ``parse_reveal_v4``), and the digest
regex — the pieces exercised by the miner and the validator's on-chain scan.
"""

import sys
from types import SimpleNamespace

import pytest

import leoma.infra.model_store as ms
from leoma.infra.model_store import (
    DIGEST_RE,
    HUB_PASSWORD_ENV_NAMES,
    HUB_TOKEN_ENV_NAMES,
    HUB_USERNAME_ENV_NAMES,
    HippiusHubAuthError,
    ModelRef,
    _call_snapshot_download,
    _prepare_upload_token,
    build_reveal_v4,
    parse_reveal_v4,
)

# A realistic 48-char ss58 hotkey and two digest shapes.
HOTKEY = "5C7LM2i42XgL2oB4x3rcmB7KDiof4B92KZzUpg5miZ6DogjU"
SHA256_DIGEST = "sha256:" + "a" * 64
HF_DIGEST = "hf:" + "b" * 40
REPO = f"user/leoma-mymodel-{HOTKEY}"


def _clear_hub_auth(monkeypatch, tmp_path):
    for name in (*HUB_TOKEN_ENV_NAMES, *HUB_USERNAME_ENV_NAMES, *HUB_PASSWORD_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ms, "HUB_TOKEN_PATH", tmp_path / "missing-token")


class TestPublicHubReads:
    def test_public_model_download_is_anonymous_without_operator_auth(
        self, monkeypatch, tmp_path
    ):
        _clear_hub_auth(monkeypatch, tmp_path)
        captured = {}

        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return kwargs["local_dir"]

        monkeypatch.setitem(
            sys.modules,
            "hippius_hub",
            SimpleNamespace(snapshot_download=snapshot_download),
        )
        target = tmp_path / "snapshot"

        result = _call_snapshot_download(
            ModelRef(REPO, SHA256_DIGEST),
            str(target),
            2,
        )

        assert result == str(target)
        assert captured["token"] is None
        assert captured["revision"] == SHA256_DIGEST

    def test_operator_read_token_is_used_when_configured(self, monkeypatch, tmp_path):
        _clear_hub_auth(monkeypatch, tmp_path)
        monkeypatch.setenv("HIPPIUS_HUB_TOKEN", "validator-read-token")
        captured = {}

        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return kwargs["local_dir"]

        monkeypatch.setitem(
            sys.modules,
            "hippius_hub",
            SimpleNamespace(snapshot_download=snapshot_download),
        )

        _call_snapshot_download(
            ModelRef(REPO, SHA256_DIGEST),
            str(tmp_path / "snapshot"),
            2,
        )

        assert captured["token"] == "validator-read-token"

    def test_upload_still_requires_authentication(self, monkeypatch, tmp_path):
        _clear_hub_auth(monkeypatch, tmp_path)

        with pytest.raises(HippiusHubAuthError, match="Uploading public model requires"):
            _prepare_upload_token("Uploading public model")


class TestDigestRegex:
    """The two accepted digest shapes."""

    def test_sha256_64hex_ok(self):
        assert DIGEST_RE.match(SHA256_DIGEST)

    def test_hf_40hex_ok(self):
        assert DIGEST_RE.match(HF_DIGEST)

    @pytest.mark.parametrize(
        "bad",
        [
            "sha256:" + "a" * 63,     # too short
            "sha256:" + "a" * 65,     # too long
            "sha256:" + "A" * 64,     # uppercase not allowed
            "hf:" + "b" * 39,         # hf too short
            "sha1:" + "a" * 64,       # wrong prefix
            "a" * 64,                 # no prefix
            "sha256:zzzz",            # non-hex
            "",
        ],
    )
    def test_rejects_bad_digests(self, bad):
        assert not DIGEST_RE.match(bad)


class TestModelRef:
    """Construction + validation of the immutable model reference."""

    def test_valid_sha256(self):
        ref = ModelRef(REPO, SHA256_DIGEST)
        assert ref.repo == REPO
        assert ref.digest == SHA256_DIGEST
        assert ref.immutable_ref == f"{REPO}@{SHA256_DIGEST}"

    def test_valid_hf(self):
        ref = ModelRef("someorg/leoma-x", HF_DIGEST)
        assert ref.immutable_ref.endswith(HF_DIGEST)

    def test_strips_whitespace(self):
        ref = ModelRef(f"  {REPO}  ", f"  {SHA256_DIGEST}  ")
        assert ref.repo == REPO
        assert ref.digest == SHA256_DIGEST

    def test_frozen(self):
        ref = ModelRef(REPO, SHA256_DIGEST)
        with pytest.raises(Exception):
            ref.repo = "other/repo"  # type: ignore[misc]

    def test_equality(self):
        assert ModelRef(REPO, SHA256_DIGEST) == ModelRef(REPO, SHA256_DIGEST)

    @pytest.mark.parametrize("bad_repo", ["", "no-slash", "/leading", "trailing/", " /x"])
    def test_invalid_repo_raises(self, bad_repo):
        with pytest.raises(ValueError):
            ModelRef(bad_repo, SHA256_DIGEST)

    @pytest.mark.parametrize("bad_digest", ["", "sha256:xyz", "deadbeef", "sha256:" + "a" * 10])
    def test_invalid_digest_raises(self, bad_digest):
        with pytest.raises(ValueError):
            ModelRef(REPO, bad_digest)


class TestRevealV4:
    """Round-trip of the on-chain reveal string."""

    def test_round_trip(self):
        ref = ModelRef(REPO, SHA256_DIGEST)
        payload = build_reveal_v4(ref, HOTKEY)
        assert payload == f"v4|{REPO}|{SHA256_DIGEST}|{HOTKEY}"
        parsed_ref, author = parse_reveal_v4(payload)
        assert parsed_ref == ref
        assert author == HOTKEY

    def test_round_trip_hf_digest(self):
        ref = ModelRef("org/leoma-x", HF_DIGEST)
        parsed_ref, author = parse_reveal_v4(build_reveal_v4(ref, HOTKEY))
        assert parsed_ref == ref
        assert author == HOTKEY

    def test_build_rejects_bad_hotkey(self):
        with pytest.raises(ValueError):
            build_reveal_v4(ModelRef(REPO, SHA256_DIGEST), "not-a-hotkey")

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "garbage",
            '{"model_name": "user/leoma-old", "model_revision": "x", "endpoint_id": "y"}',  # legacy JSON
            f"v3|{SHA256_DIGEST}|{REPO}|{SHA256_DIGEST}|{HOTKEY}",  # legacy v3 shape
            f"v4|{REPO}|{SHA256_DIGEST}",                            # too few parts
            f"v4|{REPO}|{SHA256_DIGEST}|{HOTKEY}|extra",             # too many parts
            f"v4|{REPO}|not-a-digest|{HOTKEY}",                      # bad digest
            f"v4|{REPO}|{SHA256_DIGEST}|not-a-hotkey",               # bad hotkey
            f"v4|bad_repo_no_slash|{SHA256_DIGEST}|{HOTKEY}",        # bad repo
        ],
    )
    def test_parse_rejects_malformed(self, payload):
        with pytest.raises(ValueError):
            parse_reveal_v4(payload)
