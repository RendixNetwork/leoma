"""Operator-safety checks for corpus manifest CLI commands."""

import json
from types import SimpleNamespace

from click.testing import CliRunner

import leoma.bootstrap as bootstrap
from leoma.delivery.commands import cli
from leoma.infra.chain_config import SPEC
from leoma.infra import corpus_manifest, storage_backend


def test_build_manifest_refuses_a_runtime_consensus_bucket_mismatch(monkeypatch):
    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", "wrong-bucket")

    result = CliRunner().invoke(
        cli,
        ["corpus", "build-manifest", "--corpus-id", "leoma-testnet-v1"],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "source bucket mismatch" in result.output
    assert SPEC.corpus.bucket in result.output


def test_build_manifest_reads_the_bucket_pinned_by_consensus(monkeypatch, tmp_path):
    seen = {}

    class Client:
        def list_objects(self, bucket, recursive):
            seen["listed"] = (bucket, recursive)
            return [SimpleNamespace(object_name="too-small.mp4", size=1)]

    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", SPEC.corpus.bucket)
    monkeypatch.setattr(storage_backend, "create_source_read_client", Client)
    monkeypatch.setattr(
        corpus_manifest,
        "build_manifest",
        lambda client, bucket, **kwargs: seen.setdefault("built", bucket) and [],
    )
    monkeypatch.setattr("leoma.eval.manifest.check_decode_compat", lambda manifest, gen: None)
    monkeypatch.setattr("leoma.eval.manifest.check_corpus_size", lambda manifest, n_clips: None)
    monkeypatch.setattr(corpus_manifest, "write_manifest", lambda manifest, path: "sha256:test")

    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "build-manifest",
            "--corpus-id",
            "leoma-testnet-v1",
            "--out",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["listed"] == (SPEC.corpus.bucket, True)
    assert seen["built"] == SPEC.corpus.bucket


def test_limited_build_stops_listing_after_the_requested_eligible_count(monkeypatch, tmp_path):
    seen = {"yielded": 0}

    class Client:
        def list_objects(self, bucket, recursive):
            for i in range(1000):
                seen["yielded"] += 1
                yield SimpleNamespace(object_name=f"{i:04d}.mp4", size=2_000_000)

    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", SPEC.corpus.bucket)
    monkeypatch.setattr(storage_backend, "create_source_read_client", Client)
    monkeypatch.setattr(corpus_manifest, "build_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr("leoma.eval.manifest.check_decode_compat", lambda manifest, gen: None)
    monkeypatch.setattr("leoma.eval.manifest.check_corpus_size", lambda manifest, n_clips: None)
    monkeypatch.setattr(corpus_manifest, "write_manifest", lambda manifest, path: "sha256:test")

    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "build-manifest",
            "--corpus-id",
            "leoma-testnet-v1",
            "--limit",
            "3",
            "--out",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["yielded"] == 3


def test_build_manifest_reports_missing_storage_access_without_a_traceback(monkeypatch):
    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", SPEC.corpus.bucket)

    def fail():
        raise ValueError("missing read credentials")

    monkeypatch.setattr(storage_backend, "create_source_read_client", fail)

    result = CliRunner().invoke(
        cli,
        ["corpus", "build-manifest", "--corpus-id", "leoma-testnet-v1"],
    )

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "could not list source bucket" in result.output
    assert "missing read credentials" in result.output


def test_build_manifest_refuses_to_write_an_undersized_exam(monkeypatch, tmp_path):
    class Client:
        def list_objects(self, bucket, recursive):
            return []

    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", SPEC.corpus.bucket)
    monkeypatch.setattr(storage_backend, "create_source_read_client", Client)
    monkeypatch.setattr(corpus_manifest, "build_manifest", lambda *args, **kwargs: [])
    monkeypatch.setattr("leoma.eval.manifest.check_decode_compat", lambda manifest, gen: None)

    def too_small(manifest, n_clips):
        raise ValueError("needs at least 640 clips")

    monkeypatch.setattr("leoma.eval.manifest.check_corpus_size", too_small)
    monkeypatch.setattr(
        corpus_manifest,
        "write_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not write")),
    )

    result = CliRunner().invoke(
        cli,
        [
            "corpus",
            "build-manifest",
            "--corpus-id",
            "leoma-testnet-v1",
            "--out",
            str(tmp_path / "manifest.json"),
        ],
    )

    assert result.exit_code != 0
    assert "built manifest is not duel-ready" in result.output
    assert "needs at least 640 clips" in result.output


def test_publish_manifest_refuses_an_undersized_exam_before_upload(monkeypatch, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{}")
    monkeypatch.setattr(corpus_manifest, "read_manifest", lambda path: [])
    monkeypatch.setattr("leoma.eval.manifest.check_decode_compat", lambda manifest, gen: None)

    def too_small(manifest, n_clips):
        raise ValueError("needs at least 640 clips")

    monkeypatch.setattr("leoma.eval.manifest.check_corpus_size", too_small)
    monkeypatch.setattr(
        storage_backend,
        "create_source_write_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not create an upload client")),
    )

    result = CliRunner().invoke(cli, ["corpus", "publish-manifest", str(path)])

    assert result.exit_code != 0
    assert "could not publish a duel-ready manifest" in result.output
    assert "needs at least 640 clips" in result.output


def test_v2_inventory_reports_that_a_target_exceeds_the_source_cap(monkeypatch):
    class Client:
        def list_objects(self, bucket, recursive):
            return [
                SimpleNamespace(object_name="a.mp4", size=2_000_000),
                SimpleNamespace(object_name="b.mov", size=3_000_000),
                SimpleNamespace(object_name="notes.txt", size=4_000_000),
                SimpleNamespace(object_name="tiny.mp4", size=1),
            ]

    monkeypatch.setattr(bootstrap, "SOURCE_BUCKET", SPEC.corpus.bucket)
    monkeypatch.setattr(storage_backend, "create_source_read_client", Client)
    result = CliRunner().invoke(
        cli,
        [
            "corpus", "v2", "inventory", "--target-samples", "20",
            "--windows-per-source", "8",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["eligible_videos"] == 2
    assert report["theoretical_sample_ceiling"] == 16
    assert report["minimum_accepted_windows_per_source_for_target"] == 10
    assert report["target_possible_at_configured_cap_before_filtering"] is False
