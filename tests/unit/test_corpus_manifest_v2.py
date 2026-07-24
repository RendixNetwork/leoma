"""Consensus tests for the sharded caption + first-frame corpus-v2 evaluator.

The root digest is not enough by itself. A valid v2 exam requires this entire
chain to remain intact:

    pinned root -> pinned shard -> pinned MP4 + PNG -> decoded truth/frame hashes

Every failure below is local corpus/infrastructure failure, never a miner fault,
and no selected item may be skipped or substituted.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from leoma.app.validator.seeds import select_clip_indices
from leoma.eval.dataset import (
    build_duel_clips,
    corpus_audit,
    fetch_manifest,
    select_manifest_entries,
)
from leoma.eval.digests import canonical_json, digest_file, digest_frames, sha256_bytes
from leoma.eval.errors import ConsensusConfigError, CorpusIntegrityError, TransientDuelError
from leoma.eval.manifest import (
    CorpusManifestV2,
    SampleEntryV2,
    load_pinned_manifest,
    load_pinned_manifest_v2_shard,
)
from leoma.eval.spec import CorpusSpec
from leoma.eval.video_runner import GenParams
from leoma.infra.corpus_manifest import verify_manifest
from leoma.infra.corpus_v2 import (
    CAPTION_PROMPT,
    CAPTION_PROMPT_VERSION,
    PilotLedger,
    PilotSpec,
    PreparedSample,
    build_sharded_manifest,
    publish_captioned_samples,
)


BUCKET = "videos"
DECODE = {"width": 8, "height": 8, "fps": 2, "num_frames": 4}
GEN = GenParams(num_frames=4, fps=2, width=8, height=8)
CAPTION_MODEL = "org/caption-model"
CAPTION_REVISION = "f" * 40


def _truth(index: int) -> np.ndarray:
    rng = np.random.default_rng(index)
    return rng.integers(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)


def _png(frame: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(out, format="PNG")
    return out.getvalue()


@dataclass
class _Bundle:
    manifest_key: str
    root_raw: bytes
    root_digest: str
    objects: dict[str, bytes]
    truths_by_clip_bytes: dict[bytes, np.ndarray]
    samples: list[dict]
    shard_raws: list[bytes]

    @property
    def corpus(self) -> CorpusSpec:
        return CorpusSpec(
            bucket=BUCKET,
            manifest_key=self.manifest_key,
            manifest_digest=self.root_digest,
        )


def _bundle(
    *,
    count: int = 80,
    shard_size: int = 20,
    conditioning_mismatch: bool = False,
    mutate_sample=None,
) -> _Bundle:
    base = "corpus-v2/prod"
    objects: dict[str, bytes] = {}
    truths_by_clip_bytes: dict[bytes, np.ndarray] = {}
    samples = []
    for index in range(count):
        sample_id = f"{index:064x}"
        truth = _truth(index)
        clip_bytes = f"clip-{index:04d}".encode()
        clip_sha = sha256_bytes(clip_bytes)
        frame = _truth(index + 10_000)[0] if conditioning_mismatch else truth[0]
        frame_bytes = _png(frame)
        frame_sha = sha256_bytes(frame_bytes)
        clip_key = f"{base}/clips/{clip_sha[7:9]}/{clip_sha[7:]}.mp4"
        frame_key = f"{base}/first-frames/{frame_sha[7:9]}/{frame_sha[7:]}.png"
        sample = {
            "sample_id": sample_id,
            "source_key": f"raw/source-{index:04d}.mp4",
            "source_sha256": sha256_bytes(f"source-{index}".encode()),
            "source_start_ms": index * 10_000,
            "clip_key": clip_key,
            "clip_sha256": clip_sha,
            "first_frame_key": frame_key,
            "first_frame_sha256": frame_sha,
            "first_frame_rgb_sha256": digest_frames(frame[None, ...]),
            "truth_frames_sha256": digest_frames(truth),
            "caption": (
                f"A subject number {index} moves through the scene while the camera follows."
            ),
            "caption_model": CAPTION_MODEL,
            "caption_revision": CAPTION_REVISION,
            "caption_prompt_version": CAPTION_PROMPT_VERSION,
            "caption_frame_count": 16,
            "motion_energy": float(index + 1),
            "decode": dict(DECODE),
        }
        if mutate_sample is not None:
            mutate_sample(sample, index)
        samples.append(sample)
        objects[clip_key] = clip_bytes
        objects[frame_key] = frame_bytes
        truths_by_clip_bytes[clip_bytes] = truth

    manifest_dir = f"{base}/manifests/test-v2"
    refs = []
    shard_raws = []
    for shard_index, start in enumerate(range(0, count, shard_size)):
        batch = samples[start:start + shard_size]
        raw = canonical_json(
            {
                "manifest_version": 2,
                "corpus_id": "test-v2",
                "shard_index": shard_index,
                "samples": batch,
            }
        )
        digest = sha256_bytes(raw)
        key = f"shard-{shard_index:06d}-{digest[7:]}.json"
        refs.append(
            {
                "key": key,
                "sha256": digest,
                "count": len(batch),
                "first_sample_id": batch[0]["sample_id"],
                "last_sample_id": batch[-1]["sample_id"],
            }
        )
        objects[f"{manifest_dir}/{key}"] = raw
        shard_raws.append(raw)

    root_raw = canonical_json(
        {
            "manifest_version": 2,
            "corpus_id": "test-v2",
            "sample_count": count,
            "decode": dict(DECODE),
            "caption_prompt_version": CAPTION_PROMPT_VERSION,
            "caption_prompt": CAPTION_PROMPT,
            "caption_build": {
                "model": CAPTION_MODEL,
                "revision": CAPTION_REVISION,
                "prompt_version": CAPTION_PROMPT_VERSION,
                "frame_count": 16,
                "max_new_tokens": 96,
            },
            "quality_assurance": {
                "reviewed": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": None,
            },
            "shards": refs,
        }
    )
    root_digest = sha256_bytes(root_raw)
    manifest_key = f"{manifest_dir}/root-{root_digest[7:]}.json"
    objects[manifest_key] = root_raw
    return _Bundle(
        manifest_key=manifest_key,
        root_raw=root_raw,
        root_digest=root_digest,
        objects=objects,
        truths_by_clip_bytes=truths_by_clip_bytes,
        samples=samples,
        shard_raws=shard_raws,
    )


class _Response:
    def __init__(self, raw: bytes):
        self.raw = raw

    def read(self):
        return self.raw

    def close(self):
        pass

    def release_conn(self):
        pass


class _Store:
    def __init__(self, bundle: _Bundle):
        self.objects = dict(bundle.objects)
        self.gets: list[str] = []
        self.downloads: list[str] = []
        self.fail_get: set[str] = set()
        self.fail_download: set[str] = set()

    def get_object(self, bucket, key):
        self.gets.append(key)
        if key in self.fail_get:
            raise RuntimeError("connection reset")
        try:
            return _Response(self.objects[key])
        except KeyError as exc:
            raise RuntimeError("NoSuchKey") from exc

    def fget_object(self, bucket, key, path):
        self.downloads.append(key)
        if key in self.fail_download:
            raise RuntimeError("connection reset")
        try:
            Path(path).write_bytes(self.objects[key])
        except KeyError as exc:
            raise RuntimeError("NoSuchKey") from exc


class _MissingObject(Exception):
    code = "NoSuchKey"


class _PublishingStore:
    def __init__(self):
        self.objects: dict[tuple[str, str], tuple[bytes, dict]] = {}

    def stat_object(self, bucket, key):
        try:
            raw, metadata = self.objects[(bucket, key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return SimpleNamespace(size=len(raw), metadata=metadata)

    def fput_object(self, bucket, key, path, *, content_type, metadata):
        normalized = {
            f"x-amz-meta-{name}": value for name, value in metadata.items()
        }
        self.objects[(bucket, key)] = (Path(path).read_bytes(), normalized)


def _install_decode(monkeypatch, bundle: _Bundle, *, wrong_for: set[bytes] = frozenset()):
    import leoma.infra.video_utils as video_utils

    def decode(path, **kwargs):
        clip_bytes = Path(path).read_bytes()
        if clip_bytes in wrong_for:
            return _truth(999_999)
        return bundle.truths_by_clip_bytes[clip_bytes]

    monkeypatch.setattr(video_utils, "decode_frames_rgb", decode)


def _load(bundle: _Bundle, store: _Store) -> CorpusManifestV2:
    manifest = fetch_manifest(store, bundle.corpus)
    assert isinstance(manifest, CorpusManifestV2)
    return manifest


class TestPinnedRoot:
    def test_offline_builder_output_is_accepted_by_the_evaluator_contract(self, tmp_path):
        spec = PilotSpec(
            corpus_id="builder-contract-v2",
            width=8,
            height=8,
            fps=2,
            num_frames=4,
            max_windows_per_source=1,
            min_window_gap_seconds=2.0,
        )
        ledger = PilotLedger(tmp_path / "pilot.sqlite3", spec)
        ledger.bind_caption_spec(
            model=CAPTION_MODEL,
            revision=CAPTION_REVISION,
            frame_count=16,
            max_new_tokens=96,
        )
        truth = _truth(123)
        clip_path = tmp_path / "truth.mp4"
        frame_path = tmp_path / "first-frame.png"
        clip_path.write_bytes(b"builder-contract-clip")
        frame_path.write_bytes(_png(truth[0]))
        sample = PreparedSample(
            sample_id="a" * 64,
            source_key="raw/source.mp4",
            source_sha256=sha256_bytes(b"source"),
            source_start_ms=0,
            clip_path=str(clip_path),
            clip_sha256=digest_file(clip_path),
            truth_frames_sha256=digest_frames(truth),
            first_frame_path=str(frame_path),
            first_frame_file_sha256=digest_file(frame_path),
            first_frame_sha256=digest_frames(truth[:1]),
            motion_energy=3.5,
            width=8,
            height=8,
            fps=2,
            num_frames=4,
        )
        assert ledger.add_sample(sample)
        task = ledger.claim_caption("gpu-0")
        ledger.finish_caption(
            task.sample_id,
            caption="A person walks across a room while the camera follows behind.",
            model=CAPTION_MODEL,
            revision=CAPTION_REVISION,
            worker="gpu-0",
            frame_count=16,
        )
        publish_captioned_samples(
            _PublishingStore(),
            BUCKET,
            ledger=ledger,
            prefix="corpus-v2/prod",
        )
        root_path, root_digest = build_sharded_manifest(
            ledger,
            output_dir=tmp_path / "manifest",
            shard_size=1,
        )
        manifest = load_pinned_manifest(
            Path(root_path).read_bytes(),
            root_digest,
            manifest_key=f"corpus-v2/prod/manifests/{spec.corpus_id}/{Path(root_path).name}",
        )
        assert isinstance(manifest, CorpusManifestV2)
        shard = manifest.shards[0]
        entries = load_pinned_manifest_v2_shard(
            (Path(root_path).parent / shard.key).read_bytes(),
            manifest=manifest,
            shard=shard,
        )
        assert entries[0].sample_id == sample.sample_id
        assert entries[0].caption_model == CAPTION_MODEL
        assert entries[0].clip_key.endswith(f"/{sample.clip_sha256[7:]}.mp4")

    def test_matching_root_loads_without_fetching_a_shard(self):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)

        assert manifest.corpus_id == "test-v2"
        assert len(manifest) == 80
        assert manifest.source_digest == bundle.root_digest
        assert store.gets == [bundle.manifest_key]

    def test_root_is_hashed_before_any_tampered_field_is_parsed(self):
        bundle = _bundle()
        tampered = json.loads(bundle.root_raw)
        tampered["sample_count"] = 999_999_999
        raw = canonical_json(tampered)

        with pytest.raises(CorpusIntegrityError, match="digest mismatch"):
            load_pinned_manifest(
                raw,
                bundle.root_digest,
                manifest_key=bundle.manifest_key,
            )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda root: root.update(sample_count=79), "shard counts sum"),
            (
                lambda root: root["shards"][1].update(
                    first_sample_id=root["shards"][0]["last_sample_id"]
                ),
                "overlaps",
            ),
            (
                lambda root: root["shards"][0].update(key="../shard.json"),
                "safe bucket object key",
            ),
            (lambda root: root.update(extra_field=True), "unknown"),
            (
                lambda root: root["caption_build"].update(revision="main"),
                "immutable",
            ),
        ],
    )
    def test_malformed_but_correctly_pinned_root_fails_closed(self, mutate, message):
        bundle = _bundle()
        root = json.loads(bundle.root_raw)
        mutate(root)
        raw = canonical_json(root)
        with pytest.raises(CorpusIntegrityError, match=message):
            load_pinned_manifest(
                raw,
                sha256_bytes(raw),
                manifest_key=bundle.manifest_key,
            )


class TestPinnedShards:
    def test_selection_fetches_only_shards_containing_selected_indices(self):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)
        indices = select_clip_indices(7, len(manifest), 4)

        entries = select_manifest_entries(
            manifest,
            client=store,
            bucket=BUCKET,
            indices=indices,
        )

        expected_shards = {index // 20 for index in indices}
        fetched_shards = {
            int(Path(key).name.split("-")[1])
            for key in store.gets[1:]
        }
        assert fetched_shards == expected_shards
        assert [entry.sample_id for entry in entries] == [
            f"{index:064x}" for index in indices
        ]

    def test_shard_bytes_are_hashed_before_json_is_parsed(self):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)
        shard = manifest.shards[0]
        key = f"{Path(bundle.manifest_key).parent.as_posix()}/{shard.key}"
        store.objects[key] += b" "

        with pytest.raises(CorpusIntegrityError, match="shard digest mismatch"):
            select_manifest_entries(
                manifest,
                client=store,
                bucket=BUCKET,
                indices=[0],
            )

    def test_unreachable_selected_shard_is_transient(self):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)
        key = f"{Path(bundle.manifest_key).parent.as_posix()}/{manifest.shards[0].key}"
        store.fail_get.add(key)

        with pytest.raises(TransientDuelError, match="corpus-v2 shard 0"):
            select_manifest_entries(
                manifest,
                client=store,
                bucket=BUCKET,
                indices=[0],
            )

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda shard: shard.update(corpus_id="other"), "different corpus_id"),
            (lambda shard: shard.update(shard_index=99), "wrong shard_index"),
            (
                lambda shard: shard["samples"][0].update(
                    caption_model="other/model"
                ),
                "caption settings",
            ),
            (
                lambda shard: shard["samples"][0]["decode"].update(fps=99),
                "does not match",
            ),
            (
                lambda shard: shard["samples"][0].update(
                    clip_key="corpus-v2/prod/clips/not-content-addressed.mp4"
                ),
                "content-addressed",
            ),
            (
                lambda shard: shard["samples"].reverse(),
                "unique and sorted",
            ),
        ],
    )
    def test_semantically_invalid_shard_fails_even_when_its_hash_matches(
        self, mutate, message
    ):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)
        shard_doc = json.loads(bundle.shard_raws[0])
        mutate(shard_doc)
        raw = canonical_json(shard_doc)
        ref = replace(manifest.shards[0], sha256=sha256_bytes(raw))

        with pytest.raises(CorpusIntegrityError, match=message):
            load_pinned_manifest_v2_shard(
                raw,
                manifest=manifest,
                shard=ref,
            )


class TestArtifactVerification:
    def test_builds_captioned_ti2v_clips_and_audits_selected_shards(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)

        clips, entries = build_duel_clips(
            manifest,
            client=store,
            bucket=BUCKET,
            master_seed=7,
            n_clips=4,
            gen=GEN,
            prompt_mode="manifest",
            fixed_prompt="",
        )

        assert len(clips) == 4
        assert all(isinstance(entry, SampleEntryV2) for entry in entries)
        assert [clip.prompt for clip in clips] == [entry.caption for entry in entries]
        assert all(np.array_equal(clip.first_frame, clip.truth_frames[0]) for clip in clips)
        audit = corpus_audit(manifest, entries)
        assert audit["manifest_version"] == 2
        assert audit["manifest_digest"] == bundle.root_digest
        assert audit["clip_ids"] == [entry.sample_id for entry in entries]
        assert {item["shard_index"] for item in audit["selected_shards"]} == {
            entry.shard_index for entry in entries
        }

    def test_fixed_prompt_still_overrides_the_pinned_caption(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)

        clips, _ = build_duel_clips(
            manifest,
            client=store,
            bucket=BUCKET,
            master_seed=7,
            n_clips=4,
            gen=GEN,
            prompt_mode="fixed",
            fixed_prompt="the consensus-pinned fixed prompt",
        )
        assert {clip.prompt for clip in clips} == {"the consensus-pinned fixed prompt"}

    def test_tampered_truth_clip_aborts_instead_of_skipping(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)
        selected = select_manifest_entries(
            manifest,
            client=store,
            bucket=BUCKET,
            indices=select_clip_indices(7, len(manifest), 4),
        )
        store.objects[selected[0].clip_key] += b"tampered"

        with pytest.raises(CorpusIntegrityError, match="truth clip.*does not match"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=GEN,
                prompt_mode="manifest",
                fixed_prompt="",
            )

    def test_tampered_png_aborts_instead_of_using_truth_frame_silently(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)
        selected = select_manifest_entries(
            manifest,
            client=store,
            bucket=BUCKET,
            indices=select_clip_indices(7, len(manifest), 4),
        )
        store.objects[selected[0].first_frame_key] += b"tampered"

        with pytest.raises(CorpusIntegrityError, match="first-frame PNG.*does not match"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=GEN,
                prompt_mode="manifest",
                fixed_prompt="",
            )

    def test_different_ffmpeg_decode_is_a_corpus_failure(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)
        selected = select_manifest_entries(
            manifest,
            client=store,
            bucket=BUCKET,
            indices=select_clip_indices(7, len(manifest), 4),
        )
        wrong = {store.objects[selected[0].clip_key]}
        _install_decode(monkeypatch, bundle, wrong_for=wrong)

        with pytest.raises(CorpusIntegrityError, match="decoded truth clip"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=GEN,
                prompt_mode="manifest",
                fixed_prompt="",
            )

    def test_png_must_be_the_truth_clips_exact_frame_zero(self, monkeypatch):
        bundle = _bundle(conditioning_mismatch=True)
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)

        with pytest.raises(CorpusIntegrityError, match="not truth clip frame zero"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=GEN,
                prompt_mode="manifest",
                fixed_prompt="",
            )

    def test_artifact_download_failure_is_transient(self, monkeypatch):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)
        selected = select_manifest_entries(
            manifest,
            client=store,
            bucket=BUCKET,
            indices=select_clip_indices(7, len(manifest), 4),
        )
        store.fail_download.add(selected[0].first_frame_key)

        with pytest.raises(TransientDuelError, match="could not fetch first-frame PNG"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=GEN,
                prompt_mode="manifest",
                fixed_prompt="",
            )

    def test_operator_verification_supports_v2_without_loading_every_shard(
        self, monkeypatch
    ):
        bundle = _bundle()
        store = _Store(bundle)
        _install_decode(monkeypatch, bundle)
        manifest = _load(bundle, store)

        assert verify_manifest(store, BUCKET, manifest, sample=3) == 3
        shard_gets = [key for key in store.gets if "/shard-" in key]
        assert len(shard_gets) == 1

    def test_decode_mismatch_is_rejected_before_artifact_download(self):
        bundle = _bundle()
        store = _Store(bundle)
        manifest = _load(bundle, store)

        with pytest.raises(ConsensusConfigError, match="decode parameters"):
            build_duel_clips(
                manifest,
                client=store,
                bucket=BUCKET,
                master_seed=7,
                n_clips=4,
                gen=replace(GEN, fps=99),
                prompt_mode="manifest",
                fixed_prompt="",
            )
        assert store.downloads == []
