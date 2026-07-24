"""Safety and resumability tests for the large captioned corpus pilot."""

import json
import hashlib
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from leoma.eval.digests import digest_file, digest_frames
from leoma.infra.corpus_v2 import (
    CAPTION_PROMPT_VERSION,
    PilotLedger,
    PilotSpec,
    PreparedSample,
    build_sharded_manifest,
    choose_window_starts,
    export_qa_bundle,
    import_qa_reviews,
    materialize_first_frame_images,
    normalize_caption,
    order_source_keys,
    prepare_bucket_sources,
    prepare_local_source,
    publish_captioned_samples,
    select_qa_samples,
    validate_caption,
)
from leoma.infra.video_caption import validate_model_revision


def _spec(**overrides):
    values = {"corpus_id": "leoma-caption-pilot-v1"}
    values.update(overrides)
    return PilotSpec(**values)


def _sample(tmp_path: Path, sample_id: str = "a" * 64, truth: str = "sha256:" + "b" * 64):
    clip = tmp_path / f"{sample_id}.mp4"
    clip.write_bytes(b"deterministic-test-clip")
    first_frame = tmp_path / f"{sample_id}.png"
    first_frame.write_bytes(b"deterministic-test-first-frame")
    return PreparedSample(
        sample_id=sample_id,
        source_key="raw/source.mp4",
        source_sha256="sha256:" + "1" * 64,
        source_start_ms=1_000,
        clip_path=str(clip),
        clip_sha256=digest_file(clip),
        truth_frames_sha256=truth,
        first_frame_path=str(first_frame),
        first_frame_file_sha256=digest_file(first_frame),
        first_frame_sha256="sha256:" + "3" * 64,
        motion_energy=4.25,
        width=832,
        height=480,
        fps=16,
        num_frames=81,
    )


def _bind_caption(ledger: PilotLedger, *, frame_count: int = 16):
    return ledger.bind_caption_spec(
        model="org/caption-model",
        revision="f" * 40,
        frame_count=frame_count,
        max_new_tokens=96,
    )


def test_wan_window_is_nominal_five_seconds_and_gap_cannot_overlap():
    spec = _spec()
    assert spec.clip_seconds == pytest.approx(5.0625)
    with pytest.raises(ValueError, match="at least one clip"):
        _spec(min_window_gap_seconds=5.0)


def test_window_selection_is_deterministic_capped_and_inside_shots():
    kwargs = dict(
        duration=100.0,
        scene_cuts=[30.0, 60.0],
        source_key="raw/example.mp4",
        clip_seconds=5.0625,
        max_windows=7,
        min_gap_seconds=10.0,
        boundary_margin=0.15,
    )
    first = choose_window_starts(**kwargs)
    assert first == choose_window_starts(**kwargs)
    assert len(first) == 7
    for start in first:
        assert start >= 0
        assert start + kwargs["clip_seconds"] <= kwargs["duration"]
        assert not (start < 30 < start + kwargs["clip_seconds"])
        assert not (start < 60 < start + kwargs["clip_seconds"])


def test_bounded_pilot_source_order_is_seeded_deterministic_and_deduplicated():
    keys = [f"{index:04d}.mp4" for index in range(20)] + ["0001.mp4"]
    first = order_source_keys(keys, seed="pilot-a")
    assert first == order_source_keys(reversed(keys), seed="pilot-a")
    assert len(first) == 20
    assert first != order_source_keys(keys, seed="pilot-b")


def test_parallel_preparation_keeps_the_same_capped_source_order(monkeypatch, tmp_path):
    from leoma.infra import corpus_v2

    class Client:
        def fget_object(self, bucket, key, path):
            Path(path).write_bytes(key.encode())

    def fake_prepare(source_path, *, source_key, workdir, spec):
        # Finish in reverse order to prove completion timing does not choose the
        # capped subset.
        time.sleep({"source-a": 0.03, "source-b": 0.02, "source-c": 0.01}[source_key])
        sample_id = hashlib.sha256(source_key.encode()).hexdigest()
        clip = Path(workdir) / "clips" / sample_id[:2] / f"{sample_id}.mp4"
        frame = Path(workdir) / "first-frames" / sample_id[:2] / f"{sample_id}.png"
        clip.parent.mkdir(parents=True, exist_ok=True)
        frame.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(f"clip:{source_key}".encode())
        frame.write_bytes(f"frame:{source_key}".encode())
        suffix = hashlib.sha256(f"truth:{source_key}".encode()).hexdigest()
        source_suffix = hashlib.sha256(f"source:{source_key}".encode()).hexdigest()
        return [PreparedSample(
            sample_id=sample_id,
            source_key=source_key,
            source_sha256=f"sha256:{source_suffix}",
            source_start_ms=0,
            clip_path=str(clip),
            clip_sha256=digest_file(clip),
            truth_frames_sha256=f"sha256:{suffix}",
            first_frame_path=str(frame),
            first_frame_file_sha256=digest_file(frame),
            first_frame_sha256=f"sha256:{suffix}",
            motion_energy=2.0,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            num_frames=spec.num_frames,
        )]

    monkeypatch.setattr(corpus_v2, "prepare_local_source", fake_prepare)
    selected = []
    for workers in (1, 3):
        workdir = tmp_path / f"workers-{workers}"
        ledger = PilotLedger(workdir / "pilot.sqlite3", _spec())
        prepare_bucket_sources(
            Client(),
            "bucket",
            ledger=ledger,
            workdir=workdir,
            spec=ledger.read_spec(),
            keys=["source-a", "source-b", "source-c"],
            max_samples=2,
            workers=workers,
        )
        with ledger.connect() as db:
            selected.append([
                row[0] for row in db.execute(
                    "SELECT source_key FROM samples ORDER BY rowid"
                )
            ])
    assert selected == [["source-a", "source-b"], ["source-a", "source-b"]]


def test_source_resume_retries_transport_errors_but_skips_terminal_rejections(
    monkeypatch, tmp_path
):
    from collections import Counter
    from leoma.infra import corpus_v2

    attempts = Counter()

    class Client:
        def fget_object(self, bucket, key, path):
            attempts[key] += 1
            if key == "transport.mp4" and attempts[key] == 1:
                raise TimeoutError("source read timed out")
            Path(path).write_bytes(key.encode())

    def fake_prepare(source_path, *, source_key, workdir, spec):
        if source_key == "terminal.mp4":
            raise ValueError("decoded 80 frames, need 81")
        return []

    monkeypatch.setattr(corpus_v2, "prepare_local_source", fake_prepare)
    workdir = tmp_path / "resume"
    ledger = PilotLedger(workdir / "pilot.sqlite3", _spec())
    kwargs = dict(
        client=Client(),
        bucket="bucket",
        ledger=ledger,
        workdir=workdir,
        spec=ledger.read_spec(),
        keys=["transport.mp4", "terminal.mp4"],
        max_samples=10,
        workers=2,
    )

    prepare_bucket_sources(**kwargs)
    with ledger.connect() as db:
        first = dict(db.execute("SELECT source_key,status FROM sources"))
    assert first == {
        "transport.mp4": "retryable",
        "terminal.mp4": "failed",
    }

    prepare_bucket_sources(**kwargs)
    with ledger.connect() as db:
        second = dict(db.execute("SELECT source_key,status FROM sources"))
    assert second == {
        "transport.mp4": "done",
        "terminal.mp4": "failed",
    }
    assert attempts == {"transport.mp4": 2, "terminal.mp4": 1}


def test_ledger_is_bound_to_one_immutable_spec(tmp_path):
    path = tmp_path / "pilot.sqlite3"
    PilotLedger(path, _spec())
    PilotLedger(path, _spec())
    with pytest.raises(ValueError, match="different pilot settings"):
        PilotLedger(path, _spec(max_windows_per_source=4))


def test_caption_claim_finish_and_recovery_are_transactional(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger)
    sample = _sample(tmp_path)
    assert ledger.add_sample(sample)
    assert not ledger.add_sample(sample)

    task = ledger.claim_caption("gpu-0")
    assert task and task.sample_id == sample.sample_id
    assert ledger.claim_caption("gpu-1") is None
    assert ledger.recover_caption_claims() == 1
    task = ledger.claim_caption("gpu-1")
    assert task and task.sample_id == sample.sample_id
    ledger.finish_caption(
        sample.sample_id,
        caption='  "A person walks across a room as the camera follows."  ',
        model="org/caption-model",
        revision="f" * 40,
        worker="gpu-1",
        frame_count=16,
    )
    assert ledger.stats()["samples"] == {"captioned": 1}
    row = next(ledger.captioned())
    assert row["caption"] == "A person walks across a room as the camera follows."
    assert row["caption_prompt_version"] == CAPTION_PROMPT_VERSION


def test_following_caption_worker_waits_for_a_growing_prepare_queue(
    monkeypatch, tmp_path,
):
    from leoma.infra import video_caption

    class Captioner:
        def __init__(self, *args, **kwargs):
            pass

        def caption(self, clip_path):
            return "A person walks through a room while the camera follows behind."

    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger)
    sample = _sample(tmp_path)

    def add_later():
        time.sleep(0.02)
        ledger.add_sample(sample)

    monkeypatch.setattr(video_caption, "TransformersVideoCaptioner", Captioner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    producer = threading.Thread(target=add_later)
    producer.start()
    video_caption._caption_worker(
        ledger_path=ledger.path,
        gpu="0",
        model="org/caption-model",
        revision="f" * 40,
        frame_count=16,
        max_new_tokens=96,
        max_attempts=3,
        follow=True,
        poll_seconds=0.005,
        idle_timeout_seconds=0.05,
    )
    producer.join()
    assert ledger.stats()["samples"] == {"captioned": 1}


def test_terminal_caption_failures_require_explicit_operator_retry(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    sample = _sample(tmp_path)
    ledger.add_sample(sample)
    for attempt in range(3):
        task = ledger.claim_caption("gpu-0", max_attempts=3)
        assert task is not None
        ledger.fail_caption(task.sample_id, "missing video decoder", max_attempts=3)
    assert ledger.caption_queue_size(max_attempts=3) == 0
    assert ledger.stats()["samples"] == {"failed": 1}
    assert ledger.retry_failed_captions() == 1
    assert ledger.caption_queue_size(max_attempts=3) == 1
    assert ledger.stats()["samples"] == {"prepared": 1}


def test_caption_settings_cannot_mix_without_guarded_full_reset(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger, frame_count=16)
    with pytest.raises(ValueError, match="different caption"):
        _bind_caption(ledger, frame_count=8)
    sample = _sample(tmp_path)
    ledger.add_sample(sample)
    task = ledger.claim_caption("gpu-0")
    with pytest.raises(ValueError, match="do not match"):
        ledger.finish_caption(
            task.sample_id,
            caption="A person walks through a room while the camera follows.",
            model="org/caption-model",
            revision="f" * 40,
            worker="gpu-0",
            frame_count=8,
        )
    ledger.recover_caption_claims()
    assert ledger.reset_all_captions() == 1
    _bind_caption(ledger, frame_count=8)
    assert ledger.read_caption_spec()["frame_count"] == 8


def test_exact_truth_duplicates_are_not_added_twice(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    one = _sample(tmp_path, "a" * 64, truth="sha256:" + "9" * 64)
    two = PreparedSample(
        **{
            **one.__dict__,
            "sample_id": "c" * 64,
            "source_key": "raw/copy.mp4",
            "source_sha256": "sha256:" + "2" * 64,
            "source_start_ms": 2_000,
        }
    )
    assert ledger.add_sample(one)
    assert not ledger.add_sample(two)
    assert ledger.stats()["total_samples"] == 1


def test_same_content_under_another_source_key_is_a_safe_alias(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    original = _sample(tmp_path)
    alias = PreparedSample(**{
        **original.__dict__,
        "source_key": "raw/identical-object-under-another-key.mp4",
    })
    assert ledger.add_sample(original)
    assert not ledger.add_sample(alias)
    with ledger.connect() as db:
        stored = db.execute(
            "SELECT source_key FROM samples WHERE sample_id=?",
            (original.sample_id,),
        ).fetchone()
    assert stored[0] == original.source_key

    changed_content = PreparedSample(**{
        **alias.__dict__,
        "clip_sha256": "sha256:" + "f" * 64,
    })
    with pytest.raises(ValueError, match="sample id collision"):
        ledger.add_sample(changed_content)


def test_caption_policy_rejects_mutable_models_and_boilerplate():
    validate_model_revision("a" * 40)
    for revision in ("main", "v1", "a" * 39, "sha256:" + "a" * 64):
        with pytest.raises(ValueError, match="immutable"):
            validate_model_revision(revision)
    assert normalize_caption("  A   dog runs. ") == "A dog runs."
    with pytest.raises(ValueError, match="too short"):
        validate_caption("A dog.")
    with pytest.raises(ValueError, match="boilerplate"):
        validate_caption("As an AI, I cannot inspect this video clip.")
    # A raw substring check mistakes the start of "airplane" for "AI".
    validate_caption(
        "Raindrops fall from a horizontal bar against a cloudy sky as an airplane "
        "flies past in the distance."
    )


def test_manifest_is_sharded_and_root_digest_is_reproducible(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger)
    sample = _sample(tmp_path)
    ledger.add_sample(sample)
    ledger.claim_caption("gpu-0")
    ledger.finish_caption(
        sample.sample_id,
        caption="A person walks across a room as the camera follows.",
        model="org/caption-model",
        revision="f" * 40,
        worker="gpu-0",
        frame_count=16,
    )
    ledger.mark_published(
        sample.sample_id,
        "corpus-v2/pilot/clips/aa/clip.mp4",
        "corpus-v2/pilot/first-frames/bb/frame.png",
    )

    root_path, digest = build_sharded_manifest(
        ledger, output_dir=tmp_path / "manifest", shard_size=1,
    )
    root_path_2, digest_2 = build_sharded_manifest(
        ledger, output_dir=tmp_path / "manifest", shard_size=1,
    )
    assert (root_path, digest) == (root_path_2, digest_2)
    root = json.loads(Path(root_path).read_text())
    assert root["manifest_version"] == 2
    assert root["sample_count"] == 1
    assert root["quality_assurance"] == {
        "reviewed": 0, "passed": 0, "failed": 0, "pass_rate": None,
    }
    assert root["caption_build"]["frame_count"] == 16
    assert root["shards"][0]["count"] == 1
    shard = json.loads((Path(root_path).parent / root["shards"][0]["key"]).read_text())
    assert shard["samples"][0]["caption"].startswith("A person")
    assert shard["samples"][0]["first_frame_key"].endswith("/frame.png")
    assert shard["samples"][0]["first_frame_sha256"] == sample.first_frame_file_sha256
    assert shard["samples"][0]["first_frame_rgb_sha256"] == sample.first_frame_sha256


def test_qa_export_is_deterministic_and_import_quarantines_failures(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger)
    for index in range(6):
        original = _sample(
            tmp_path,
            sample_id=f"{index + 1:064x}",
            truth="sha256:" + f"{index + 10:064x}",
        )
        sample = PreparedSample(**{
            **original.__dict__,
            "source_key": f"raw/source-{index}.mp4",
            "source_sha256": "sha256:" + f"{index + 20:064x}",
            "source_start_ms": index * 10_000,
            "motion_energy": float(index + 1),
        })
        ledger.add_sample(sample)
        ledger.claim_caption("gpu-0")
        ledger.finish_caption(
            sample.sample_id,
            caption=f"A subject number {index} moves through the scene while the camera follows.",
            model="org/caption-model",
            revision="f" * 40,
            worker="gpu-0",
            frame_count=16,
        )

    first = select_qa_samples(ledger.qa_candidates(), count=4, seed="fixed")
    second = select_qa_samples(ledger.qa_candidates(), count=4, seed="fixed")
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len({row["source_key"] for row in first}) == 4

    review_path, html_path = export_qa_bundle(
        ledger, output_dir=tmp_path / "qa", count=4, seed="fixed",
    )
    assert Path(html_path).is_file()
    records = [json.loads(line) for line in Path(review_path).read_text().splitlines()]
    for index, record in enumerate(records):
        record["verdict"] = "fail" if index == 0 else "pass"
        record["reviewer"] = "alice"
    Path(review_path).write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    report = import_qa_reviews(ledger, review_path)
    assert report == {"reviewed": 4, "passed": 3, "failed": 1, "pass_rate": 0.75}
    assert ledger.stats()["samples"] == {"captioned": 5, "qa_rejected": 1}
    with pytest.raises(ValueError, match="below required"):
        ledger.assert_qa_gate(min_reviews=4, min_pass_rate=0.95)
    assert ledger.assert_qa_gate(min_reviews=4, min_pass_rate=0.70) == report
    assert ledger.retry_qa_rejected() == 1
    assert ledger.stats()["samples"] == {"captioned": 5, "prepared": 1}


def test_real_ffmpeg_preparation_pins_first_frame_and_exact_length(tmp_path):
    source = tmp_path / "moving-source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=128x72:rate=16:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
        capture_output=True,
    )
    spec = _spec(
        width=128,
        height=72,
        max_windows_per_source=1,
        min_motion_energy=0.1,
    )
    samples = prepare_local_source(
        str(source), source_key="raw/synthetic.mp4", workdir=tmp_path, spec=spec,
    )
    assert len(samples) == 1
    sample = samples[0]
    assert Path(sample.clip_path).is_file()
    assert Path(sample.first_frame_path).is_file()
    assert digest_file(sample.first_frame_path) == sample.first_frame_file_sha256
    assert sample.num_frames == 81
    assert sample.first_frame_sha256 != sample.truth_frames_sha256
    assert sample.motion_energy > spec.min_motion_energy

    from PIL import Image
    import numpy as np
    from leoma.infra.video_utils import decode_frames_rgb

    png_rgb = np.asarray(Image.open(sample.first_frame_path).convert("RGB"))
    truth = decode_frames_rgb(
        sample.clip_path,
        start_seconds=0.0,
        duration_seconds=spec.clip_seconds,
        fps=spec.fps,
        num_frames=spec.num_frames,
        width=spec.width,
        height=spec.height,
    )
    assert np.array_equal(png_rgb, truth[0])
    assert digest_frames([png_rgb]) == sample.first_frame_sha256

    ledger = PilotLedger(tmp_path / "backfill.sqlite3", spec)
    assert ledger.add_sample(sample)
    Path(sample.first_frame_path).unlink()
    with ledger.connect() as db:
        db.execute(
            """UPDATE samples SET first_frame_path='',first_frame_file_sha256=''
               WHERE sample_id=?""",
            (sample.sample_id,),
        )
    stats = materialize_first_frame_images(
        ledger, output_dir=tmp_path / "backfilled-first-frames",
    )
    assert stats["materialized_this_run"] == 1
    with ledger.connect() as db:
        row = db.execute(
            "SELECT * FROM samples WHERE sample_id=?", (sample.sample_id,),
        ).fetchone()
    assert Path(row["first_frame_path"]).is_file()
    assert digest_file(row["first_frame_path"]) == row["first_frame_file_sha256"]
    assert row["first_frame_sha256"] == sample.first_frame_sha256


class _MissingObject(Exception):
    code = "NoSuchKey"


class _MemoryObjectStore:
    def __init__(self):
        self.objects = {}

    def stat_object(self, bucket, key):
        try:
            body, metadata = self.objects[(bucket, key)]
        except KeyError as exc:
            raise _MissingObject() from exc
        return SimpleNamespace(size=len(body), metadata=metadata)

    def fput_object(self, bucket, key, path, *, content_type, metadata):
        stored = {f"x-amz-meta-{name}": value for name, value in metadata.items()}
        self.objects[(bucket, key)] = (Path(path).read_bytes(), stored)


def test_approved_publish_batch_resumes_without_requiring_the_audit_twice(tmp_path):
    ledger = PilotLedger(tmp_path / "pilot.sqlite3", _spec())
    _bind_caption(ledger)
    for index in range(3):
        original = _sample(
            tmp_path,
            sample_id=f"{index + 100:064x}",
            truth="sha256:" + f"{index + 200:064x}",
        )
        sample = PreparedSample(**{
            **original.__dict__,
            "source_key": f"raw/{index}.mp4",
            "source_sha256": "sha256:" + f"{index + 300:064x}",
            "source_start_ms": index * 10_000,
        })
        ledger.add_sample(sample)
        task = ledger.claim_caption("gpu-0")
        ledger.finish_caption(
            task.sample_id,
            caption="A person walks through a room while the camera follows behind.",
            model="org/caption-model", revision="f" * 40, worker="gpu-0",
            frame_count=16,
        )
    reviewed_id = ledger.qa_candidates()[0]["sample_id"]
    ledger.record_qa_review(reviewed_id, verdict="pass", reviewer="alice")

    store = _MemoryObjectStore()
    first = publish_captioned_samples(
        store, "bucket", ledger=ledger, prefix="corpus-v2/pilot",
        limit=1, qa_min_reviews=1, qa_min_pass_rate=1.0,
    )
    assert first["samples"] == {"captioned": 2, "published": 1}
    # The audited row may already be published. The frozen approval bit, rather
    # than the still-pending review count, makes the interrupted batch resumable.
    second = publish_captioned_samples(
        store, "bucket", ledger=ledger, prefix="corpus-v2/pilot",
        qa_min_reviews=1, qa_min_pass_rate=1.0,
    )
    assert second["samples"] == {"published": 3}
