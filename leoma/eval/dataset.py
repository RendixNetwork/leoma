"""Build a duel's clips from the pinned corpus manifest — deterministically, or not at all.

What this module used to do, and why each step was a consensus hole:

1. ``list_objects`` on a live bucket, then permute ``range(len(keys))``. The try
   order was a function of the corpus *size*, so uploading one video reshuffled
   every validator's clip set — and only for the validators who had seen the
   upload.
2. Run ffmpeg **scene detection at duel time** to pick the window inside each
   video. Scene-cut timestamps vary by ffmpeg build, so two validators could carve
   a *different five seconds out of the same video* and never know.
3. ``except Exception: return None`` on any clip failure, then move to the next
   candidate. One flaky S3 read shifted the whole clip set — and a duel could
   quietly run on 3 clips instead of 32 and be recorded as a normal result.

Now: the manifest fixes the clip list and the window *offline*; the seed picks
**indices into a fixed-size list**; each clip's ground truth is **verified against
its hash**; and any failure **aborts the duel**. A clip is never skipped and never
substituted — if we cannot build the exact exam the chain specifies, we do not
hand out a grade.

Fault attribution is the other half. A download that fails is
:class:`~leoma.eval.errors.TransientDuelError` — retry, and never blame the miner.
A truth that decodes to the *wrong hash* is
:class:`~leoma.eval.errors.CorpusIntegrityError` — this validator's ffmpeg or bucket
is wrong, and it must not emit a verdict at all.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Iterator, Optional, Sequence

from leoma.app.validator.seeds import select_clip_indices
from leoma.eval.digests import digest_frames, digest_file
from leoma.eval.errors import CorpusIntegrityError, TransientDuelError
from leoma.eval.manifest import (
    ClipEntry,
    CorpusManifestAny,
    CorpusManifestV2,
    ManifestEntryAny,
    ManifestShardV2,
    SampleEntryV2,
    check_corpus_size,
    check_decode_compat,
    clip_keys_digest,
    load_pinned_manifest,
    load_pinned_manifest_v2_shard,
    resolve_manifest_v2_shard_key,
)
from leoma.eval.video_runner import Clip, GenParams

#: Called after each clip is built — the eval server's forward-progress heartbeat
#: for a phase that is otherwise silent for minutes.
ProgressFn = Callable[[int, int, ManifestEntryAny], None]


def _read_object(client, bucket: str, key: str, *, description: str) -> bytes:
    """Read one small manifest object and close the S3 response on every path."""
    try:
        response = client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    except Exception as e:  # noqa: BLE001 — a fetch failure is the network's fault
        raise TransientDuelError(
            f"could not fetch {description} {bucket}/{key}: {e}"
        ) from e


def fetch_manifest(client, corpus) -> CorpusManifestAny:
    """Fetch the manifest from the corpus bucket and verify it against the pin."""
    raw = _read_object(
        client,
        corpus.bucket,
        corpus.manifest_key,
        description="the corpus manifest",
    )
    return load_pinned_manifest(
        raw,
        corpus.manifest_digest,
        manifest_key=corpus.manifest_key,
    )


def _fetch_v2_shard(
    client,
    bucket: str,
    manifest: CorpusManifestV2,
    shard: ManifestShardV2,
) -> tuple[SampleEntryV2, ...]:
    key = resolve_manifest_v2_shard_key(manifest, shard)
    raw = _read_object(
        client,
        bucket,
        key,
        description=f"corpus-v2 shard {shard.shard_index}",
    )
    return load_pinned_manifest_v2_shard(
        raw,
        manifest=manifest,
        shard=shard,
    )


def select_manifest_entries(
    manifest: CorpusManifestAny,
    *,
    client,
    bucket: str,
    indices: Sequence[int],
) -> list[ManifestEntryAny]:
    """Resolve exact global manifest indices, fetching only selected v2 shards."""
    if not isinstance(manifest, CorpusManifestV2):
        return list(manifest.select(indices))

    requested: dict[int, list[tuple[int, int]]] = {}
    shards: dict[int, ManifestShardV2] = {}
    for output_position, global_index in enumerate(indices):
        shard = manifest.shard_for_index(global_index)
        local_index = global_index - shard.start_index
        requested.setdefault(shard.shard_index, []).append(
            (output_position, local_index)
        )
        shards[shard.shard_index] = shard

    selected: list[Optional[SampleEntryV2]] = [None] * len(indices)
    for shard_index in sorted(requested):
        entries = _fetch_v2_shard(
            client,
            bucket,
            manifest,
            shards[shard_index],
        )
        for output_position, local_index in requested[shard_index]:
            try:
                selected[output_position] = entries[local_index]
            except IndexError as e:  # defensive: the shard parser already checks count
                raise CorpusIntegrityError(
                    f"corpus-v2 shard {shard_index} does not contain local index {local_index}"
                ) from e
    if any(entry is None for entry in selected):
        raise CorpusIntegrityError("corpus-v2 selection did not resolve every requested sample")
    return [entry for entry in selected if entry is not None]


def iter_manifest_entries(
    manifest: CorpusManifestAny,
    *,
    client,
    bucket: str,
    limit: Optional[int] = None,
) -> Iterator[ManifestEntryAny]:
    """Iterate a pinned corpus in global index order without loading every v2 shard."""
    remaining = len(manifest) if limit is None else min(max(0, int(limit)), len(manifest))
    if not isinstance(manifest, CorpusManifestV2):
        yield from manifest.clips[:remaining]
        return
    for shard in manifest.shards:
        if remaining <= 0:
            return
        entries = _fetch_v2_shard(client, bucket, manifest, shard)
        take = min(remaining, len(entries))
        yield from entries[:take]
        remaining -= take


def _fetch_and_decode(client, bucket: str, entry: ClipEntry, gen: GenParams):
    """Download one source video and decode its pinned window, verifying both hashes."""
    from leoma.infra.video_utils import decode_frames_rgb

    tmpdir = tempfile.mkdtemp(prefix="leoma-duel-")
    src = os.path.join(tmpdir, "src.mp4")
    try:
        try:
            client.fget_object(bucket, entry.video_key, src)
        except Exception as e:  # noqa: BLE001
            raise TransientDuelError(
                f"could not fetch source video {entry.video_key} for clip {entry.clip_id}: {e}"
            ) from e

        actual_video = digest_file(src)
        if actual_video != entry.video_sha256:
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: source video {entry.video_key} does not match the "
                f"manifest (pinned {entry.video_sha256}, got {actual_video}). The bucket's "
                "contents have drifted from the pinned corpus."
            )

        try:
            truth = decode_frames_rgb(
                src,
                start_seconds=entry.clip_start,
                duration_seconds=entry.clip_seconds,
                fps=gen.fps,
                num_frames=gen.num_frames,
                width=gen.width,
                height=gen.height,
            )
        except Exception as e:  # noqa: BLE001 — ffmpeg failed on a file that hashed correctly
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: ffmpeg could not decode the pinned window "
                f"({entry.clip_start:.3f}s +{entry.clip_seconds:.3f}s): {e}"
            ) from e

        actual_truth = digest_frames(truth)
        if actual_truth != entry.truth_sha256:
            raise CorpusIntegrityError(
                f"clip {entry.clip_id}: decoded ground truth does not match the manifest "
                f"(pinned {entry.truth_sha256}, got {actual_truth}). This box decodes video "
                "differently from the one that built the corpus — its distances would be "
                "measured against different ground truth. Refusing to duel.\n"
                "Run `leoma corpus verify` to check this box's ffmpeg against the manifest."
            )
        return truth
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _fetch_and_decode_v2(
    client,
    bucket: str,
    entry: SampleEntryV2,
    gen: GenParams,
):
    """Download and verify a v2 truth clip plus its separately-pinned first frame."""
    import numpy as np
    from PIL import Image

    from leoma.infra.video_utils import decode_frames_rgb

    with tempfile.TemporaryDirectory(prefix="leoma-duel-v2-") as tmpdir:
        clip_path = os.path.join(tmpdir, "truth.mp4")
        frame_path = os.path.join(tmpdir, "first-frame.png")
        for key, path, label in (
            (entry.clip_key, clip_path, "truth clip"),
            (entry.first_frame_key, frame_path, "first-frame PNG"),
        ):
            try:
                client.fget_object(bucket, key, path)
            except Exception as e:  # noqa: BLE001
                raise TransientDuelError(
                    f"could not fetch {label} {key} for sample {entry.sample_id}: {e}"
                ) from e

        actual_clip = digest_file(clip_path)
        if actual_clip != entry.clip_sha256:
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: truth clip {entry.clip_key} does not match "
                f"the manifest (pinned {entry.clip_sha256}, got {actual_clip})"
            )
        actual_frame_file = digest_file(frame_path)
        if actual_frame_file != entry.first_frame_sha256:
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: first-frame PNG {entry.first_frame_key} does "
                f"not match the manifest (pinned {entry.first_frame_sha256}, "
                f"got {actual_frame_file})"
            )

        try:
            truth = decode_frames_rgb(
                clip_path,
                start_seconds=0.0,
                duration_seconds=gen.num_frames / gen.fps,
                fps=gen.fps,
                num_frames=gen.num_frames,
                width=gen.width,
                height=gen.height,
            )
        except Exception as e:  # noqa: BLE001
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: ffmpeg could not decode the pinned truth clip: {e}"
            ) from e
        actual_truth = digest_frames(truth)
        if actual_truth != entry.truth_frames_sha256:
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: decoded truth clip does not match the manifest "
                f"(pinned {entry.truth_frames_sha256}, got {actual_truth}). This box "
                "decodes the corpus differently from the builder."
            )

        try:
            with Image.open(frame_path) as image:
                first_frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        except Exception as e:  # noqa: BLE001
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: could not decode its pinned first-frame PNG: {e}"
            ) from e
        expected_shape = (gen.height, gen.width, 3)
        if first_frame.shape != expected_shape:
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: first-frame PNG shape {first_frame.shape} "
                f"does not match {expected_shape}"
            )
        actual_frame_rgb = digest_frames(first_frame[None, ...])
        if actual_frame_rgb != entry.first_frame_rgb_sha256:
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: decoded first-frame PNG does not match the "
                f"manifest (pinned {entry.first_frame_rgb_sha256}, got {actual_frame_rgb})"
            )
        if not np.array_equal(first_frame, truth[0]):
            raise CorpusIntegrityError(
                f"sample {entry.sample_id}: conditioning PNG is not truth clip frame zero"
            )
        return truth, first_frame


def build_duel_clips(
    manifest: CorpusManifestAny,
    *,
    client,
    bucket: str,
    master_seed: int,
    n_clips: int,
    gen: GenParams,
    prompt_mode: str,
    fixed_prompt: str,
    on_progress: Optional[ProgressFn] = None,
) -> tuple[list[Clip], list[ManifestEntryAny]]:
    """The exam: ``n_clips`` clips selected by seed from the pinned manifest.

    Returns the runnable clips **and** the manifest entries they came from (the
    caller needs the entries to publish the audit block). Raises rather than
    returning a short list — a duel on fewer clips than the chain specifies is not
    a valid duel, and silently accepting one is how the old code turned a flaky
    bucket into a crown.
    """
    check_decode_compat(manifest, gen)
    check_corpus_size(manifest, n_clips)

    indices = select_clip_indices(master_seed, len(manifest), n_clips)
    if len(indices) != n_clips:
        raise CorpusIntegrityError(
            f"selected {len(indices)} clips from a corpus of {len(manifest)} but the duel "
            f"requires exactly {n_clips}"
        )
    entries = select_manifest_entries(
        manifest,
        client=client,
        bucket=bucket,
        indices=indices,
    )

    clips: list[Clip] = []
    for position, (index, entry) in enumerate(zip(indices, entries)):
        if isinstance(entry, SampleEntryV2):
            truth, first_frame = _fetch_and_decode_v2(client, bucket, entry, gen)
        else:
            truth = _fetch_and_decode(client, bucket, entry, gen)
            first_frame = truth[0]
        prompt = fixed_prompt if prompt_mode == "fixed" else entry.prompt
        clips.append(
            Clip(
                # The manifest index — NOT the position in this duel. The seed is
                # derived from it, so it must identify the clip in the corpus, or
                # the same clip would generate from different noise depending on
                # where it happened to land in the selection.
                clip_index=index,
                clip_id=entry.clip_id,
                first_frame=first_frame,
                prompt=prompt,
                truth_frames=truth,
                params=gen,
            )
        )
        if on_progress:
            on_progress(position + 1, n_clips, entry)

    return clips, list(entries)


def corpus_audit(
    manifest: CorpusManifestAny,
    entries: Sequence[ManifestEntryAny],
) -> dict:
    """The corpus half of the verdict's audit block — enough to replay the exam."""
    audit = {
        "manifest_version": manifest.manifest_version,
        "corpus_id": manifest.corpus_id,
        "manifest_digest": manifest.source_digest,
        "corpus_size": len(manifest),
        "clip_ids": [e.clip_id for e in entries],
        "clip_keys_digest": clip_keys_digest(entries),
    }
    if isinstance(manifest, CorpusManifestV2):
        selected_shards = {}
        for entry in entries:
            if isinstance(entry, SampleEntryV2):
                selected_shards[entry.shard_index] = {
                    "shard_index": entry.shard_index,
                    "key": entry.shard_key,
                    "sha256": entry.shard_sha256,
                }
        audit["selected_shards"] = [
            selected_shards[index] for index in sorted(selected_shards)
        ]
    return audit


__all__ = [
    "ProgressFn",
    "build_duel_clips",
    "corpus_audit",
    "fetch_manifest",
    "iter_manifest_entries",
    "select_manifest_entries",
]
