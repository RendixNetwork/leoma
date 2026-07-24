"""Build, publish and verify the corpus manifest.

This is the offline half of the consensus surface. Everything expensive,
build-dependent or non-deterministic about choosing duel clips happens **here,
once**, and its results are frozen into a file whose digest goes into
``chain.toml``. At duel time nothing is detected, searched or listed.

What moves offline, and what that buys:

* **scene detection** — ffmpeg's scene-cut timestamps vary by build. Deciding the
  clip window once and recording it as ``clip_start`` means two validators can no
  longer carve a different five seconds out of the same video.
* **the "is this video usable" question** — a video too short to hold a clip, or
  with no single-shot window, used to be discovered *at duel time* and silently
  skipped, shifting every subsequent index. It is a property of the video, so it
  is decided here and the video simply never enters the list.
* **the ground truth itself** — decoded and hashed, so a validator whose ffmpeg
  produces different pixels finds out immediately instead of silently scoring
  against different reality.
* **static clips** — dropped by ``motion_energy``, because the freeze cheat only
  pays on clips that barely move. Remove the surface rather than only detecting
  the attack.

Building requires read access to the source bucket and a working ffmpeg. Verifying
requires the same, and is what an operator runs on a **new eval box** before it is
allowed to duel: it proves this box decodes the pinned corpus byte-identically.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Callable, Iterable, Optional

from leoma.eval.digests import canonical_json, digest_file, digest_frames, sha256_bytes
from leoma.eval.manifest import (
    MANIFEST_VERSION,
    ClipEntry,
    CorpusManifest,
    CorpusManifestAny,
    DecodeParams,
    SampleEntryV2,
    load_pinned_manifest,
    parse_manifest,
)

#: Clips whose mean inter-frame delta is below this are excluded: they are the
#: clips a freeze-frame cheat scores well on. Calibrate against a real corpus —
#: too high and the corpus starves, too low and the cheat surface survives.
MIN_MOTION_ENERGY = 1.5

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _window_seed(video_key: str) -> int:
    """Reproducible per-video seed for the clip-window choice."""
    import hashlib

    return int.from_bytes(hashlib.blake2b(video_key.encode(), digest_size=8).digest(), "little")


def serialize(manifest: CorpusManifest) -> bytes:
    """The exact bytes that get hashed and published.

    Canonical JSON: the digest in ``chain.toml`` is a digest of *these* bytes, so
    two people serializing the same manifest must produce the same file.
    """
    return canonical_json(manifest.as_dict())


def manifest_digest(manifest: CorpusManifest) -> str:
    return sha256_bytes(serialize(manifest))


def build_clip_entry(
    local_path: str,
    *,
    clip_id: str,
    video_key: str,
    decode: DecodeParams,
    seed: int,
    prompt: str = "",
) -> Optional[ClipEntry]:
    """Decide one video's clip window and hash its ground truth.

    Returns ``None`` when the video cannot yield a usable clip — too short, no
    single-shot window long enough, or too static to be worth dueling on. That is a
    permanent property of the video, so excluding it here is exactly right: it can
    never surprise a duel later.
    """
    import asyncio

    from leoma.infra.video_utils import (
        choose_one_shot_clip_start,
        decode_frames_rgb,
        motion_energy,
    )

    clip_seconds = decode.num_frames / max(1, decode.fps)

    selection = asyncio.run(choose_one_shot_clip_start(local_path, clip_seconds, seed=seed))
    if selection is None:
        return None

    try:
        truth = decode_frames_rgb(
            local_path,
            start_seconds=selection.clip_start_seconds,
            duration_seconds=clip_seconds,
            fps=decode.fps,
            num_frames=decode.num_frames,
            width=decode.width,
            height=decode.height,
        )
    except Exception:  # noqa: BLE001 — a video we cannot decode is a video we exclude
        return None

    energy = motion_energy(truth)
    if energy < MIN_MOTION_ENERGY:
        return None

    return ClipEntry(
        clip_id=clip_id,
        video_key=video_key,
        video_sha256=digest_file(local_path),
        clip_start=round(float(selection.clip_start_seconds), 3),
        clip_seconds=round(float(clip_seconds), 3),
        prompt=prompt,
        truth_sha256=digest_frames(truth),
        motion_energy=round(float(energy), 4),
    )


def build_manifest(
    client,
    bucket: str,
    *,
    corpus_id: str,
    decode: DecodeParams,
    keys: Iterable[str],
    log: LogFn = _noop,
) -> CorpusManifest:
    """Build a manifest from source videos in ``bucket``.

    ``clip_id`` is derived from the video key, so it is stable across rebuilds: a
    rebuilt manifest with the same inputs produces the same ids in the same order,
    and a diff shows only what actually changed.
    """
    entries: list[ClipEntry] = []
    skipped = 0

    for key in sorted(keys):  # sorted: the manifest's order must not depend on S3's
        clip_id = os.path.splitext(os.path.basename(key))[0]
        with tempfile.TemporaryDirectory(prefix="leoma-manifest-") as tmpdir:
            local = os.path.join(tmpdir, "src.mp4")
            try:
                client.fget_object(bucket, key, local)
            except Exception as e:  # noqa: BLE001
                log(f"skip {key}: download failed ({e})")
                skipped += 1
                continue
            entry = build_clip_entry(
                local,
                clip_id=clip_id,
                video_key=key,
                decode=decode,
                # Seeded from the key, so a rebuild from the same corpus picks the
                # same window and the diff shows only what genuinely changed.
                seed=_window_seed(key),
            )
        if entry is None:
            log(f"skip {key}: no usable one-shot window, or too static")
            skipped += 1
            continue
        entries.append(entry)
        log(f"add  {clip_id}  start={entry.clip_start:.3f}s  motion={entry.motion_energy:.2f}")

    if not entries:
        raise RuntimeError(f"no usable clips found in {bucket} ({skipped} skipped)")

    entries.sort(key=lambda e: e.clip_id)
    log(f"built {len(entries)} clips ({skipped} skipped)")
    return CorpusManifest(
        corpus_id=corpus_id,
        decode=decode,
        clips=tuple(entries),
        manifest_version=MANIFEST_VERSION,
    )


def verify_manifest(
    client,
    bucket: str,
    manifest: CorpusManifestAny,
    *,
    sample: Optional[int] = None,
    log: LogFn = _noop,
) -> int:
    """Re-decode clips on THIS box and check them against the manifest's hashes.

    The preflight an eval box must pass before it is trusted with a duel. A box
    whose ffmpeg decodes even slightly differently will measure every distance
    against different ground truth — silently, confidently, and wrongly. Finding
    that out here costs a minute; finding it out in production costs consensus.

    Returns the number of clips verified. Raises on the first mismatch.
    """
    from leoma.eval.dataset import (
        _fetch_and_decode,
        _fetch_and_decode_v2,
        iter_manifest_entries,
    )
    from leoma.eval.video_runner import GenParams

    gen = GenParams(
        num_frames=manifest.decode.num_frames,
        fps=manifest.decode.fps,
        width=manifest.decode.width,
        height=manifest.decode.height,
    )

    total = len(manifest) if sample is None else min(int(sample), len(manifest))
    entries = iter_manifest_entries(
        manifest,
        client=client,
        bucket=bucket,
        limit=total,
    )
    checked = 0
    for checked, entry in enumerate(entries, 1):
        # Raises CorpusIntegrityError on any hash mismatch — artifact bytes,
        # decoded truth, conditioning PNG, or their frame-zero relationship.
        if isinstance(entry, SampleEntryV2):
            _fetch_and_decode_v2(client, bucket, entry, gen)
        else:
            _fetch_and_decode(client, bucket, entry, gen)
        log(f"[{checked}/{total}] ok  {entry.clip_id}")
    if checked != total:
        raise RuntimeError(
            f"manifest iteration returned {checked} entries but verification expected {total}"
        )
    return checked


def write_manifest(manifest: CorpusManifest, path: str) -> str:
    """Write the manifest and return its digest — the value to paste into chain.toml."""
    raw = serialize(manifest)
    with open(path, "wb") as f:
        f.write(raw)
    return sha256_bytes(raw)


def read_manifest(path: str) -> CorpusManifest:
    with open(path, "rb") as f:
        raw = f.read()
    return parse_manifest(json.loads(raw), source_digest=sha256_bytes(raw))


def publish_manifest(client, bucket: str, key: str, manifest: CorpusManifest) -> str:
    """Upload the manifest and return its digest."""
    import io

    raw = serialize(manifest)
    client.put_object(
        bucket, key, io.BytesIO(raw), length=len(raw), content_type="application/json"
    )
    return sha256_bytes(raw)


__all__ = [
    "MIN_MOTION_ENERGY",
    "build_clip_entry",
    "build_manifest",
    "load_pinned_manifest",
    "manifest_digest",
    "publish_manifest",
    "read_manifest",
    "serialize",
    "verify_manifest",
    "write_manifest",
]
