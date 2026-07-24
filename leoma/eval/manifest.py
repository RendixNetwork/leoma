"""The corpus manifest: the duel's exam paper, fixed in advance and hash-anchored.

The original evaluator derived clips from a **live bucket listing** and chose
windows with **ffmpeg scene detection at duel time**. Both were consensus holes:

* the try-order permutation is a function of the corpus *size*, so one extra video
  — or one flaky S3 read that makes a clip get skipped — changes the entire clip
  set, and
* scene-cut timestamps vary across ffmpeg builds, so two validators can carve a
  *different window out of the same video* and score against different ground
  truth, silently.

A manifest closes both. Version 1 lists source-video windows directly. Version 2
uses a small pinned root plus digest-pinned shards whose samples each carry a
caption, a normalized truth clip, and a lossless conditioning PNG. At duel time
nobody detects, searches, or lists anything: the validator selects global manifest
indices, verifies every object in the root-to-artifact chain, and refuses the exam
if its decoded truth or first frame differs.

Pre-filtering offline is sound because the reasons a video is unusable — too short,
no single-shot window long enough — are **seed-independent** properties of the
video. The seed only ever picked *among* viable candidates. So "is this video
usable" is decidable once, at build time, which turns "silently skip it and shift
everyone else's index" into "it was never in the list".

The manifest also carries the ``decode`` block it was hashed under. A manifest built
at 16 fps × 81 frames cannot be used to run a duel at 24 fps: the truth hashes
would all miss. Rather than discover that clip by clip, :func:`check_decode_compat`
rejects it up front.
"""
from __future__ import annotations

import json
import math
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from leoma.eval.digests import digest_obj, sha256_bytes
from leoma.eval.errors import ConsensusConfigError, CorpusIntegrityError

MANIFEST_VERSION = 1
MANIFEST_VERSION_V2 = 2

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAMPLE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: A manifest must be much larger than one duel's clip count, or a miner can simply
#: memorize the whole exam. The duel samples ``n_clips`` from it per challenge.
MIN_CORPUS_MULTIPLE = 20


@dataclass(frozen=True)
class DecodeParams:
    """The decode settings ``truth_sha256`` was computed under.

    Part of the manifest, not of ``chain.toml``, because the hashes are only
    meaningful *relative to these numbers*. Keeping them together makes a
    mismatched pair impossible to construct by accident.
    """

    width: int
    height: int
    fps: int
    num_frames: int

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fps": self.fps, "num_frames": self.num_frames}


@dataclass(frozen=True)
class ClipEntry:
    """One held-out duel item, fully determined before any duel runs."""

    clip_id: str
    video_key: str
    video_sha256: str      # the source .mp4's hash — proves we fetched the right file
    clip_start: float      # seconds; CHOSEN OFFLINE, never re-detected at duel time
    clip_seconds: float
    prompt: str
    truth_sha256: str      # hash of the decoded RGB ground truth under DecodeParams
    motion_energy: float   # mean abs inter-frame delta; static clips are excluded

    def as_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "video_key": self.video_key,
            "video_sha256": self.video_sha256,
            "clip_start": self.clip_start,
            "clip_seconds": self.clip_seconds,
            "prompt": self.prompt,
            "truth_sha256": self.truth_sha256,
            "motion_energy": self.motion_energy,
        }


@dataclass(frozen=True)
class CorpusManifest:
    """A versioned, hash-anchored list of duel clips."""

    corpus_id: str
    decode: DecodeParams
    clips: tuple[ClipEntry, ...]
    manifest_version: int = MANIFEST_VERSION
    #: Digest of the manifest bytes as loaded. Set by :func:`load_pinned_manifest`;
    #: empty for a manifest built in memory (nothing has been serialized to hash yet).
    source_digest: str = field(default="", compare=False)

    def __len__(self) -> int:
        return len(self.clips)

    def as_dict(self) -> dict:
        return {
            "manifest_version": self.manifest_version,
            "corpus_id": self.corpus_id,
            "decode": self.decode.as_dict(),
            "clips": [c.as_dict() for c in self.clips],
        }

    def select(self, indices: Sequence[int]) -> list[ClipEntry]:
        """The clips at ``indices`` — the duel's actual exam."""
        try:
            return [self.clips[i] for i in indices]
        except IndexError as e:
            raise CorpusIntegrityError(
                f"clip index out of range for a corpus of {len(self.clips)} clips: {e}"
            ) from e


@dataclass(frozen=True)
class CaptionBuildV2:
    """The immutable offline caption build bound into a corpus-v2 root."""

    model: str
    revision: str
    prompt_version: str
    frame_count: int
    max_new_tokens: int

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "revision": self.revision,
            "prompt_version": self.prompt_version,
            "frame_count": self.frame_count,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class ManifestShardV2:
    """One digest-pinned shard referenced by the corpus-v2 root."""

    key: str
    sha256: str
    count: int
    first_sample_id: str
    last_sample_id: str
    shard_index: int
    start_index: int

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "sha256": self.sha256,
            "count": self.count,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
        }


@dataclass(frozen=True)
class SampleEntryV2:
    """One caption + first-frame + truth-clip exam item from a verified shard."""

    sample_id: str
    source_key: str
    source_sha256: str
    source_start_ms: int
    clip_key: str
    clip_sha256: str
    first_frame_key: str
    first_frame_sha256: str
    first_frame_rgb_sha256: str
    truth_frames_sha256: str
    caption: str
    caption_model: str
    caption_revision: str
    caption_prompt_version: str
    caption_frame_count: int
    motion_energy: float
    decode: DecodeParams
    shard_key: str = field(compare=False)
    shard_sha256: str = field(compare=False)
    shard_index: int = field(compare=False)

    # The duel runner and audit code intentionally consume one common interface
    # across v1 and v2. These aliases keep that interface explicit without
    # pretending the serialized v2 field names are the old v1 names.
    @property
    def clip_id(self) -> str:
        return self.sample_id

    @property
    def truth_sha256(self) -> str:
        return self.truth_frames_sha256

    @property
    def prompt(self) -> str:
        return self.caption


@dataclass(frozen=True)
class CorpusManifestV2:
    """A small pinned root whose shards contain the large corpus."""

    corpus_id: str
    decode: DecodeParams
    sample_count: int
    caption_prompt_version: str
    caption_prompt: str
    caption_build: CaptionBuildV2
    quality_assurance: dict
    shards: tuple[ManifestShardV2, ...]
    manifest_version: int = MANIFEST_VERSION_V2
    source_digest: str = field(default="", compare=False)
    manifest_key: str = field(default="", compare=False)

    def __len__(self) -> int:
        return self.sample_count

    def shard_for_index(self, index: int) -> ManifestShardV2:
        if index < 0 or index >= self.sample_count:
            raise CorpusIntegrityError(
                f"sample index {index} out of range for a corpus of {self.sample_count} samples"
            )
        # Roots contain only tens to hundreds of shard refs. Walking them avoids
        # maintaining a second offset structure that could disagree with the refs.
        for shard in self.shards:
            if shard.start_index <= index < shard.start_index + shard.count:
                return shard
        raise CorpusIntegrityError(
            f"sample index {index} is not covered by any corpus-v2 shard"
        )


CorpusManifestAny = CorpusManifest | CorpusManifestV2
ManifestEntryAny = ClipEntry | SampleEntryV2


def clip_keys_digest(clips: Sequence[ManifestEntryAny]) -> str:
    """Digest identifying *exactly which exam* was set.

    Binds each selected clip's id **and** its expected truth hash, in order. Two
    validators publishing the same ``clip_keys_digest`` have provably scored
    against the same ground truth — which is what makes a distance disagreement
    attributable to generation noise rather than to a different question.
    """
    return digest_obj([[c.clip_id, c.truth_sha256] for c in clips])


def _exact_keys(data: dict, expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise CorpusIntegrityError(f"{context} has invalid fields ({'; '.join(details)})")


def _strict_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusIntegrityError(f"{context} must be an integer")
    if value < minimum:
        raise CorpusIntegrityError(f"{context} must be at least {minimum}")
    return value


def _nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CorpusIntegrityError(f"{context} must be a non-empty trimmed string")
    if "\x00" in value:
        raise CorpusIntegrityError(f"{context} contains a NUL byte")
    return value


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CorpusIntegrityError(f"{context} must be sha256:<64 lowercase hex>")
    return value


def _sample_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _SAMPLE_ID_RE.fullmatch(value):
        raise CorpusIntegrityError(f"{context} must be 64 lowercase hex characters")
    return value


def _decode_v2(data: Any, *, context: str) -> DecodeParams:
    if not isinstance(data, dict):
        raise CorpusIntegrityError(f"{context} must be an object")
    _exact_keys(data, {"width", "height", "fps", "num_frames"}, context)
    return DecodeParams(
        width=_strict_int(data["width"], context=f"{context}.width", minimum=8),
        height=_strict_int(data["height"], context=f"{context}.height", minimum=8),
        fps=_strict_int(data["fps"], context=f"{context}.fps", minimum=1),
        num_frames=_strict_int(data["num_frames"], context=f"{context}.num_frames", minimum=2),
    )


def _safe_object_key(value: Any, *, context: str, suffix: str = "") -> str:
    key = _nonempty_string(value, context=context)
    parts = key.split("/")
    if key.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise CorpusIntegrityError(f"{context} is not a safe bucket object key")
    if suffix and not key.endswith(suffix):
        raise CorpusIntegrityError(f"{context} must end with {suffix}")
    return key


def _content_addressed_key(
    value: Any,
    *,
    digest: str,
    suffix: str,
    context: str,
) -> str:
    key = _safe_object_key(value, context=context, suffix=suffix)
    expected = digest.split(":", 1)[1] + suffix
    if posixpath.basename(key) != expected:
        raise CorpusIntegrityError(
            f"{context} is not content-addressed by its pinned digest"
        )
    return key


def _parse_quality_assurance(data: Any) -> dict:
    context = "corpus-v2 quality_assurance"
    if not isinstance(data, dict):
        raise CorpusIntegrityError(f"{context} must be an object")
    _exact_keys(data, {"reviewed", "passed", "failed", "pass_rate"}, context)
    reviewed = _strict_int(data["reviewed"], context=f"{context}.reviewed")
    passed = _strict_int(data["passed"], context=f"{context}.passed")
    failed = _strict_int(data["failed"], context=f"{context}.failed")
    if passed + failed != reviewed:
        raise CorpusIntegrityError(
            f"{context} counts disagree: passed + failed != reviewed"
        )
    rate = data["pass_rate"]
    if reviewed == 0:
        if rate is not None:
            raise CorpusIntegrityError(f"{context}.pass_rate must be null with zero reviews")
    else:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise CorpusIntegrityError(f"{context}.pass_rate must be numeric")
        rate = float(rate)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise CorpusIntegrityError(f"{context}.pass_rate must be between zero and one")
        expected = round(passed / reviewed, 6)
        if rate != expected:
            raise CorpusIntegrityError(
                f"{context}.pass_rate {rate} does not match counts ({expected})"
            )
    return {
        "reviewed": reviewed,
        "passed": passed,
        "failed": failed,
        "pass_rate": rate,
    }


def _parse_caption_build(data: Any) -> CaptionBuildV2:
    context = "corpus-v2 caption_build"
    if not isinstance(data, dict):
        raise CorpusIntegrityError(f"{context} must be an object")
    _exact_keys(
        data,
        {"model", "revision", "prompt_version", "frame_count", "max_new_tokens"},
        context,
    )
    revision = _nonempty_string(data["revision"], context=f"{context}.revision")
    if not _HF_REVISION_RE.fullmatch(revision):
        raise CorpusIntegrityError(
            f"{context}.revision must be an immutable 40-character Hugging Face commit"
        )
    return CaptionBuildV2(
        model=_nonempty_string(data["model"], context=f"{context}.model"),
        revision=revision,
        prompt_version=_nonempty_string(
            data["prompt_version"], context=f"{context}.prompt_version"
        ),
        frame_count=_strict_int(
            data["frame_count"], context=f"{context}.frame_count", minimum=2
        ),
        max_new_tokens=_strict_int(
            data["max_new_tokens"], context=f"{context}.max_new_tokens", minimum=1
        ),
    )


def parse_manifest_v2_root(
    data: Any,
    *,
    source_digest: str = "",
    manifest_key: str = "",
) -> CorpusManifestV2:
    """Parse a verified corpus-v2 root without fetching or trusting any shard."""
    context = "corpus-v2 root"
    if not isinstance(data, dict):
        raise CorpusIntegrityError(f"{context} must be a JSON object")
    _exact_keys(
        data,
        {
            "manifest_version",
            "corpus_id",
            "sample_count",
            "decode",
            "caption_prompt_version",
            "caption_prompt",
            "caption_build",
            "quality_assurance",
            "shards",
        },
        context,
    )
    if data["manifest_version"] != MANIFEST_VERSION_V2:
        raise CorpusIntegrityError(
            f"unsupported manifest_version {data['manifest_version']!r} "
            f"(this parser reads {MANIFEST_VERSION_V2})"
        )
    corpus_id = _nonempty_string(data["corpus_id"], context=f"{context}.corpus_id")
    sample_count = _strict_int(
        data["sample_count"], context=f"{context}.sample_count", minimum=1
    )
    decode = _decode_v2(data["decode"], context=f"{context}.decode")
    caption_build = _parse_caption_build(data["caption_build"])
    prompt_version = _nonempty_string(
        data["caption_prompt_version"],
        context=f"{context}.caption_prompt_version",
    )
    if prompt_version != caption_build.prompt_version:
        raise CorpusIntegrityError(
            "corpus-v2 root caption_prompt_version does not match caption_build"
        )
    caption_prompt = _nonempty_string(
        data["caption_prompt"], context=f"{context}.caption_prompt"
    )
    quality_assurance = _parse_quality_assurance(data["quality_assurance"])

    raw_shards = data["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise CorpusIntegrityError(f"{context}.shards must be a non-empty list")
    shards: list[ManifestShardV2] = []
    offset = 0
    previous_last = ""
    for index, raw in enumerate(raw_shards):
        shard_context = f"{context}.shards[{index}]"
        if not isinstance(raw, dict):
            raise CorpusIntegrityError(f"{shard_context} must be an object")
        _exact_keys(
            raw,
            {"key", "sha256", "count", "first_sample_id", "last_sample_id"},
            shard_context,
        )
        digest = _sha256(raw["sha256"], context=f"{shard_context}.sha256")
        key = _safe_object_key(raw["key"], context=f"{shard_context}.key", suffix=".json")
        expected_name = f"shard-{index:06d}-{digest.split(':', 1)[1]}.json"
        if key != expected_name:
            raise CorpusIntegrityError(
                f"{shard_context}.key must be {expected_name!r}"
            )
        first = _sample_id(
            raw["first_sample_id"], context=f"{shard_context}.first_sample_id"
        )
        last = _sample_id(
            raw["last_sample_id"], context=f"{shard_context}.last_sample_id"
        )
        if first > last:
            raise CorpusIntegrityError(f"{shard_context} has an inverted sample-id range")
        if previous_last and first <= previous_last:
            raise CorpusIntegrityError(
                f"{shard_context} overlaps or is not ordered after the previous shard"
            )
        count = _strict_int(raw["count"], context=f"{shard_context}.count", minimum=1)
        shards.append(
            ManifestShardV2(
                key=key,
                sha256=digest,
                count=count,
                first_sample_id=first,
                last_sample_id=last,
                shard_index=index,
                start_index=offset,
            )
        )
        offset += count
        previous_last = last
    if offset != sample_count:
        raise CorpusIntegrityError(
            f"corpus-v2 root sample_count={sample_count} but shard counts sum to {offset}"
        )
    return CorpusManifestV2(
        corpus_id=corpus_id,
        decode=decode,
        sample_count=sample_count,
        caption_prompt_version=prompt_version,
        caption_prompt=caption_prompt,
        caption_build=caption_build,
        quality_assurance=quality_assurance,
        shards=tuple(shards),
        source_digest=source_digest,
        manifest_key=manifest_key,
    )


def resolve_manifest_v2_shard_key(
    manifest: CorpusManifestV2,
    shard: ManifestShardV2,
) -> str:
    """Resolve a safe shard filename beside the pinned root object."""
    base = posixpath.dirname(manifest.manifest_key)
    return f"{base}/{shard.key}" if base else shard.key


def _parse_sample_v2(
    raw: Any,
    *,
    manifest: CorpusManifestV2,
    shard: ManifestShardV2,
    position: int,
) -> SampleEntryV2:
    context = f"corpus-v2 shard {shard.shard_index} sample {position}"
    if not isinstance(raw, dict):
        raise CorpusIntegrityError(f"{context} must be an object")
    _exact_keys(
        raw,
        {
            "sample_id",
            "source_key",
            "source_sha256",
            "source_start_ms",
            "clip_key",
            "clip_sha256",
            "first_frame_key",
            "first_frame_sha256",
            "first_frame_rgb_sha256",
            "truth_frames_sha256",
            "caption",
            "caption_model",
            "caption_revision",
            "caption_prompt_version",
            "caption_frame_count",
            "motion_energy",
            "decode",
        },
        context,
    )
    sample_id = _sample_id(raw["sample_id"], context=f"{context}.sample_id")
    clip_sha = _sha256(raw["clip_sha256"], context=f"{context}.clip_sha256")
    frame_sha = _sha256(
        raw["first_frame_sha256"], context=f"{context}.first_frame_sha256"
    )
    decode = _decode_v2(raw["decode"], context=f"{context}.decode")
    if decode != manifest.decode:
        raise CorpusIntegrityError(f"{context}.decode does not match the corpus-v2 root")

    caption = _nonempty_string(raw["caption"], context=f"{context}.caption")
    if len(caption) < 12 or len(caption) > 600 or " ".join(caption.split()) != caption:
        raise CorpusIntegrityError(
            f"{context}.caption is not a normalized 12-600 character caption"
        )
    caption_model = _nonempty_string(
        raw["caption_model"], context=f"{context}.caption_model"
    )
    caption_revision = _nonempty_string(
        raw["caption_revision"], context=f"{context}.caption_revision"
    )
    caption_prompt_version = _nonempty_string(
        raw["caption_prompt_version"], context=f"{context}.caption_prompt_version"
    )
    caption_frame_count = _strict_int(
        raw["caption_frame_count"],
        context=f"{context}.caption_frame_count",
        minimum=2,
    )
    if (
        caption_model != manifest.caption_build.model
        or caption_revision != manifest.caption_build.revision
        or caption_prompt_version != manifest.caption_build.prompt_version
        or caption_frame_count != manifest.caption_build.frame_count
    ):
        raise CorpusIntegrityError(
            f"{context} caption settings do not match the corpus-v2 root"
        )
    motion = raw["motion_energy"]
    if isinstance(motion, bool) or not isinstance(motion, (int, float)):
        raise CorpusIntegrityError(f"{context}.motion_energy must be numeric")
    motion = float(motion)
    if not math.isfinite(motion) or motion < 0:
        raise CorpusIntegrityError(
            f"{context}.motion_energy must be finite and non-negative"
        )

    return SampleEntryV2(
        sample_id=sample_id,
        source_key=_nonempty_string(raw["source_key"], context=f"{context}.source_key"),
        source_sha256=_sha256(
            raw["source_sha256"], context=f"{context}.source_sha256"
        ),
        source_start_ms=_strict_int(
            raw["source_start_ms"], context=f"{context}.source_start_ms"
        ),
        clip_key=_content_addressed_key(
            raw["clip_key"],
            digest=clip_sha,
            suffix=".mp4",
            context=f"{context}.clip_key",
        ),
        clip_sha256=clip_sha,
        first_frame_key=_content_addressed_key(
            raw["first_frame_key"],
            digest=frame_sha,
            suffix=".png",
            context=f"{context}.first_frame_key",
        ),
        first_frame_sha256=frame_sha,
        first_frame_rgb_sha256=_sha256(
            raw["first_frame_rgb_sha256"],
            context=f"{context}.first_frame_rgb_sha256",
        ),
        truth_frames_sha256=_sha256(
            raw["truth_frames_sha256"],
            context=f"{context}.truth_frames_sha256",
        ),
        caption=caption,
        caption_model=caption_model,
        caption_revision=caption_revision,
        caption_prompt_version=caption_prompt_version,
        caption_frame_count=caption_frame_count,
        motion_energy=motion,
        decode=decode,
        shard_key=shard.key,
        shard_sha256=shard.sha256,
        shard_index=shard.shard_index,
    )


def load_pinned_manifest_v2_shard(
    raw: bytes,
    *,
    manifest: CorpusManifestV2,
    shard: ManifestShardV2,
) -> tuple[SampleEntryV2, ...]:
    """Hash a selected shard before parsing any of its attacker-controlled fields."""
    actual = sha256_bytes(raw)
    if actual != shard.sha256:
        raise CorpusIntegrityError(
            f"corpus-v2 shard digest mismatch for {shard.key}: "
            f"pinned {shard.sha256}, fetched {actual}"
        )
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise CorpusIntegrityError(
            f"corpus-v2 shard {shard.key} is not valid JSON: {e}"
        ) from e
    context = f"corpus-v2 shard {shard.shard_index}"
    if not isinstance(data, dict):
        raise CorpusIntegrityError(f"{context} must be a JSON object")
    _exact_keys(data, {"manifest_version", "corpus_id", "shard_index", "samples"}, context)
    if data["manifest_version"] != MANIFEST_VERSION_V2:
        raise CorpusIntegrityError(f"{context} has the wrong manifest_version")
    if data["corpus_id"] != manifest.corpus_id:
        raise CorpusIntegrityError(f"{context} belongs to a different corpus_id")
    if data["shard_index"] != shard.shard_index:
        raise CorpusIntegrityError(f"{context} carries the wrong shard_index")
    raw_samples = data["samples"]
    if not isinstance(raw_samples, list) or len(raw_samples) != shard.count:
        raise CorpusIntegrityError(
            f"{context} contains {len(raw_samples) if isinstance(raw_samples, list) else 'invalid'} "
            f"samples but the root pins {shard.count}"
        )
    samples = tuple(
        _parse_sample_v2(
            sample,
            manifest=manifest,
            shard=shard,
            position=position,
        )
        for position, sample in enumerate(raw_samples)
    )
    ids = [sample.sample_id for sample in samples]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise CorpusIntegrityError(
            f"{context} sample_ids must be unique and sorted"
        )
    if ids[0] != shard.first_sample_id or ids[-1] != shard.last_sample_id:
        raise CorpusIntegrityError(
            f"{context} sample-id range does not match the root"
        )
    return samples


def parse_manifest(data: Any, *, source_digest: str = "") -> CorpusManifest:
    """Turn manifest JSON into a :class:`CorpusManifest`, fail-closed.

    Every invariant here exists because violating it would let two validators build
    different clip sets from the same file: duplicate ids (ambiguous selection),
    unsorted clips (index N means a different clip), a missing hash (nothing to
    verify against).
    """
    if not isinstance(data, dict):
        raise CorpusIntegrityError("manifest must be a JSON object")

    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise CorpusIntegrityError(
            f"unsupported manifest_version {version!r} (this build reads {MANIFEST_VERSION})"
        )

    corpus_id = str(data.get("corpus_id") or "").strip()
    if not corpus_id:
        raise CorpusIntegrityError("manifest has no corpus_id (needed to rotate the corpus)")

    raw_decode = data.get("decode")
    if not isinstance(raw_decode, dict):
        raise CorpusIntegrityError("manifest has no decode block")
    try:
        decode = DecodeParams(
            width=int(raw_decode["width"]),
            height=int(raw_decode["height"]),
            fps=int(raw_decode["fps"]),
            num_frames=int(raw_decode["num_frames"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CorpusIntegrityError(f"manifest decode block is invalid: {e}") from e

    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise CorpusIntegrityError("manifest has no clips")

    clips: list[ClipEntry] = []
    for i, raw in enumerate(raw_clips):
        if not isinstance(raw, dict):
            raise CorpusIntegrityError(f"clip {i} is not an object")
        try:
            entry = ClipEntry(
                clip_id=str(raw["clip_id"]),
                video_key=str(raw["video_key"]),
                video_sha256=str(raw["video_sha256"]),
                clip_start=float(raw["clip_start"]),
                clip_seconds=float(raw["clip_seconds"]),
                prompt=str(raw.get("prompt", "")),
                truth_sha256=str(raw["truth_sha256"]),
                motion_energy=float(raw.get("motion_energy", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CorpusIntegrityError(f"clip {i} is invalid: {e}") from e
        if not entry.truth_sha256.startswith("sha256:"):
            raise CorpusIntegrityError(f"clip {entry.clip_id} has no usable truth_sha256")
        if entry.clip_start < 0:
            raise CorpusIntegrityError(f"clip {entry.clip_id} has a negative clip_start")
        clips.append(entry)

    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        raise CorpusIntegrityError("manifest contains duplicate clip_ids")
    if ids != sorted(ids):
        raise CorpusIntegrityError(
            "manifest clips must be sorted by clip_id — selection is by index, so an "
            "unsorted manifest means index N is a different clip on different boxes"
        )

    return CorpusManifest(
        corpus_id=corpus_id,
        decode=decode,
        clips=tuple(clips),
        manifest_version=MANIFEST_VERSION,
        source_digest=source_digest,
    )


def load_pinned_manifest(
    raw: bytes,
    expected_digest: str,
    *,
    manifest_key: str = "",
) -> CorpusManifestAny:
    """Verify the manifest **bytes** against the pinned digest, then parse.

    Hash first, parse second. Parsing an unverified manifest to "see what's in it"
    would mean the file has already influenced our behavior (allocations, error
    paths) before we know it is the file the chain pinned.
    """
    actual = sha256_bytes(raw)
    if actual != expected_digest:
        raise CorpusIntegrityError(
            f"corpus manifest digest mismatch: pinned {expected_digest}, fetched {actual}. "
            "The bucket's manifest is not the one chain.toml pins — refusing to duel "
            "on an unknown corpus."
        )
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise CorpusIntegrityError(f"corpus manifest is not valid JSON: {e}") from e
    if isinstance(data, dict) and data.get("manifest_version") == MANIFEST_VERSION_V2:
        return parse_manifest_v2_root(
            data,
            source_digest=actual,
            manifest_key=manifest_key,
        )
    return parse_manifest(data, source_digest=actual)


def check_decode_compat(manifest: CorpusManifestAny, gen) -> None:
    """The manifest's truth hashes are only valid under its own decode params."""
    d = manifest.decode
    mismatched = [
        f"{name}: manifest={mine} chain={theirs}"
        for name, mine, theirs in (
            ("width", d.width, gen.width),
            ("height", d.height, gen.height),
            ("fps", d.fps, gen.fps),
            ("num_frames", d.num_frames, gen.num_frames),
        )
        if mine != theirs
    ]
    if mismatched:
        raise ConsensusConfigError(
            "corpus manifest was built under different decode parameters than "
            f"chain.toml [gen] pins ({'; '.join(mismatched)}). Every truth_sha256 in "
            "the manifest would fail to verify. Rebuild the manifest or fix [gen]."
        )


def check_corpus_size(manifest: CorpusManifestAny, n_clips: int) -> None:
    """A corpus barely larger than one duel is a memorizable exam, not a test set."""
    needed = MIN_CORPUS_MULTIPLE * n_clips
    if len(manifest) < needed:
        raise ConsensusConfigError(
            f"corpus has {len(manifest)} clips but a {n_clips}-clip duel needs at least "
            f"{needed} ({MIN_CORPUS_MULTIPLE}x) so miners cannot memorize the held-out set"
        )


__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_VERSION_V2",
    "MIN_CORPUS_MULTIPLE",
    "CaptionBuildV2",
    "ClipEntry",
    "CorpusManifest",
    "CorpusManifestAny",
    "CorpusManifestV2",
    "DecodeParams",
    "ManifestEntryAny",
    "ManifestShardV2",
    "SampleEntryV2",
    "check_corpus_size",
    "check_decode_compat",
    "clip_keys_digest",
    "load_pinned_manifest",
    "load_pinned_manifest_v2_shard",
    "parse_manifest",
    "parse_manifest_v2_root",
    "resolve_manifest_v2_shard_key",
]
