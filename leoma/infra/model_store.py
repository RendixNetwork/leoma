"""Hippius Hub model references and local materialization.

Miners upload their video-generation model weights (safetensors + config) to
Hippius Hub — a content-addressed OCI registry — and commit a compact
``v4|<repo>|<digest>|<hotkey>`` reveal on-chain. Validators read the reveal,
resolve it to an immutable ``repo@digest`` reference, and download the exact
bytes themselves (no miner-hosted inference endpoint).

``hippius_hub`` is imported lazily inside the functions that need it so that
``ModelRef`` and the reveal parsing/serialisation (the parts exercised by unit
tests and by the on-chain scan) import cleanly without the package installed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from leoma.eval.errors import ChallengerFault

MODEL_CACHE_DIR = os.environ.get("LEOMA_MODEL_CACHE_DIR", "/tmp/leoma/hippius_models")
HUB_TOKEN_PATH = Path("~/.cache/hippius/hub/token").expanduser()

REVEAL_V4_PREFIX = "v4"
REPO_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._/-]*$")
# Two digest shapes accepted:
#   - "sha256:<64hex>"  Hippius OCI manifest digest (challenger uploads via
#                       hippius_hub, the canonical Hippius reference)
#   - "hf:<40hex>"      HuggingFace commit SHA (a genesis king pinned to a
#                       vanilla HF repo without a Hippius mirror)
DIGEST_RE = re.compile(r"^(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")
SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{47,48}$")

# Files pulled for a model snapshot. Covers diffusers layout (config.json,
# model_index.json, scheduler/*, text_encoder + tokenizer configs) plus the
# safetensors weights. No code files are ever downloaded/executed.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "special_tokens*", "*.model", "*.txt"]
CONFIG_ONLY_PATTERNS = ALLOW_PATTERNS[1:]

HUB_TOKEN_ENV_NAMES = (
    "HIPPIUS_HUB_TOKEN",
    "HIPPIUS_TOKEN",
    "LEOMA_HIPPIUS_TOKEN",
)
# S3-style Hippius creds (used for the video corpus + state buckets) are NOT
# valid for Hub/OCI registry auth; detected only to produce a clearer error.
S3_ONLY_ENV_NAMES = (
    "HIPPIUS_VIDEOS_READ_ACCESS_KEY",
    "HIPPIUS_VIDEOS_READ_SECRET_KEY",
    "HIPPIUS_VIDEOS_WRITE_ACCESS_KEY",
    "HIPPIUS_VIDEOS_WRITE_SECRET_KEY",
    "HIPPIUS_ACCESS_KEY",
    "HIPPIUS_SECRET_KEY",
)
HUB_USERNAME_ENV_NAMES = (
    "HIPPIUS_HUB_USERNAME",
    "HIPPIUS_REGISTRY_USERNAME",
    "LEOMA_HIPPIUS_USERNAME",
)
HUB_PASSWORD_ENV_NAMES = (
    "HIPPIUS_HUB_PASSWORD",
    "HIPPIUS_REGISTRY_PASSWORD",
    "LEOMA_HIPPIUS_PASSWORD",
)


class HippiusHubAuthError(RuntimeError):
    """Raised when an authenticated Hub operation has no usable credentials."""


class UnsafeSnapshotError(ChallengerFault):
    """A downloaded model contains executable or otherwise forbidden files."""

    reason = "unsafe_snapshot"


def _get_first_env(names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def get_hub_token() -> Optional[str]:
    token = _get_first_env(HUB_TOKEN_ENV_NAMES)
    if token:
        return token
    if HUB_TOKEN_PATH.exists():
        cached = HUB_TOKEN_PATH.read_text().strip()
        if cached:
            return cached
    return None


def get_hub_basic_auth() -> Optional[tuple[str, str]]:
    username = _get_first_env(HUB_USERNAME_ENV_NAMES)
    password = _get_first_env(HUB_PASSWORD_ENV_NAMES)
    if username and password:
        return username, password
    return None


def _s3_auth_detail() -> str:
    present = [name for name in S3_ONLY_ENV_NAMES if (os.environ.get(name) or "").strip()]
    if not present:
        return ""
    return (
        " Found only S3-style Hippius credentials "
        f"({', '.join(present)}), which are not valid for Hub/OCI registry auth."
    )


def _resolve_hub_token(action: Optional[str] = None) -> Optional[str]:
    """Return optional read credentials, or ``None`` for a public repository.

    Miner submissions are public by protocol, so config inspection, OCI copy
    checks, and snapshot downloads must work anonymously. Operators may still
    provide a read token to avoid anonymous rate limits or access a private
    genesis repository. Uploads remain authenticated through
    :func:`_prepare_upload_token`.
    """
    token = get_hub_token()
    if token:
        return token

    basic_auth = get_hub_basic_auth()
    if basic_auth:
        from hippius_hub import login as hub_login

        username, password = basic_auth
        hub_login(username=username, password=password)
        token = get_hub_token()
        if token:
            return token
        if action:
            raise HippiusHubAuthError(f"{action} could not read cached Hippius Hub auth after login.")
        return None

    # Public miner repositories are the normal read path. Passing token=None lets
    # hippius_hub request anonymous pull credentials from the registry.
    return None


def _prepare_upload_token(action: str) -> Optional[str]:
    basic_auth = get_hub_basic_auth()
    if basic_auth:
        from hippius_hub import login as hub_login

        username, password = basic_auth
        hub_login(username=username, password=password)
        return None

    token = get_hub_token()
    if token:
        return token

    raise HippiusHubAuthError(
        f"{action} requires Hippius Hub auth via token {HUB_TOKEN_ENV_NAMES} "
        f"or username/password envs {HUB_USERNAME_ENV_NAMES} + {HUB_PASSWORD_ENV_NAMES}."
        f"{_s3_auth_detail()}"
    )


@dataclass(frozen=True)
class ModelRef:
    """Immutable Hippius Hub model reference (``repo@digest``)."""

    repo: str
    digest: str

    def __post_init__(self) -> None:
        repo = (self.repo or "").strip()
        digest = (self.digest or "").strip()
        if not REPO_RE.match(repo):
            raise ValueError(f"invalid Hippius repo id: {self.repo!r}")
        if not DIGEST_RE.match(digest):
            raise ValueError(f"invalid Hippius OCI digest: {self.digest!r}")
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "digest", digest)

    @property
    def immutable_ref(self) -> str:
        return f"{self.repo}@{self.digest}"


def _normalise_digest(value: str) -> str:
    digest = (value or "").strip()
    if not DIGEST_RE.match(digest):
        raise ValueError(f"invalid OCI digest: {value!r}")
    return digest


# v4 payload: `v4|<challenger_repo>|<challenger_digest>|<author_hotkey>`.
# challenger_digest carries its format prefix (sha256:/hf:) so the validator can
# dispatch to the right download path. author_hotkey is the 48-char ss58 of the
# submitter, kept for cross-check against the chain-side commitment signer key.
# Longest case: `v4|<repo-50>|sha256:<64>|<ss58-48>` ≈ 160 chars.

def build_reveal_v4(challenger_ref: ModelRef, author_hotkey: str) -> str:
    hk = (author_hotkey or "").strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid author hotkey ss58: {author_hotkey!r}")
    return f"{REVEAL_V4_PREFIX}|{challenger_ref.repo}|{challenger_ref.digest}|{hk}"


def parse_reveal_v4(payload: str) -> tuple[ModelRef, str]:
    """Returns (ModelRef(challenger_repo, challenger_digest), author_hotkey).

    Raises ValueError for any non-v4 / malformed payload (e.g. a legacy JSON
    commitment), so the scanner can treat those as skippable.
    """
    parts = (payload or "").strip().split("|")
    if len(parts) != 4 or parts[0] != REVEAL_V4_PREFIX:
        raise ValueError("expected v4|repo|challenger_digest|author_hotkey reveal")
    hk = parts[3].strip()
    if not SS58_RE.match(hk):
        raise ValueError(f"invalid v4 author hotkey: {parts[3]!r}")
    return ModelRef(parts[1], _normalise_digest(parts[2])), hk


def _cache_snapshot_path(ref: ModelRef) -> Path:
    repo_key = ref.repo.replace("/", "--")
    digest_key = ref.digest.replace(":", "-")
    return Path(MODEL_CACHE_DIR) / repo_key / "snapshots" / digest_key


def local_snapshot_path(ref: ModelRef) -> str:
    path = _cache_snapshot_path(ref)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return str(path)


def cache_path(ref: ModelRef, *, config_only: bool = False) -> str:
    """Where this ref's snapshot lives — whether or not it has been downloaded yet.

    Unlike :func:`local_snapshot_path` this does not require the directory to exist,
    because the caller that needs it most is the download-progress watcher: it polls
    the directory *while it is being filled*, which is precisely the window in which
    "does it exist yet" is the wrong question.
    """
    base = _cache_snapshot_path(ref)
    return str(base.with_name(base.name + "_cfg") if config_only else base)


def _call_snapshot_download(ref: ModelRef, local_dir: Optional[str], max_workers: Optional[int],
                            *, allow_patterns=ALLOW_PATTERNS) -> str:
    if ref.digest.startswith("hf:"):
        from huggingface_hub import snapshot_download as hf_snapshot_download

        return str(hf_snapshot_download(
            repo_id=ref.repo, revision=ref.digest[3:], local_dir=local_dir,
            allow_patterns=allow_patterns, max_workers=max_workers or 8,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        ))
    from hippius_hub import snapshot_download

    return str(snapshot_download(
        repo_id=ref.repo, revision=ref.digest, local_dir=local_dir,
        allow_patterns=allow_patterns, max_workers=max_workers or 8,
        token=_resolve_hub_token(f"Downloading {ref.immutable_ref}"),
    ))


#: Written into a snapshot directory once its download has *finished*.
#:
#: The cache check used to be "does ``config.json`` exist, and is there a
#: ``*.safetensors`` at the top level" — a **transformers** layout. A diffusers
#: snapshot has neither: no root ``config.json`` (it has ``model_index.json``), and
#: the weights live in ``transformer/``, ``vae/``, ``text_encoder/`` subfolders. So
#: the check **never hit**, and every single duel `rmtree`d the cache and
#: re-downloaded the king's ~30-70 GB of weights. The king is the same model for
#: every challenger in the queue; it was being pulled again for each one.
#:
#: A completion marker fixes the deeper bug too: an *interrupted* download left a
#: directory that looked exactly like a valid cache. Presence of files was never
#: evidence that all the files were there.
COMPLETION_MARKER = ".leoma_complete.json"

#: Snapshots to keep on disk. The king plus an in-flight challenger is the working
#: set; anything beyond that is history. At ~30-70 GB per snapshot, "never evict"
#: fills any disk.
MAX_CACHED_SNAPSHOTS = int(os.environ.get("LEOMA_MAX_CACHED_SNAPSHOTS", "4"))

#: Free bytes to keep available. A download that fills the disk takes the box down
#: with it, and a 70 GB pull needs the headroom *before* it starts, not after.
MIN_FREE_BYTES = int(os.environ.get("LEOMA_MIN_FREE_BYTES", str(150 * 1024**3)))


def _marker_path(target: Path) -> Path:
    return target / COMPLETION_MARKER


def is_complete(target: str | os.PathLike[str]) -> bool:
    """Has this snapshot finished downloading?

    Layout-agnostic on purpose: it asks about the *download*, not about the model.
    A check that inspects file names has to know whether it is looking at a
    transformers repo or a diffusers one — and gets it wrong the moment the pinned
    architecture changes.
    """
    return _marker_path(Path(target)).is_file()


def _mark_complete(target: Path, ref: ModelRef) -> None:
    payload = {
        "ref": ref.immutable_ref,
        "files": len(list_snapshot_files(target)),
        "bytes": snapshot_size(target),
    }
    _marker_path(target).write_text(json.dumps(payload, indent=2))


def validate_snapshot_files(
    target: str | os.PathLike[str], *, require_weights: bool = True
) -> None:
    """Reject executable files, links, and non-safetensors model weights.

    Miner repositories are untrusted input. Diffusers versions before 0.38 have
    known remote-code guard bypasses, so the evaluator does not rely on that guard
    alone: the registry download allowlist is enforced again on disk before every
    load. A poisoned local cache therefore cannot add ``.py``/pickle files after the
    original download and become executable on the next duel.
    """
    root = Path(target)
    if not root.is_dir() or root.is_symlink():
        raise UnsafeSnapshotError(f"model snapshot is not a real directory: {root}")

    found_weights = False
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in dirnames:
            path = directory_path / name
            if path.is_symlink():
                raise UnsafeSnapshotError(f"model snapshot contains a directory symlink: {path}")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise UnsafeSnapshotError(f"model snapshot contains a non-regular file: {path}")
            if name == COMPLETION_MARKER:
                continue
            if not any(fnmatch(name, pattern) for pattern in ALLOW_PATTERNS):
                relative = path.relative_to(root).as_posix()
                raise UnsafeSnapshotError(
                    f"model snapshot contains forbidden file {relative!r}; only configs, "
                    "tokenizer data, and .safetensors weights are accepted"
                )
            if fnmatch(name, "*.safetensors"):
                found_weights = True

    if require_weights and not found_weights:
        raise UnsafeSnapshotError("model snapshot contains no .safetensors weights")


def _cached_snapshots(root: Path) -> list[tuple[float, Path]]:
    """Completed snapshots on disk, oldest use first. Incomplete ones are junk."""
    found: list[tuple[float, Path]] = []
    if not root.exists():
        return found
    for marker in root.glob(f"*/snapshots/*/{COMPLETION_MARKER}"):
        try:
            found.append((marker.stat().st_mtime, marker.parent))
        except OSError:
            continue
    return sorted(found)


def evict_snapshots(keep: Iterable[str] = (), *, root: Optional[str] = None) -> list[str]:
    """Evict cached snapshots (LRU) until we are under budget. Returns what went.

    Runs **inline, immediately before a download** rather than on a timer. Disk
    pressure only matters at the moment we are about to consume disk, and a sweeper
    process is one more thing to deploy, monitor, and get wrong. The keep-set is the
    working set (king + the challenger we are about to duel), which must never be
    evicted no matter how cold it looks.
    """
    base = Path(root or MODEL_CACHE_DIR)
    protected = {str(Path(k).resolve()) for k in keep}
    evicted: list[str] = []

    snapshots = _cached_snapshots(base)
    for _, path in snapshots:
        if str(path.resolve()) in protected:
            continue
        over_count = len([p for _, p in _cached_snapshots(base)]) > MAX_CACHED_SNAPSHOTS
        try:
            free = shutil.disk_usage(base).free
        except OSError:
            free = MIN_FREE_BYTES
        if not over_count and free >= MIN_FREE_BYTES:
            break
        shutil.rmtree(path, ignore_errors=True)
        evicted.append(str(path))

    return evicted


def materialize_model(ref: ModelRef, local_dir: Optional[str] = None, max_workers: Optional[int] = None,
                      *, config_only: bool = False, keep: Iterable[str] = ()) -> str:
    """Download or reuse an immutable Hippius Hub snapshot; returns the local dir.

    ``repo@digest`` is immutable, so a **complete** snapshot is valid forever: if the
    marker is there, the bytes are the right bytes and there is nothing to re-check.

    ``config_only=True`` skips the large ``*.safetensors`` files — used by the
    validator's pre-dispatch arch/lock check, which only needs the configs. The cache
    dir is suffixed with ``_cfg`` so a config-only fetch can never be mistaken for a
    full one.
    """
    if config_only:
        base = Path(local_dir) if local_dir else _cache_snapshot_path(ref)
        target = base.with_name(base.name + "_cfg")
    else:
        target = Path(local_dir) if local_dir else _cache_snapshot_path(ref)

    if is_complete(target):
        validate_snapshot_files(target, require_weights=not config_only)
        return str(target)

    # Not complete: either absent, or the debris of an interrupted download. Either
    # way it cannot be trusted, and a partial snapshot that *looks* usable is worse
    # than none at all.
    if target.exists():
        shutil.rmtree(target)

    # Make room before we pull tens of gigabytes, not after.
    if not config_only:
        evict_snapshots(keep=[*keep, str(target)])

    target.parent.mkdir(parents=True, exist_ok=True)
    patterns = CONFIG_ONLY_PATTERNS if config_only else ALLOW_PATTERNS
    result = _call_snapshot_download(ref, str(target), max_workers, allow_patterns=patterns)
    result_path = Path(result)
    if result_path.resolve() != target.resolve():
        raise UnsafeSnapshotError(
            f"registry returned snapshot outside the requested cache path: {result_path}"
        )

    validate_snapshot_files(result_path, require_weights=not config_only)
    _mark_complete(result_path, ref)
    return result


def list_snapshot_files(snapshot: str | os.PathLike[str]) -> list[str]:
    root = Path(snapshot)
    return sorted(
        str(p.relative_to(root)).replace(os.sep, "/")
        for p in root.rglob("*")
        if p.is_file()
    )


def snapshot_size(snapshot: str | os.PathLike[str], files: Optional[Iterable[str]] = None) -> int:
    root = Path(snapshot)
    paths = (root / f for f in files) if files is not None else (p for p in root.rglob("*") if p.is_file())
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except FileNotFoundError:
            continue
    return total


def sha256_safetensors(path: str | os.PathLike[str]) -> str:
    """Content hash of a snapshot's weights — used to detect a copy of the king.

    The old implementation globbed ``*.safetensors`` **non-recursively**. A diffusers
    snapshot keeps its weights in ``transformer/``, ``vae/`` and ``text_encoder/``
    subfolders, so it matched **zero files** and returned the sha256 of the empty
    string: the same constant, for every model on earth. As a copy-detector it would
    have flagged every model as identical to every other one.

    Now: recursive, path-sensitive (so the same bytes under a different filename hash
    differently), and it **raises** rather than hashing nothing — returning a valid
    -looking digest for a snapshot with no weights is how the original bug hid.
    """
    import hashlib

    root = Path(path)
    weights = sorted(p for p in root.rglob("*.safetensors") if p.is_file())
    if not weights:
        raise FileNotFoundError(
            f"no .safetensors found anywhere under {root}. Refusing to return a digest "
            "of nothing — that is indistinguishable from a real one."
        )

    h = hashlib.sha256()
    for p in weights:
        # Hash the PATH as well as the bytes: two snapshots that contain the same
        # tensors under different component names are not the same model.
        h.update(str(p.relative_to(root)).replace(os.sep, "/").encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def fetch_oci_copy_info(ref: "ModelRef") -> Optional[dict]:
    """Per-layer weight digests + the registry's own push timestamp, from OCI metadata.

    This is what lets the validator catch a *repackaged* copy of the king — identical
    weights re-uploaded with a changed README or tokenizer, which yields a new
    top-level manifest digest but the SAME per-layer safetensor digests — for the cost
    of one manifest fetch (a few KB), with **no weight download at all**. On a
    content-addressed registry, identical layer digests mean identical bytes.

    Returns ``{"safetensor_layers": {title: digest}, "committed_at": iso|None,
    "timestamp_source": str|None}``, or **None** when the check cannot be performed
    (an ``hf:`` ref, the registry is unreachable, the manifest is absent). None means
    "don't know" — the caller must fail *open*, never blocking a valid submission on a
    metadata hiccup.

    ``committed_at`` is deliberately the **registry-observed** push time (Harbor's
    ``push_time`` or the manifest's ``Last-Modified``), never a client-supplied
    annotation like ``org.opencontainers.image.created`` — a miner can backdate the
    latter to steal an earlier-author claim. Ported from Teutonic's
    ``_fetch_model_oci_info``.
    """
    if ref.digest.startswith("hf:"):
        return None
    try:
        import httpx
        from hippius_hub._harbor import harbor_get_artifact, split_repo_id
        from hippius_hub._oci import manifest_url, oci_headers
        from hippius_hub.auth import (
            get_oci_bearer_token,
            resolve_auth_header,
            resolve_token_value,
        )
        from hippius_hub.constants import resolve_registry
        from hippius_hub.file_download import _oci_repo_path

        registry = resolve_registry(None)
        oci_repo = _oci_repo_path(ref.repo, None)
        raw_token = _resolve_hub_token(f"copy-check manifest {ref.repo}")
        oci_token = get_oci_bearer_token(oci_repo, resolve_token_value(raw_token), push=False)

        resp = httpx.get(
            manifest_url(registry, oci_repo, ref.digest),
            headers=oci_headers(oci_token),
            timeout=httpx.Timeout(15.0),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        manifest = resp.json()

        safetensor_layers: dict[str, str] = {}
        for layer in manifest.get("layers", []):
            title = layer.get("annotations", {}).get("org.opencontainers.image.title", "")
            if title.endswith(".safetensors") and "digest" in layer:
                safetensor_layers[title] = layer["digest"]

        artifact = None
        auth_header = resolve_auth_header(raw_token)
        if auth_header:
            try:
                project, repo = split_repo_id(oci_repo)
                artifact = harbor_get_artifact(auth_header, project, repo, ref.digest, endpoint=None)
            except Exception:
                pass  # timestamp metadata is best-effort; layer digests are the load-bearing part

        committed_at = None
        timestamp_source = None
        if isinstance(artifact, dict) and artifact.get("push_time"):
            committed_at, timestamp_source = artifact["push_time"], "harbor_artifact.push_time"
        elif resp.headers.get("Last-Modified"):
            committed_at, timestamp_source = resp.headers["Last-Modified"], "manifest_last_modified"

        return {
            "safetensor_layers": safetensor_layers,
            "committed_at": committed_at,
            "timestamp_source": timestamp_source,
        }
    except Exception:
        # Fail OPEN: a metadata hiccup must never block a valid submission. The
        # caller treats None as "cannot check", not "not a copy".
        return None


def upload_model_folder(
    folder_path: str | os.PathLike[str],
    repo: str,
    revision: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> ModelRef:
    """Upload a model folder to Hippius Hub and return its immutable digest."""
    from hippius_hub import upload_folder

    token = _prepare_upload_token(f"Uploading {folder_path} to {repo}")
    result = upload_folder(
        repo_id=repo, folder_path=str(folder_path), revision=revision,
        commit_message=commit_message, allow_patterns=ALLOW_PATTERNS, token=token,
    )
    return ModelRef(repo, _normalise_digest(str(result.oid)))
