"""
S3-compatible object storage (Hippius or Cloudflare R2): clients + buckets.

Backend is selected with OBJECT_STORAGE_BACKEND=r2|hippius (default r2; see
env.example). Used for the source-video corpus (read/write) and each
validator's own king-state bucket (write).
"""
import asyncio
import os
from typing import Optional, Tuple

import certifi
import urllib3
from minio import Minio

from leoma.bootstrap import emit_log
from leoma.bootstrap.runtime import normalize_s3_endpoint_host, settings


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _pool_manager(*, connect: float, read: float, retries: int, maxsize: int) -> urllib3.PoolManager:
    """An HTTP pool with BOUNDED timeouts.

    minio-py's default client is Timeout(connect=300, read=300) with 5 retries —
    i.e. a hung socket can stall a caller for ~25 minutes. For the validator's
    state bucket that means stalling the whole event loop, so we always pass an
    explicit client.
    """
    return urllib3.PoolManager(
        timeout=urllib3.util.Timeout(connect=connect, read=read),
        maxsize=maxsize,
        cert_reqs="CERT_REQUIRED",
        ca_certs=os.environ.get("SSL_CERT_FILE") or certifi.where(),
        retries=urllib3.Retry(
            total=retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=None,  # retry idempotent verbs and PUT
        ),
    )


def _state_pool() -> urllib3.PoolManager:
    """Small JSON objects; must never stall the validator loop."""
    return _pool_manager(
        connect=_env_float("LEOMA_S3_CONNECT_TIMEOUT", 5.0),
        read=_env_float("LEOMA_S3_READ_TIMEOUT", 20.0),
        retries=_env_int("LEOMA_S3_RETRIES", 3),
        maxsize=4,
    )


def _corpus_pool() -> urllib3.PoolManager:
    """Multi-hundred-MB fget_object; the read timeout is per socket read, not total."""
    return _pool_manager(
        connect=_env_float("LEOMA_S3_CONNECT_TIMEOUT", 10.0),
        read=_env_float("LEOMA_S3_CORPUS_READ_TIMEOUT", 120.0),
        retries=_env_int("LEOMA_S3_RETRIES", 3),
        maxsize=16,
    )


def _active_s3_host_and_region() -> Tuple[str, str]:
    if settings.object_storage_backend == "r2":
        raw = settings.r2_endpoint_raw
        if not raw or not raw.strip():
            raise ValueError(
                "R2_ENDPOINT is required when OBJECT_STORAGE_BACKEND=r2 "
                "(e.g. https://<ACCOUNT_ID>.r2.cloudflarestorage.com)"
            )
        return normalize_s3_endpoint_host(raw), settings.r2_region
    return settings.hippius_endpoint, settings.hippius_region


def _create_minio_client(
    access_key: str | None,
    secret_key: str | None,
    *,
    purpose: str,
    http_client: Optional[urllib3.PoolManager] = None,
) -> Minio:
    if not access_key or not secret_key:
        backend = settings.object_storage_backend
        raise ValueError(
            f"Missing object storage credentials for {purpose} (backend={backend}). "
            "Set the matching access key and secret key environment variables "
            f"({'R2_*' if backend == 'r2' else 'HIPPIUS_*'})."
        )
    endpoint, region = _active_s3_host_and_region()
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region,
        http_client=http_client or _corpus_pool(),
    )


def _build_minio_client(
    endpoint_raw: str | None,
    region: str | None,
    access_key: str | None,
    secret_key: str | None,
    *,
    purpose: str,
    http_client: Optional[urllib3.PoolManager] = None,
) -> Minio:
    """Build a Minio client from an explicit endpoint/region/creds (own bucket)."""
    if not endpoint_raw or not endpoint_raw.strip():
        raise ValueError(f"Missing object storage endpoint for {purpose}")
    if not access_key or not secret_key:
        raise ValueError(
            f"Missing object storage credentials for {purpose}. "
            "Set the matching access key and secret key (HIPPIUS_OWN_* for Hippius, "
            "R2_OWN_* for R2)."
        )
    return Minio(
        normalize_s3_endpoint_host(endpoint_raw),
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        region=region or "auto",
        http_client=http_client or _state_pool(),
    )


def create_own_write_client() -> Minio:
    """Minio client for this validator's OWN state bucket (write creds from env).

    Uses the tight state-bucket timeouts: these calls sit on the validator's hot
    path and must never stall the loop.
    """
    return _build_minio_client(
        settings.r2_own_endpoint,
        settings.r2_own_region,
        settings.r2_own_write_access_key,
        settings.r2_own_write_secret_key,
        purpose="own state bucket write access",
        http_client=_state_pool(),
    )


def create_dashboard_write_client() -> Minio:
    """Minio client for the public dashboard delivery bucket.

    Most S3 providers can reuse the validator owner's state credentials. Providers
    that issue bucket-scoped keys can set the optional LEOMA_DASHBOARD_* endpoint,
    region, and write credential overrides instead.
    """
    return _build_minio_client(
        settings.dashboard_endpoint,
        settings.dashboard_region,
        settings.dashboard_write_access_key,
        settings.dashboard_write_secret_key,
        purpose="dashboard bucket write access",
        http_client=_state_pool(),
    )


def create_source_read_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak, sk = settings.r2_videos_read_access_key, settings.r2_videos_read_secret_key
    else:
        ak, sk = settings.hippius_videos_read_access_key, settings.hippius_videos_read_secret_key
    return _create_minio_client(ak, sk, purpose="source bucket read access", http_client=_corpus_pool())


def create_source_write_client() -> Minio:
    if settings.object_storage_backend == "r2":
        ak, sk = settings.r2_videos_write_access_key, settings.r2_videos_write_secret_key
    else:
        ak, sk = settings.hippius_videos_write_access_key, settings.hippius_videos_write_secret_key
    return _create_minio_client(ak, sk, purpose="source bucket write access", http_client=_corpus_pool())


async def ensure_bucket_exists(minio_client: Minio, bucket_name: str) -> None:
    exists = await asyncio.to_thread(minio_client.bucket_exists, bucket_name)
    if not exists:
        await asyncio.to_thread(minio_client.make_bucket, bucket_name)
        emit_log(f"Created bucket: {bucket_name}", "success")
