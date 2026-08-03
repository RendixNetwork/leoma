"""
Runtime environment and logging.

Loads configuration from the environment and provides structured logging helpers.
"""

import inspect
import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Callable, TypeVar, ParamSpec

from dotenv import load_dotenv

load_dotenv()

# Standard logger for leoma; use log_exception() in production to avoid full tracebacks
logger = logging.getLogger("leoma")

# Context variable for request/correlation ID tracking
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})

P = ParamSpec("P")
R = TypeVar("R")


def _is_production() -> bool:
    env = os.environ.get("LEOMA_ENV", os.environ.get("ENVIRONMENT", "")).lower()
    return env == "production"


def _is_debug() -> bool:
    return os.environ.get("LOG_LEVEL", "INFO").upper() == "DEBUG"


# ANSI color codes.

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"

_BLACK = "\033[30m"
_DARK_GRAY = "\033[90m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"

_HEADER_WIDTH = 80

_LEVEL_COLORS = {
    "DEBUG": _DIM,
    "INFO": _BLUE,
    "SUCCESS": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
    "CRITICAL": f"{_BOLD}{_RED}",
}

_LEVEL_TOKENS = {
    "DEBUG": f"{_DIM}··{_RESET}",
    "INFO": f"{_BLUE}●{_RESET}",
    "SUCCESS": f"{_GREEN}✓{_RESET}",
    "WARNING": f"{_YELLOW}▲{_RESET}",
    "ERROR": f"{_RED}✗{_RESET}",
    "CRITICAL": f"{_BOLD}{_RED}✗{_RESET}",
    "START": f"{_BLUE}▶{_RESET}",
}


class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    START = "START"


def _get_caller_info(skip_frames: int = 2) -> Dict[str, Any]:
    """Get detailed caller information (file, function, line)."""
    for i, frame_info in enumerate(inspect.stack()):
        if i < skip_frames:
            continue
        # Skip frames inside runtime.py
        if "runtime.py" in frame_info.filename:
            continue
        # Get relative path from workspace
        filename = frame_info.filename
        if "/leoma_subnet/" in filename:
            filename = filename.split("/leoma_subnet/")[-1]
        elif "/leoma/" in filename:
            filename = filename.split("/leoma/")[-1]
        
        return {
            "file": filename,
            "function": frame_info.function,
            "line": frame_info.lineno,
        }
    return {"file": "unknown", "function": "unknown", "line": 0}


def _get_component_name(caller_info: Dict[str, Any]) -> str:
    """Extract component/module name from caller file path."""
    filepath = caller_info.get("file", "")
    # Extract meaningful component names
    parts = filepath.split("/")
    if len(parts) >= 2:
        # Return something like "app.validator.main" or "infra.judge"
        return ".".join(parts[-3:]) if len(parts) >= 3 else ".".join(parts[-2:])
    return filepath


def _wall_clock() -> str:
    """Return full datetime string: YYYY-MM-DD HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 1000:03d}"


def _format_prefix(level: str, ts: str, caller_info: Dict[str, Any], component: str) -> str:
    """Format log line prefix with timestamp, component, and level.
    
    Format: [TIMESTAMP] COMP_MODULE  LEVEL_TOKEN  MESSAGE
    """
    level_token = _LEVEL_TOKENS.get(level, " ")
    timestamp = f"{_DIM}[{ts}]{_RESET}"
    component_str = f"{_MAGENTA}{component}{_RESET}"
    line_str = f"{_DIM}:{caller_info['line']}{_RESET}"
    
    return f"{timestamp} {component_str}{line_str} {level_token}"


def _format_data(data: Optional[Dict[str, Any]], indent: bool = False) -> str:
    """Format structured data for logging."""
    if not data:
        return ""
    
    if _is_debug() or indent:
        try:
            return "\n  " + json.dumps(data, indent=2, default=str).replace("\n", "\n  ")
        except Exception:
            return f" {data}"
    else:
        try:
            return f" | {json.dumps(data, default=str)}"
        except Exception:
            return f" | {data}"


def set_log_context(**kwargs: Any) -> None:
    """Set context variables to be included in all subsequent logs."""
    current = _request_context.get().copy()
    current.update(kwargs)
    _request_context.set(current)


def clear_log_context() -> None:
    """Clear all context variables."""
    _request_context.set({})


def get_log_context() -> Dict[str, Any]:
    """Get current log context."""
    return _request_context.get().copy()


def emit_log(
    msg: str,
    level: str = "INFO",
    data: Optional[Dict[str, Any]] = None,
    exc: Optional[Exception] = None,
) -> None:
    """Print a structured log line with timestamp, component, level, and optional data."""
    caller_info = _get_caller_info()
    component = _get_component_name(caller_info)
    ts = _wall_clock()

    context = get_log_context()
    all_data = {**context, **(data or {})}
    if exc:
        all_data["exception"] = str(exc)

    prefix = _format_prefix(level, ts, caller_info, component)
    data_str = _format_data(all_data) if all_data else ""

    print(f"{prefix} {msg}{data_str}")

    # Also log to standard logger for ERROR/CRITICAL
    if level in ("ERROR", "CRITICAL"):
        logger.error("%s %s", msg, data_str)


def emit_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a bold section header with optional subtitle.
    
    Compact format: **** TITLE ****
    """
    if subtitle:
        print(f"\n{_BOLD}{_BLUE}#### {title} ####{_RESET}")
        print(f"{_DIM}    {subtitle}{_RESET}")
    else:
        print(f"\n{_BOLD}{_BLUE}#### {title} ####{_RESET}")


def emit_section(title: str) -> None:
    """Print a minor section header.
    
    Compact format: ---- TITLE ----
    """
    print(f"\n{_WHITE}---- {title} ----{_RESET}")


def log_debug(msg: str, **data: Any) -> None:
    emit_log(msg, level=LogLevel.DEBUG, data=data if data else None)


def log_info(msg: str, **data: Any) -> None:
    emit_log(msg, level=LogLevel.INFO, data=data if data else None)


def log_success(msg: str, **data: Any) -> None:
    emit_log(msg, level=LogLevel.SUCCESS, data=data if data else None)


def log_warning(msg: str, **data: Any) -> None:
    emit_log(msg, level=LogLevel.WARNING, data=data if data else None)


def log_error(msg: str, exc: Optional[Exception] = None, **data: Any) -> None:
    emit_log(msg, level=LogLevel.ERROR, data=data if data else None, exc=exc)


def log_critical(msg: str, exc: Optional[Exception] = None, **data: Any) -> None:
    emit_log(msg, level=LogLevel.CRITICAL, data=data if data else None, exc=exc)


def log_start(msg: str, **data: Any) -> None:
    emit_log(msg, level=LogLevel.START, data=data if data else None)


def log_exception(message: str, exc: Optional[BaseException] = None) -> None:
    """Log an exception; in production omit full traceback to avoid leaking paths."""
    if _is_production():
        detail = str(exc) if exc else ""
        emit_log(message, level=LogLevel.ERROR, data={"exception": detail})
    else:
        emit_log(message, level=LogLevel.ERROR, exc=exc if isinstance(exc, Exception) else None)
        # Also log to standard logger for full traceback in development
        logger.exception("%s", message, exc_info=exc is None or True)


class LogTimer:
    """Context manager for timing operations and logging the duration."""
    
    def __init__(self, operation: str, **data: Any):
        self.operation = operation
        self.data = data
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def __enter__(self) -> "LogTimer":
        self.start_time = time.monotonic()
        log_debug(f"Starting: {self.operation}", **self.data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_ms = (time.monotonic() - (self.start_time or 0)) * 1000
        
        if exc_type:
            log_error(
                f"Failed: {self.operation}",
                exc=exc_val,
                duration_ms=f"{self.duration_ms:.1f}",
                **self.data
            )
        else:
            log_success(
                f"Completed: {self.operation}",
                duration_ms=f"{self.duration_ms:.1f}",
                **self.data
            )


def timed(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to time a function and log the duration."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with LogTimer(operation, function=func.__name__):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with LogTimer(operation, function=func.__name__):
                return await func(*args, **kwargs)  # type: ignore
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper
    
    return decorator



def _read_str(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def _read_int(name: str, fallback: int) -> int:
    return int(os.environ.get(name, str(fallback)))


def _read_float(name: str, fallback: float) -> float:
    return float(os.environ.get(name, str(fallback)))


def _read_optional_str(name: str) -> Optional[str]:
    return os.environ.get(name)


def normalize_s3_endpoint_host(raw: str) -> str:
    """Strip URL scheme and trailing slash for Minio ``endpoint`` (Hippius or Cloudflare R2 S3 API)."""
    s = raw.strip()
    if s.startswith("https://"):
        s = s[8:]
    elif s.startswith("http://"):
        s = s[7:]
    return s.rstrip("/")


def _parse_object_storage_backend(raw: str) -> str:
    v = raw.strip().lower()
    if v in ("hippius", "hippius-s3", "s3-hippius"):
        return "hippius"
    if v in ("r2", "cloudflare", "cloudflare-r2", "cf-r2", ""):
        return "r2"
    raise ValueError(
        "OBJECT_STORAGE_BACKEND must be 'hippius' or 'r2' "
        f"(got {raw!r}). See env.example."
    )


class Settings:
    """Central settings loaded from the environment."""

    def __init__(self) -> None:
        self.netuid = _read_int("NETUID", 36)
        self.epoch_len = _read_int("EPOCH_LEN", 180)
        self.request_timeout = _read_int("REQUEST_TIMEOUT", 300)
        self.wallet_name = _read_str("WALLET_NAME", "default")
        self.hotkey_name = _read_str("HOTKEY_NAME", "default")
        self.network = _read_str("NETWORK", "finney")
        self.hippius_endpoint = _read_str("HIPPIUS_ENDPOINT", "s3.hippius.com")
        self.hippius_region = _read_str("HIPPIUS_REGION", "decentralized")
        self.hippius_videos_read_access_key = _read_optional_str("HIPPIUS_VIDEOS_READ_ACCESS_KEY")
        self.hippius_videos_read_secret_key = _read_optional_str("HIPPIUS_VIDEOS_READ_SECRET_KEY")
        self.hippius_videos_write_access_key = _read_optional_str("HIPPIUS_VIDEOS_WRITE_ACCESS_KEY")
        self.hippius_videos_write_secret_key = _read_optional_str("HIPPIUS_VIDEOS_WRITE_SECRET_KEY")
        self.source_bucket = _read_str("HIPPIUS_SOURCE_BUCKET", "videos")
        # Was hardcoded to "r2", which made the entire Hippius branch of
        # storage_backend unreachable — even though OBJECT_STORAGE_BACKEND is
        # compose-wired, re-exported, documented, and set in the tests. The parser
        # below it was written and then never called. Leoma's whole story is
        # decentralized storage; the corpus living only on Cloudflare was an accident
        # of a one-line hardcode, not a decision.
        self.object_storage_backend = _parse_object_storage_backend(
            _read_str("OBJECT_STORAGE_BACKEND", "r2")
        )

        # ── Hippius Hub (OCI model registry) — miners upload weights here and
        #    validators download them by immutable digest. Distinct from the
        #    HIPPIUS_VIDEOS_* S3 creds above (those are for the video corpus).
        self.hippius_hub_token = _read_optional_str("HIPPIUS_HUB_TOKEN")
        self.hippius_hub_username = _read_optional_str("HIPPIUS_HUB_USERNAME")
        self.hippius_hub_password = _read_optional_str("HIPPIUS_HUB_PASSWORD")
        # Local cache for downloaded model snapshots (keyed by repo@digest).
        self.model_cache_dir = _read_str("LEOMA_MODEL_CACHE_DIR", "/tmp/leoma/hippius_models")

        # Env-driven, with today's live values as the defaults — so nothing changes for
        # an existing operator, but a validator is no longer *unable* to point at a
        # different account. These were hardcoded, which meant the whole fleet's corpus
        # access was baked into the source.
        self.r2_endpoint_raw = _read_str(
            "R2_ENDPOINT", "https://cce499ad4f3a4703b069771d8ff4215a.r2.cloudflarestorage.com"
        )
        self.r2_region = _read_str("R2_REGION", "auto")
        self.r2_source_bucket = _read_str("R2_SOURCE_BUCKET", "leoma-videos")
        self.r2_videos_read_access_key = _read_optional_str("R2_VIDEOS_READ_ACCESS_KEY")
        self.r2_videos_read_secret_key = _read_optional_str("R2_VIDEOS_READ_SECRET_KEY")
        self.r2_videos_write_access_key = _read_optional_str("R2_VIDEOS_WRITE_ACCESS_KEY")
        self.r2_videos_write_secret_key = _read_optional_str("R2_VIDEOS_WRITE_SECRET_KEY")

        # ── King-of-the-hill: this validator's own state bucket ──
        # SAMPLING_ROTATION_INTERVAL is retained only for the (unused) allowlist snapshot.
        self.sampling_rotation_interval = _read_int("SAMPLING_ROTATION_INTERVAL", 100)
        # This validator's own private bucket (write creds) for durable king state.
        # ``R2_OWN_*`` predates Hippius support and remains a compatibility fallback;
        # a Hippius deployment should not have to describe its state bucket as R2.
        if self.object_storage_backend == "hippius":
            self.r2_own_endpoint = (
                _read_optional_str("HIPPIUS_OWN_ENDPOINT")
                or _read_optional_str("R2_OWN_ENDPOINT")
                or self.hippius_endpoint
            )
            self.r2_own_region = (
                _read_optional_str("HIPPIUS_OWN_REGION")
                or _read_optional_str("R2_OWN_REGION")
                or self.hippius_region
            )
            self.r2_own_bucket = (
                _read_optional_str("HIPPIUS_OWN_BUCKET")
                or _read_optional_str("R2_OWN_BUCKET")
            )
            self.r2_own_write_access_key = (
                _read_optional_str("HIPPIUS_OWN_WRITE_ACCESS_KEY")
                or _read_optional_str("R2_OWN_WRITE_ACCESS_KEY")
            )
            self.r2_own_write_secret_key = (
                _read_optional_str("HIPPIUS_OWN_WRITE_SECRET_KEY")
                or _read_optional_str("R2_OWN_WRITE_SECRET_KEY")
            )
        else:
            self.r2_own_endpoint = _read_str("R2_OWN_ENDPOINT", self.r2_endpoint_raw)
            self.r2_own_region = _read_str("R2_OWN_REGION", self.r2_region)
            self.r2_own_bucket = _read_optional_str("R2_OWN_BUCKET")
            self.r2_own_write_access_key = _read_optional_str("R2_OWN_WRITE_ACCESS_KEY")
            self.r2_own_write_secret_key = _read_optional_str("R2_OWN_WRITE_SECRET_KEY")
        # Public dashboard delivery is intentionally separated from canonical king
        # state. Endpoint + write credentials default to this validator owner's state
        # credentials, while optional LEOMA_DASHBOARD_* overrides support providers
        # (including bucket-scoped S3 keys) that issue a distinct key per bucket.
        # Leaving the bucket unset preserves the legacy same-bucket behavior for
        # existing operators; preflight warns because that layout cannot safely make
        # the dashboard public without also exposing validator state.
        self.dashboard_bucket = _read_optional_str("LEOMA_DASHBOARD_BUCKET")
        self.dashboard_endpoint = (
            _read_optional_str("LEOMA_DASHBOARD_ENDPOINT") or self.r2_own_endpoint
        )
        self.dashboard_region = (
            _read_optional_str("LEOMA_DASHBOARD_REGION") or self.r2_own_region
        )
        explicit_dashboard_access_key = _read_optional_str(
            "LEOMA_DASHBOARD_WRITE_ACCESS_KEY"
        )
        explicit_dashboard_secret_key = _read_optional_str(
            "LEOMA_DASHBOARD_WRITE_SECRET_KEY"
        )
        # Bucket-scoped Hippius credentials can safely serve both roles only when the
        # operator explicitly chose the same bucket. State objects remain private;
        # JsonBucketStore applies public-read solely to dashboard.json.
        if self.r2_own_bucket and self.r2_own_bucket == self.dashboard_bucket:
            # An explicitly selected Hippius bucket must not accidentally inherit
            # stale legacy R2 credentials. If no canonical HIPPIUS_OWN key was set,
            # the key scoped to this exact dashboard bucket is the authoritative one.
            canonical_hippius_key = _read_optional_str("HIPPIUS_OWN_WRITE_ACCESS_KEY")
            canonical_hippius_secret = _read_optional_str("HIPPIUS_OWN_WRITE_SECRET_KEY")
            if self.object_storage_backend == "hippius" and _read_optional_str(
                "HIPPIUS_OWN_BUCKET"
            ):
                self.r2_own_write_access_key = (
                    canonical_hippius_key
                    or explicit_dashboard_access_key
                    or self.r2_own_write_access_key
                )
                self.r2_own_write_secret_key = (
                    canonical_hippius_secret
                    or explicit_dashboard_secret_key
                    or self.r2_own_write_secret_key
                )
            else:
                self.r2_own_write_access_key = (
                    self.r2_own_write_access_key or explicit_dashboard_access_key
                )
                self.r2_own_write_secret_key = (
                    self.r2_own_write_secret_key or explicit_dashboard_secret_key
                )
        self.dashboard_write_access_key = (
            explicit_dashboard_access_key or self.r2_own_write_access_key
        )
        self.dashboard_write_secret_key = (
            explicit_dashboard_secret_key or self.r2_own_write_secret_key
        )

        self.openai_api_key = _read_optional_str("OPENAI_API_KEY")
        self.gemini_api_key = _read_optional_str("GEMINI_API_KEY")
        self.required_video_width = _read_int("REQUIRED_VIDEO_WIDTH", 832)
        self.required_video_height = _read_int("REQUIRED_VIDEO_HEIGHT", 480)
        self.video_resolution_tolerance = _read_int("VIDEO_RESOLUTION_TOLERANCE", 32)
        self.min_video_size = _read_int("MIN_VIDEO_SIZE", 1_000_000)
        self.max_video_size = _read_int("MAX_VIDEO_SIZE", 200_000_000)
        self.clip_duration = _read_int("CLIP_DURATION", 5)
        self.max_concurrent_miners = _read_int("MAX_CONCURRENT_MINERS", 5)
        self.hf_token = _read_optional_str("HF_TOKEN")
        self.corpus_min_duration = _read_int("CORPUS_MIN_DURATION", 5)
        self.corpus_max_duration = _read_int("CORPUS_MAX_DURATION", 300)
        self.corpus_target_resolution = _read_str("CORPUS_TARGET_RESOLUTION", "720")
        self.corpus_max_filesize = _read_int("CORPUS_MAX_FILESIZE", 200_000_000)


_settings_instance = Settings()
settings = _settings_instance

NETUID = settings.netuid
EPOCH_LEN = settings.epoch_len
REQUEST_TIMEOUT = settings.request_timeout
WALLET_NAME = settings.wallet_name
HOTKEY_NAME = settings.hotkey_name
NETWORK = settings.network
HIPPIUS_ENDPOINT = settings.hippius_endpoint
HIPPIUS_REGION = settings.hippius_region
HIPPIUS_VIDEOS_READ_ACCESS_KEY = settings.hippius_videos_read_access_key
HIPPIUS_VIDEOS_READ_SECRET_KEY = settings.hippius_videos_read_secret_key
HIPPIUS_VIDEOS_WRITE_ACCESS_KEY = settings.hippius_videos_write_access_key
HIPPIUS_VIDEOS_WRITE_SECRET_KEY = settings.hippius_videos_write_secret_key
HIPPIUS_HUB_TOKEN = settings.hippius_hub_token
HIPPIUS_HUB_USERNAME = settings.hippius_hub_username
HIPPIUS_HUB_PASSWORD = settings.hippius_hub_password
MODEL_CACHE_DIR = settings.model_cache_dir
OBJECT_STORAGE_BACKEND = settings.object_storage_backend
SOURCE_BUCKET = (
    settings.r2_source_bucket
    if settings.object_storage_backend == "r2"
    else settings.source_bucket
)
OPENAI_API_KEY = settings.openai_api_key
GEMINI_API_KEY = settings.gemini_api_key
REQUIRED_VIDEO_WIDTH = settings.required_video_width
REQUIRED_VIDEO_HEIGHT = settings.required_video_height
VIDEO_RESOLUTION_TOLERANCE = settings.video_resolution_tolerance
MIN_VIDEO_SIZE = settings.min_video_size
MAX_VIDEO_SIZE = settings.max_video_size
CLIP_DURATION = settings.clip_duration
MAX_CONCURRENT_MINERS = settings.max_concurrent_miners
HF_TOKEN = settings.hf_token
CORPUS_MIN_DURATION = settings.corpus_min_duration
CORPUS_MAX_DURATION = settings.corpus_max_duration
CORPUS_TARGET_RESOLUTION = settings.corpus_target_resolution
CORPUS_MAX_FILESIZE = settings.corpus_max_filesize
SAMPLING_ROTATION_INTERVAL = settings.sampling_rotation_interval
R2_OWN_BUCKET = settings.r2_own_bucket
DASHBOARD_BUCKET = settings.dashboard_bucket

# Ensure leoma logger has a handler when not configured by application
if not logger.handlers:
    _log_level = getattr(
        logging,
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.setLevel(_log_level)
    logger.addHandler(_handler)
