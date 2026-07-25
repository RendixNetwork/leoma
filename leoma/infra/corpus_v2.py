"""Offline, resumable corpus-v2 preparation for large TI2V exams.

The active evaluator still reads the small v1 manifest.  This module deliberately
does not change that consensus path: it builds an immutable, sharded pilot corpus
which can be inspected and calibrated before a later consensus migration.

The work directory contains one SQLite ledger plus content-addressed five-second
clips and lossless first-frame PNGs.  SQLite is used instead of millions of marker
files so eight caption workers can claim work atomically and a stopped run can
resume without guessing what finished.  Each PNG is written from decoded truth
frame zero, and both its file digest and decoded RGB digest are pinned.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from leoma.eval.digests import canonical_json, digest_file, digest_frames, sha256_bytes
from leoma.infra.corpus_manifest import MIN_MOTION_ENERGY
from leoma.infra.video_utils import (
    decode_frames_rgb,
    detect_scene_cuts_sync,
    get_video_duration_sync,
    motion_energy,
)


CORPUS_V2_VERSION = 2
LEDGER_SCHEMA_VERSION = 5
DEFAULT_SHARD_SIZE = 5_000
CAPTION_PROMPT_VERSION = "leoma-motion-caption-v1"
CAPTION_PROMPT = (
    "Describe this short video in one concise sentence for a text-guided "
    "image-to-video model. Mention the main subject, visible action, environment, "
    "and camera motion when apparent. Describe only visible facts. Do not mention "
    "frames, timestamps, image quality, or that you are an AI."
)


@dataclass(frozen=True)
class PilotSpec:
    """Pinned preparation settings for one pilot work directory."""

    corpus_id: str
    width: int = 832
    height: int = 480
    fps: int = 16
    num_frames: int = 81
    max_windows_per_source: int = 8
    min_window_gap_seconds: float = 10.0
    min_motion_energy: float = MIN_MOTION_ENERGY
    scene_threshold: float = 0.18
    boundary_margin_seconds: float = 0.15
    encode_crf: int = 18
    encode_preset: str = "medium"

    def __post_init__(self) -> None:
        if not self.corpus_id.strip():
            raise ValueError("corpus_id is required")
        if self.width < 8 or self.height < 8 or self.fps < 1 or self.num_frames < 2:
            raise ValueError("invalid decode dimensions/frame settings")
        if self.max_windows_per_source < 1:
            raise ValueError("max_windows_per_source must be positive")
        if self.min_window_gap_seconds < self.clip_seconds:
            raise ValueError("min_window_gap_seconds must be at least one clip duration")

    @property
    def clip_seconds(self) -> float:
        # Wan video frame counts are 4n+1.  81 frames at 16fps is the subnet's
        # nominal five-second window and remains compatible with Wan2.2.
        return self.num_frames / self.fps

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    source_key: str
    source_sha256: str
    source_start_ms: int
    clip_path: str
    clip_sha256: str
    truth_frames_sha256: str
    first_frame_path: str
    first_frame_file_sha256: str
    first_frame_sha256: str
    motion_energy: float
    width: int
    height: int
    fps: int
    num_frames: int


@dataclass(frozen=True)
class CaptionTask:
    sample_id: str
    clip_path: str


def _now() -> float:
    return time.time()


def _canonical_text(value: dict) -> str:
    return canonical_json(value).decode("utf-8")


_RETRYABLE_SOURCE_ERROR_MARKERS = (
    "connect timeout",
    "connection aborted",
    "connection refused",
    "connection reset",
    "connection timed out",
    "internalerror",
    "max retries exceeded",
    "read timed out",
    "request timeout",
    "requesttimeout",
    "service unavailable",
    "serviceunavailable",
    "slowdown",
    "temporary failure",
    "temporarily unavailable",
)


def _source_error_is_retryable(error: BaseException) -> bool:
    """Keep transport/storage outages distinct from terminal video rejection."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _RETRYABLE_SOURCE_ERROR_MARKERS)


class PilotLedger:
    """Transactional work ledger shared by preparation/caption/publish stages."""

    def __init__(self, path: str | os.PathLike[str], spec: Optional[PilotSpec] = None):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        if spec is not None:
            self.bind_spec(spec)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=60.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=60000")
        return db

    def _init_schema(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    samples INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    source_key TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    source_start_ms INTEGER NOT NULL,
                    clip_path TEXT NOT NULL,
                    clip_sha256 TEXT NOT NULL,
                    truth_frames_sha256 TEXT NOT NULL,
                    first_frame_path TEXT NOT NULL,
                    first_frame_file_sha256 TEXT NOT NULL,
                    first_frame_sha256 TEXT NOT NULL,
                    motion_energy REAL NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    fps INTEGER NOT NULL,
                    num_frames INTEGER NOT NULL,
                    caption TEXT NOT NULL DEFAULT '',
                    caption_model TEXT NOT NULL DEFAULT '',
                    caption_revision TEXT NOT NULL DEFAULT '',
                    caption_prompt_version TEXT NOT NULL DEFAULT '',
                    caption_frame_count INTEGER NOT NULL DEFAULT 0,
                    caption_worker TEXT NOT NULL DEFAULT '',
                    clip_key TEXT NOT NULL DEFAULT '',
                    first_frame_key TEXT NOT NULL DEFAULT '',
                    publish_approved INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'prepared',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    UNIQUE(source_sha256, source_start_ms, width, height, fps, num_frames)
                );
                CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status, sample_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_truth_unique
                    ON samples(truth_frames_sha256);
                CREATE TABLE IF NOT EXISTS qa_reviews (
                    sample_id TEXT PRIMARY KEY,
                    verdict TEXT NOT NULL CHECK(verdict IN ('pass', 'fail')),
                    reviewer TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
                );
                """
            )
            version = db.execute("SELECT value FROM meta WHERE key='ledger_schema_version'").fetchone()
            if version is None:
                db.execute(
                    "INSERT INTO meta(key, value) VALUES('ledger_schema_version', ?)",
                    (str(LEDGER_SCHEMA_VERSION),),
                )
            elif int(version[0]) in {1, 2, 3, 4} and LEDGER_SCHEMA_VERSION == 5:
                # Schema 2 added qa_reviews (created above); schema 3 makes an
                # interrupted, QA-approved upload batch resumable; schema 4 pins
                # caption frame counts; schema 5 stores real first-frame images.
                columns = {row[1] for row in db.execute("PRAGMA table_info(samples)")}
                if "publish_approved" not in columns:
                    db.execute(
                        "ALTER TABLE samples ADD COLUMN publish_approved "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                if "caption_frame_count" not in columns:
                    db.execute(
                        "ALTER TABLE samples ADD COLUMN caption_frame_count "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                if "first_frame_path" not in columns:
                    db.execute(
                        "ALTER TABLE samples ADD COLUMN first_frame_path "
                        "TEXT NOT NULL DEFAULT ''"
                    )
                if "first_frame_file_sha256" not in columns:
                    db.execute(
                        "ALTER TABLE samples ADD COLUMN first_frame_file_sha256 "
                        "TEXT NOT NULL DEFAULT ''"
                    )
                if "first_frame_key" not in columns:
                    db.execute(
                        "ALTER TABLE samples ADD COLUMN first_frame_key "
                        "TEXT NOT NULL DEFAULT ''"
                    )
                db.execute(
                    "UPDATE meta SET value=? WHERE key='ledger_schema_version'",
                    (str(LEDGER_SCHEMA_VERSION),),
                )
            elif int(version[0]) != LEDGER_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported pilot ledger schema {version[0]} (expected {LEDGER_SCHEMA_VERSION})"
                )

    def bind_spec(self, spec: PilotSpec) -> None:
        encoded = _canonical_text(spec.as_dict())
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='pilot_spec'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta(key, value) VALUES('pilot_spec', ?)", (encoded,))
            elif row[0] != encoded:
                raise ValueError(
                    "work directory is already bound to different pilot settings; "
                    "use a new work directory instead of mixing two exams"
                )

    def read_spec(self) -> PilotSpec:
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='pilot_spec'").fetchone()
        if row is None:
            raise ValueError("pilot ledger has no bound spec")
        return PilotSpec(**json.loads(row[0]))

    def bind_caption_spec(
        self,
        *,
        model: str,
        revision: str,
        frame_count: int,
        max_new_tokens: int,
    ) -> dict:
        value = {
            "model": str(model),
            "revision": str(revision),
            "prompt_version": CAPTION_PROMPT_VERSION,
            "frame_count": int(frame_count),
            "max_new_tokens": int(max_new_tokens),
        }
        encoded = _canonical_text(value)
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='caption_spec'").fetchone()
            if row is None:
                db.execute("INSERT INTO meta(key,value) VALUES('caption_spec',?)", (encoded,))
            elif row[0] != encoded:
                raise ValueError(
                    "work directory is already bound to a different caption model/settings; "
                    "use caption-reset-all or a new work directory"
                )
        return value

    def read_caption_spec(self) -> Optional[dict]:
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='caption_spec'").fetchone()
        return json.loads(row[0]) if row is not None else None

    def source_finished(self, source_key: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT status FROM sources WHERE source_key=?", (source_key,)).fetchone()
        # Invalid/too-short videos are terminal corpus rejections. Retrying them
        # on every resume wastes bandwidth and can hide a real storage outage.
        # Interrupted and explicitly retryable rows remain eligible.
        return bool(row and row[0] in {"done", "failed"})

    def record_source(self, source_key: str, *, status: str, samples: int = 0, error: str = "") -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO sources(source_key,status,samples,error,updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(source_key) DO UPDATE SET
                     status=excluded.status, samples=excluded.samples,
                     error=excluded.error, updated_at=excluded.updated_at""",
                (source_key, status, int(samples), str(error)[:2000], _now()),
            )

    def add_sample(self, sample: PreparedSample) -> bool:
        values = asdict(sample)
        values["updated_at"] = _now()
        columns = list(values)
        placeholders = ",".join("?" for _ in columns)
        with self.connect() as db:
            cur = db.execute(
                f"INSERT OR IGNORE INTO samples({','.join(columns)}) VALUES({placeholders})",
                tuple(values[c] for c in columns),
            )
            if cur.rowcount:
                return True
            row = db.execute("SELECT * FROM samples WHERE sample_id=?", (sample.sample_id,)).fetchone()
            if row is None:
                duplicate = db.execute(
                    "SELECT sample_id FROM samples WHERE truth_frames_sha256=?",
                    (sample.truth_frames_sha256,),
                ).fetchone()
                if duplicate is not None:
                    return False
        if row is not None:
            changed = [
                key for key in asdict(sample)
                if row[key] != getattr(sample, key)
            ]
            # sample_id is derived from source content + window + encode settings,
            # not the mutable bucket key. Two keys can legitimately alias the same
            # source object; keep the first provenance key and deduplicate the alias.
            if not changed or changed == ["source_key"]:
                return False
        raise ValueError(f"sample id collision or changed immutable sample {sample.sample_id}")

    def has_sample(self, sample_id: str) -> bool:
        with self.connect() as db:
            row = db.execute(
                "SELECT 1 FROM samples WHERE sample_id=?", (sample_id,),
            ).fetchone()
        return row is not None

    def recover_caption_claims(self) -> int:
        """Return interrupted caption claims to the queue before workers start."""
        with self.connect() as db:
            cur = db.execute(
                "UPDATE samples SET status='prepared', caption_worker='', updated_at=? "
                "WHERE status='captioning'",
                (_now(),),
            )
            return int(cur.rowcount)

    def caption_queue_size(self, *, max_attempts: int = 3) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM samples WHERE status='prepared' AND attempts < ?",
                (int(max_attempts),),
            ).fetchone()
        return int(row[0])

    def retry_failed_captions(self) -> int:
        """Retry terminal caption failures after an operator fixes their cause."""
        with self.connect() as db:
            cur = db.execute(
                """UPDATE samples SET status='prepared',attempts=0,error='',
                   caption_worker='',updated_at=? WHERE status='failed'""",
                (_now(),),
            )
            return int(cur.rowcount)

    def reset_all_captions(self) -> int:
        """Clear every caption/QA decision before deliberately changing settings."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            blocked = db.execute(
                "SELECT COUNT(*) FROM samples WHERE status='published' OR publish_approved=1"
            ).fetchone()
            if int(blocked[0]):
                raise ValueError("cannot reset captions after a batch is approved or published")
            db.execute("DELETE FROM qa_reviews")
            cur = db.execute(
                """UPDATE samples SET status='prepared',caption='',caption_model='',
                   caption_revision='',caption_prompt_version='',caption_frame_count=0,
                   caption_worker='',attempts=0,error='',updated_at=?""",
                (_now(),),
            )
            db.execute("DELETE FROM meta WHERE key='caption_spec'")
            return int(cur.rowcount)

    def claim_caption(self, worker: str, *, max_attempts: int = 3) -> Optional[CaptionTask]:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT sample_id, clip_path FROM samples
                   WHERE status='prepared' AND attempts < ?
                   ORDER BY sample_id LIMIT 1""",
                (int(max_attempts),),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            db.execute(
                """UPDATE samples SET status='captioning', caption_worker=?,
                   attempts=attempts+1, error='', updated_at=? WHERE sample_id=?""",
                (worker, _now(), row["sample_id"]),
            )
            db.commit()
            return CaptionTask(row["sample_id"], row["clip_path"])
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish_caption(
        self,
        sample_id: str,
        *,
        caption: str,
        model: str,
        revision: str,
        worker: str,
        frame_count: int,
        prompt_version: str = CAPTION_PROMPT_VERSION,
    ) -> None:
        caption = normalize_caption(caption)
        validate_caption(caption)
        actual = {
            "model": str(model),
            "revision": str(revision),
            "prompt_version": str(prompt_version),
            "frame_count": int(frame_count),
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            bound = db.execute("SELECT value FROM meta WHERE key='caption_spec'").fetchone()
            if bound is None:
                raise ValueError("caption settings are not bound to this work directory")
            caption_spec = json.loads(bound[0])
            expected = {key: caption_spec[key] for key in actual}
            if actual != expected:
                raise ValueError(f"caption settings do not match bound workdir settings: {actual}")
            cur = db.execute(
                """UPDATE samples SET caption=?,caption_model=?,caption_revision=?,
                   caption_prompt_version=?,caption_frame_count=?,caption_worker=?,
                   status='captioned',error='',updated_at=?
                   WHERE sample_id=? AND status='captioning'""",
                (
                    caption, model, revision, prompt_version, int(frame_count), worker,
                    _now(), sample_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError(f"sample {sample_id} is not claimed for captioning")

    def fail_caption(self, sample_id: str, error: str, *, max_attempts: int = 3) -> None:
        with self.connect() as db:
            row = db.execute("SELECT attempts FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown sample {sample_id}")
            status = "failed" if int(row[0]) >= int(max_attempts) else "prepared"
            db.execute(
                "UPDATE samples SET status=?,caption_worker='',error=?,updated_at=? WHERE sample_id=?",
                (status, str(error)[:2000], _now(), sample_id),
            )

    def captioned(self, *, limit: Optional[int] = None) -> Iterator[sqlite3.Row]:
        query = "SELECT * FROM samples WHERE status='captioned' ORDER BY sample_id"
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        with self.connect() as db:
            rows = list(db.execute(query, params))
        yield from rows

    def published(self) -> Iterator[sqlite3.Row]:
        with self.connect() as db:
            rows = list(db.execute("SELECT * FROM samples WHERE status='published' ORDER BY sample_id"))
        yield from rows

    def qa_candidates(self) -> list[sqlite3.Row]:
        """Return locally reviewable captions (never already-published samples)."""
        with self.connect() as db:
            rows = list(db.execute(
                """SELECT s.*,q.verdict AS qa_verdict,q.reviewer AS qa_reviewer,
                          q.notes AS qa_notes
                   FROM samples s LEFT JOIN qa_reviews q USING(sample_id)
                   WHERE (s.status='captioned' AND s.publish_approved=0)
                      OR s.status='qa_rejected'
                   ORDER BY s.sample_id"""
            ))
        return rows

    def record_qa_review(
        self,
        sample_id: str,
        *,
        verdict: str,
        reviewer: str,
        notes: str = "",
    ) -> None:
        verdict = str(verdict).strip().lower()
        reviewer = str(reviewer).strip()
        if verdict not in {"pass", "fail"}:
            raise ValueError("QA verdict must be 'pass' or 'fail'")
        if not reviewer:
            raise ValueError("QA reviewer is required")
        with self.connect() as db:
            row = db.execute(
                "SELECT status,publish_approved FROM samples WHERE sample_id=?", (sample_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown QA sample {sample_id}")
            if row[0] not in {"captioned", "qa_rejected"}:
                raise ValueError(
                    f"sample {sample_id} cannot be reviewed while status={row[0]!r}"
                )
            if row[0] == "captioned" and int(row[1]):
                raise ValueError(
                    f"sample {sample_id} is already approved for publishing"
                )
            db.execute(
                """INSERT INTO qa_reviews(sample_id,verdict,reviewer,notes,updated_at)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(sample_id) DO UPDATE SET
                     verdict=excluded.verdict,reviewer=excluded.reviewer,
                     notes=excluded.notes,updated_at=excluded.updated_at""",
                (sample_id, verdict, reviewer, str(notes)[:2000], _now()),
            )
            status = "captioned" if verdict == "pass" else "qa_rejected"
            db.execute(
                "UPDATE samples SET status=?,updated_at=? WHERE sample_id=?",
                (status, _now(), sample_id),
            )

    def retry_qa_rejected(self) -> int:
        """Clear rejected captions and put their clips back in the caption queue."""
        with self.connect() as db:
            cur = db.execute(
                """UPDATE samples SET status='prepared',caption='',caption_model='',
                   caption_revision='',caption_prompt_version='',caption_worker='',
                   attempts=0,error='',updated_at=? WHERE status='qa_rejected'""",
                (_now(),),
            )
            db.execute(
                "DELETE FROM qa_reviews WHERE sample_id IN "
                "(SELECT sample_id FROM samples WHERE status='prepared')"
            )
            return int(cur.rowcount)

    def qa_report(self) -> dict:
        with self.connect() as db:
            counts = {row[0]: int(row[1]) for row in db.execute(
                "SELECT verdict,COUNT(*) FROM qa_reviews GROUP BY verdict"
            )}
        passed = counts.get("pass", 0)
        failed = counts.get("fail", 0)
        reviewed = passed + failed
        return {
            "reviewed": reviewed,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / reviewed, 6) if reviewed else None,
        }

    def pending_qa_report(self) -> dict:
        """Report reviews for the not-yet-published caption batch only."""
        with self.connect() as db:
            counts = {row[0]: int(row[1]) for row in db.execute(
                """SELECT q.verdict,COUNT(*) FROM qa_reviews q
                   JOIN samples s USING(sample_id)
                   WHERE (s.status='captioned' AND s.publish_approved=0)
                      OR s.status='qa_rejected'
                   GROUP BY q.verdict"""
            )}
        passed = counts.get("pass", 0)
        failed = counts.get("fail", 0)
        reviewed = passed + failed
        return {
            "reviewed": reviewed,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / reviewed, 6) if reviewed else None,
        }

    def assert_qa_gate(self, *, min_reviews: int, min_pass_rate: float) -> dict:
        if min_reviews < 0:
            raise ValueError("QA minimum review count cannot be negative")
        if not 0.0 <= min_pass_rate <= 1.0:
            raise ValueError("QA minimum pass rate must be between zero and one")
        report = self.pending_qa_report()
        if report["reviewed"] < min_reviews:
            raise ValueError(
                f"QA gate needs {min_reviews} reviews; only {report['reviewed']} are recorded"
            )
        if min_reviews and report["pass_rate"] < min_pass_rate:
            raise ValueError(
                f"QA pass rate {report['pass_rate']:.3%} is below required "
                f"{min_pass_rate:.3%}"
            )
        return report

    def approve_publish_batch(self, *, min_reviews: int, min_pass_rate: float) -> dict:
        """Freeze the current QA-passed batch so interrupted uploads can resume."""
        if min_reviews < 0:
            raise ValueError("QA minimum review count cannot be negative")
        if not 0.0 <= min_pass_rate <= 1.0:
            raise ValueError("QA minimum pass rate must be between zero and one")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT COUNT(*) FROM samples
                   WHERE (status='captioned' AND publish_approved=0)
                      OR status='qa_rejected'"""
            ).fetchone()
            pending = int(row[0])
            if not pending:
                return {
                    "reviewed": 0, "passed": 0, "failed": 0, "pass_rate": None,
                    "approved": 0, "excluded": 0,
                }
            counts = {result[0]: int(result[1]) for result in db.execute(
                """SELECT q.verdict,COUNT(*) FROM qa_reviews q
                   JOIN samples s USING(sample_id)
                   WHERE (s.status='captioned' AND s.publish_approved=0)
                      OR s.status='qa_rejected'
                   GROUP BY q.verdict"""
            )}
            passed = counts.get("pass", 0)
            failed = counts.get("fail", 0)
            reviewed = passed + failed
            report = {
                "reviewed": reviewed,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / reviewed, 6) if reviewed else None,
            }
            if reviewed < min_reviews:
                raise ValueError(
                    f"QA gate needs {min_reviews} reviews; only {reviewed} are recorded"
                )
            if min_reviews and report["pass_rate"] < min_pass_rate:
                raise ValueError(
                    f"QA pass rate {report['pass_rate']:.3%} is below required "
                    f"{min_pass_rate:.3%}"
                )
            rejected = db.execute(
                "UPDATE samples SET status='excluded',updated_at=? WHERE status='qa_rejected'",
                (_now(),),
            )
            approved = db.execute(
                """UPDATE samples SET publish_approved=1,updated_at=?
                   WHERE status='captioned' AND publish_approved=0""",
                (_now(),),
            )
        return {
            **report,
            "approved": int(approved.rowcount),
            "excluded": int(rejected.rowcount),
        }

    def record_first_frame(
        self,
        sample_id: str,
        *,
        path: str,
        file_sha256: str,
        rgb_sha256: str,
    ) -> None:
        """Attach a verified PNG to a legacy row prepared before schema 5."""
        with self.connect() as db:
            row = db.execute(
                "SELECT first_frame_sha256,status FROM samples WHERE sample_id=?",
                (sample_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown sample {sample_id}")
            if row["status"] == "published":
                raise ValueError(f"cannot change published sample {sample_id}")
            if row["first_frame_sha256"] != rgb_sha256:
                raise ValueError(
                    f"first frame RGB digest does not match prepared sample {sample_id}"
                )
            db.execute(
                """UPDATE samples SET first_frame_path=?,first_frame_file_sha256=?,
                   updated_at=? WHERE sample_id=?""",
                (str(path), str(file_sha256), _now(), sample_id),
            )

    def mark_published(self, sample_id: str, clip_key: str, first_frame_key: str) -> None:
        if not clip_key or not first_frame_key:
            raise ValueError("both clip and first-frame keys are required")
        with self.connect() as db:
            cur = db.execute(
                """UPDATE samples SET status='published',clip_key=?,first_frame_key=?,
                   error='',updated_at=?
                   WHERE sample_id=? AND status='captioned'""",
                (clip_key, first_frame_key, _now(), sample_id),
            )
            if cur.rowcount != 1:
                raise ValueError(f"sample {sample_id} is not ready to publish")

    def stats(self) -> dict:
        with self.connect() as db:
            samples = {row[0]: int(row[1]) for row in db.execute(
                "SELECT status,COUNT(*) FROM samples GROUP BY status"
            )}
            sources = {row[0]: int(row[1]) for row in db.execute(
                "SELECT status,COUNT(*) FROM sources GROUP BY status"
            )}
        return {
            "samples": samples,
            "sources": sources,
            "total_samples": sum(samples.values()),
            "qa": self.qa_report(),
        }


def normalize_caption(text: str) -> str:
    text = " ".join(str(text).replace("\x00", " ").split()).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def validate_caption(text: str) -> None:
    if len(text) < 12:
        raise ValueError("caption is too short")
    if len(text) > 600:
        raise ValueError("caption is too long")
    boilerplate = (
        r"\bas an ai\b",
        r"\bi cannot\b",
        r"\bi'm unable\b",
        r"\bthe provided frames\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in boilerplate):
        raise ValueError("caption contains model boilerplate")


def select_qa_samples(
    rows: Sequence[sqlite3.Row],
    *,
    count: int,
    seed: str,
) -> list[sqlite3.Row]:
    """Select a deterministic, motion-stratified, source-diverse audit sample."""
    import hashlib

    if count < 1:
        raise ValueError("QA sample count must be positive")
    if not rows:
        raise ValueError("no captioned samples are ready for QA")
    ordered = sorted(rows, key=lambda row: (float(row["motion_energy"]), row["sample_id"]))
    strata_count = min(10, len(ordered))
    strata: list[list[sqlite3.Row]] = [[] for _ in range(strata_count)]
    for rank, row in enumerate(ordered):
        index = min(strata_count - 1, rank * strata_count // len(ordered))
        strata[index].append(row)
    for stratum in strata:
        stratum.sort(key=lambda row: hashlib.sha256(
            f"{seed}\0{row['source_key']}\0{row['sample_id']}".encode()
        ).digest())

    selected: list[sqlite3.Row] = []
    used_sources: set[str] = set()
    target = min(count, len(ordered))
    while len(selected) < target:
        progressed = False
        for stratum in strata:
            if not stratum or len(selected) >= target:
                continue
            # Prefer an unseen source without making low-cardinality corpora fail.
            choice = next(
                (index for index, row in enumerate(stratum)
                 if row["source_key"] not in used_sources),
                0,
            )
            row = stratum.pop(choice)
            selected.append(row)
            used_sources.add(str(row["source_key"]))
            progressed = True
        if not progressed:
            break
    return selected


def export_qa_bundle(
    ledger: PilotLedger,
    *,
    output_dir: str | os.PathLike[str],
    count: int,
    seed: str = "leoma-corpus-v2-qa-v1",
) -> tuple[str, str]:
    """Write an editable JSONL review sheet and a local HTML video gallery."""
    import html

    rows = select_qa_samples(ledger.qa_candidates(), count=count, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    review_path = out / "reviews.jsonl"
    records = []
    cards = []
    for row in rows:
        path = Path(row["clip_path"]).resolve()
        first_frame_path = Path(str(row["first_frame_path"])).resolve()
        if not row["first_frame_path"] or not first_frame_path.is_file():
            raise FileNotFoundError(
                f"first-frame PNG is missing for {row['sample_id']}; "
                "run corpus v2 materialize-first-frames"
            )
        record = {
            "sample_id": row["sample_id"],
            "source_key": row["source_key"],
            "motion_energy": row["motion_energy"],
            "caption": row["caption"],
            "first_frame_path": str(first_frame_path),
            "verdict": row["qa_verdict"] or "",
            "reviewer": row["qa_reviewer"] or "",
            "notes": row["qa_notes"] or "",
        }
        records.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
        cards.append(
            '<article><img loading="lazy" src="'
            + html.escape(first_frame_path.as_uri(), quote=True)
            + '" alt="First conditioning frame"><video controls preload="metadata" src="'
            + html.escape(path.as_uri(), quote=True)
            + '"></video><p class="caption">'
            + html.escape(str(row["caption"]))
            + '</p><p class="meta">sample '
            + html.escape(str(row["sample_id"]))
            + ' · motion '
            + html.escape(str(row["motion_energy"]))
            + '</p></article>'
        )
    review_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    html_path = out / "review.html"
    html_path.write_text(
        """<!doctype html><meta charset=\"utf-8\"><title>Leoma corpus QA</title>
<style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}
.note{padding:1rem;background:#eef3ff;border-radius:8px}section{display:grid;
grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}article{border:1px
solid #ccc;border-radius:8px;padding:10px}img,video{width:100%}.caption{font-size:17px}
.meta{font:12px monospace;color:#555;overflow-wrap:anywhere}</style>
<h1>Leoma corpus-v2 caption audit</h1><p class=\"note\">Review every video against its
first conditioning frame and caption. In <code>reviews.jsonl</code>, set each verdict
to <code>pass</code> or <code>fail</code>, set reviewer, and add notes for failures. Then run
<code>leoma corpus v2 qa-import</code>.</p><section>"""
        + "".join(cards)
        + "</section>",
        encoding="utf-8",
    )
    return str(review_path), str(html_path)


def import_qa_reviews(ledger: PilotLedger, path: str | os.PathLike[str]) -> dict:
    """Validate and import a reviewer-edited JSONL sheet transactionally per row."""
    seen: set[str] = set()
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid QA JSON on line {line_number}: {exc}") from exc
            sample_id = str(record.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"missing or duplicate sample_id on QA line {line_number}")
            seen.add(sample_id)
            verdict = str(record.get("verdict") or "").strip().lower()
            reviewer = str(record.get("reviewer") or "").strip()
            if verdict not in {"pass", "fail"} or not reviewer:
                raise ValueError(
                    f"QA line {line_number} needs verdict pass/fail and a reviewer"
                )
            records.append((sample_id, verdict, reviewer, str(record.get("notes") or "")))
    if not records:
        raise ValueError("QA review file is empty")
    for sample_id, verdict, reviewer, notes in records:
        ledger.record_qa_review(
            sample_id, verdict=verdict, reviewer=reviewer, notes=notes,
        )
    return ledger.qa_report()


def choose_window_starts(
    *,
    duration: float,
    scene_cuts: Sequence[float],
    source_key: str,
    clip_seconds: float,
    max_windows: int,
    min_gap_seconds: float,
    boundary_margin: float,
) -> list[float]:
    """Choose deterministic, non-overlapping single-shot windows for one source."""
    import hashlib

    if duration < clip_seconds:
        return []
    cuts = sorted({float(c) for c in scene_cuts if 0.0 < float(c) < duration})
    bounds = [0.0, *cuts, float(duration)]
    candidates: list[float] = []
    step = max(float(min_gap_seconds), float(clip_seconds))
    for index, (left, right) in enumerate(zip(bounds, bounds[1:])):
        usable_left = left + (boundary_margin if index else 0.0)
        usable_right = right - (boundary_margin if index < len(bounds) - 2 else 0.0)
        last_start = usable_right - clip_seconds
        if last_start < usable_left:
            continue
        count = int(math.floor((last_start - usable_left) / step)) + 1
        if count == 1:
            starts = [(usable_left + last_start) / 2.0]
        else:
            # Spread candidates across a long shot instead of clustering at its head.
            starts = [usable_left + i * (last_start - usable_left) / (count - 1) for i in range(count)]
        candidates.extend(starts)

    ranked = sorted(
        candidates,
        key=lambda start: hashlib.sha256(f"{source_key}\0{start:.3f}".encode()).digest(),
    )[:max_windows]
    # Quantize toward the beginning. Normal rounding can push a mathematically valid
    # last start a fraction of a millisecond beyond the source/shot boundary.
    return sorted(math.floor(float(start) * 1000.0) / 1000.0 for start in ranked)


def order_source_keys(keys: Iterable[str], *, seed: str) -> list[str]:
    """Deterministically shuffle source keys so bounded pilots are representative."""
    import hashlib

    unique = {str(key) for key in keys}
    return sorted(
        unique,
        key=lambda key: (hashlib.sha256(f"{seed}\0{key}".encode()).digest(), key),
    )


def _encode_window(source: str, output: str, *, start: float, spec: PilotSpec) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{output}.partial"
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-i", source,
        "-vf", f"fps={spec.fps},scale={spec.width}:{spec.height}:flags=bicubic",
        "-frames:v", str(spec.num_frames), "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", spec.encode_preset, "-crf", str(spec.encode_crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", tmp,
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        message = result.stderr.decode(errors="ignore")[:500]
        raise RuntimeError(f"ffmpeg could not encode window at {start:.3f}s: {message}")
    os.replace(tmp, output)


def _write_first_frame_png(frame, output: str | os.PathLike[str]) -> None:
    """Atomically store an RGB truth frame as a lossless PNG."""
    from PIL import Image

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.partial")
    try:
        image = Image.fromarray(frame).convert("RGB")
        image.save(tmp, format="PNG", compress_level=9, optimize=False)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def materialize_first_frame_images(
    ledger: PilotLedger,
    *,
    output_dir: str | os.PathLike[str],
    limit: Optional[int] = None,
    log=lambda _: None,
) -> dict:
    """Backfill lossless first-frame PNGs for pre-schema-5 retained clips."""
    query = """SELECT * FROM samples
               WHERE status!='published'
                 AND (first_frame_path='' OR first_frame_file_sha256='')
               ORDER BY sample_id"""
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    with ledger.connect() as db:
        rows = list(db.execute(query, params))

    materialized = 0
    out = Path(output_dir)
    for row in rows:
        clip_path = str(row["clip_path"])
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"prepared clip is missing: {clip_path}")
        truth = decode_frames_rgb(
            clip_path,
            start_seconds=0.0,
            duration_seconds=float(row["num_frames"]) / float(row["fps"]),
            fps=int(row["fps"]),
            num_frames=int(row["num_frames"]),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        rgb_sha256 = digest_frames(truth[:1])
        if rgb_sha256 != row["first_frame_sha256"]:
            raise ValueError(
                f"first decoded frame changed for {row['sample_id']}: "
                f"{rgb_sha256} != {row['first_frame_sha256']}"
            )
        path = out / row["sample_id"][:2] / f"{row['sample_id']}.png"
        _write_first_frame_png(truth[0], path)
        file_sha256 = digest_file(str(path))
        ledger.record_first_frame(
            row["sample_id"],
            path=str(path.resolve()),
            file_sha256=file_sha256,
            rgb_sha256=rgb_sha256,
        )
        materialized += 1
        log(f"materialized first frame {row['sample_id']} -> {path}")
    return {**ledger.stats(), "materialized_this_run": materialized}


def prepare_local_source(
    source_path: str,
    *,
    source_key: str,
    workdir: str | os.PathLike[str],
    spec: PilotSpec,
) -> list[PreparedSample]:
    """Extract all accepted windows from one local source video."""
    source_sha = digest_file(source_path)
    duration = get_video_duration_sync(source_path)
    cuts = detect_scene_cuts_sync(source_path, scene_threshold=spec.scene_threshold)
    starts = choose_window_starts(
        duration=duration,
        scene_cuts=cuts,
        source_key=source_key,
        clip_seconds=spec.clip_seconds,
        max_windows=spec.max_windows_per_source,
        min_gap_seconds=spec.min_window_gap_seconds,
        boundary_margin=spec.boundary_margin_seconds,
    )

    accepted: list[PreparedSample] = []
    for start in starts:
        identity = {
            "source_sha256": source_sha,
            "source_start_ms": int(round(start * 1000)),
            "width": spec.width,
            "height": spec.height,
            "fps": spec.fps,
            "num_frames": spec.num_frames,
            "encode_crf": spec.encode_crf,
            "encode_preset": spec.encode_preset,
        }
        sample_id = sha256_bytes(canonical_json(identity)).split(":", 1)[1]
        clip_path = Path(workdir) / "clips" / sample_id[:2] / f"{sample_id}.mp4"
        first_frame_path = (
            Path(workdir) / "first-frames" / sample_id[:2] / f"{sample_id}.png"
        )
        if not clip_path.exists():
            _encode_window(source_path, str(clip_path), start=start, spec=spec)
        try:
            truth = decode_frames_rgb(
                str(clip_path), start_seconds=0.0, duration_seconds=spec.clip_seconds,
                fps=spec.fps, num_frames=spec.num_frames, width=spec.width, height=spec.height,
            )
            energy = float(motion_energy(truth))
            if energy < spec.min_motion_energy:
                clip_path.unlink(missing_ok=True)
                first_frame_path.unlink(missing_ok=True)
                continue
            _write_first_frame_png(truth[0], first_frame_path)
            accepted.append(
                PreparedSample(
                    sample_id=sample_id,
                    source_key=source_key,
                    source_sha256=source_sha,
                    source_start_ms=identity["source_start_ms"],
                    clip_path=str(clip_path.resolve()),
                    clip_sha256=digest_file(str(clip_path)),
                    truth_frames_sha256=digest_frames(truth),
                    first_frame_path=str(first_frame_path.resolve()),
                    first_frame_file_sha256=digest_file(str(first_frame_path)),
                    first_frame_sha256=digest_frames(truth[:1]),
                    motion_energy=round(energy, 6),
                    width=spec.width,
                    height=spec.height,
                    fps=spec.fps,
                    num_frames=spec.num_frames,
                )
            )
        except Exception:
            clip_path.unlink(missing_ok=True)
            first_frame_path.unlink(missing_ok=True)
            raise
    return accepted


def prepare_bucket_sources(
    client,
    bucket: str,
    *,
    ledger: PilotLedger,
    workdir: str | os.PathLike[str],
    spec: PilotSpec,
    keys: Iterable[str],
    max_samples: int,
    workers: int = 1,
    log=lambda _: None,
) -> dict:
    """Prepare sources concurrently while committing them in deterministic order."""
    from concurrent.futures import ThreadPoolExecutor

    workers = int(workers)
    if workers < 1:
        raise ValueError("preparation worker count must be positive")

    def prepare_one(key: str) -> tuple[str, list[PreparedSample], Optional[Exception]]:
        try:
            with tempfile.TemporaryDirectory(prefix="leoma-corpus-v2-") as tmpdir:
                local = os.path.join(tmpdir, "source.mp4")
                client.fget_object(bucket, key, local)
                samples = prepare_local_source(
                    local, source_key=key, workdir=workdir, spec=spec,
                )
            return key, samples, None
        except Exception as exc:
            return key, [], exc

    def discard(samples: Sequence[PreparedSample]) -> None:
        for sample in samples:
            if ledger.has_sample(sample.sample_id):
                continue
            Path(sample.clip_path).unlink(missing_ok=True)
            Path(sample.first_frame_path).unlink(missing_ok=True)

    total_added = ledger.stats()["total_samples"]
    from collections import deque

    key_iterator = iter(keys)

    def next_pending_key() -> Optional[str]:
        for key in key_iterator:
            if ledger.source_finished(key):
                continue
            ledger.record_source(key, status="processing")
            return key
        return None

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="corpus-v2") as pool:
        # A rolling ordered queue keeps workers busy behind a slow source while
        # still committing strictly in source order. Four waves cap speculative
        # disk work near the final sample high-water mark.
        queued = deque()
        prefetch = workers * 4
        while len(queued) < prefetch:
            key = next_pending_key()
            if key is None:
                break
            queued.append((key, pool.submit(prepare_one, key)))

        while queued:
            expected_key, future = queued.popleft()
            key, samples, error = future.result()
            if key != expected_key:
                raise RuntimeError("preparation worker returned a different source key")
            if total_added >= max_samples:
                discard(samples)
                ledger.record_source(key, status="pending")
            elif error is not None:
                status = "retryable" if _source_error_is_retryable(error) else "failed"
                ledger.record_source(key, status=status, error=str(error))
                log(f"{key}: {status} ({error})")
            else:
                added = 0
                for sample in samples:
                    if total_added >= max_samples:
                        # The sample is valid but beyond this pilot's cap. Remove
                        # both artifacts so a capped run has bounded disk use.
                        if not ledger.has_sample(sample.sample_id):
                            Path(sample.clip_path).unlink(missing_ok=True)
                            Path(sample.first_frame_path).unlink(missing_ok=True)
                        continue
                    if ledger.add_sample(sample):
                        added += 1
                        total_added += 1
                ledger.record_source(key, status="done", samples=added)
                log(f"{key}: {added} accepted ({total_added}/{max_samples})")

            if total_added < max_samples:
                next_key = next_pending_key()
                if next_key is not None:
                    queued.append((next_key, pool.submit(prepare_one, next_key)))
    return ledger.stats()


def sample_manifest_entry(row: sqlite3.Row) -> dict:
    return {
        "sample_id": row["sample_id"],
        "source_key": row["source_key"],
        "source_sha256": row["source_sha256"],
        "source_start_ms": row["source_start_ms"],
        "clip_key": row["clip_key"],
        "clip_sha256": row["clip_sha256"],
        "first_frame_key": row["first_frame_key"],
        "first_frame_sha256": row["first_frame_file_sha256"],
        "first_frame_rgb_sha256": row["first_frame_sha256"],
        "truth_frames_sha256": row["truth_frames_sha256"],
        "caption": row["caption"],
        "caption_model": row["caption_model"],
        "caption_revision": row["caption_revision"],
        "caption_prompt_version": row["caption_prompt_version"],
        "caption_frame_count": row["caption_frame_count"],
        "motion_energy": row["motion_energy"],
        "decode": {
            "width": row["width"], "height": row["height"], "fps": row["fps"],
            "num_frames": row["num_frames"],
        },
    }


def _missing_object(exc: Exception) -> bool:
    return str(getattr(exc, "code", "")) in {
        "NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound", "404",
    }


def _object_matches(client, bucket: str, key: str, *, size: int, sha256: str) -> bool:
    try:
        stat = client.stat_object(bucket, key)
    except Exception as exc:
        if _missing_object(exc):
            return False
        raise
    metadata = {str(k).lower(): str(v) for k, v in (getattr(stat, "metadata", {}) or {}).items()}
    remote_sha = metadata.get("x-amz-meta-sha256") or metadata.get("sha256")
    return int(getattr(stat, "size", -1)) == int(size) and remote_sha == sha256


def publish_captioned_samples(
    client,
    bucket: str,
    *,
    ledger: PilotLedger,
    prefix: str,
    limit: Optional[int] = None,
    workers: int = 1,
    qa_min_reviews: int = 0,
    qa_min_pass_rate: float = 0.0,
    delete_local_after_upload: bool = False,
    log=lambda _: None,
) -> dict:
    """Publish each content-addressed clip/PNG pair and checkpoint each success."""
    clean_prefix = prefix.strip("/")
    if not clean_prefix:
        raise ValueError("a versioned publish prefix is required")
    if workers < 1:
        raise ValueError("publish workers must be at least one")
    ledger.approve_publish_batch(
        min_reviews=qa_min_reviews,
        min_pass_rate=qa_min_pass_rate,
    )
    published = 0
    query = "SELECT * FROM samples WHERE status='captioned' AND publish_approved=1 ORDER BY sample_id"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    with ledger.connect() as db:
        approved_rows = list(db.execute(query, params))

    def upload_pair(row: sqlite3.Row) -> tuple[str, str, str, str, str]:
        clip_path = str(row["clip_path"])
        first_frame_path = str(row["first_frame_path"])
        if not os.path.isfile(clip_path):
            raise FileNotFoundError(f"prepared clip is missing: {clip_path}")
        if not first_frame_path or not os.path.isfile(first_frame_path):
            raise FileNotFoundError(
                f"first-frame PNG is missing for {row['sample_id']}; "
                "run corpus v2 materialize-first-frames"
            )
        clip_sha256 = digest_file(clip_path)
        if clip_sha256 != row["clip_sha256"]:
            raise ValueError(
                f"clip {row['sample_id']} changed after preparation: "
                f"{clip_sha256} != {row['clip_sha256']}"
            )
        first_frame_sha256 = digest_file(first_frame_path)
        if first_frame_sha256 != row["first_frame_file_sha256"]:
            raise ValueError(
                f"first-frame PNG {row['sample_id']} changed after preparation: "
                f"{first_frame_sha256} != {row['first_frame_file_sha256']}"
            )

        clip_hex = clip_sha256.split(":", 1)[1]
        clip_key = f"{clean_prefix}/clips/{clip_hex[:2]}/{clip_hex}.mp4"
        clip_size = os.path.getsize(clip_path)
        if not _object_matches(
            client, bucket, clip_key, size=clip_size, sha256=clip_sha256,
        ):
            client.fput_object(
                bucket,
                clip_key,
                clip_path,
                content_type="video/mp4",
                metadata={"sha256": clip_sha256, "sample-id": row["sample_id"]},
            )
            if not _object_matches(
                client, bucket, clip_key, size=clip_size, sha256=clip_sha256,
            ):
                raise IOError(
                    f"uploaded clip did not verify: s3://{bucket}/{clip_key}"
                )

        frame_hex = first_frame_sha256.split(":", 1)[1]
        first_frame_key = (
            f"{clean_prefix}/first-frames/{frame_hex[:2]}/{frame_hex}.png"
        )
        first_frame_size = os.path.getsize(first_frame_path)
        if not _object_matches(
            client,
            bucket,
            first_frame_key,
            size=first_frame_size,
            sha256=first_frame_sha256,
        ):
            client.fput_object(
                bucket,
                first_frame_key,
                first_frame_path,
                content_type="image/png",
                metadata={"sha256": first_frame_sha256, "sample-id": row["sample_id"]},
            )
            if not _object_matches(
                client,
                bucket,
                first_frame_key,
                size=first_frame_size,
                sha256=first_frame_sha256,
            ):
                raise IOError(
                    f"uploaded first frame did not verify: "
                    f"s3://{bucket}/{first_frame_key}"
                )

        return (
            str(row["sample_id"]),
            clip_key,
            first_frame_key,
            clip_path,
            first_frame_path,
        )

    def checkpoint(result: tuple[str, str, str, str, str]) -> None:
        nonlocal published
        sample_id, clip_key, first_frame_key, clip_path, first_frame_path = result
        ledger.mark_published(sample_id, clip_key, first_frame_key)
        if delete_local_after_upload:
            Path(clip_path).unlink(missing_ok=True)
            Path(first_frame_path).unlink(missing_ok=True)
        published += 1
        log(f"published {sample_id} -> {clip_key} + {first_frame_key}")

    if workers == 1:
        for row in approved_rows:
            checkpoint(upload_pair(row))
    else:
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        # Keep the in-flight set bounded. If a worker fails, at most twice the
        # configured concurrency can have uploaded but not yet checkpointed.
        # Resumption safely verifies and reuses those content-addressed objects.
        pending = set()
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="corpus-v2-publish",
        ) as pool:
            for row in approved_rows:
                pending.add(pool.submit(upload_pair, row))
                if len(pending) < workers * 2:
                    continue
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    checkpoint(future.result())
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    checkpoint(future.result())
    return {**ledger.stats(), "published_this_run": published}


def build_sharded_manifest(
    ledger: PilotLedger,
    *,
    output_dir: str | os.PathLike[str],
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[str, str]:
    """Write immutable local shards and a root; return ``(root_path, digest)``."""
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    spec = ledger.read_spec()
    caption_spec = ledger.read_caption_spec()
    if caption_spec is None:
        raise ValueError("cannot manifest a corpus without bound caption settings")
    rows = list(ledger.published())
    if not rows:
        raise ValueError("no published samples to manifest")
    for row in rows:
        if not row["clip_key"] or not row["first_frame_key"]:
            raise ValueError(
                f"published sample {row['sample_id']} is missing an artifact key"
            )
        if not row["first_frame_file_sha256"]:
            raise ValueError(
                f"published sample {row['sample_id']} is missing its PNG digest"
            )
        actual_caption = {
            "model": row["caption_model"],
            "revision": row["caption_revision"],
            "prompt_version": row["caption_prompt_version"],
            "frame_count": row["caption_frame_count"],
        }
        expected_caption = {key: caption_spec[key] for key in actual_caption}
        if actual_caption != expected_caption:
            raise ValueError(
                f"published sample {row['sample_id']} does not match bound caption settings"
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_refs = []
    for index in range(0, len(rows), shard_size):
        batch = rows[index:index + shard_size]
        doc = {
            "manifest_version": CORPUS_V2_VERSION,
            "corpus_id": spec.corpus_id,
            "shard_index": index // shard_size,
            "samples": [sample_manifest_entry(row) for row in batch],
        }
        raw = canonical_json(doc)
        digest = sha256_bytes(raw)
        name = f"shard-{index // shard_size:06d}-{digest.split(':', 1)[1]}.json"
        path = out / name
        path.write_bytes(raw)
        shard_refs.append({
            "key": name,
            "sha256": digest,
            "count": len(batch),
            "first_sample_id": batch[0]["sample_id"],
            "last_sample_id": batch[-1]["sample_id"],
        })

    root = {
        "manifest_version": CORPUS_V2_VERSION,
        "corpus_id": spec.corpus_id,
        "sample_count": len(rows),
        "decode": {
            "width": spec.width, "height": spec.height, "fps": spec.fps,
            "num_frames": spec.num_frames,
        },
        "caption_prompt_version": CAPTION_PROMPT_VERSION,
        "caption_prompt": CAPTION_PROMPT,
        "caption_build": caption_spec,
        "quality_assurance": ledger.qa_report(),
        "shards": shard_refs,
    }
    raw = canonical_json(root)
    digest = sha256_bytes(raw)
    path = out / f"root-{digest.split(':', 1)[1]}.json"
    path.write_bytes(raw)
    return str(path), digest


def publish_manifest_bundle(
    client,
    bucket: str,
    *,
    root_path: str | os.PathLike[str],
    prefix: str,
) -> tuple[str, str]:
    """Publish all referenced shards, then the immutable root object last."""
    root_file = Path(root_path)
    root = json.loads(root_file.read_text())
    corpus_id = str(root.get("corpus_id") or "").strip()
    if root.get("manifest_version") != CORPUS_V2_VERSION or not corpus_id:
        raise ValueError("not a corpus-v2 root manifest")
    clean_prefix = prefix.strip("/")
    base = f"{clean_prefix}/manifests/{corpus_id}"

    for ref in root.get("shards") or []:
        path = root_file.parent / ref["key"]
        actual = digest_file(path)
        if actual != ref["sha256"]:
            raise ValueError(f"manifest shard changed after build: {path}")
        key = f"{base}/{path.name}"
        size = path.stat().st_size
        if not _object_matches(client, bucket, key, size=size, sha256=actual):
            client.fput_object(
                bucket, key, str(path), content_type="application/json",
                metadata={"sha256": actual},
            )
            if not _object_matches(client, bucket, key, size=size, sha256=actual):
                raise IOError(f"uploaded manifest shard did not verify: s3://{bucket}/{key}")

    root_digest = digest_file(root_file)
    root_key = f"{base}/{root_file.name}"
    size = root_file.stat().st_size
    if not _object_matches(client, bucket, root_key, size=size, sha256=root_digest):
        client.fput_object(
            bucket, root_key, str(root_file), content_type="application/json",
            metadata={"sha256": root_digest},
        )
        if not _object_matches(client, bucket, root_key, size=size, sha256=root_digest):
            raise IOError(f"uploaded root manifest did not verify: s3://{bucket}/{root_key}")
    return root_key, root_digest


__all__ = [
    "CAPTION_PROMPT",
    "CAPTION_PROMPT_VERSION",
    "CORPUS_V2_VERSION",
    "CaptionTask",
    "PilotLedger",
    "PilotSpec",
    "PreparedSample",
    "build_sharded_manifest",
    "choose_window_starts",
    "export_qa_bundle",
    "import_qa_reviews",
    "materialize_first_frame_images",
    "normalize_caption",
    "order_source_keys",
    "prepare_bucket_sources",
    "prepare_local_source",
    "publish_captioned_samples",
    "publish_manifest_bundle",
    "sample_manifest_entry",
    "select_qa_samples",
    "validate_caption",
]
