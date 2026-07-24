"""
CLI interface for Leoma.

Provides commands for running the validator and individual services.
"""

import asyncio
from itertools import islice

import click

from leoma import __version__


def _run_async(coroutine) -> None:
    """Run an async coroutine from CLI commands."""
    asyncio.run(coroutine)


@click.group()
@click.version_option(version=__version__, prog_name="leoma")
def cli():
    """Leoma - Image-to-Video Generation Subnet

    A king-of-the-hill TI2V subnet: miners upload model weights to Hippius Hub,
    validators download them and duel challenger vs king on held-out clips.
    """


@cli.command()
def serve():
    """Start the king-of-the-hill validator.

    Scans on-chain miner reveals, duels each new challenger against the reigning
    king on the GPU eval server (deterministic, block-hash-seeded), crowns
    winners, and sets equal weights across the king chain (else burns UID 0).
    Requires R2_OWN_BUCKET and a reachable EVAL_SERVER_URL.
    """
    from leoma.app.validator.main import main
    _run_async(main())


@cli.group()
def servers():
    """Start individual services.
    
    Use these commands to run specific components separately.
    """


@servers.command("validator")
def start_validator():
    """Start the king-of-the-hill validator (duel + weight-setter).

    Scans on-chain reveals, duels new challengers against the king on the eval
    server, crowns winners, and sets equal weights across the king chain (else
    burns to UID 0). Same as `leoma serve`.
    """
    from leoma.app.validator.main import main
    _run_async(main())


@servers.command("eval-server")
def start_eval_server():
    """Start the GPU eval server (video-generation duel).

    FastAPI service the validator dispatches duels to (POST /eval, SSE stream).
    Listens on EVAL_SERVER_HOST:EVAL_SERVER_PORT (default 0.0.0.0:9000).
    """
    from leoma.eval_server import main as eval_main
    eval_main()


@cli.command()
def preflight():
    """Check whether this validator is ready to launch — a hard gate, not a hope.

    Verifies every pin the subnet needs before it will crown anyone: the genesis
    king, the corpus manifest, the consensus surface, and (if configured) that the
    eval box is on the same chain.toml + scoring code. Exits non-zero if anything is
    a hard FAIL, so it can gate a launch script.
    """
    import os

    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.preflight import PASS, WARN, FAIL, run_preflight
    from leoma.eval.codehash import eval_code_digest
    from leoma.infra.chain_config import CONSENSUS_DIGEST, SEED_DIGEST, SPEC

    log_header("Leoma Preflight")

    # --- optional I/O the pure checks consume ---
    corpus_fetched_digest = None
    corpus_error = None
    if SPEC.corpus.pinned:
        try:
            from leoma.eval.dataset import fetch_manifest
            from leoma.infra.storage_backend import create_source_read_client

            manifest = fetch_manifest(create_source_read_client(), SPEC.corpus)
            corpus_fetched_digest = manifest.source_digest
        except Exception as e:  # noqa: BLE001 — a fetch failure is a WARN, not a crash
            corpus_error = str(e)

    # EVAL_SERVER_URLS (plural) lets an operator configure several independently-run
    # eval-server processes; checking only EVAL_SERVER_URL here would silently skip
    # every server but the first. _parse_eval_server_urls is the same pure parser the
    # validator itself uses, so the two never drift apart.
    eval_url_env = os.environ.get("EVAL_SERVER_URL", "")
    eval_urls_env = os.environ.get("EVAL_SERVER_URLS", "")
    eval_probes = ()
    if eval_url_env or eval_urls_env.strip():
        import httpx

        from leoma.app.preflight import EvalServerProbe
        from leoma.app.validator.main import _parse_eval_server_urls

        urls = _parse_eval_server_urls(eval_urls_env, eval_url_env or "http://localhost:9000")
        probes = []
        for url in urls:
            try:
                token = os.environ.get("LEOMA_EVAL_TOKEN", "")
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                health = httpx.get(
                    f"{url}/health", timeout=httpx.Timeout(15.0), headers=headers
                ).json()
                probes.append(EvalServerProbe(url, health, None))
            except Exception as e:  # noqa: BLE001
                probes.append(EvalServerProbe(url, None, str(e)))
        eval_probes = tuple(probes)

    report = run_preflight(
        seed_digest=SEED_DIGEST,
        corpus_pinned=SPEC.corpus.pinned,
        manifest_digest=SPEC.corpus.manifest_digest,
        consensus_digest=CONSENSUS_DIGEST,
        eval_code_digest=eval_code_digest(),
        own_bucket=os.environ.get("R2_OWN_BUCKET"),
        wallet_name=os.environ.get("WALLET_NAME"),
        hotkey_name=os.environ.get("HOTKEY_NAME"),
        corpus_fetched_digest=corpus_fetched_digest,
        corpus_error=corpus_error,
        eval_servers=eval_probes,
    )

    icon = {PASS: "✓", WARN: "▲", FAIL: "✗"}
    level = {PASS: "success", WARN: "warn", FAIL: "error"}
    for c in report.checks:
        log(f"{icon[c.status]} {c.name}: {c.detail}", level[c.status])

    log_header("Ready" if report.ready else "NOT READY")
    if report.ready:
        msg = "all checks pass" if not report.warnings else f"ready ({len(report.warnings)} warning(s) — review above)"
        log(msg, "success")
    else:
        log(f"{len(report.failures)} blocking failure(s) — the validator will burn to UID 0 until fixed", "error")
        raise SystemExit(1)


@cli.command()
@click.argument("source", default="")
@click.option("--require-live", is_flag=True, help="Also require an in-flight duel at snapshot time")
def smoke(source, require_live):
    """Confirm a testnet run actually exercised each expected outcome.

    Reads a validator's published dashboard.json (a URL, a file path, or
    LEOMA_DASHBOARD_URL) and checks that the rehearsal scenarios were observed: a
    genuine crown, a fair rejection, a quarantined broken model, a copy-of-king
    rejection, a freeze-cheat rejection. Reports exactly which behaviors have and
    haven't been exercised yet, so you can drive the missing ones.
    """
    import json
    import os

    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.smoke import run_smoke

    src = source or os.environ.get("LEOMA_DASHBOARD_URL", "")
    if not src:
        raise click.ClickException("give a dashboard.json URL or path (or set LEOMA_DASHBOARD_URL)")

    log_header("Leoma Smoke — testnet scenarios")
    if src.startswith("http://") or src.startswith("https://"):
        import httpx

        try:
            resp = httpx.get(src, timeout=httpx.Timeout(20.0))
            resp.raise_for_status()
            dash = resp.json()
        except httpx.HTTPError as e:
            raise click.ClickException(f"could not fetch dashboard from {src}: {e}")
        except ValueError as e:  # json.JSONDecodeError subclasses ValueError
            raise click.ClickException(f"{src} did not return valid JSON: {e}")
    else:
        try:
            with open(src) as f:
                dash = json.load(f)
        except OSError as e:
            raise click.ClickException(f"could not read dashboard file {src}: {e}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"{src} is not valid JSON: {e}")

    report = run_smoke(dash, require_live=require_live)
    for s in report.scenarios:
        log(f"{'✓' if s.observed else '·'} {s.description} — {s.evidence}",
            "success" if s.observed else "warn")

    log_header(f"{report.passed}/{report.total} scenarios observed")
    if report.complete:
        log("every rehearsal scenario was exercised and handled correctly", "success")
    else:
        log("still to drive: " + ", ".join(s.key for s in report.missing), "warn")
        raise SystemExit(1)


@cli.group()
def corpus():
    """Video corpus management commands.
    
    Commands for ingesting, listing, and managing videos in the
    Hippius "videos" bucket used for I2V evaluation.
    """

@corpus.command("expand")
@click.option("--count", "-n", default=10, help="Number of videos to add")
@click.option("--concurrent", "-c", default=2, help="Max concurrent downloads")
@click.option("--query", "-q", multiple=True, help="Custom search queries (can specify multiple)")
def corpus_expand(count, concurrent, query):
    """Expand corpus with random YouTube videos.
    
    Searches YouTube using diverse queries and ingests suitable videos.
    Uses default queries for nature, cooking, travel, dance, sports, etc.
    
    Examples:
    
        leoma corpus expand --count 20
        
        leoma corpus expand -n 10 -q "nature documentary" -q "cooking tutorial"
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.storage_backend import create_source_write_client
    from leoma.infra.corpus import expand_corpus_random
    
    queries = list(query) if query else None
    
    async def run():
        log_header("Expanding Video Corpus")
        log(f"Target: {count} videos, {concurrent} concurrent downloads", "info")
        if queries:
            log(f"Custom queries: {queries}", "info")
        
        minio_client = create_source_write_client()
        results = await expand_corpus_random(minio_client, count, queries, concurrent)
        
        log_header("Expansion Results")
        log(f"Success: {results['success']}/{results['total']}", "success")
        
        if results["uploaded"]:
            log(f"Uploaded {len(results['uploaded'])} videos", "info")
        
        if results["errors"]:
            log(f"Failed: {results['failed']}", "warn")
            for err in results["errors"][:5]:
                url = err.get("url", "unknown")[:50]
                log(f"  {url}: {err['error']}", "warn")
            if len(results["errors"]) > 5:
                log(f"  ... and {len(results['errors']) - 5} more errors", "warn")
    
    _run_async(run())


@corpus.group("v2")
def corpus_v2():
    """Build the resumable, captioned, sharded corpus-v2 pilot."""


def _pilot_ledger_path(workdir: str) -> str:
    import os

    return os.path.join(os.path.abspath(workdir), "pilot.sqlite3")


@corpus_v2.command("inventory")
@click.option("--target-samples", type=click.IntRange(min=1), default=2_000_000, show_default=True)
@click.option("--windows-per-source", type=click.IntRange(min=1), default=8, show_default=True)
def corpus_v2_inventory(target_samples, windows_per_source):
    """Report whether the live source inventory can support a proposed sample target."""
    import json
    import math

    from leoma.bootstrap import MIN_VIDEO_SIZE, SOURCE_BUCKET
    from leoma.infra.chain_config import SPEC
    from leoma.infra.storage_backend import create_source_read_client

    if SOURCE_BUCKET != SPEC.corpus.bucket:
        raise click.ClickException(
            f"source bucket mismatch: runtime={SOURCE_BUCKET!r}, pinned={SPEC.corpus.bucket!r}"
        )
    objects = total_bytes = eligible = eligible_bytes = 0
    try:
        for obj in create_source_read_client().list_objects(SOURCE_BUCKET, recursive=True):
            objects += 1
            size = int(getattr(obj, "size", 0))
            total_bytes += size
            key = str(getattr(obj, "object_name", ""))
            if key.lower().endswith((".mp4", ".mov", ".mkv", ".webm")) and size > MIN_VIDEO_SIZE:
                eligible += 1
                eligible_bytes += size
    except Exception as exc:
        raise click.ClickException(f"could not inventory {SOURCE_BUCKET!r}: {exc}") from None
    minimum = math.ceil(target_samples / eligible) if eligible else None
    report = {
        "bucket": SOURCE_BUCKET,
        "objects": objects,
        "total_bytes": total_bytes,
        "eligible_videos": eligible,
        "eligible_bytes": eligible_bytes,
        "windows_per_source": windows_per_source,
        "theoretical_sample_ceiling": eligible * windows_per_source,
        "target_samples": target_samples,
        "minimum_accepted_windows_per_source_for_target": minimum,
        "target_possible_at_configured_cap_before_filtering": bool(
            eligible * windows_per_source >= target_samples
        ),
    }
    click.echo(json.dumps(report, sort_keys=True))


@corpus_v2.command("prepare")
@click.option("--corpus-id", required=True, help="Immutable corpus version label")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--max-samples", type=click.IntRange(min=1), default=10_000, show_default=True)
@click.option("--max-sources", type=click.IntRange(min=1), default=None)
@click.option("--windows-per-source", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--min-gap-seconds", type=click.FloatRange(min=5.0625), default=10.0, show_default=True)
@click.option("--min-motion", type=click.FloatRange(min=0), default=1.5, show_default=True)
@click.option(
    "--workers", type=click.IntRange(min=1, max=64), default=8, show_default=True,
    help="Concurrent source download/FFmpeg workers; results still commit deterministically",
)
def corpus_v2_prepare(
    corpus_id,
    workdir,
    max_samples,
    max_sources,
    windows_per_source,
    min_gap_seconds,
    min_motion,
    workers,
):
    """Prepare a bounded pilot from source videos, resuming from its SQLite ledger."""
    import os
    from leoma.bootstrap import MIN_VIDEO_SIZE, SOURCE_BUCKET, emit_header, emit_log
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_v2 import (
        PilotLedger,
        PilotSpec,
        order_source_keys,
        prepare_bucket_sources,
    )
    from leoma.infra.storage_backend import create_source_read_client

    if SOURCE_BUCKET != SPEC.corpus.bucket:
        raise click.ClickException(
            f"source bucket mismatch: runtime={SOURCE_BUCKET!r}, pinned={SPEC.corpus.bucket!r}"
        )
    spec = PilotSpec(
        corpus_id=corpus_id,
        width=SPEC.gen.width,
        height=SPEC.gen.height,
        fps=SPEC.gen.fps,
        num_frames=SPEC.gen.num_frames,
        max_windows_per_source=windows_per_source,
        min_window_gap_seconds=min_gap_seconds,
        min_motion_energy=min_motion,
    )
    os.makedirs(workdir, exist_ok=True)
    ledger = PilotLedger(_pilot_ledger_path(workdir), spec)
    client = create_source_read_client()
    eligible = [
        obj.object_name
        for obj in client.list_objects(SOURCE_BUCKET, recursive=True)
        if obj.object_name.lower().endswith((".mp4", ".mov", ".mkv", ".webm"))
        and int(getattr(obj, "size", 0)) > MIN_VIDEO_SIZE
    ]
    keys = order_source_keys(eligible, seed=corpus_id)
    if max_sources is not None:
        keys = keys[:max_sources]
    emit_header("Corpus v2 — prepare pilot")
    emit_log(
        f"workdir={os.path.abspath(workdir)} target={max_samples} "
        f"windows/source<={windows_per_source} workers={workers}", "info",
    )
    stats = prepare_bucket_sources(
        client,
        SOURCE_BUCKET,
        ledger=ledger,
        workdir=workdir,
        spec=spec,
        keys=keys,
        max_samples=max_samples,
        workers=workers,
        log=lambda message: emit_log(message, "info"),
    )
    emit_log(f"pilot state: {stats}", "success")


@corpus_v2.command("materialize-first-frames")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
def corpus_v2_materialize_first_frames(workdir, limit):
    """Backfill lossless frame-zero PNGs for an existing retained pilot."""
    import os

    from leoma.bootstrap import emit_header, emit_log
    from leoma.infra.corpus_v2 import PilotLedger, materialize_first_frame_images

    ledger = PilotLedger(_pilot_ledger_path(workdir))
    output_dir = os.path.join(os.path.abspath(workdir), "first-frames")
    emit_header("Corpus v2 — materialize first frames")
    try:
        stats = materialize_first_frame_images(
            ledger,
            output_dir=output_dir,
            limit=limit,
            log=lambda message: emit_log(message, "info"),
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    emit_log(f"pilot state: {stats}", "success")


@corpus_v2.command("caption")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option(
    "--model", default="Qwen/Qwen3-VL-8B-Instruct", show_default=True,
    help="Open-source native-video caption model repository",
)
@click.option(
    "--revision", default="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", show_default=True,
    help="Immutable 40-character Hugging Face commit hash",
)
@click.option("--gpus", default="0,1,2,3,4,5,6,7", show_default=True)
@click.option("--frames", "frame_count", type=click.IntRange(min=2, max=32), default=16, show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=16, max=256), default=96, show_default=True)
@click.option("--max-attempts", type=click.IntRange(min=1, max=10), default=3, show_default=True)
@click.option(
    "--follow/--drain", default=False, show_default=True,
    help="Keep loaded GPU workers waiting for samples added by a running prepare job",
)
@click.option(
    "--idle-timeout", type=click.FloatRange(min=10.0), default=600.0, show_default=True,
    help="In follow mode, exit after the queue stays empty for this many seconds",
)
def corpus_v2_caption(
    workdir,
    model,
    revision,
    gpus,
    frame_count,
    max_new_tokens,
    max_attempts,
    follow,
    idle_timeout,
):
    """Caption pending pilot clips with one persistent process per selected GPU."""
    from leoma.bootstrap import emit_header, emit_log
    from leoma.infra.video_caption import run_caption_workers

    gpu_ids = [part.strip() for part in gpus.split(",") if part.strip()]
    emit_header("Corpus v2 — caption")
    emit_log(f"model={model}@{revision} GPUs={','.join(gpu_ids)}", "info")
    try:
        stats = run_caption_workers(
            _pilot_ledger_path(workdir),
            gpus=gpu_ids,
            model=model,
            revision=revision,
            frame_count=frame_count,
            max_new_tokens=max_new_tokens,
            max_attempts=max_attempts,
            follow=follow,
            idle_timeout_seconds=idle_timeout,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    emit_log(f"pilot state: {stats}", "success")


@corpus_v2.command("caption-retry-failed")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
def corpus_v2_caption_retry_failed(workdir):
    """Return terminal caption failures to the queue after fixing their cause."""
    from leoma.infra.corpus_v2 import PilotLedger

    count = PilotLedger(_pilot_ledger_path(workdir)).retry_failed_captions()
    click.echo(f"queued {count} failed samples for captioning")


@corpus_v2.command("caption-reset-all")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--yes", is_flag=True, help="Confirm all captions and QA decisions will be cleared")
def corpus_v2_caption_reset_all(workdir, yes):
    """Clear all unpublished captions before deliberately changing caption settings."""
    if not yes:
        raise click.ClickException("refusing reset without --yes")
    from leoma.infra.corpus_v2 import PilotLedger

    try:
        count = PilotLedger(_pilot_ledger_path(workdir)).reset_all_captions()
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f"reset {count} samples; caption settings may now be rebound")


@corpus_v2.command("qa-export")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--out", "output_dir", default="corpus-v2-qa", show_default=True)
@click.option("--count", type=click.IntRange(min=1), default=200, show_default=True)
@click.option("--seed", default="leoma-corpus-v2-qa-v1", show_default=True)
def corpus_v2_qa_export(workdir, output_dir, count, seed):
    """Export a deterministic, motion-stratified caption review sheet."""
    from leoma.bootstrap import emit_header, emit_log
    from leoma.infra.corpus_v2 import PilotLedger, export_qa_bundle

    emit_header("Corpus v2 — export QA audit")
    try:
        review_path, html_path = export_qa_bundle(
            PilotLedger(_pilot_ledger_path(workdir)),
            output_dir=output_dir,
            count=count,
            seed=seed,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    emit_log(f"review sheet: {review_path}", "success")
    emit_log(f"video gallery: {html_path}", "success")
    emit_log("fill every verdict/reviewer field in reviews.jsonl, then run qa-import", "warn")


@corpus_v2.command("qa-import")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
def corpus_v2_qa_import(workdir, input_path):
    """Import pass/fail decisions; failed captions are quarantined from publishing."""
    import json

    from leoma.infra.corpus_v2 import PilotLedger, import_qa_reviews

    try:
        report = import_qa_reviews(PilotLedger(_pilot_ledger_path(workdir)), input_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(json.dumps(report, sort_keys=True))


@corpus_v2.command("qa-retry-rejected")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
def corpus_v2_qa_retry_rejected(workdir):
    """Return QA-rejected clips to the caption queue for a corrected model/prompt."""
    from leoma.infra.corpus_v2 import PilotLedger

    count = PilotLedger(_pilot_ledger_path(workdir)).retry_qa_rejected()
    click.echo(f"queued {count} QA-rejected samples for captioning")


@corpus_v2.command("publish")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--prefix", required=True, help="Versioned bucket prefix, e.g. corpus-v2/pilot-10k")
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--qa-min-reviews", type=click.IntRange(min=0), default=200, show_default=True)
@click.option(
    "--qa-min-pass-rate", type=click.FloatRange(min=0.0, max=1.0),
    default=0.95, show_default=True,
)
@click.option("--delete-local-after-upload/--keep-local", default=True, show_default=True)
def corpus_v2_publish(
    workdir, prefix, limit, qa_min_reviews, qa_min_pass_rate, delete_local_after_upload,
):
    """Upload captioned clip/first-frame pairs and checkpoint each verified sample."""
    from leoma.bootstrap import SOURCE_BUCKET, emit_header, emit_log
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_v2 import PilotLedger, publish_captioned_samples
    from leoma.infra.storage_backend import create_source_write_client

    if SOURCE_BUCKET != SPEC.corpus.bucket:
        raise click.ClickException(
            f"source bucket mismatch: runtime={SOURCE_BUCKET!r}, pinned={SPEC.corpus.bucket!r}"
        )
    ledger = PilotLedger(_pilot_ledger_path(workdir))
    emit_header("Corpus v2 — publish clips and first frames")
    try:
        stats = publish_captioned_samples(
            create_source_write_client(),
            SOURCE_BUCKET,
            ledger=ledger,
            prefix=prefix,
            limit=limit,
            qa_min_reviews=qa_min_reviews,
            qa_min_pass_rate=qa_min_pass_rate,
            delete_local_after_upload=delete_local_after_upload,
            log=lambda message: emit_log(message, "info"),
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    emit_log(f"pilot state: {stats}", "success")


@corpus_v2.command("manifest")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
@click.option("--prefix", required=True, help="Same versioned prefix used by publish")
@click.option("--out", "output_dir", default="corpus-v2-manifest", show_default=True)
@click.option("--shard-size", type=click.IntRange(min=100, max=100_000), default=5_000, show_default=True)
@click.option("--upload/--no-upload", default=False, show_default=True)
def corpus_v2_manifest(workdir, prefix, output_dir, shard_size, upload):
    """Build manifest shards and optionally publish the immutable root last."""
    from leoma.bootstrap import SOURCE_BUCKET, emit_header, emit_log
    from leoma.infra.corpus_v2 import PilotLedger, build_sharded_manifest, publish_manifest_bundle
    from leoma.infra.storage_backend import create_source_write_client

    ledger = PilotLedger(_pilot_ledger_path(workdir))
    stats = ledger.stats()
    unfinished = sum(count for status, count in stats["samples"].items()
                     if status not in {"published", "excluded"})
    if unfinished:
        raise click.ClickException(
            f"refusing a partial root manifest: {unfinished} samples are not published ({stats})"
        )
    emit_header("Corpus v2 — build manifest")
    try:
        root_path, digest = build_sharded_manifest(
            ledger, output_dir=output_dir, shard_size=shard_size,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from None
    emit_log(f"root={root_path}", "success")
    emit_log(f"digest={digest}", "success")
    if upload:
        try:
            key, uploaded_digest = publish_manifest_bundle(
                create_source_write_client(), SOURCE_BUCKET,
                root_path=root_path, prefix=prefix,
            )
        except Exception as exc:
            raise click.ClickException(str(exc)) from None
        emit_log(f"published root: s3://{SOURCE_BUCKET}/{key}", "success")
        emit_log(f"pin this digest only after evaluator-v2 support: {uploaded_digest}", "warn")


@corpus_v2.command("status")
@click.option("--workdir", default="corpus-v2-pilot", show_default=True)
def corpus_v2_status(workdir):
    """Print resumable preparation/caption/publish counts."""
    import json

    from leoma.infra.corpus_v2 import PilotLedger

    ledger = PilotLedger(_pilot_ledger_path(workdir))
    click.echo(json.dumps(ledger.stats(), sort_keys=True))


@corpus.command("build-manifest")
@click.option("--corpus-id", required=True, help="Version label for this corpus, e.g. leoma-corpus-v1")
@click.option("--out", "-o", default="manifest.json", help="Where to write the manifest")
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Only consider the first N eligible source videos",
)
def corpus_build_manifest(corpus_id, out, limit):
    """Build the pinned corpus manifest from the source bucket.

    Decides each clip's window ONCE, decodes its ground truth, and records the
    hashes. This is what makes a duel reproducible: at duel time nobody detects
    scenes, lists buckets, or skips a clip — they just decode the pinned window and
    check it against the hash recorded here.

    Prints the manifest digest to paste into chain.toml [corpus].manifest_digest.
    """
    from leoma.bootstrap import (
        MAX_VIDEO_SIZE,
        MIN_VIDEO_SIZE,
        SOURCE_BUCKET,
        emit_log as log,
        emit_header as log_header,
    )
    from leoma.eval.manifest import DecodeParams, check_corpus_size, check_decode_compat
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_manifest import build_manifest, write_manifest
    from leoma.infra.storage_backend import create_source_read_client

    log_header("Building Corpus Manifest")

    if SOURCE_BUCKET != SPEC.corpus.bucket:
        raise click.ClickException(
            "source bucket mismatch: runtime configuration selects "
            f"{SOURCE_BUCKET!r}, but chain.toml [corpus].bucket pins "
            f"{SPEC.corpus.bucket!r}. Building from one bucket and evaluating from "
            "another would make every duel fail. Point R2_SOURCE_BUCKET or "
            "HIPPIUS_SOURCE_BUCKET at the pinned bucket before building."
        )

    # The decode params come from the PINNED [gen] block, never from flags: the
    # truth hashes are only meaningful relative to them, and a manifest built under
    # different numbers would fail every hash check at duel time.
    decode = DecodeParams(
        width=SPEC.gen.width,
        height=SPEC.gen.height,
        fps=SPEC.gen.fps,
        num_frames=SPEC.gen.num_frames,
    )
    log(f"Decode: {decode.width}x{decode.height} @ {decode.fps}fps x {decode.num_frames} frames", "info")

    try:
        client = create_source_read_client()
        eligible_keys = (
            obj.object_name
            for obj in client.list_objects(SPEC.corpus.bucket, recursive=True)
            if obj.object_name.endswith(".mp4") and MIN_VIDEO_SIZE < obj.size < MAX_VIDEO_SIZE
        )
        if limit is None:
            keys = sorted(eligible_keys)
        else:
            # S3 ListObjectsV2 returns keys in lexicographic order. Preserve that
            # deterministic order while stopping after N eligible objects; collecting
            # the entire archive before slicing made a bounded testnet build unbounded.
            keys = list(islice(eligible_keys, limit))
            if keys != sorted(keys):
                raise RuntimeError(
                    "object storage returned a non-lexicographic listing; refusing a "
                    "limited build because its first N inputs would not be reproducible"
                )
    except Exception as e:
        raise click.ClickException(
            f"could not list source bucket {SPEC.corpus.bucket!r}: {e}"
        ) from None
    log(f"Considering {len(keys)} source videos from {SPEC.corpus.bucket}", "info")

    manifest = build_manifest(
        client, SPEC.corpus.bucket, corpus_id=corpus_id, decode=decode, keys=keys,
        log=lambda m: log(m, "info"),
    )
    try:
        check_decode_compat(manifest, SPEC.gen)
        check_corpus_size(manifest, SPEC.duel.n_clips)
    except Exception as e:
        raise click.ClickException(f"built manifest is not duel-ready: {e}") from None
    digest = write_manifest(manifest, out)

    log_header("Manifest Built")
    log(f"{len(manifest)} clips -> {out}", "success")
    log(f"digest: {digest}", "success")
    log("Paste into chain.toml [corpus].manifest_digest, then `leoma corpus publish-manifest`", "info")


@corpus.command("publish-manifest")
@click.argument("path", type=click.Path(exists=True))
def corpus_publish_manifest(path):
    """Upload a built manifest to the corpus bucket at the pinned key."""
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.infra.chain_config import SPEC
    from leoma.eval.manifest import check_corpus_size, check_decode_compat
    from leoma.infra.corpus_manifest import publish_manifest, read_manifest
    from leoma.infra.storage_backend import create_source_write_client

    log_header("Publishing Corpus Manifest")
    try:
        manifest = read_manifest(path)
        check_decode_compat(manifest, SPEC.gen)
        check_corpus_size(manifest, SPEC.duel.n_clips)
        client = create_source_write_client()
        digest = publish_manifest(client, SPEC.corpus.bucket, SPEC.corpus.manifest_key, manifest)
    except Exception as e:
        raise click.ClickException(f"could not publish a duel-ready manifest: {e}") from None

    log(f"Published {len(manifest)} clips to {SPEC.corpus.bucket}/{SPEC.corpus.manifest_key}", "success")
    log(f"digest: {digest}", "success")
    if digest != SPEC.corpus.manifest_digest:
        log("chain.toml [corpus].manifest_digest does NOT match this manifest — "
            "validators will refuse to duel until it is pinned to the digest above", "warn")


@corpus.command("verify")
@click.option("--sample", type=int, default=4, help="How many clips to re-decode (0 = all)")
def corpus_verify(sample):
    """Prove THIS box decodes the pinned corpus byte-identically.

    Run this on every new eval box before it duels. A box whose ffmpeg produces even
    slightly different pixels measures every distance against different ground truth
    — silently, confidently, and wrongly. A minute here, or a broken consensus in
    production.
    """
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.eval.dataset import fetch_manifest
    from leoma.infra.chain_config import SPEC
    from leoma.infra.corpus_manifest import verify_manifest
    from leoma.infra.storage_backend import create_source_read_client

    log_header("Verifying Corpus")
    SPEC.require_duel_ready()

    client = create_source_read_client()
    manifest = fetch_manifest(client, SPEC.corpus)
    log(f"Manifest {manifest.corpus_id}: {len(manifest)} clips, digest matches chain.toml", "success")

    checked = verify_manifest(
        client, SPEC.corpus.bucket, manifest,
        sample=None if sample == 0 else sample,
        log=lambda m: log(m, "info"),
    )
    log_header("Corpus Verified")
    log(f"{checked} clips decoded byte-identically to the manifest — this box can duel", "success")


@cli.group()
def calibrate():
    """Measure the cross-GPU noise floor so delta_threshold can be set, not guessed.

    Leoma crowns a challenger whose bootstrap LCB beats the king by delta_threshold.
    That margin only means anything if it is wider than the hardware noise: the same
    model generates slightly different frames on different GPUs, hence different
    distances, hence potentially a different verdict. This is the subnet's largest open
    consensus risk. Two steps:

      1. `leoma calibrate generate` on EACH GPU type in the fleet -> one record per box.
      2. `leoma calibrate analyze box_*.json` -> the measured floor + a delta recommendation.
    """


@calibrate.command("generate")
@click.option("--model", default="", help="repo@digest to calibrate on (default: the pinned seed king)")
@click.option("--n-clips", type=int, default=32, help="How many clips to generate")
@click.option("--gpu", "gpu_name", default="", help="Label for this box (default: the detected GPU name)")
@click.option("--out", "-o", default="calibration.json", help="Where to write this box's record")
def calibrate_generate(model, n_clips, gpu_name, out):
    """Generate the calibration model on this box and record per-clip distances.

    Run this ONCE ON EACH GPU TYPE in the fleet. Every box uses the same model, clips
    and seed, so any distance difference between two boxes is pure hardware noise.
    """
    import json

    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.eval.calibrate import box_record
    from leoma.eval.determinism import runtime_env
    from leoma.infra.chain_config import SEED_REPO, SEED_DIGEST, SPEC
    from leoma.infra.model_store import ModelRef

    log_header("Calibration — generate")
    SPEC.require_duel_ready()

    if model:
        repo, _, digest = model.partition("@")
        ref = ModelRef(repo, digest)
    elif SEED_DIGEST:
        ref = ModelRef(SEED_REPO, SEED_DIGEST)
    else:
        raise click.ClickException(
            "no --model given and chain.toml [seed].seed_digest is not pinned; "
            "pass --model repo@digest to calibrate on a specific model"
        )

    label = gpu_name or (runtime_env().get("gpu") or "unknown-gpu")
    log(f"Model: {ref.immutable_ref}  GPU: {label}  clips: {n_clips}", "info")
    log("Generating (this uses the real duel path — expect it to take a while)...", "info")

    record = box_record(ref, spec=SPEC, n_clips=n_clips, gpu_name=label)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)

    log_header("Done")
    log(f"Wrote {len(record['clips'])} clip distances for {label} -> {out}", "success")
    log("Run this on every GPU type, then: leoma calibrate analyze <all the records>", "info")


@calibrate.command("analyze")
@click.argument("records", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--safety", type=float, default=3.0, help="Multiple of the noise floor to recommend")
def calibrate_analyze(records, safety):
    """Compare box records and print the measured floor + a delta_threshold recommendation.

    Pure analysis — no GPU. Give it the JSON records from `generate` on each box.
    """
    import json

    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.eval.calibrate import analyze
    from leoma.infra.chain_config import SPEC

    log_header("Calibration — analyze")
    loaded = []
    for path in records:
        with open(path) as f:
            loaded.append(json.load(f))
        log(f"loaded {path}: {loaded[-1].get('gpu')} ({len(loaded[-1].get('clips', []))} clips)", "info")

    report = analyze(loaded, current_delta_threshold=SPEC.duel.delta_threshold, safety_factor=safety)

    log_header("Cross-GPU noise")
    log(f"boxes: {report.n_boxes}  pairs: {report.n_pairs}  gpus: {', '.join(report.gpus)}", "info")
    for p in report.pairs:
        repro = "bit-identical" if p.bit_identical_clips == p.n_clips else \
                f"{p.bit_identical_clips}/{p.n_clips} clips identical"
        log(f"  {p.gpu_a} vs {p.gpu_b}: |Δ| max={p.abs_delta_max:.2e} p99={p.abs_delta_p99:.2e}  "
            f"mu_hat={p.mu_hat:+.2e} lcb={p.lcb:+.2e} ucb={p.ucb:+.2e}  ({repro})", "info")

    log(f"worst spurious mean-advantage (any pair): {report.max_abs_mu_hat:.3e}", "info")
    log(f"worst single-clip disagreement:           {report.max_abs_delta:.3e}", "info")
    log(f"recommended delta_threshold (>= {safety}x): {report.recommended_delta_threshold}", "success")

    level = "success" if report.verdict.startswith("PASS") else \
            "error" if report.verdict.startswith("FAIL") else "info"
    log(report.verdict, level)


@cli.group()
def miner():
    """Miner management commands.

    Commands for uploading model weights to Hippius Hub and committing
    the model reveal to the blockchain for validator discovery.
    """


@miner.command("push")
@click.option("--model-dir", required=True, help="Local folder with the model weights (safetensors + config)")
@click.option("--repo", required=True, help="Hippius Hub repo id (must start with 'leoma' and end with your hotkey), e.g. user/leoma-<name>-<hotkey>")
@click.option("--revision", default=None, help="Optional revision/branch label for the upload")
@click.option("--message", "commit_message", default=None, help="Optional upload commit message")
def miner_push(model_dir, repo, revision, commit_message):
    """Upload model weights to Hippius Hub.

    Uploads the safetensors + config from a local folder to Hippius Hub and
    prints the immutable repo@digest to commit.

    Example:

        leoma miner push --model-dir ./out --repo user/leoma-mymodel-5GRW...
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import push_command

    async def run():
        log_header("Uploading to Hippius Hub")
        log(f"Model dir: {model_dir}", "info")
        log(f"Repo: {repo}", "info")

        result = await push_command(
            model_dir=model_dir,
            repo=repo,
            revision=revision,
            commit_message=commit_message,
        )

        print(json.dumps(result, indent=2))

        if result.get("success"):
            log_header("Upload Complete")
            log(f"Immutable ref: {result.get('immutable_ref')}", "success")
            log("Next: leoma miner commit --repo <repo> --digest <digest>", "info")
        else:
            log(f"Upload failed: {result.get('error')}", "error")

    _run_async(run())


@miner.command("commit")
@click.option("--repo", required=True, help="Hippius Hub repo id (from push output)")
@click.option("--digest", required=True, help="Immutable digest, e.g. sha256:<64hex> (from push output)")
@click.option("--coldkey", help="Wallet coldkey name (optional, from env WALLET_NAME)")
@click.option("--hotkey", help="Wallet hotkey name (optional, from env HOTKEY_NAME)")
def miner_commit(repo, digest, coldkey, hotkey):
    """Commit the model reveal to the blockchain.

    Reveals ``v4|<repo>|<digest>|<hotkey>`` on-chain so validators can discover
    and download the exact model weights.

    Example:

        leoma miner commit --repo user/leoma-mymodel-5GRW... --digest sha256:abc...
    """
    import json
    from leoma.bootstrap import emit_log as log, emit_header as log_header
    from leoma.app.miner.main import commit_command

    async def run():
        log_header("Committing to Chain")
        log(f"Repo: {repo}", "info")
        log(f"Digest: {digest[:24]}...", "info")

        result = await commit_command(
            repo=repo,
            digest=digest,
            coldkey=coldkey,
            hotkey=hotkey,
        )

        print(json.dumps(result, indent=2))

        if result.get("success"):
            log_header("Commit Complete")
            log("Model reveal committed to chain", "success")
        else:
            log(f"Commit failed: {result.get('error')}", "error")

    _run_async(run())


if __name__ == "__main__":
    cli()
