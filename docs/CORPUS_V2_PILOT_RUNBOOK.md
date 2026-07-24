# TI2V corpus-v2 pilot and staged build

This pipeline prepares caption + first-frame PNG + five-second truth-clip samples
offline. Every published sample has all three artifacts:

- the caption in its immutable manifest entry;
- a lossless, content-addressed `first-frames/<sha256>.png`;
- a content-addressed `clips/<sha256>.mp4`.

It does **not** change `chain.toml`, the active corpus-v1 manifest, or validator
consensus. The evaluator has version-gated v1/v2 readers, but corpus-v2 must still
pass human QA, rights review, remote verification, testnet rehearsal, and calibration
before its root key and digest can become active in a coordinated consensus release.

The PNG contains exactly decoded truth frame 0. Every manifest entry pins the PNG
object digest and its decoded RGB digest, so the evaluator can verify both the stored
image and its exact relationship to the truth clip.

## Safety limits

Start with 10,000 samples. Do not launch a two-million-sample build until the pilot
has measured accepted windows per source, caption throughput, object size, rejection
rate, and end-to-end cost. With the default maximum of eight windows per source,
two million samples require at least 250,000 eligible source videos. Raising that
limit is possible, but increases source correlation and must be an explicit choice.

Run the live inventory check before each sizing decision:

```bash
venv/bin/leoma corpus v2 inventory --target-samples 2000000 --windows-per-source 8
```

On 2026-07-23, `leoma-source` contained 151,247 eligible videos (1.608 TB). The
eight-window cap therefore had a theoretical ceiling of 1,209,976 samples before any
filtering. Reaching two million would require at least 14 accepted windows per eligible
source on average; short duration, scene boundaries, low motion, and duplicates push
the required configured cap higher. Treat this as evidence to measure the 10k pilot,
not as permission to weaken non-overlap or quality filters.

A bounded follow-up on the same date allowed 20 windows/source and needed 78 source
objects to produce 100 accepted samples: 73 sources completed and 5 failed closed on
short 79/80-frame decodes. The observed yield was 1.28 accepted samples/source, which
would project to roughly 194k samples if representative of the full lexicographic
inventory. This is a sizing sample, not a statistically randomized final estimate.

Public availability is not a redistribution license. Before publishing derived
YouTube or other third-party clips, retain source/license provenance and confirm that
the intended use and redistribution are permitted. Pexels and each source platform
may impose different terms.

## 1. Use the 8xH100 host as an offline builder

Stop the four eval-server processes while captioning so all eight GPUs are available.
Use local NVMe for the work directory and install the dedicated caption extra:

```bash
uv sync --active --frozen --extra caption --no-dev
ffmpeg -version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

The default caption model is the open-source native-video model
`Qwen/Qwen3-VL-8B-Instruct`, pinned to immutable Hugging Face commit
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. One persistent model process runs on
each selected GPU. The model is an offline build dependency, not a validator or duel
dependency.

The shell must contain Hippius read credentials for preparation and write credentials
for publication:

```dotenv
OBJECT_STORAGE_BACKEND=hippius
HIPPIUS_ENDPOINT=s3.hippius.com
HIPPIUS_REGION=decentralized
HIPPIUS_SOURCE_BUCKET=leoma-source
HIPPIUS_VIDEOS_READ_ACCESS_KEY=<read-key>
HIPPIUS_VIDEOS_READ_SECRET_KEY=<read-secret>
HIPPIUS_VIDEOS_WRITE_ACCESS_KEY=<write-key>
HIPPIUS_VIDEOS_WRITE_SECRET_KEY=<write-secret>
```

## 2. Prepare the bounded 10k pilot

```bash
WORK=/var/lib/leoma/corpus-v2-pilot-10k
venv/bin/leoma corpus v2 prepare \
  --corpus-id leoma-ti2v-pilot-10k-v1 \
  --workdir "$WORK" \
  --max-samples 10000 \
  --windows-per-source 8 \
  --workers 8
venv/bin/leoma corpus v2 status --workdir "$WORK"
```

Preparation processes eight source objects concurrently by default. The worker count
is operational and may change on resume; completed source results are committed in
the deterministic source order, while a bounded rolling prefetch queue keeps workers
busy behind slow sources. Worker timing therefore cannot change a capped corpus.
Accepted windows are single-shot, nominally five seconds (81 frames at 16 fps),
832x480, motion-filtered, and content/digest pinned. MP4s are written under `clips/`,
and lossless frame-zero PNGs are written under `first-frames/`. `pilot.sqlite3` is the
resumable ledger; do not copy it while its `-wal` file is active.

The builder lists the eligible namespace once and hash-orders source keys using the
`corpus-id` as a seed. This makes a bounded pilot representative across the bucket
while remaining exactly reproducible on resume; it does not depend on mutable bucket
listing order.

## 3. Caption on all eight GPUs

For maximum end-to-end throughput, start captioning while `prepare` is still
running. Follow mode keeps each loaded model resident and polls the growing SQLite
queue instead of unloading whenever it briefly becomes empty:

```bash
venv/bin/leoma corpus v2 caption \
  --workdir "$WORK" \
  --gpus 0,1,2,3,4,5,6,7 \
  --follow \
  --idle-timeout 600
```

Stop the live eval service before this command; it deliberately reserves all eight
GPUs. The CPU preparation workers can continue concurrently.

If preparation has already finished, drain the fixed queue normally:

```bash
venv/bin/leoma corpus v2 caption \
  --workdir "$WORK" \
  --gpus 0,1,2,3,4,5,6,7
venv/bin/leoma corpus v2 status --workdir "$WORK"
```

Each worker atomically claims one clip. A restart returns interrupted claims to the
queue, and invalid captions receive at most three attempts. Captions are generated
deterministically from sixteen temporally positioned video frames. One immutable
model/revision/frame-count/token configuration is bound to the workdir, so a resumed
build cannot silently mix caption policies.

With the Qwen snapshot already cached, the same 2026-07-23 host captioned the 100-sample
benchmark on eight H100 PCIe GPUs in 43.9 seconds wall time (about 2.28 samples/second
aggregate). That small run extrapolates to about 10.2 continuous days for two million
captions, before preparation, QA, retries, or upload. Use the 10k pilot for a sustained
throughput estimate rather than treating the warm 100-sample number as a guarantee.

If all three attempts fail because of an infrastructure issue, fix the dependency or
host first and then run `leoma corpus v2 caption-retry-failed --workdir "$WORK"`.
The next caption command will retry those rows from attempt zero.

Changing the caption model or settings requires either a new workdir or the guarded
`caption-reset-all --yes` command. Resetting is refused after a batch has been approved
for publication.

For a retained pilot prepared before first-frame PNGs were introduced, materialize
them directly from the pinned local clips before QA or publishing:

```bash
venv/bin/leoma corpus v2 materialize-first-frames --workdir "$WORK"
```

The command decodes each clip using its pinned settings, refuses any RGB-digest
mismatch, and then checkpoints the PNG path and file digest in the ledger. New
preparation runs create the PNGs automatically.

## 4. Human QA before upload

```bash
venv/bin/leoma corpus v2 qa-export \
  --workdir "$WORK" --out /var/lib/leoma/corpus-v2-qa --count 200
```

Open `review.html`. Each card shows the conditioning PNG, truth clip, and caption.
In `reviews.jsonl`, set every `verdict` to `pass` or `fail`, add the reviewer name,
and explain failures. A pass means the first image matches clip frame zero and the
caption accurately covers the visible subject, action, environment, and apparent
camera motion without invented facts. Import the completed sheet:

```bash
venv/bin/leoma corpus v2 qa-import \
  --workdir "$WORK" --input /var/lib/leoma/corpus-v2-qa/reviews.jsonl
```

Publishing is blocked unless at least 200 reviews exist for the current unpublished
batch and at least 95% pass. Failed samples are quarantined. To caption them again
after deliberately changing the model or prompt, run `qa-retry-rejected`; otherwise
the successful publish gate permanently excludes them.

## 5. Publish verified sample pairs one by one

```bash
venv/bin/leoma corpus v2 publish \
  --workdir "$WORK" \
  --prefix corpus-v2/leoma-ti2v-pilot-10k-v1
```

Every MP4 and PNG is uploaded under its own content digest, then checked by size and
SHA-256 object metadata. A ledger row advances to `published` only after both objects
verify. The default removes both local files only after that checkpoint. An
interrupted command resumes the already-approved batch without requiring the same
200 audited rows to remain unpublished. Use `--keep-local` only when NVMe capacity
has been planned.

For a non-production smoke test with fewer than 200 total clips, the QA threshold can
be explicitly reduced. Never use this override for the real corpus:

```bash
venv/bin/leoma corpus v2 publish ... --qa-min-reviews 0
```

## 6. Build shards, then publish the root last

```bash
venv/bin/leoma corpus v2 manifest \
  --workdir "$WORK" \
  --prefix corpus-v2/leoma-ti2v-pilot-10k-v1 \
  --out /var/lib/leoma/corpus-v2-manifest

# After inspecting the local root and shards:
venv/bin/leoma corpus v2 manifest \
  --workdir "$WORK" \
  --prefix corpus-v2/leoma-ti2v-pilot-10k-v1 \
  --out /var/lib/leoma/corpus-v2-manifest \
  --upload
```

Shards are uploaded and verified first; the immutable root is the final object. Keep
the printed root digest as an artifact. Do not paste it into `chain.toml` yet.

## 7. Rehearse evaluator-v2 before activating the root

Captioning is an offline build step. The evaluator reads the pinned caption and
first-frame PNG from the selected manifest entries, so no Qwen process or dedicated
caption GPU runs during a duel. All eight H100s can remain assigned to the four
king/challenger eval pairs.

On a testnet release branch, set both corpus fields to the uploaded immutable root:

```toml
[corpus]
bucket = "leoma-source"
manifest_key = "<printed-root-object-key>"
manifest_digest = "<printed-sha256-root-digest>"
```

Changing either field changes the consensus digest. Deploy the exact same commit and
`chain.toml` to the validator and all eval processes, then run:

```bash
venv/bin/leoma corpus verify --sample 4
venv/bin/leoma corpus verify --sample 0
venv/bin/leoma preflight
```

The v2 verifier checks the root, every visited shard, MP4 and PNG file digests, decoded
truth-frame digest, decoded PNG RGB digest, and that the PNG is exactly truth frame
zero. `--sample 0` reads every published MP4 and PNG and is therefore a deliberate,
large production gate rather than a quick health check.

Run the same-model controls and real challenger scenarios from the production runbook,
then regenerate all calibration records against this exact root. Keep the prior
manifest key, manifest digest, consensus digest, and deployment image as the rollback
unit; never roll back only one validator or one eval process.

## 8. Scale only in audited high-watermark batches

After the pilot is accepted, reuse the same work directory and increase
`--max-samples` by a fixed batch size, for example 10,000 at a time. For every batch,
run prepare, caption, QA export/import, and publish before increasing the high-water
mark. Published clips and PNGs have been removed locally, while the compact SQLite
ledger retains all manifest data. The QA gate applies only to the new unpublished
batch, so an old audit cannot approve new captions.

At each 100k checkpoint, record yield/source, caption clips/GPU-hour, mean and p95
clip bytes, source concentration, QA failure categories, and total projected storage.
Stop if the distribution drifts or the projected two-million-sample corpus no longer
fits the source diversity, time, or storage budget.
