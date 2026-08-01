# Production rehearsal: Wan2.2 on one 8xH100 host

This is the launch gate for Leoma's pinned Wan2.2 genesis model. Do not start the
validator on mainnet until every assertion below passes.

## Pinned model

The mutable Hippius revision `v1` has been resolved and is never used at runtime:

```text
leoma/wan2.2-i2v-a14b-diffusers@sha256:79e4cc89edd935e8ea843ae2bd80fb85d3444128bb946990109a33ffafb2184e
```

The snapshot is approximately 126.2 GB. Wan2.2 contains two transformer experts;
Leoma pins Diffusers model-level CPU offload so each 80 GB H100 retains activation
headroom. A full four-pair fleet can hold one king and four challengers on disk and
eight loaded pipelines in host memory. Use at least 2 TB of local NVMe and 768 GB of
system RAM; keep `LEOMA_MIN_FREE_BYTES` at 300 GiB or higher.

## 1. Host and dependency checks

Install Python 3.12, ffmpeg, Node/PM2, and a driver that supports the locked CUDA
12.4 PyTorch stack. Confirm all devices are healthy:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
ffmpeg -version
df -h /var/lib/leoma
```

Exactly eight H100s must be visible and no other process may consume their VRAM.
Install the frozen project environment rather than resolving dependencies again:

```bash
python3.12 -m venv venv
. venv/bin/activate
uv sync --active --frozen --extra eval --no-dev
python -c "import torch,diffusers,transformers,hippius_hub; print(torch.__version__, diffusers.__version__, transformers.__version__, hippius_hub.__version__)"
```

Expected core versions are PyTorch 2.6.0, Diffusers 0.35.2, Transformers 5.14.1,
and Hippius Hub 0.6.1.

## 2. Eval-host secrets

Create `.env` on the GPU host and never commit it:

```dotenv
# Optional for public miner repositories; set for private genesis access/rate limits.
HIPPIUS_HUB_TOKEN=
OBJECT_STORAGE_BACKEND=hippius
HIPPIUS_ENDPOINT=s3.hippius.com
HIPPIUS_REGION=decentralized
HIPPIUS_SOURCE_BUCKET=leoma-source
HIPPIUS_VIDEOS_READ_ACCESS_KEY=<read-access-key>
HIPPIUS_VIDEOS_READ_SECRET_KEY=<read-secret-key>
LEOMA_EVAL_TOKEN=<random-high-entropy-shared-token>
LEOMA_MODEL_CACHE_DIR=/var/lib/leoma/models
LEOMA_MAX_CACHED_SNAPSHOTS=8
LEOMA_MIN_FREE_BYTES=322122547200
```

Load it into the shell used to start PM2:

```bash
set -a
. ./.env
set +a
install -d -m 0750 /var/lib/leoma/models
```

## 3. Verify the pinned corpus on this box

First verify a quick sample, then the complete corpus before production:

```bash
venv/bin/leoma corpus verify --sample 4
venv/bin/leoma corpus verify --sample 0
```

Both commands must report byte-identical decoding. A failure means this host cannot
participate in consensus, even if another box decodes successfully.

## 4. Prewarm the immutable genesis snapshot

All four eval processes share one cache. Download the 126 GB genesis model once,
before starting any of them, so no processes race over a partial snapshot:

```bash
venv/bin/python - <<'PY'
from leoma.infra.chain_config import SEED_DIGEST, SEED_REPO
from leoma.infra.model_store import ModelRef, materialize_model

path = materialize_model(ModelRef(SEED_REPO, SEED_DIGEST))
print(path)
PY
```

Do not proceed until the snapshot directory contains `.leoma_complete.json` and the
filesystem still has more than 150 GiB free.

## 5. Calibrate every physical H100

Identical model names do not prove identical numerical behavior. Run two records on
each physical GPU to measure both repeatability and card-to-card noise:

```bash
mkdir -p calibration
for run in 1 2; do
  for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$gpu" \
      venv/bin/leoma calibrate generate \
      --gpu "h100-${gpu}-run${run}" \
      --n-clips 32 \
      --out "calibration/h100-${gpu}-run${run}.json"
  done
done
venv/bin/leoma calibrate analyze calibration/h100-*.json
```

`analyze` must return PASS. If it recommends a larger `delta_threshold`, update
`chain.toml`, redeploy the exact same config to validator and eval hosts, and rerun
preflight. Do not launch through a calibration failure.

## 6. Start one GPU pair first

The four-process PM2 file maps physical pairs 0/1, 2/3, 4/5 and 6/7 to local ports
9000 through 9003. Inside each isolated process the pair appears as cuda:0/cuda:1.

```bash
export LEOMA_BIN="$PWD/venv/bin/leoma"
pm2 start ecosystem.eval.8xh100.config.js --only leoma-eval-0 --update-env
curl -fsS -H "Authorization: Bearer $LEOMA_EVAL_TOKEN" http://127.0.0.1:9000/health
```

Run one real seed/self-evaluation or compatible challenger through this pair and
watch VRAM, RAM, disk and progress before enabling the remaining six GPUs. The
genesis model alone is not a full on-chain challenger test; a challenger must have
different weights and pass the naming rule below.

## 7. Start the four-process fleet

```bash
pm2 start ecosystem.eval.8xh100.config.js --update-env
pm2 save
pm2 status
```

All servers bind loopback and require the Bearer token. If the validator is on a
different host, expose them only through an SSH tunnel:

```bash
ssh -N \
  -L 9000:127.0.0.1:9000 \
  -L 9001:127.0.0.1:9001 \
  -L 9002:127.0.0.1:9002 \
  -L 9003:127.0.0.1:9003 \
  <gpu-host>
```

Never publish ports 9000-9003 directly to the internet.

## 8. Validator configuration and preflight

The validator needs its wallet, durable state bucket, corpus read credentials, the
same Bearer token, and all four URLs. A Hippius Hub read token is optional because
public miner configs are fetched anonymously:

```dotenv
NETWORK=finney
NETUID=99
WALLET_NAME=<coldkey-name>
HOTKEY_NAME=<validator-hotkey-name>
EVAL_SERVER_URLS=http://127.0.0.1:9000,http://127.0.0.1:9001,http://127.0.0.1:9002,http://127.0.0.1:9003
LEOMA_EVAL_TOKEN=<same-shared-token>
HIPPIUS_HUB_TOKEN=
OBJECT_STORAGE_BACKEND=hippius
HIPPIUS_ENDPOINT=s3.hippius.com
HIPPIUS_REGION=decentralized
HIPPIUS_SOURCE_BUCKET=leoma-source
HIPPIUS_VIDEOS_READ_ACCESS_KEY=<read-access-key>
HIPPIUS_VIDEOS_READ_SECRET_KEY=<read-secret-key>
R2_OWN_BUCKET=<validator-state-bucket>
LEOMA_DASHBOARD_BUCKET=<public-dashboard-bucket>
# Optional when the dashboard bucket has its own S3 credentials:
# LEOMA_DASHBOARD_ENDPOINT=<dashboard-bucket-s3-endpoint>
# LEOMA_DASHBOARD_REGION=<dashboard-bucket-region>
# LEOMA_DASHBOARD_WRITE_ACCESS_KEY=<dashboard-write-access-key>
# LEOMA_DASHBOARD_WRITE_SECRET_KEY=<dashboard-write-secret-key>
R2_OWN_ENDPOINT=<state-bucket-s3-endpoint>
R2_OWN_REGION=<state-bucket-region>
R2_OWN_WRITE_ACCESS_KEY=<state-write-access-key>
R2_OWN_WRITE_SECRET_KEY=<state-write-secret-key>
```

The `R2_OWN_*` names are legacy names for the validator's independent S3-compatible
state bucket; its endpoint may be Hippius. Then gate startup:

```bash
venv/bin/leoma preflight
venv/bin/leoma serve
```

Preflight must pass every configured eval URL with identical consensus and scoring
code digests.

## 9. Submit a real challenger

The genesis repo itself is not a valid miner submission name. A challenger repo must
start with `leoma` and end with the submitting hotkey, for example:

```text
<namespace>/leoma-wan22-test-<miner-hotkey>
```

Upload with `leoma miner push`, capture the returned immutable `sha256:` digest, then
commit that digest with `leoma miner commit`. Never commit the mutable revision name.

## 10. Production gates

Before mainnet, all of these must be true:

- full corpus verification passes on the H100 host;
- all 16 calibration records analyze to PASS;
- one complete real duel finishes on a single pair without OOM or watchdog timeout;
- all four authenticated health endpoints pass validator preflight;
- validator restart and one eval-process restart recover correctly;
- `leoma smoke <dashboard-url>` observes every required scenario;
- disk monitoring alerts before free space reaches 150 GiB;
- the deployed validator and eval images are pinned by image digest, not `latest`.
