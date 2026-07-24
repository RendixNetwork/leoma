# Leoma Subnet

Leoma is an **AI video subnet** on [Bittensor](https://docs.learnbittensor.org/). It runs as a
**king of the hill**: a single reigning **king** model holds the crown, and miners submit
**challengers** that must beat the king in a head-to-head duel to take it. Emission is split equally
across the king plus recent prior kings; when there is no king, it burns.

**Model type:** **Image/Text-Image to Video (I2V/TI2V)** — all miners fine-tune a single pinned base
architecture (see `chain.toml`).

---

## Contents

- [How it works](#how-it-works)
- [The duel](#the-duel)
- [Validator setup](#validator-setup)
- [Eval server setup](#eval-server-setup)
- [Miner setup](#miner-setup)
- [License](#license)

---

## How it works

Miners do **not** host inference. They upload model **weights** to a content-addressed registry
(**Hippius Hub**) and commit a compact reveal on-chain; validators download the weights and run the
model themselves. There is no owner-api and no rotation — every validator is independent and, because
the duel is deterministic (seeded from the block hash), they all converge on the same king.

1. **Miner submits.** Upload weights to Hippius Hub → immutable `repo@digest`, then commit
   `v4|<repo>|<digest>|<hotkey>` on-chain (`set_reveal_commitment`). The repo name must start with
   `leoma` and end with your hotkey.
2. **Validator discovers.** Each validator scans the chain's revealed commitments, parses each into an
   immutable `(repo, digest)` model reference, and queues new challengers.
3. **Validator duels.** For each challenger, the validator dispatches a duel to its GPU **eval server**,
   which downloads king + challenger by digest and scores them (below).
4. **Crown + weights.** A challenger that wins by a confident margin is crowned; the deposed king slides
   onto a bounded **king chain**. The validator sets **equal weights** across the king + up to 4 prior
   kings still registered (else burns 100% to UID 0). King state persists to the validator's own bucket.

## The duel

Leoma's duel metric is **reference-based and deterministic**. Because the source corpus is real video,
the ground-truth continuation of each clip is known. On the same block-hash-seeded held-out clips, king
and challenger each generate a continuation from the clip's first frame + prompt using the **same
per-clip seed**; each generation is scored against the **real continuation** with a reference distance
(LPIPS by default; MSE/SSIM available) — lower is better.

The challenger is crowned only if it is **confidently** better: the per-clip advantage
`king_distance − challenger_distance` is bootstrapped, and the crown passes iff the lower-confidence
bound `lcb > delta_threshold`.

Every input that can change a verdict — the corpus, the prompt, the frame count, the resolution, the
metric, the threshold — is pinned in **`chain.toml`**, hashed into a `consensus_digest` that is sent
with each eval request and **echoed back in the verdict**. A validator running a different config
cannot quietly disagree with the rest of the subnet: the mismatch is refused at the door. The held-out
clips come from a **digest-pinned corpus manifest** (not a live bucket listing), with each clip's
window and ground-truth hash fixed offline, so two validators provably grade the same exam. The block
hash is unpredictable until mined, so miners cannot overfit to the test set, yet every validator
reproduces the identical verdict.

---

## Validator setup

The validator scans reveals, dispatches duels to an eval server, crowns winners, and sets weights.

### Prerequisites

- **Bittensor wallet** (coldkey + hotkey) registered as a validator on the subnet.
- **A reachable eval server** (`EVAL_SERVER_URL`), typically an SSH tunnel to a GPU box.
- **An own bucket** (`R2_OWN_BUCKET` + `R2_OWN_WRITE_*`) for durable king state.

### Run

```bash
# Requires Python 3.12+
pip install -e .            # or: uv pip install -e .
cp env.validator.example .env   # fill in wallet, EVAL_SERVER_URL, R2_OWN_*
leoma serve                 # scan reveals -> duel -> crown -> set weights
```

Or with Docker: `cp env.validator.example .env && docker compose up -d validator`. Mount your
Bittensor wallets so the container can sign weight-setting transactions.

---

## Eval server setup

The eval server is the GPU box that downloads and runs miner models. Install the `[eval]` extra
(torch/diffusers/lpips) and provide Hippius Hub credentials + source-corpus read keys.

```bash
pip install -e '.[eval]'
cp env.eval.example .env    # fill in HIPPIUS_HUB_TOKEN, HIPPIUS_VIDEOS_READ_*
leoma servers eval-server   # FastAPI on EVAL_SERVER_PORT (default 9000)
```

The validator reaches it over `EVAL_SERVER_URL` (default `http://localhost:9000`, usually an SSH
tunnel). One duel runs at a time. See `ecosystem.eval.config.js` for a PM2 launcher.

Before a new eval box is allowed to duel, prove it decodes the pinned corpus byte-identically:

```bash
leoma corpus verify --sample 4
```

A box whose ffmpeg produces even slightly different pixels measures every distance against different
ground truth — silently, confidently, and wrongly. This takes a minute and rules that out.

---

## Corpus (subnet operator)

The duel's held-out clips come from a **pinned manifest**, not a live bucket listing. The manifest
fixes which videos, the window inside each one, and the hash of the decoded ground truth — so every
validator provably grades the same exam. Building it once, offline, is what removes the whole class of
"two honest validators disagree" bugs: nothing is scene-detected, listed or skipped at duel time.

```bash
leoma corpus build-manifest --corpus-id leoma-corpus-v1   # decides windows, hashes truth
leoma corpus publish-manifest manifest.json               # uploads; prints the digest
# paste that digest into chain.toml [corpus].manifest_digest, then ship it
```

**Until `[corpus].manifest_digest` and `[seed].seed_digest` are pinned, validators refuse to duel and
burn 100% to UID 0.** That is deliberate: an unpinned corpus is not reproducible, and an unevaluated
first challenger must never be crowned. Rotate the corpus by rebuilding with a new `corpus-id` and
re-pinning — a version bump, auditable in git.

The captioned, sharded TI2V corpus-v2 builder is deliberately offline from the active
v1 evaluator. Its bounded 10k pilot, eight-GPU captioning, human-QA gate, incremental
publication, and staged scale-up procedure are documented in
[`docs/CORPUS_V2_PILOT_RUNBOOK.md`](docs/CORPUS_V2_PILOT_RUNBOOK.md). Do not pin a v2
root until evaluator-v2 support and calibration have landed.

---

## Miner setup

Fine-tune the pinned base architecture (see `chain.toml`), then upload the weights and commit.

### 1. Upload weights to Hippius Hub

```bash
leoma miner push --model-dir ./out --repo <user>/leoma-<name>-<your-hotkey-ss58>
```

The repo name must **start with `leoma`** and **end with your hotkey** (SS58). The command prints the
immutable `repo@digest` to commit.

### 2. Commit the reveal on-chain

```bash
leoma miner commit --repo <user>/leoma-<name>-<hotkey> --digest sha256:<...> \
  --coldkey <wallet-name> --hotkey <hotkey-name>
```

Your wallet must be registered on the subnet. Validators will discover the reveal, duel your model
against the king, and crown it if it wins.

---

## License

MIT
