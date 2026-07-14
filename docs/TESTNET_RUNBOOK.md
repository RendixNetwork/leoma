# Testnet dress-rehearsal runbook

A repeatable rehearsal that proves the subnet does the right thing **before** mainnet.
Every step ends in an assertion, not an eyeball — `leoma preflight` gates the launch,
and `leoma smoke` confirms each scenario was actually exercised and handled correctly.

Run this on testnet with at least one validator + one eval box (a GPU) and a handful of
miner hotkeys you control.

---

## 0. Prerequisites — pin the consensus surface

The subnet deliberately burns 100% to UID 0 until these are pinned. This is not optional.

1. **Pick the base-model revision** and pin `chain.toml [seed].seed_digest` (the genesis
   king) to the exact Wan2.1-I2V-14B-480P revision you will run — either a Hippius OCI
   digest (`sha256:<64hex>`) or a HuggingFace commit SHA (`hf:<40hex>`). `preflight`
   rejects anything else as unresolvable.
2. **Build and publish the corpus:**
   ```bash
   leoma corpus build-manifest --corpus-id leoma-testnet-v1   # decides windows, hashes truth
   leoma corpus publish-manifest manifest.json                # prints the digest
   ```
   Paste the printed digest into `chain.toml [corpus].manifest_digest`.
3. **Verify each eval box** decodes the corpus byte-identically to the manifest:
   ```bash
   leoma corpus verify --sample 4
   ```
   A box that fails this must not duel — its distances would not be reproducible.

## 1. Calibrate `delta_threshold` (the load-bearing measurement)

This is the single largest open consensus risk. On **each GPU type** in the fleet:

```bash
leoma calibrate generate --gpu <label> -o box-<label>.json
```

Then, once, compare them all:

```bash
leoma calibrate analyze box-*.json
```

- **PASS** → the current `delta_threshold` clears the measured cross-GPU noise floor.
  Proceed.
- **FAIL** → `delta_threshold` is *below* the noise floor. Two honest validators can
  fork. Raise it to the recommended value **or**, if the recommendation is implausibly
  large, treat it as a signal that LPIPS-on-generated-frames is too noisy for
  cross-hardware consensus and make a structural decision (a more reproducible metric,
  or pin the fleet to one GPU class). **Do not launch through a FAIL.**

## 2. Preflight — the launch gate

On the validator box, with `EVAL_SERVER_URL`, `R2_OWN_BUCKET`, `WALLET_NAME`,
`HOTKEY_NAME` set:

```bash
leoma preflight
```

It exits non-zero (and says exactly why) if the seed or corpus is unpinned, the
consensus surface is invalid, or the eval box is on a different `chain.toml` / scoring
code than the validator. **Gate your launch script on it:**

```bash
leoma preflight && leoma serve
```

Running several eval-server processes (one per GPU pair on an 8×H100 box)? Set
`EVAL_SERVER_URLS` (comma-separated) instead of the single `EVAL_SERVER_URL` —
`preflight` checks every configured URL independently and labels each finding by its
box, so one stale server can't hide behind a healthy sibling.

## 3. Start the services

```bash
# GPU box(es) — one process per pair of GPUs to duel on, each pinned via
# LEOMA_KING_DEVICE/LEOMA_CHALLENGER_DEVICE if running more than one on the same host
leoma servers eval-server           # binds 127.0.0.1; validator reaches it over an SSH tunnel

# validator box — set EVAL_SERVER_URLS to the comma-separated list if running more than one
leoma serve
```

## 4. Drive the scenarios

Submit each of these as a miner (`leoma miner push` + `leoma miner commit`) and let the
validator pick it up. The goal is to exercise every handling path once.

| # | Scenario | How to produce it | Expected outcome |
|---|----------|-------------------|------------------|
| A | **Genuine crown** | A model that actually beats the genesis king | Crowned; king chain grows; weights shift |
| B | **Fair rejection** | A model weaker than the king | Scored, `lcb < delta`, king holds |
| C | **Broken repo (quarantine)** | Commit a reveal pointing at a non-existent repo | `error` row, `model_not_found`; quarantined after 2 sightings; **later challengers still run** |
| D | **Wrong architecture** | A model whose `transformer/config.json` shape differs from the base | Rejected pre-dispatch (`arch_mismatch`) in ~seconds, no GPU spent |
| E | **Copy of the king** | Re-upload the king's weights under a new hotkey (change only the README) | Rejected pre-duel (`copy_of_king`); no multi-hour duel |
| F | **Freeze cheat** | A model that emits the conditioning frame repeated | Rejected by the freeze gate (`FROZE OUT`) even if it beat a weak king |

## 5. Smoke — assert the outcomes

Point `smoke` at the validator's published `dashboard.json`:

```bash
leoma smoke https://<your-state-bucket>/dashboard.json
```

It reports which rehearsal scenarios have been observed and exits non-zero until every
one has. Re-run it as you drive more scenarios; a clean run means:

- ✓ a challenger beat the king and was crowned
- ✓ a challenger was scored and lost fairly
- ✓ a broken model was recorded as an error (not silently dropped)
- ✓ a copy of the king was rejected pre-duel
- ✓ a freeze cheat was rejected by the gate
- ✓ the validator is dueling, not degraded

Check the **dashboard** during a real duel too — the "in the arena" panel should show
the live challenger-vs-king, and the quality-over-reigns chart should show models
sitting well below the dashed freeze cheat floor.

## 6. Resilience spot-checks (optional but recommended)

- **Restart the validator mid-duel.** It should re-attach to the in-flight slot and
  settle the same duel, not orphan it.
- **Kill the eval box mid-duel.** The validator should classify it transient and retry;
  the box should come back with a fresh CUDA context.
- **Point one eval box at a stale `chain.toml`.** `preflight` and the validator's
  dispatch preflight should both refuse it (`consensus_mismatch`). With a single
  configured eval server the dashboard should show the `degraded` reason; with several
  configured (`EVAL_SERVER_URLS`), the stale one should simply be skipped in favor of a
  healthy sibling — the validator should **not** show `degraded` as long as at least one
  configured box is healthy.

---

## Definition of done

- `leoma calibrate analyze` returns **PASS** for the fleet's `delta_threshold`.
- `leoma preflight` exits 0 on every validator.
- `leoma smoke` reports **all** scenarios observed.
- The dashboard shows a live duel and the cheat-floor chart.

Only then is the subnet ready for mainnet.
