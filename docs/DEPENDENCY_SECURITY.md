# Dependency security and numerical-runtime policy

Last reviewed: 2026-08-02.

Leoma has two different upgrade classes. Validator, network, HTTP, crypto, and CLI
packages may be patched after the normal test/build gate. Packages that decode
frames, load weights, generate video, or calculate scores are consensus-sensitive:
their upgrade also requires a coordinated runtime-digest release and H100
calibration. An unattended dependency bot must never merge the latter class.

## Current audit result

The locked validator graph has no known advisory after upgrading FastAPI/Starlette,
Pillow, cryptography, urllib3, idna, msgpack, pyasn1, Pygments, Click, setuptools,
and yt-dlp. Pillow 12.3.0 is a frame-processing change, so this release still needs
the calibration procedure below before mainnet rollout.

The evaluator intentionally retains Torch 2.6.0 and Diffusers 0.35.2 until the
replacement stack passes a real Wan2.2 compatibility/calibration run. The current
advisory database reports findings in both. Their most serious miner-reachable class
is malicious repository code or pickle deserialization. Leoma applies independent
controls before either library sees a miner snapshot:

- the registry request allows only JSON/config/tokenizer data and `.safetensors`;
- the materialized directory is rechecked before every load, including completed
  cache entries; executable files, pickle weights, symlinks, and special files fail;
- Diffusers is forced offline with `local_files_only=True`,
  `trust_remote_code=False`, and `use_safetensors=True`;
- the pipeline, component libraries/classes, shape-critical config, and total model
  size must match the immutable Wan2.2 base architecture.

These controls substantially reduce reachability but do not make an old ML runtime
equivalent to a patched one. The evaluator remains a dedicated, authenticated box;
its ports bind loopback, model repositories are treated as hostile, and no wallet or
state-bucket write credentials belong on that host.

The dashboard lock has all available non-breaking fixes. npm currently reports one
React Router RSC action-CSRF advisory with no non-vulnerable published version; its
suggested downgrade reintroduces older advisories. Leoma is a client-rendered Vite
SPA using `BrowserRouter`/`Routes`, with no RSC server, route actions, loaders, or
document request handler, so the affected server path is absent. Recheck and remove
this exception as soon as a fixed release is published.

## Consensus-sensitive upgrade gate

For Pillow, Torch, Diffusers, Transformers, Accelerate, safetensors, NumPy, SciPy,
OpenCV, torchvision, LPIPS, or OpenCLIP changes:

1. Update exact versions in `uv.lock` and `leoma/eval/runtime_lock.py`.
2. Set `[runtime].eval_lock_digest` in `chain.toml` to the new `uv.lock` SHA-256.
3. Build and publish the eval image, then record its immutable registry digest.
4. On every physical H100, run two same-model control records and analyze all 16.
5. Run a complete seed-versus-seed duel and a known real challenger regression.
6. Only after every check passes, deploy that exact image digest to all four eval
   pairs and the matching consensus release to every validator.

Do not mix old/new runtime digests. Evaluator health and validator dispatch are
fail-closed specifically to prevent that partial rollout from producing verdicts.
