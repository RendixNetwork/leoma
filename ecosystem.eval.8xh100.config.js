// Four isolated Leoma eval-server processes for one 8xH100 host.
//
// Load secrets into the shell before starting PM2:
//   set -a; . ./.env; set +a
//   pm2 start ecosystem.eval.8xh100.config.js --update-env
//
// The shared cache is safe only after the immutable genesis model has been
// prewarmed before these processes start. See docs/PRODUCTION_8XH100_RUNBOOK.md.

const pairs = [
  [0, 1],
  [2, 3],
  [4, 5],
  [6, 7],
];

const required = [
  "HIPPIUS_VIDEOS_READ_ACCESS_KEY",
  "HIPPIUS_VIDEOS_READ_SECRET_KEY",
  "LEOMA_EVAL_TOKEN",
];
const missing = required.filter((name) => !process.env[name]);
if (missing.length) {
  throw new Error(`Missing required eval-fleet environment: ${missing.join(", ")}`);
}

const shared = {
  EVAL_SERVER_HOST: "127.0.0.1",
  LEOMA_EVAL_TOKEN: process.env.LEOMA_EVAL_TOKEN,
  // Optional for public miner repositories; useful for private genesis repos or
  // avoiding anonymous registry rate limits.
  HIPPIUS_HUB_TOKEN: process.env.HIPPIUS_HUB_TOKEN || "",
  OBJECT_STORAGE_BACKEND: "hippius",
  HIPPIUS_ENDPOINT: process.env.HIPPIUS_ENDPOINT || "s3.hippius.com",
  HIPPIUS_REGION: process.env.HIPPIUS_REGION || "decentralized",
  HIPPIUS_SOURCE_BUCKET: process.env.HIPPIUS_SOURCE_BUCKET || "leoma-source",
  HIPPIUS_VIDEOS_READ_ACCESS_KEY: process.env.HIPPIUS_VIDEOS_READ_ACCESS_KEY,
  HIPPIUS_VIDEOS_READ_SECRET_KEY: process.env.HIPPIUS_VIDEOS_READ_SECRET_KEY,
  LEOMA_MODEL_CACHE_DIR: process.env.LEOMA_MODEL_CACHE_DIR || "/var/lib/leoma/models",
  LEOMA_MAX_CACHED_SNAPSHOTS: process.env.LEOMA_MAX_CACHED_SNAPSHOTS || "8",
  LEOMA_MIN_FREE_BYTES: process.env.LEOMA_MIN_FREE_BYTES || "322122547200",
  LEOMA_UV_LOCK_PATH: process.env.LEOMA_UV_LOCK_PATH || `${__dirname}/uv.lock`,
  LEOMA_CONCURRENT_GENERATION: "1",
  LEOMA_KING_DEVICE: "cuda:0",
  LEOMA_CHALLENGER_DEVICE: "cuda:1",
  CUDA_DEVICE_ORDER: "PCI_BUS_ID",
  CUBLAS_WORKSPACE_CONFIG: ":4096:8",
};

module.exports = {
  apps: pairs.map(([kingGpu, challengerGpu], index) => ({
    name: `leoma-eval-${index}`,
    script: process.env.LEOMA_BIN || "leoma",
    args: "servers eval-server",
    interpreter: "none",
    exec_mode: "fork",
    instances: 1,
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    time: true,
    env: {
      ...shared,
      EVAL_SERVER_PORT: String(9000 + index),
      CUDA_VISIBLE_DEVICES: `${kingGpu},${challengerGpu}`,
    },
  })),
};
