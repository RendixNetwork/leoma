// PM2 config for the Leoma EVAL SERVER (runs on the GPU box).
//
// Downloads king + challenger weights from Hippius Hub by digest, loads the
// pinned diffusers I2V pipeline, generates on the deterministic held-out clips,
// and scores each generation against the real continuation. One duel at a time.
// The validator reaches this over EVAL_SERVER_URL (SSH tunnel to :9000).
// For a full 8xH100 fleet use ecosystem.eval.8xh100.config.js instead.
module.exports = {
  apps: [
    {
      name: "leoma-eval-server",
      script: "leoma",
      args: "servers eval-server",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      env: {
        EVAL_SERVER_HOST: "127.0.0.1",
        EVAL_SERVER_PORT: "9000",
        LEOMA_EVAL_TOKEN: process.env.LEOMA_EVAL_TOKEN || "",

        // Optional read auth; public miner repositories download anonymously.
        HIPPIUS_HUB_TOKEN: process.env.HIPPIUS_HUB_TOKEN || "",
        HIPPIUS_HUB_USERNAME: process.env.HIPPIUS_HUB_USERNAME || "",
        HIPPIUS_HUB_PASSWORD: process.env.HIPPIUS_HUB_PASSWORD || "",
        LEOMA_MODEL_CACHE_DIR: process.env.LEOMA_MODEL_CACHE_DIR || "/var/lib/leoma/models",
        LEOMA_UV_LOCK_PATH: process.env.LEOMA_UV_LOCK_PATH || `${__dirname}/uv.lock`,

        // Hippius S3 source-video corpus (ground-truth continuations).
        OBJECT_STORAGE_BACKEND: "hippius",
        HIPPIUS_ENDPOINT: process.env.HIPPIUS_ENDPOINT || "s3.hippius.com",
        HIPPIUS_REGION: process.env.HIPPIUS_REGION || "decentralized",
        HIPPIUS_SOURCE_BUCKET: process.env.HIPPIUS_SOURCE_BUCKET || "leoma-source",
        HIPPIUS_VIDEOS_READ_ACCESS_KEY: process.env.HIPPIUS_VIDEOS_READ_ACCESS_KEY || "",
        HIPPIUS_VIDEOS_READ_SECRET_KEY: process.env.HIPPIUS_VIDEOS_READ_SECRET_KEY || "",
      },
    },
  ],
};
