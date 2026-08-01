// PM2 config for the Leoma king-of-the-hill VALIDATOR (runs on the validator host).
//
// Scans on-chain miner reveals, dispatches duels to the eval server (reached at
// EVAL_SERVER_URL — typically an SSH tunnel to the GPU box), crowns winners, and
// sets weights across the king chain. Fill wallet + bucket + Hippius Hub creds
// from your secrets manager; do not commit real secrets.
module.exports = {
  apps: [
    {
      name: "leoma-validator",
      script: process.env.LEOMA_BIN || "leoma",
      args: "serve",
      interpreter: "none",
      autorestart: true,
      max_restarts: 1000,
      env: {
        // PM2 pipes stdout, so Python otherwise block-buffers successful tick logs
        // until process exit. Production health must be observable in real time.
        PYTHONUNBUFFERED: process.env.PYTHONUNBUFFERED || "1",
        NETWORK: process.env.NETWORK || "finney",
        NETUID: process.env.NETUID || "99",
        WALLET_NAME: process.env.WALLET_NAME || "default",
        HOTKEY_NAME: process.env.HOTKEY_NAME || "default",

        // Where duels are dispatched (SSH-tunnel to the GPU eval box).
        EVAL_SERVER_URL: process.env.EVAL_SERVER_URL || "http://localhost:9000",
        EVAL_SERVER_URLS: process.env.EVAL_SERVER_URLS || "",
        LEOMA_EVAL_TOKEN: process.env.LEOMA_EVAL_TOKEN || "",

        // Prescreen + corpus access.
        HIPPIUS_HUB_TOKEN: process.env.HIPPIUS_HUB_TOKEN || "",
        OBJECT_STORAGE_BACKEND: process.env.OBJECT_STORAGE_BACKEND || "hippius",
        HIPPIUS_ENDPOINT: process.env.HIPPIUS_ENDPOINT || "s3.hippius.com",
        HIPPIUS_REGION: process.env.HIPPIUS_REGION || "decentralized",
        HIPPIUS_SOURCE_BUCKET: process.env.HIPPIUS_SOURCE_BUCKET || "leoma-source",
        HIPPIUS_VIDEOS_READ_ACCESS_KEY: process.env.HIPPIUS_VIDEOS_READ_ACCESS_KEY || "",
        HIPPIUS_VIDEOS_READ_SECRET_KEY: process.env.HIPPIUS_VIDEOS_READ_SECRET_KEY || "",

        // Duel parameters live in chain.toml (the consensus surface), not here.
        LEOMA_KING_CHAIN_SIZE: "5",
        LEOMA_WEIGHT_INTERVAL: "300",
        LEOMA_BURN_UID: "0",

        // This validator's own state bucket (durable king state).
        R2_OWN_BUCKET: process.env.R2_OWN_BUCKET || "",
        R2_OWN_ENDPOINT: process.env.R2_OWN_ENDPOINT || "",
        R2_OWN_REGION: process.env.R2_OWN_REGION || "auto",
        R2_OWN_WRITE_ACCESS_KEY: process.env.R2_OWN_WRITE_ACCESS_KEY || "",
        R2_OWN_WRITE_SECRET_KEY: process.env.R2_OWN_WRITE_SECRET_KEY || "",
        // Public-read delivery bucket. Defaults to the owner endpoint/credentials
        // above; optional overrides support providers with bucket-scoped keys.
        LEOMA_DASHBOARD_BUCKET: process.env.LEOMA_DASHBOARD_BUCKET || "",
        LEOMA_DASHBOARD_ENDPOINT: process.env.LEOMA_DASHBOARD_ENDPOINT || "",
        LEOMA_DASHBOARD_REGION: process.env.LEOMA_DASHBOARD_REGION || "",
        LEOMA_DASHBOARD_WRITE_ACCESS_KEY: process.env.LEOMA_DASHBOARD_WRITE_ACCESS_KEY || "",
        LEOMA_DASHBOARD_WRITE_SECRET_KEY: process.env.LEOMA_DASHBOARD_WRITE_SECRET_KEY || "",
      },
    },
  ],
};
