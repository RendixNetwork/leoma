# Leoma VALIDATOR image (slim, CPU).
#
# The validator scans on-chain reveals, dispatches duels to the GPU eval server,
# crowns winners and sets weights. It never loads a model, so it needs no torch.
# The GPU eval server uses Dockerfile.eval instead.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# ffprobe/ffmpeg for the video utilities; curl for healthchecks;
# build-essential for any deps that need compiling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    LEOMA_UV_LOCK_PATH=/app/uv.lock

WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.6

COPY pyproject.toml uv.lock README.md ./
COPY leoma ./leoma

# Install the package (non-editable for a production image). chain.toml ships
# INSIDE the package via [tool.setuptools.package-data] — it is consensus-critical
# and is read at import time, so it must exist in site-packages, not just in git.
RUN uv sync --active --frozen --no-dev --no-editable --no-cache

# Fail at BUILD time if the package cannot import. This is exactly the bug class
# that shipped before (a missing chain.toml / missing numpy only surfaced when the
# container started and crash-looped).
RUN python -c "import leoma.infra.chain_config as c, leoma.app.validator.main; print('validator image OK, chain =', c.NAME)"

CMD ["leoma", "serve"]
