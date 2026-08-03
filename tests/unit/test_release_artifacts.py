"""Production artifacts cannot silently resolve mutable runtime inputs."""

import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def test_container_bases_uv_and_python_graph_are_locked():
    for name in ("Dockerfile", "Dockerfile.eval"):
        dockerfile = _read(name)
        first_from = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
        assert re.search(r"@sha256:[0-9a-f]{64}$", first_from)
        assert "pip install --no-cache-dir uv==0.11.6" in dockerfile
        assert "ENV VIRTUAL_ENV=/opt/venv" in dockerfile
        assert "uv sync --active --frozen" in dockerfile
        assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile


def test_production_compose_is_pull_only_and_digest_addressed():
    for name, variable in (
        ("docker-compose.validator.production.yml", "LEOMA_VALIDATOR_IMAGE_DIGEST"),
        ("docker-compose.eval.8xh100.production.yml", "LEOMA_EVAL_IMAGE_DIGEST"),
    ):
        compose = _read(name)
        assert "build:" not in compose
        assert ":latest" not in compose
        assert f"rendixnetwork/leoma@${{{variable}:?" in compose
        assert "pull_policy: always" in compose


def test_h100_compose_has_four_isolated_pairs_and_loopback_ports():
    compose = _read("docker-compose.eval.8xh100.production.yml")
    for pair in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert f'device_ids: ["{pair[0]}", "{pair[1]}"]' in compose
    for port in range(9000, 9004):
        assert f'"127.0.0.1:{port}:9000"' in compose
