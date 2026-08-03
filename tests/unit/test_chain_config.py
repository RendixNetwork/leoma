"""
Unit tests for chain.toml resolution.

chain.toml is consensus-critical and is read at IMPORT time, so it must ship
inside the package (leoma/chain.toml) — not merely exist in the git checkout.
It previously lived at the repo root and was resolved via ``parents[2]``, which
meant a non-editable install resolved it into ``site-packages/`` and crashed with
FileNotFoundError on startup. These tests pin the resolution contract.
"""

import pathlib

import pytest

from leoma.infra import chain_config
from leoma.infra.chain_config import _resolve_toml_path


class TestPackagedLocation:
    def test_chain_toml_lives_inside_the_package(self):
        """The file must sit next to the package, so package-data ships it."""
        pkg_dir = pathlib.Path(chain_config.__file__).resolve().parents[1]  # .../leoma/
        assert (pkg_dir / "chain.toml").is_file(), (
            "chain.toml must live at leoma/chain.toml so it ships in the wheel"
        )

    def test_resolved_path_is_the_packaged_one(self):
        resolved = _resolve_toml_path()
        pkg_dir = pathlib.Path(chain_config.__file__).resolve().parents[1]
        assert resolved.is_file()
        assert resolved.resolve() == (pkg_dir / "chain.toml").resolve()

    def test_config_actually_loaded(self):
        assert chain_config.NAME == "leoma"
        assert chain_config.ARCH_PIPELINE  # the one live [arch] key

    def test_wan22_base_and_genesis_are_immutably_pinned_together(self):
        assert chain_config.ARCH_BASE_REPO == "leoma/wan2.2-i2v-a14b-diffusers"
        assert chain_config.ARCH_BASE_DIGEST.startswith("sha256:")
        assert chain_config.ARCH_BASE_DIGEST == chain_config.SEED_DIGEST
        assert chain_config.SEED_REPO == chain_config.ARCH_BASE_REPO
        assert chain_config.SEED_REPO_BACKEND == "hippius"

    def test_eval_runtime_is_pinned_to_the_h100_lock(self):
        runtime = chain_config.SPEC.runtime
        assert runtime.python == "3.12"
        assert runtime.cuda == "12.4"
        assert runtime.compute_capability == "9.0"
        assert runtime.eval_lock_digest.startswith("sha256:")


class TestOverride:
    def test_override_absolute_path(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt.toml"
        alt.write_text('[chain]\nname = "alt"\nseed_repo = "x/y"\n')
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", str(alt))
        assert _resolve_toml_path() == alt

    def test_override_relative_to_cwd(self, tmp_path, monkeypatch):
        alt = tmp_path / "rel.toml"
        alt.write_text('[chain]\nname = "rel"\nseed_repo = "x/y"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", "rel.toml")
        assert _resolve_toml_path().resolve() == alt.resolve()

    def test_missing_override_raises_clearly(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LEOMA_CHAIN_OVERRIDE", str(tmp_path / "nope.toml"))
        with pytest.raises(RuntimeError, match="missing file"):
            _resolve_toml_path()
