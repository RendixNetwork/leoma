"""The storage backend was compose-wired, documented, and hardcoded.

`OBJECT_STORAGE_BACKEND` is set in docker-compose, re-exported from `bootstrap`, and
even set by the existing tests — but `Settings.__init__` assigned the literal `"r2"`,
so the entire Hippius branch of `storage_backend` was unreachable. The parser
(`_parse_object_storage_backend`) was written and then never called.

Leoma's whole story is decentralized storage. The corpus living only on Cloudflare was
an accident of a one-line hardcode, not a decision.
"""

import importlib

import pytest


def _settings(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import leoma.bootstrap.runtime as rt

    importlib.reload(rt)
    return rt


class TestMainnetDefaults:
    def test_default_netuid_is_leoma_mainnet(self, monkeypatch):
        monkeypatch.delenv("NETUID", raising=False)
        rt = _settings(monkeypatch)
        assert rt.settings.netuid == 36


class TestBackendSelection:
    def test_the_default_is_still_r2_so_nothing_changes_for_operators(self, monkeypatch):
        monkeypatch.delenv("OBJECT_STORAGE_BACKEND", raising=False)
        rt = _settings(monkeypatch)
        assert rt.settings.object_storage_backend == "r2"

    def test_hippius_can_finally_be_selected(self, monkeypatch):
        rt = _settings(monkeypatch, OBJECT_STORAGE_BACKEND="hippius")
        assert rt.settings.object_storage_backend == "hippius"

    @pytest.mark.parametrize("alias", ["hippius", "hippius-s3", "s3-hippius"])
    def test_the_aliases_the_parser_always_supported_now_work(self, monkeypatch, alias):
        rt = _settings(monkeypatch, OBJECT_STORAGE_BACKEND=alias)
        assert rt.settings.object_storage_backend == "hippius"

    def test_an_unknown_backend_is_a_loud_error(self, monkeypatch):
        monkeypatch.setenv("OBJECT_STORAGE_BACKEND", "dropbox")
        import leoma.bootstrap.runtime as rt

        with pytest.raises(ValueError, match="must be 'hippius' or 'r2'"):
            importlib.reload(rt)

        monkeypatch.delenv("OBJECT_STORAGE_BACKEND")
        importlib.reload(rt)   # leave the module importable for everyone else


class TestR2ConfigIsNoLongerBakedIntoTheSource:
    def test_the_endpoint_is_env_driven_with_the_live_value_as_the_default(self, monkeypatch):
        monkeypatch.delenv("R2_ENDPOINT", raising=False)
        rt = _settings(monkeypatch)
        assert "r2.cloudflarestorage.com" in rt.settings.r2_endpoint_raw

        rt = _settings(monkeypatch, R2_ENDPOINT="https://example.r2.cloudflarestorage.com")
        assert rt.settings.r2_endpoint_raw == "https://example.r2.cloudflarestorage.com"

    def test_the_source_bucket_is_env_driven(self, monkeypatch):
        rt = _settings(monkeypatch, R2_SOURCE_BUCKET="my-corpus")
        assert rt.settings.r2_source_bucket == "my-corpus"


class TestDashboardBucketCredentials:
    def test_dashboard_defaults_to_state_endpoint_and_credentials(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            R2_OWN_ENDPOINT="s3.owner.example",
            R2_OWN_REGION="owner-region",
            R2_OWN_WRITE_ACCESS_KEY="owner-key",
            R2_OWN_WRITE_SECRET_KEY="owner-secret",
            LEOMA_DASHBOARD_ENDPOINT="",
            LEOMA_DASHBOARD_REGION="",
            LEOMA_DASHBOARD_WRITE_ACCESS_KEY="",
            LEOMA_DASHBOARD_WRITE_SECRET_KEY="",
        )

        assert rt.settings.dashboard_endpoint == "s3.owner.example"
        assert rt.settings.dashboard_region == "owner-region"
        assert rt.settings.dashboard_write_access_key == "owner-key"
        assert rt.settings.dashboard_write_secret_key == "owner-secret"

    def test_dashboard_can_use_bucket_scoped_credentials(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            R2_OWN_WRITE_ACCESS_KEY="owner-key",
            R2_OWN_WRITE_SECRET_KEY="owner-secret",
            LEOMA_DASHBOARD_ENDPOINT="s3.dashboard.example",
            LEOMA_DASHBOARD_REGION="dashboard-region",
            LEOMA_DASHBOARD_WRITE_ACCESS_KEY="dashboard-key",
            LEOMA_DASHBOARD_WRITE_SECRET_KEY="dashboard-secret",
        )

        assert rt.settings.dashboard_endpoint == "s3.dashboard.example"
        assert rt.settings.dashboard_region == "dashboard-region"
        assert rt.settings.dashboard_write_access_key == "dashboard-key"
        assert rt.settings.dashboard_write_secret_key == "dashboard-secret"
