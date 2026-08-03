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


class TestHippiusValidatorState:
    def test_hippius_own_names_select_the_hippius_state_bucket(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            OBJECT_STORAGE_BACKEND="hippius",
            HIPPIUS_ENDPOINT="s3.hippius.example",
            HIPPIUS_REGION="decentralized",
            HIPPIUS_OWN_BUCKET="leoma-state",
            HIPPIUS_OWN_WRITE_ACCESS_KEY="state-key",
            HIPPIUS_OWN_WRITE_SECRET_KEY="state-secret",
        )

        assert rt.settings.r2_own_bucket == "leoma-state"
        assert rt.settings.r2_own_endpoint == "s3.hippius.example"
        assert rt.settings.r2_own_region == "decentralized"
        assert rt.settings.r2_own_write_access_key == "state-key"
        assert rt.settings.r2_own_write_secret_key == "state-secret"

    def test_same_hippius_bucket_can_reuse_its_bucket_scoped_dashboard_key(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            OBJECT_STORAGE_BACKEND="hippius",
            HIPPIUS_OWN_BUCKET="leoma-dashboard",
            LEOMA_DASHBOARD_BUCKET="leoma-dashboard",
            LEOMA_DASHBOARD_WRITE_ACCESS_KEY="bucket-key",
            LEOMA_DASHBOARD_WRITE_SECRET_KEY="bucket-secret",
        )

        assert rt.settings.r2_own_write_access_key == "bucket-key"
        assert rt.settings.r2_own_write_secret_key == "bucket-secret"

    def test_canonical_hippius_bucket_does_not_inherit_stale_legacy_r2_keys(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            OBJECT_STORAGE_BACKEND="hippius",
            HIPPIUS_OWN_BUCKET="leoma-dashboard",
            R2_OWN_WRITE_ACCESS_KEY="wrong-legacy-key",
            R2_OWN_WRITE_SECRET_KEY="wrong-legacy-secret",
            LEOMA_DASHBOARD_BUCKET="leoma-dashboard",
            LEOMA_DASHBOARD_WRITE_ACCESS_KEY="bucket-key",
            LEOMA_DASHBOARD_WRITE_SECRET_KEY="bucket-secret",
        )

        assert rt.settings.r2_own_write_access_key == "bucket-key"
        assert rt.settings.r2_own_write_secret_key == "bucket-secret"

    def test_legacy_r2_own_names_remain_a_hippius_fallback(self, monkeypatch):
        rt = _settings(
            monkeypatch,
            OBJECT_STORAGE_BACKEND="hippius",
            R2_OWN_BUCKET="legacy-state",
            R2_OWN_ENDPOINT="legacy.hippius.example",
            R2_OWN_WRITE_ACCESS_KEY="legacy-key",
            R2_OWN_WRITE_SECRET_KEY="legacy-secret",
        )

        assert rt.settings.r2_own_bucket == "legacy-state"
        assert rt.settings.r2_own_endpoint == "legacy.hippius.example"
        assert rt.settings.r2_own_write_access_key == "legacy-key"

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
