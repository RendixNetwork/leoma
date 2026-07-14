"""CLI-level crash handling for `leoma smoke`.

Before this, an unreachable URL or malformed JSON crashed with a raw traceback
(httpx.ConnectError / json.JSONDecodeError bubbling straight out of the command) instead
of a clean, actionable CLI error — the exact opposite of what an operator needs while
driving a testnet rehearsal. These tests pin that both failure modes, for both the URL
and file-path sources, now exit cleanly with a readable message.
"""
import json

import httpx
import pytest
from click.testing import CliRunner

from leoma.delivery.commands import cli


@pytest.fixture
def runner():
    return CliRunner()


def _good_dashboard():
    return {
        "history": [{"accepted": True, "verdict": "challenger", "hotkey": "5win",
                     "model_repo": "u/leoma-win", "block": 100}],
        "live_duels": [],
        "degraded": None,
    }


class TestSmokeFromFile:
    def test_a_valid_dashboard_file_runs_cleanly(self, runner, tmp_path):
        path = tmp_path / "dashboard.json"
        path.write_text(json.dumps(_good_dashboard()))
        result = runner.invoke(cli, ["smoke", str(path)])
        assert "scenarios observed" in result.output

    def test_a_missing_file_exits_cleanly_not_a_traceback(self, runner, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        result = runner.invoke(cli, ["smoke", str(missing)])
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "could not read dashboard file" in result.output

    def test_malformed_json_exits_cleanly_not_a_traceback(self, runner, tmp_path):
        path = tmp_path / "dashboard.json"
        path.write_text("{not valid json")
        result = runner.invoke(cli, ["smoke", str(path)])
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "not valid JSON" in result.output


class TestSmokeFromUrl:
    def test_an_unreachable_url_exits_cleanly_not_a_traceback(self, runner, monkeypatch):
        def _raise(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(httpx, "get", _raise)
        result = runner.invoke(cli, ["smoke", "http://nonexistent-eval-box:9000/dashboard.json"])
        assert result.exit_code != 0
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "could not fetch dashboard" in result.output

    def test_a_non_200_response_exits_cleanly(self, runner, monkeypatch):
        def _fake_get(url, timeout=None):
            return httpx.Response(500, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", _fake_get)
        result = runner.invoke(cli, ["smoke", "http://eval-box:9000/dashboard.json"])
        assert result.exit_code != 0
        assert "could not fetch dashboard" in result.output

    def test_a_valid_url_response_runs_cleanly(self, runner, monkeypatch):
        def _fake_get(url, timeout=None):
            return httpx.Response(200, json=_good_dashboard(), request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", _fake_get)
        result = runner.invoke(cli, ["smoke", "http://eval-box:9000/dashboard.json"])
        assert "scenarios observed" in result.output

    def test_non_json_body_exits_cleanly(self, runner, monkeypatch):
        def _fake_get(url, timeout=None):
            return httpx.Response(200, content=b"not json", request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", _fake_get)
        result = runner.invoke(cli, ["smoke", "http://eval-box:9000/dashboard.json"])
        assert result.exit_code != 0
        assert "did not return valid JSON" in result.output


class TestSmokeMissingSource:
    def test_no_source_and_no_env_var_is_a_clean_usage_error(self, runner, monkeypatch):
        monkeypatch.delenv("LEOMA_DASHBOARD_URL", raising=False)
        result = runner.invoke(cli, ["smoke"])
        assert result.exit_code != 0
        assert "dashboard.json URL or path" in result.output
