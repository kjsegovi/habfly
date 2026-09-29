"""Only parse local start options; serve is injected and never launches browsers."""

import json

import pytest
from test_runtime_browser_project import options
from typer.testing import CliRunner

import habfly.runtime as runtime_module
from habfly.cli import app


@pytest.fixture
def runner(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime_module, "serve", lambda **kwargs: calls.append(kwargs))
    return CliRunner(), calls


def test_bare_jsonl_is_unchanged(runner):
    cli, calls = runner
    result = cli.invoke(app, ["runtime", "--jsonl"])
    assert result.exit_code == 0, result.output
    assert calls == [{}]


def test_project_start_options_are_validated_and_forwarded(runner, tmp_path):
    cli, calls = runner
    path = tmp_path / "start.json"
    path.write_text(json.dumps(options(tmp_path)))
    result = cli.invoke(app, ["runtime", "--jsonl", "--start-options", str(path)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1 and calls[0]["start_options"]["task"] == "browser_project"
    assert calls[0]["start_options"]["paused"] is True
    assert "password" not in calls[0]["start_options"]


@pytest.mark.parametrize(
    "raw",
    [
        "{secret",
        "[]",
        "null",
        '"credential-secret"',
        "{" + "x" * 65536,
        '{"task":"browser_project","password":"credential-secret"}',
        '{"task":"browser_project","browser_config":"http://localhost?private-session"}',
    ],
)
def test_invalid_file_never_reaches_runtime_or_echoes_contents(runner, tmp_path, raw):
    cli, calls = runner
    path = tmp_path / "start.json"
    path.write_text(raw)
    result = cli.invoke(app, ["runtime", "--jsonl", "--start-options", str(path)])
    assert result.exit_code != 0 and not calls
    assert "credential-secret" not in result.output and "private-session" not in result.output


def test_missing_file_and_required_jsonl_are_clear(runner, tmp_path):
    cli, calls = runner
    path = tmp_path / "missing.json"
    result = cli.invoke(app, ["runtime", "--jsonl", "--start-options", str(path)])
    assert result.exit_code != 0 and "Invalid start-options" in result.output
    result = cli.invoke(app, ["runtime", "--start-options", str(path)])
    assert result.exit_code != 0 and "--jsonl" in result.output and not calls
