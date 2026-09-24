"""CLI intent/credential handling without launching browsers, Cargo, or training."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("browser_launcher_under_test", "scripts/browser_tui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "load_browser_policy",
        lambda options: (
            None,
            {
                "browser_execution": options.browser_execution,
                "optimizer_updates": 0,
            },
        ),
    )
    monkeypatch.setattr(module, "browser_config", lambda _: None)
    monkeypatch.setattr(module.os, "chdir", lambda _: None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("HABFLY_PREVIEW_URL", "http://localhost/activity")
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", "launcher@example.invalid")
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", "launcher-test-secret")

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected external process or interactive credential prompt")

    monkeypatch.setattr(module.subprocess, "call", forbidden)
    monkeypatch.setattr(module.getpass, "getpass", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    return module


def invoke(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["browser_tui.py", *args])
    return module.main()


def test_autonomous_check_is_offline_without_credentials_or_artifacts(
    launcher, monkeypatch, tmp_path, capsys
):
    monkeypatch.delenv("HABFLY_LOGIN_EMAIL")
    monkeypatch.delenv("HABFLY_LOGIN_PASSWORD")
    output = tmp_path / "batch"
    invoke(launcher, monkeypatch, "--autonomous", "--runs", "3", "--output", str(output), "--check")
    report = json.loads(capsys.readouterr().out)
    assert report["browser_execution"] == "autonomous" and not report["paused"]
    assert report["requested_runs"] == 3 and report["optimizer_updates"] == 0
    assert not output.exists()


@pytest.mark.parametrize(
    "flags",
    [["--runs", "3"], ["--autonomous", "--runs", "0"], ["--autonomous", "--runs", "11"], ["--output", "no"]],
)
def test_launcher_requires_explicit_bounded_batch_intent(launcher, monkeypatch, flags):
    with pytest.raises(SystemExit) as exc:
        invoke(launcher, monkeypatch, *flags)
    assert exc.value.code == 2


@pytest.mark.parametrize("autonomous", [False, True])
def test_tui_launch_modes_keep_credentials_out_of_json_and_arguments(launcher, monkeypatch, autonomous):
    calls = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda command, env: calls.append((command, env)) or 0)
    monkeypatch.setenv("DEBUG", "pw:api")
    with pytest.raises(SystemExit) as exc:
        invoke(launcher, monkeypatch, "--autonomous" if autonomous else "--auto-setup")
    assert exc.value.code == 0
    command, env = calls[0]
    payload = json.loads(command[command.index("--start-payload") + 1])
    assert payload.get("browser_execution", "supervised") == ("autonomous" if autonomous else "supervised")
    assert payload["paused"] is not autonomous and payload["browser_setup"] == "automatic"
    assert payload["stars"] == 1 if autonomous else "stars" not in payload
    assert "launcher-test-secret" not in " ".join(command) and "launcher@example.invalid" not in json.dumps(
        payload
    )
    assert env["HABFLY_LOGIN_PASSWORD"] == "launcher-test-secret" and "DEBUG" not in env


def test_profile_cannot_silently_enable_autonomous_writes(launcher, monkeypatch, tmp_path):
    payload = json.loads(Path("configs/browser_numeric_tui.json").read_text())
    payload["browser_execution"] = "autonomous"
    profile = tmp_path / "unsafe.json"
    profile.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as exc:
        invoke(launcher, monkeypatch, "--profile", str(profile), "--check")
    assert exc.value.code == 2


def test_batch_launch_has_no_tui_and_reports_real_requested_budget(launcher, monkeypatch, tmp_path, capsys):
    captured = []

    def run(options, output, *, runs, credentials):
        captured.append((options, output, runs, credentials))
        return {"passed_runs": 3, "requested_runs": 3, "unique_stars": 2, "all_requested_runs_passed": True}

    monkeypatch.setattr("habfly.browser_reliability.run_reliability", run)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    output = tmp_path / "batch"
    with pytest.raises(SystemExit) as exc:
        invoke(launcher, monkeypatch, "--autonomous", "--runs", "3", "--output", str(output))
    assert exc.value.code == 0
    options, destination, count, credentials = captured[0]
    assert options.browser_execution == "autonomous" and not options.paused
    assert count == 3 and destination == output
    assert credentials == ("launcher@example.invalid", "launcher-test-secret")
    text = capsys.readouterr().out
    assert "3/3" in text and "2 distinct stars" in text
    assert "launcher-test-secret" not in text and "launcher@example.invalid" not in text
