"""The combined launcher never silently enables autonomous writes."""

import importlib.util
import json
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("four_field_launcher", "scripts/browser_four_field_tui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("Unexpected input prompt"))
    monkeypatch.setattr(module.getpass, "getpass", lambda *_: pytest.fail("Unexpected password prompt"))
    monkeypatch.setattr(module.subprocess, "call", lambda *_a, **_k: pytest.fail("Unexpected TUI launch"))
    return module


@pytest.mark.parametrize(
    "flags",
    [
        ["--autonomous"],
        ["--runs", "10"],
        ["--autonomous", "--runs", "11"],
        ["--autonomous", "--runs", "0"],
        ["--autonomous", "--runs", "2", "--manual-setup"],
        ["--output", "unused"],
    ],
)
def test_requires_bounded_explicit_batch_flags(launcher, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["browser_four_field_tui.py", *flags])
    monkeypatch.setattr(launcher, "load_four_field_policy", lambda *_: pytest.fail("Loaded model"))
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_check_no_credentials_or_browser(launcher, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["browser_four_field_tui.py", "--check"])
    monkeypatch.setattr(launcher, "load_four_field_policy", lambda *_: (None, {"optimizer_updates": 0}))
    monkeypatch.setattr(launcher, "browser_config", lambda *_: None)
    launcher.main()
    report = json.loads(capsys.readouterr().out)
    assert report["paused"] and report["max_native_writes"] == report["human_approvals_required"] == 4
    assert report["optimizer_updates"] == 0


def test_unsafe_profile_rejected_before_prompt(launcher, tmp_path, monkeypatch):
    path = tmp_path / "unsafe.json"
    payload = json.loads(launcher.Path("configs/browser_four_field_tui.json").read_text())
    payload["browser_execution"] = "autonomous"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", ["browser_four_field_tui.py", "--profile", str(path)])
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_autonomous_check_is_offline_and_records_budget(launcher, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["browser_four_field_tui.py", "--autonomous", "--runs", "10", "--check"])
    monkeypatch.setattr(launcher, "load_four_field_policy", lambda *_: (None, {}))
    monkeypatch.setattr(launcher, "browser_config", lambda *_: None)
    launcher.main()
    report = json.loads(capsys.readouterr().out)
    assert not report["paused"] and report["requested_runs"] == 10
    assert report["human_approvals_required"] == 0 and report["max_native_writes"] == 4


@pytest.mark.parametrize("passed", [True, False])
def test_explicit_batch_dispatch_and_exit_status(launcher, tmp_path, monkeypatch, passed):
    from habfly import browser_four_field_reliability

    output = tmp_path / "batch"
    monkeypatch.setattr(
        sys, "argv", ["browser_four_field_tui.py", "--autonomous", "--runs", "10", "--output", str(output)]
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("HABFLY_PREVIEW_URL", "http://localhost/fixture")
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", "fixture-secret")
    monkeypatch.setattr(launcher, "load_four_field_policy", lambda *_: (None, {}))
    monkeypatch.setattr(launcher, "browser_config", lambda *_: None)

    def run(options, destination, *, runs, credentials):
        assert options.task == "browser_four_field" and options.browser_execution == "autonomous"
        assert options.browser_setup == "automatic" and not options.paused and options.stars == 1
        assert destination == output and runs == 10
        assert credentials == ("fixture@example.invalid", "fixture-secret")
        return {
            "passed_runs": 10 if passed else 0,
            "requested_runs": 10,
            "four_field_transport_verified_runs": 10 if passed else 0,
            "unique_stars": 1,
            "all_requested_runs_passed": passed,
        }

    monkeypatch.setattr(browser_four_field_reliability, "run_four_field_reliability", run)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == (0 if passed else 1)


@pytest.mark.parametrize("manual", [False, True])
def test_interactive_dispatch_sanitizes_credentials_and_debug(launcher, monkeypatch, capsys, manual):
    monkeypatch.setattr(sys, "argv", ["browser_four_field_tui.py", *(["--manual-setup"] if manual else [])])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("HABFLY_PREVIEW_URL", "http://localhost/fixture")
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", "fixture-secret")
    monkeypatch.setenv("DEBUG", "pw:api")
    monkeypatch.setenv("PWDEBUG", "1")
    monkeypatch.setenv("DEBUG_FILE", "unsafe.log")
    monkeypatch.setattr(launcher, "load_four_field_policy", lambda *_: (None, {}))
    monkeypatch.setattr(launcher, "browser_config", lambda *_: None)

    def dispatch(argv, *, env):
        assert "--offline" in argv
        payload = json.loads(argv[argv.index("--start-payload") + 1])
        assert payload["task"] == "browser_four_field" and payload["paused"]
        assert payload["browser_execution"] == "supervised"
        assert payload["browser_setup"] == ("manual" if manual else "automatic")
        assert all(key not in env for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"))
        assert ("HABFLY_LOGIN_PASSWORD" in env) is not manual
        assert "fixture-secret" not in str(argv) and "fixture@example.invalid" not in str(argv)
        return 0

    monkeypatch.setattr(launcher.subprocess, "call", dispatch)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "fixture-secret" not in output and "fixture@example.invalid" not in output
    assert "No Save" in output and "not course completion" in output
