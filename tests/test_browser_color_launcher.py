"""CLI authorization checks happen before credentials, browser, or output writes."""

import importlib.util
import json
import sys

import pytest


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("browser_color_launcher", "scripts/browser_color_tui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("Unexpected input prompt"))
    monkeypatch.setattr(module.getpass, "getpass", lambda *_: pytest.fail("Unexpected credential prompt"))
    monkeypatch.setattr(module.subprocess, "call", lambda *_a, **_k: pytest.fail("Unexpected TUI launch"))
    return module


@pytest.mark.parametrize(
    "args",
    [
        ["--autonomous"],
        ["--runs", "10"],
        ["--autonomous", "--runs", "0"],
        ["--autonomous", "--runs", "11"],
        ["--autonomous", "--runs", "2", "--manual-setup"],
        ["--output", "unused"],
    ],
)
def test_invalid_flags_fail_before_model_or_credentials(launcher, monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["browser_color_tui.py", *args])
    monkeypatch.setattr(
        launcher, "load_browser_color_policy", lambda _: pytest.fail("Model loaded before flag validation")
    )
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


def test_profile_cannot_silently_opt_in(launcher, tmp_path, monkeypatch):
    path = tmp_path / "autonomous.json"
    path.write_text(json.dumps({"browser_execution": "autonomous"}))
    monkeypatch.setattr(sys, "argv", ["browser_color_tui.py", "--profile", str(path), "--check"])
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("autonomous", [False, True])
def test_check_is_offline_and_does_not_start_run(launcher, monkeypatch, capsys, autonomous):
    args = ["--autonomous", "--runs", "10"] if autonomous else []
    monkeypatch.setattr(sys, "argv", ["browser_color_tui.py", "--check", *args])
    monkeypatch.setattr(
        launcher,
        "load_browser_color_policy",
        lambda options: (None, {"browser_execution": options.browser_execution}),
    )
    monkeypatch.setattr(launcher, "browser_config", lambda _: None)
    launcher.main()
    report = json.loads(capsys.readouterr().out)
    assert report["requested_runs"] == (10 if autonomous else None)
    assert report["paused"] is not autonomous
    assert report["max_native_writes_per_run"] == 1


@pytest.mark.parametrize("passed", [True, False])
def test_explicit_batch_dispatch_never_launches_tui(launcher, tmp_path, monkeypatch, capsys, passed):
    from habfly import browser_color_reliability

    destination = tmp_path / "batch"
    monkeypatch.setattr(
        sys, "argv", ["browser_color_tui.py", "--autonomous", "--runs", "10", "--output", str(destination)]
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("HABFLY_PREVIEW_URL", "http://localhost/fixture")
    monkeypatch.setenv("HABFLY_LOGIN_EMAIL", "fixture@example.invalid")
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", "private-fixture-only")
    monkeypatch.setattr(launcher, "load_browser_color_policy", lambda _: (None, {}))
    monkeypatch.setattr(launcher, "browser_config", lambda _: None)

    def run(options, output, *, runs, credentials):
        assert options.task == "browser_color" and options.browser_execution == "autonomous"
        assert options.browser_setup == "automatic" and not options.paused and options.stars == 1
        assert output == destination and runs == 10
        assert credentials == ("fixture@example.invalid", "private-fixture-only")
        return {
            "passed_runs": 10 if passed else 0,
            "requested_runs": 10,
            "color_transport_verified_runs": 10 if passed else 0,
            "reference_matching_runs": 10 if passed else 0,
            "unique_stars": 1,
            "all_requested_runs_passed": passed,
        }

    monkeypatch.setattr(browser_color_reliability, "run_color_reliability", run)
    with pytest.raises(SystemExit) as exc:
        launcher.main()
    assert exc.value.code == (0 if passed else 1)
    text = capsys.readouterr().out
    assert "fixture@example.invalid" not in text and "private-fixture-only" not in text
    assert "no n/y presses needed" in text
