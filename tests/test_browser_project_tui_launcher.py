"""Launcher-only checks: no Cargo, browser, network, inference or training."""

import importlib.util
import json
import sys

import pytest
from test_runtime_browser_project import options as fixture_options

from habfly.autonomous_validation import validate_autonomous_decision_sources
from habfly.runtime import parse_run_options


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("project_tui_launcher", "scripts/browser_project_tui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = fixture_options(tmp_path, stars=3, project_campaign=True, project_campaign_max_seconds=5400)
    payload["project_autonomous_decisions"] = True
    profile = tmp_path / "autonomous.json"
    profile.write_text(json.dumps(payload))
    module.test_profile, module.test_payload = profile, payload
    monkeypatch.setattr(module, "DEFAULT_PROFILE", profile)

    def parse(value):
        # This legacy options fixture lacks the planet/habitability source tree.
        # Real autonomous profiles and source validation are exercised separately.
        assert value["project_autonomous_decisions"] is True
        return parse_run_options({k: v for k, v in value.items() if k != "project_autonomous_decisions"})

    monkeypatch.setattr(module, "parse_run_options", parse)
    module.test_source_checks = []

    def validate(options):
        module.test_source_checks.append(options)
        return {
            "mode": "offline_autonomous_source_validation_v1",
            "sources_verified": True,
            "source_files": {"fixture": {"path": "not-a-real-model"}},
            "model_loaded": False,
            "inference_executed": False,
            "training_executed": False,
            "learned_readiness_verified": False,
            "complete_promotion_chains_verified": False,
            "browser_readiness_verified": False,
            "launch_authorized": False,
            "thirty_star_launch_authorized": False,
            "limitations": ["Injected source-check seam; no weights loaded."],
        }

    monkeypatch.setattr(module, "validate_autonomous_decision_sources", validate)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    for key in (*module.PRIVATE_ENV, *module.DEBUG_ENV):
        monkeypatch.delenv(key, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected prompt, process or network")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module.subprocess, "call", forbidden)
    monkeypatch.setattr(module.getpass, "getpass", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("habfly.runtime.Runtime.start", forbidden)
    return module


def test_check_validates_sources_without_prompts_processes_or_artifacts(launcher, capsys):
    before = launcher.test_profile.read_bytes()
    assert launcher.main(["--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["status"] == "local_sources_validated"
        and result["validation_scope"] == "local_reference_and_checkpoint_sources"
    )
    assert result["stars"] == 3 and result["paused"] is True
    assert result["sources_verified"] is True and result["source_files_verified"] == 1
    assert len(launcher.test_source_checks) == 1
    for key in (
        "model_loaded",
        "inference_executed",
        "training_executed",
        "learned_readiness_verified",
        "complete_promotion_chains_verified",
        "browser_readiness_verified",
        "launch_authorized",
        "thirty_star_launch_authorized",
    ):
        assert result[key] is False
    assert result["limitations"] == ["Injected source-check seam; no weights loaded."]
    assert result["submission_acknowledgement_grounded"] is False
    assert result["score_checkpoint_completed"] is False and result["reported_score"] is None
    assert result["browser_launched"] is result["credentials_requested"] is False
    assert result["control_capture"] == "fresh_locators"
    assert launcher.test_profile.read_bytes() == before
    assert not launcher.Path(launcher.test_payload["artifact_dir"]).exists()


def test_mascot_check_is_explicit_without_editing_profile_or_launching(launcher, capsys):
    before = launcher.test_profile.read_bytes()
    assert launcher.main(["--check", "--mascot"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mascot_enabled"] is True
    assert result["browser_launched"] is False
    assert launcher.test_profile.read_bytes() == before
    assert not launcher.Path(launcher.test_payload["artifact_dir"]).exists()


def test_custom_submission_authorization_is_not_described_as_score_only(launcher, capsys):
    launcher.test_profile.write_text(
        json.dumps(
            {
                **launcher.test_payload,
                "stars": 30,
                "project_campaign_max_seconds": 10800,
                "project_allow_scoring": True,
                "project_allow_submission": True,
            }
        )
    )
    assert launcher.main(["--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["finish_checkpoint"] == "formal_submission_separate_acknowledgement_required"
    assert result["submission_enabled"] is True
    assert result["score_checkpoint_completed"] is False and result["reported_score"] is None


def test_replay_cannot_enable_live_mascot(launcher, tmp_path):
    trace = tmp_path / "replay.jsonl"
    trace.write_text("")
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--replay", str(trace), "--mascot"])
    assert exc.value.code == 2


def test_check_discloses_recorded_private_proof_reads_without_claiming_new_run(launcher, monkeypatch, capsys):
    validate = launcher.validate_autonomous_decision_sources
    monkeypatch.setattr(
        launcher,
        "validate_autonomous_decision_sources",
        lambda options: {
            **validate(options),
            "recorded_transfer_cases_opened": True,
            "dataset_cases_opened": True,
            "final_evaluation_rerun": False,
        },
    )
    assert launcher.main(["--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recorded_transfer_cases_opened"] is result["dataset_cases_opened"] is True
    assert result["supplied_stellar_inputs_enabled"] is True
    assert result["final_evaluation_rerun"] is result["browser_launched"] is False
    assert not launcher.Path(launcher.test_payload["artifact_dir"]).exists()


@pytest.mark.parametrize(
    "profile", ["browser_project_autonomous_three_star.json", "browser_project_thirty_star.json"]
)
def test_actual_runtime_autonomous_option_checks_existing_inputs_only(
    launcher, monkeypatch, tmp_path, capsys, profile
):
    original = launcher.ROOT / "configs" / profile
    before = original.read_bytes()
    payload = json.loads(before)
    payload["project_autonomous_decisions"] = True
    payload["artifact_dir"] = str(tmp_path / "never-created")
    local = tmp_path / "real-options.json"
    local.write_text(json.dumps(payload))
    monkeypatch.setattr(launcher, "parse_run_options", parse_run_options)
    monkeypatch.setattr(
        launcher, "validate_autonomous_decision_sources", validate_autonomous_decision_sources
    )
    monkeypatch.setattr("torch.load", lambda *_a, **_k: pytest.fail("Deserialized model weights"))
    assert launcher.main(["--profile", str(local), "--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stars"] == payload["stars"] and result["autonomous_decisions_enabled"] is True
    assert result["sources_verified"] is True and result["source_files_verified"] > 0
    assert result["model_loaded"] is result["browser_readiness_verified"] is False
    assert result["launch_authorized"] is result["thirty_star_launch_authorized"] is False
    if payload["stars"] == 30:
        assert result["supplied_stellar_inputs_enabled"] is True
        assert result["recorded_transfer_cases_opened"] is True
        assert result["final_evaluation_rerun"] is False
        assert result["submission_acknowledgement_grounded"] is False
        assert result["scoring_enabled"] is True and result["submission_enabled"] is False
        assert result["finish_checkpoint"] == "thirty_verified_workflows_assessments_and_visible_update_score"
        assert result["thirty_star_launch_requires_explicit_flag"] is True
        # Use the real promoted profile/parser/source checks, but never permit
        # a build, credential prompt, process, model or browser after --check.
        with pytest.raises(SystemExit) as exc:
            launcher.main(["--profile", str(local)])
        assert exc.value.code == 2
        assert "30-star launch requires --allow-thirty-star" in capsys.readouterr().err
    assert not (tmp_path / "never-created").exists() and original.read_bytes() == before


def test_prepared_held_thirty_scope_matches_latest_three_without_expanding_budgets(launcher):
    configs = launcher.ROOT / "configs"
    three = json.loads((configs / "browser_project_single_event_three_star.json").read_bytes())
    thirty = json.loads((configs / "browser_project_thirty_star.json").read_bytes())
    expected_differences = {
        "stars": 30,
        "project_allow_scoring": True,
        "project_allow_submission": False,
        "project_campaign_max_seconds": "uncapped",
        "artifact_dir": "experiments/browser-project-thirty-star",
    }
    assert thirty == {**three, **expected_differences}
    assert thirty["paused"] is True
    assert thirty["project_max_advances"] == 512 and thirty["project_max_seconds"] == 1800
    assert thirty["project_supplied_stellar_inputs"] is True
    assert thirty["project_baseline_edge_reference"] is True
    assert thirty["project_single_event_reference"] is True
    assert thirty["browser_planet_supplied_evaluation"] == "experiments/planet-supplied-input-transfer-001"
    assert thirty["browser_habitability_supplied_evaluation"] == (
        "experiments/habitability-supplied-input-transfer-001"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"project_autonomous_decisions": False},
        {"project_autonomous_decisions": 1},
        {"paused": False},
        {"task": "stellar"},
        {"class_source": {"selected_class": "main_sequence"}},
    ],
)
def test_profile_cannot_infer_optin_unpause_or_static_decision(launcher, change):
    launcher.test_profile.write_text(json.dumps({**launcher.test_payload, **change}))
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--check"])
    assert exc.value.code == 2


def test_missing_runtime_support_fails_before_prompt(launcher, monkeypatch, capsys):
    def unsupported(_):
        raise ValueError("private URL or credential")

    monkeypatch.setattr(launcher, "parse_run_options", unsupported)
    with pytest.raises(SystemExit):
        launcher.main(["--check"])
    assert "private URL" not in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--check"], []])
@pytest.mark.parametrize("error", [ValueError, OSError, RuntimeError, TypeError])
def test_source_failure_stops_before_build_prompts_or_dispatch(
    launcher, monkeypatch, capsys, arguments, error
):
    def invalid(_):
        raise error("private URL /home/private/credential.json")

    monkeypatch.setattr(launcher, "validate_autonomous_decision_sources", invalid)
    with pytest.raises(SystemExit) as exc:
        launcher.main(arguments)
    assert exc.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "source validation failed" in output.err
    assert "private URL" not in output.err and "credential.json" not in output.err
    assert not launcher.Path(launcher.test_payload["artifact_dir"]).exists()


def test_thirty_check_does_not_authorize_launch(launcher, capsys):
    launcher.test_payload.update(
        stars=30,
        project_campaign_max_seconds=10800,
        project_allow_scoring=True,
        project_allow_submission=True,
    )
    launcher.test_profile.write_text(json.dumps(launcher.test_payload))
    assert launcher.main(["--check"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stars"] == 30 and result["thirty_star_launch_requires_explicit_flag"] is True
    with pytest.raises(SystemExit) as exc:
        launcher.main([])
    assert exc.value.code == 2


def test_thirty_flag_cannot_expand_three_star_profile(launcher):
    with pytest.raises(SystemExit):
        launcher.main(["--allow-thirty-star"])
    assert json.loads(launcher.test_profile.read_bytes())["stars"] == 3


@pytest.mark.parametrize("target", [3, 30])
def test_native_launch_is_only_injected_and_secrets_not_in_cargo_or_arguments(
    launcher, monkeypatch, capsys, target
):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    launcher.test_payload.update(stars=target, project_campaign_max_seconds=5400 if target == 3 else 10800)
    launcher.test_profile.write_text(json.dumps(launcher.test_payload))
    values = {
        "HABFLY_PREVIEW_URL": "http://localhost/activity",
        "HABFLY_LOGIN_EMAIL": "fixture@example.invalid",
        "HABFLY_LOGIN_PASSWORD": "test-private-password",
        "DEBUG": "pw:api",
        "PWDEBUG": "1",
        "DEBUG_FILE": "private.log",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    calls = []

    def build(argv, *, env, check):
        assert len(launcher.test_source_checks) == 1
        calls.append("build")
        assert argv[:2] == ["cargo", "build"] and "--offline" in argv and "--locked" in argv and check
        assert all(key not in env for key in values)

    def dispatch(argv, *, env):
        calls.append("dispatch")
        assert calls == ["build", "dispatch"] and argv[0].endswith("/habfly-tui")
        payload = json.loads(argv[argv.index("--start-payload") + 1])
        assert payload == launcher.test_payload and payload["paused"] is True
        assert all(env[key] == values[key] for key in launcher.PRIVATE_ENV)
        assert all(key not in env for key in launcher.DEBUG_ENV)
        assert "test-private-password" not in str(argv) and "fixture@example.invalid" not in str(argv)
        return 7

    monkeypatch.setattr(launcher.subprocess, "run", build)
    monkeypatch.setattr(launcher.subprocess, "call", dispatch)
    assert launcher.main(["--allow-thirty-star"] if target == 30 else []) == 7
    text = capsys.readouterr().out
    assert "test-private-password" not in text and "fixture@example.invalid" not in text
    assert "NOT learned classification" in text
    assert not launcher.Path(launcher.test_payload["artifact_dir"]).exists()


def test_build_precedes_hidden_prompts_and_prompted_url_never_enters_payload(launcher, monkeypatch):
    calls = []

    def build(*_a, **_k):
        assert len(launcher.test_source_checks) == 1
        calls.append("build")

    monkeypatch.setattr(launcher.subprocess, "run", build)
    answers = iter(["http://localhost/activity", "fixture@example.invalid", "hidden-password"])

    def prompt(_):
        assert calls[0] == "build"
        calls.append("prompt")
        return next(answers)

    def dispatch(argv, *, env):
        assert calls == ["build", "prompt", "prompt", "prompt"]
        assert env["HABFLY_LOGIN_PASSWORD"] == "hidden-password"
        assert "hidden-password" not in str(argv) and "http://localhost/activity" not in str(argv)
        return 0

    monkeypatch.setattr(launcher.getpass, "getpass", prompt)
    monkeypatch.setattr(launcher.subprocess, "call", dispatch)
    assert launcher.main([]) == 0


@pytest.mark.parametrize("failure", ["build", "url", "empty_password", "noninteractive"])
def test_prelaunch_failures_never_dispatch_or_leak(launcher, monkeypatch, capsys, failure):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    def build(*args, **kwargs):
        if failure == "build":
            raise OSError("private build detail")

    monkeypatch.setattr(launcher.subprocess, "run", build)
    if failure == "noninteractive":
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    else:
        answers = iter(
            [
                "http://outside.invalid/private" if failure == "url" else "http://localhost/activity",
                "fixture@example.invalid",
                "",
            ]
        )
        monkeypatch.setattr(launcher.getpass, "getpass", lambda *_: next(answers))
    with pytest.raises(SystemExit) as exc:
        launcher.main([])
    assert exc.value.code == 2
    text = capsys.readouterr().err
    assert "private build detail" not in text and "outside.invalid" not in text
    expected = {
        "build": "Offline TUI build failed",
        "url": "Preview URL must match",
        "empty_password": "Both login credentials",
        "noninteractive": "interactive terminal",
    }
    assert expected[failure] in text


def test_replay_needs_neither_profile_nor_credentials(launcher, monkeypatch, tmp_path):
    replay = tmp_path / "saved.jsonl"
    replay.write_text("")
    monkeypatch.setattr(launcher, "parse_run_options", lambda *_: pytest.fail("Loaded launch profile"))
    monkeypatch.setattr(
        launcher, "validate_autonomous_decision_sources", lambda *_: pytest.fail("Validated launch sources")
    )
    monkeypatch.setattr(launcher.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setenv("HABFLY_LOGIN_PASSWORD", "never-pass-to-replay")

    def dispatch(argv, *, env):
        assert argv[-2:] == ["--replay", str(replay)]
        assert "--autostart" not in argv and "--start-payload" not in argv
        assert all(key not in env for key in launcher.PRIVATE_ENV)
        return 0

    monkeypatch.setattr(launcher.subprocess, "call", dispatch)
    assert launcher.main(["--replay", str(replay)]) == 0


@pytest.mark.parametrize("extra", [["--check"], ["--profile", "unused"], ["--allow-thirty-star"]])
def test_replay_rejects_launch_and_check_options(launcher, extra):
    with pytest.raises(SystemExit):
        launcher.main(["--replay", "unused", *extra])
