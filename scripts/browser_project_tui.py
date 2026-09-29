"""Paused, explicit autonomous-reference project launcher; never a training command."""

import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

from habfly.autonomous_validation import validate_autonomous_decision_sources
from habfly.browser_policy import browser_config
from habfly.browser_probe import public_url
from habfly.runtime import parse_run_options

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = Path("configs/browser_project_autonomous_three_star.json")
PRIVATE_ENV = ("HABFLY_PREVIEW_URL", "HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD")
DEBUG_ENV = ("DEBUG", "PWDEBUG", "DEBUG_FILE")


def _clean_environment():
    return {key: value for key, value in os.environ.items() if key not in (*PRIVATE_ENV, *DEBUG_ENV)}


def _build(environment):
    """Build without credentials even when the parent shell already has them."""
    subprocess.run(
        [
            "cargo",
            "build",
            "--manifest-path",
            str(ROOT / "tui/Cargo.toml"),
            "--locked",
            "--offline",
            "--target-dir",
            str(ROOT / "tui/target"),
        ],
        env=environment,
        check=True,
    )
    return ROOT / "tui/target/debug/habfly-tui"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, help=f"RunOptions JSON (default {DEFAULT_PROFILE})")
    parser.add_argument(
        "--mascot",
        action="store_true",
        help="Show presentation-only Sporky cursor/status bubbles after login; excluded from evidence",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Offline configuration and source checks; no prompts, processes or browser",
    )
    mode.add_argument("--replay", type=Path, help="Offline saved-JSONL playback; no profile or credentials")
    parser.add_argument(
        "--allow-thirty-star",
        action="store_true",
        help="Explicitly authorize launching a selected 30-star profile",
    )
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.replay is not None:
        if args.profile is not None or args.allow_thirty_star or args.mascot:
            parser.error("Replay cannot be combined with profile or launch authorization")
        if not args.replay.is_file():
            parser.error("Replay file is unavailable")
        payload = options = None
    else:
        try:
            profile = args.profile or DEFAULT_PROFILE
            if profile.stat().st_size > 1_000_000:
                raise ValueError("oversized")
            payload = json.loads(profile.read_bytes())
            if args.mascot and isinstance(payload, dict):
                payload = {**payload, "project_mascot": True}
            if not (
                isinstance(payload, dict)
                and payload.get("task") == "browser_project"
                and payload.get("project_autonomous_decisions") is True
                and payload.get("paused") is True
            ):
                raise ValueError("wrong launcher profile")
            options = parse_run_options(payload)
            config = browser_config(options)
        except (OSError, ValueError, TypeError):
            parser.error(
                "Invalid project profile or unavailable inputs; requires explicit autonomous decisions and paused start"
            )
        try:
            sources = validate_autonomous_decision_sources(options)
        except (OSError, ValueError, TypeError, RuntimeError):
            parser.error(
                "Local reference/checkpoint source validation failed; no build, credentials or browser"
            )
        if args.check:
            print(
                json.dumps(
                    {
                        "status": "local_sources_validated",
                        "validation_scope": "local_reference_and_checkpoint_sources",
                        "sources_verified": sources["sources_verified"],
                        "source_validation_mode": sources["mode"],
                        "source_files_verified": len(sources["source_files"]),
                        **(
                            {
                                "supplied_stellar_inputs_enabled": True,
                                "recorded_transfer_cases_opened": True,
                                "dataset_cases_opened": sources["dataset_cases_opened"],
                                "final_evaluation_rerun": sources["final_evaluation_rerun"],
                            }
                            if sources.get("recorded_transfer_cases_opened") is True
                            else {}
                        ),
                        "model_loaded": sources["model_loaded"],
                        "inference_executed": sources["inference_executed"],
                        "training_executed": sources["training_executed"],
                        "learned_readiness_verified": sources["learned_readiness_verified"],
                        "complete_promotion_chains_verified": sources["complete_promotion_chains_verified"],
                        "browser_readiness_verified": sources["browser_readiness_verified"],
                        "launch_authorized": sources["launch_authorized"],
                        "thirty_star_launch_authorized": sources["thirty_star_launch_authorized"],
                        "limitations": sources["limitations"],
                        "browser_launched": False,
                        "control_capture": "fresh_pinned_handles"
                        if config.pinned_control_capture
                        else "fresh_locators",
                        "credentials_requested": False,
                        **(
                            {
                                "save_strategy": "autosave",
                                "save_interpretation": "autosave assumed; fresh visible answers and workflow/inventory readback still required; persistence unverified",
                                "persistence_verified": False,
                            }
                            if options.project_save_strategy == "autosave"
                            else {}
                        ),
                        "task": options.task,
                        "stars": options.stars,
                        "paused": options.paused,
                        "mascot_enabled": options.project_mascot,
                        **(
                            {"two_event_reference_enabled": True}
                            if options.project_two_event_reference
                            else {}
                        ),
                        **(
                            {
                                "baseline_band_reference_enabled": True,
                                "baseline_band_interpretation": "completed 5000-day baseline-band compatibility; assumed No, not proven absence or learned perception",
                            }
                            if options.project_baseline_band_reference
                            else {}
                        ),
                        "autonomous_decisions_enabled": True,
                        **(
                            {
                                "single_event_reference_enabled": True,
                                "single_event_interpretation": "completed 5000-day single-event shortcut; assumed No despite one possible dip, not proven absence or learned perception",
                            }
                            if options.project_single_event_reference
                            else {}
                        ),
                        "decision_authority": "automatic_reference_not_learned",
                        "campaign_max_seconds": options.project_campaign_max_seconds,
                        **(
                            {"campaign_timer": "no_overall_timer"}
                            if options.project_campaign_max_seconds == "uncapped"
                            else {}
                        ),
                        "per_star_max_seconds": options.project_max_seconds,
                        "scoring_enabled": options.project_allow_scoring,
                        "submission_enabled": options.project_allow_submission,
                        "submission_acknowledgement_grounded": False,
                        "finish_checkpoint": "thirty_verified_workflows_assessments_and_visible_update_score"
                        if options.stars == 30
                        and options.project_allow_scoring
                        and not options.project_allow_submission
                        else "formal_submission_separate_acknowledgement_required"
                        if options.project_allow_submission
                        else "bounded_workflow_handoff",
                        "score_checkpoint_completed": False,
                        "reported_score": None,
                        "thirty_star_launch_requires_explicit_flag": options.stars == 30,
                    }
                )
            )
            return 0
        if options.stars == 30 and not args.allow_thirty_star:
            parser.error("30-star launch requires --allow-thirty-star; --check does not launch")
        if args.allow_thirty_star and options.stars != 30:
            parser.error("--allow-thirty-star requires a selected 30-star profile")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Use an interactive terminal, or --check for offline source validation")
    environment = _clean_environment()
    try:
        binary = _build(environment)
    except (OSError, subprocess.CalledProcessError):
        parser.error("Offline TUI build failed; no credentials requested or browser opened")
    command = [str(binary), "--python", sys.executable]
    if args.replay is not None:
        command += ["--replay", str(args.replay)]
    else:
        try:
            url = (
                os.environ.get("HABFLY_PREVIEW_URL")
                or getpass.getpass("Existing preview URL (hidden; plain URL): ").strip()
            )
            candidate = type(config).model_validate({**config.model_dump(), "url": url})
            if not url or public_url(candidate.url) != public_url(config.url):
                raise ValueError("outside activity")
        except ValueError:
            parser.error("Preview URL must match the configured activity origin/path; nothing opened")
        environment["HABFLY_PREVIEW_URL"] = url
        environment["HABFLY_LOGIN_EMAIL"] = (
            os.environ.get("HABFLY_LOGIN_EMAIL") or getpass.getpass("Local author email (hidden): ").strip()
        )
        environment["HABFLY_LOGIN_PASSWORD"] = os.environ.get("HABFLY_LOGIN_PASSWORD") or getpass.getpass(
            "Local author password (hidden): "
        )
        if not environment["HABFLY_LOGIN_EMAIL"] or not environment["HABFLY_LOGIN_PASSWORD"]:
            parser.error("Both login credentials are required; nothing opened or saved")
        command += ["--start-payload", json.dumps(payload), "--autostart"]
        print(f"Paused {options.stars}-star project: r runs, n advances one bounded stage; p/space pauses.")
        print("a aborts; t saves trace; q closes Chromium. Pause/abort apply between bounded calls.")
        if options.project_campaign_max_seconds == "uncapped":
            print(
                "No overall campaign timer. The 30-star target, per-star/action limits, scoring guards and manual abort remain; this is not permission to retry or loop indefinitely."
            )
        print("Automatic reference decisions are NOT learned classification. No b/y approvals or training.")
        if options.project_save_strategy == "autosave":
            print(
                "Autosave assumed; fresh visible answers and workflow/inventory readback still required; persistence unverified. No explicit Save click or timer-as-proof."
            )
        if options.project_baseline_band_reference:
            print(
                "Baseline-band reference: completed 5000-day compatibility may select assumed No; not proven absence or learned perception."
            )
        if options.project_single_event_reference:
            print(
                "Single-event reference: one possible dip in a completed 5000-day window may select assumed No; not proven absence or learned perception."
            )
        if options.project_mascot:
            print("Sporky shows runtime status, not hidden thoughts; evidence captures exclude the mascot.")
        print(
            "Saving follows the profile; assessment/submission are separate flags. Unknown submission stays unknown."
        )
        if options.stars == 30 and options.project_allow_scoring and not options.project_allow_submission:
            print(
                "Finish checkpoint: 30 verified workflows, assessments, Update Score and its visible score. Formal Submit is disabled; a perfect score is not required."
            )
    try:
        return subprocess.call(command, env=environment)
    except OSError:
        parser.error("Could not start the TUI; no automatic retry")
    finally:
        for key in PRIVATE_ENV:
            environment.pop(key, None)


if __name__ == "__main__":
    raise SystemExit(main())
