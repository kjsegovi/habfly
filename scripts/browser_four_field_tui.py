"""Supervised four-field transfer or an explicitly enabled 1–10-run reliability batch."""

import argparse
import getpass
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from habfly.browser_four_field_policy import load_four_field_policy
from habfly.browser_policy import browser_config
from habfly.runtime import RunOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/browser_four_field_tui.json"))
    parser.add_argument("--check", action="store_true", help="Offline gates only; no browser or credentials")
    parser.add_argument(
        "--manual-setup", action="store_true", help="Manually reach Stellar in fresh Chromium"
    )
    parser.add_argument(
        "--autonomous", action="store_true", help="Enable four automatic writes per run; requires --runs"
    )
    parser.add_argument("--runs", type=int, help="1–10 sequential runs; requires --autonomous")
    parser.add_argument("--output", type=Path, help="New batch artifact directory; requires --runs")
    args = parser.parse_args()
    if args.autonomous != (args.runs is not None):
        parser.error("Use --autonomous and --runs together for a bounded reliability batch")
    if args.runs is not None and (not 1 <= args.runs <= 10 or args.manual_setup):
        parser.error("Choose --runs 1 through 10 with automatic setup")
    if args.output is not None and args.runs is None:
        parser.error("--output requires --runs")
    os.chdir(Path(__file__).resolve().parents[1])
    payload = json.loads(args.profile.read_text())
    if args.manual_setup:
        payload["browser_setup"] = "manual"
    if args.autonomous:
        payload.update(browser_execution="autonomous", browser_setup="automatic", paused=False, stars=1)
    elif payload.get("browser_execution", "supervised") != "supervised":
        parser.error("Automatic four-field entry requires explicit --autonomous --runs flags")
    options = RunOptions.model_validate(payload)
    _, provenance = load_four_field_policy(options)
    if args.check:
        browser_config(options)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "paused": options.paused,
                    "requested_runs": args.runs,
                    "max_native_writes": 4,
                    "human_approvals_required": 0 if args.autonomous else 4,
                    **provenance,
                }
            )
        )
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Use an interactive terminal or --check")
    output = args.output or Path("experiments/browser-four-field-reliability") / uuid.uuid4().hex
    if args.runs is not None and output.exists():
        parser.error("Batch output already exists; choose a new directory")
    if not os.environ.get("HABFLY_PREVIEW_URL"):
        os.environ["HABFLY_PREVIEW_URL"] = input("Existing preview URL (plain URL, not Markdown): ").strip()
        if not os.environ["HABFLY_PREVIEW_URL"]:
            parser.error("A preview URL is required")
    try:
        browser_config(options)
    except ValueError:
        parser.error("Preview URL must match the configured activity origin/path; nothing was opened")
    child_env = os.environ.copy()
    for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
        child_env.pop(key, None)
    if options.browser_setup == "automatic":
        child_env["HABFLY_LOGIN_EMAIL"] = (
            child_env.get("HABFLY_LOGIN_EMAIL") or input("Local author email: ").strip()
        )
        child_env["HABFLY_LOGIN_PASSWORD"] = child_env.get("HABFLY_LOGIN_PASSWORD") or getpass.getpass(
            "Local author password (hidden): "
        )
        if not child_env["HABFLY_LOGIN_EMAIL"] or not child_env["HABFLY_LOGIN_PASSWORD"]:
            parser.error("Both login credentials are required; nothing was saved")
    else:
        for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD"):
            child_env.pop(key, None)
    if args.runs is not None:
        from habfly.browser_four_field_reliability import run_four_field_reliability

        credentials = tuple(child_env.pop(key) for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD"))
        print(
            "AUTONOMOUS batch: fresh Chromium per run, three numeric copies + one color selection; no n/y presses."
        )
        print("No Save, scoring, submission or training. Normal site autosave may occur.")
        print("Stops on the first failure; no retries. Ctrl-C ends the batch. Repeated stars are reported.")
        try:
            report = run_four_field_reliability(options, output, runs=args.runs, credentials=credentials)
        finally:
            credentials = None
        print(
            f"Passed {report['passed_runs']}/{report['requested_runs']}; "
            f"{report['four_field_transport_verified_runs']} four-field readbacks; "
            f"{report['unique_stars']} distinct stars. Report: {output / 'report.json'}"
        )
        raise SystemExit(0 if report["all_requested_runs_passed"] else 1)
    print("Fresh Chromium; Firefox untouched. Automatic setup picks a star, then pauses.")
    print("n advances one learned decision; y approves each pending numeric copy (three total).")
    print("At the handoff, n rechecks this same star and switches to the frozen color policy.")
    print("Then n selects the source, n proposes color, y approves color, n verifies readback.")
    print("Manual setup only: b captures first. a aborts; q closes Chromium after inspection.")
    print("No Save, assessment, score updates, submission or training. Normal site autosave may occur.")
    print("Four verified readbacks are not course completion; browser confidence remains uncalibrated.")
    raise SystemExit(
        subprocess.call(
            [
                "cargo",
                "run",
                "--manifest-path",
                "tui/Cargo.toml",
                "--locked",
                "--offline",
                "--",
                "--python",
                sys.executable,
                "--start-payload",
                json.dumps(payload),
                "--autostart",
            ],
            env=child_env,
        )
    )


if __name__ == "__main__":
    main()
