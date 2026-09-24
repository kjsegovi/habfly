"""Launch a three-field browser diagnostic or a bounded autonomous reliability batch."""

import argparse
import getpass
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from habfly.browser_policy import browser_config, load_browser_policy
from habfly.runtime import RunOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/browser_numeric_tui.json"))
    parser.add_argument("--check", action="store_true", help="Offline validation; no browser or writes")
    parser.add_argument(
        "--auto-setup", action="store_true", help="Script login, introduction, and visible-star selection"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Explicitly enable automatic setup, learned decisions and up to three exact copies",
    )
    parser.add_argument(
        "--runs", type=int, help="Run 1–10 autonomous sessions sequentially (console report, no TUI)"
    )
    parser.add_argument("--output", type=Path, help="New reliability output directory; requires --runs")
    args = parser.parse_args()
    if args.runs is not None and (not args.autonomous or not 1 <= args.runs <= 10):
        parser.error("--runs requires --autonomous and a count from 1 to 10")
    if args.output is not None and args.runs is None:
        parser.error("--output requires --runs")
    os.chdir(Path(__file__).resolve().parents[1])
    payload = json.loads(args.profile.read_text())
    if args.auto_setup:
        payload["browser_setup"] = "automatic"
    if args.autonomous:
        payload.update(browser_setup="automatic", browser_execution="autonomous", paused=False, stars=1)
    elif payload.get("browser_execution", "supervised") != "supervised":
        parser.error("Automatic answer entry requires an explicit --autonomous flag")
    options = RunOptions.model_validate(payload)
    if options.task != "browser_numeric":
        parser.error("This launcher requires a browser_numeric profile")
    _, provenance = load_browser_policy(options)
    try:
        browser_config(options)
    except ValueError:
        parser.error("HABFLY_PREVIEW_URL must be the plain, exact allowed preview URL (not Markdown)")
    if args.check:
        print(
            json.dumps(
                {"status": "ready", "paused": options.paused, "requested_runs": args.runs, **provenance}
            )
        )
        return
    if args.runs is None and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        parser.error("Use an interactive terminal or --check")
    if not os.environ.get("HABFLY_PREVIEW_URL"):
        parser.error("Set HABFLY_PREVIEW_URL to the existing preview's plain URL first")
    child_env = os.environ.copy()
    if options.browser_setup == "automatic":
        for key in ("DEBUG", "PWDEBUG", "DEBUG_FILE"):
            child_env.pop(key, None)
            os.environ.pop(key, None)
        if not sys.stdin.isatty() and not all(
            child_env.get(key) for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD")
        ):
            parser.error(
                "Use an interactive terminal to enter credentials, or supply the secure login environment"
            )
        child_env["HABFLY_LOGIN_EMAIL"] = (
            child_env.get("HABFLY_LOGIN_EMAIL") or input("Local author email: ").strip()
        )
        child_env["HABFLY_LOGIN_PASSWORD"] = child_env.get("HABFLY_LOGIN_PASSWORD") or getpass.getpass(
            "Local author password (hidden): "
        )
        if not child_env["HABFLY_LOGIN_EMAIL"] or not child_env["HABFLY_LOGIN_PASSWORD"]:
            parser.error("Both local login credentials are required; nothing was saved")
        print("Fresh Chromium: scripted login, introduction and one visible star.")
    else:
        print("Fresh Chromium session; Firefox is not touched. Manually open one star's Stellar tab.")
    if args.autonomous:
        print(
            "AUTONOMOUS: the frozen policy will select, calculate and enter up to three numbers without confirmation."
        )
        print("Only distance, luminosity and temperature. No retries or training.")
    else:
        print("TUI: b captures ready page; n steps learned decisions; y confirms each exact copy + Tab.")
    print("No Save, score update, assessment, or submission. All runs are recorded.")
    if args.runs is not None:
        from habfly.browser_reliability import run_reliability

        output = args.output or Path("experiments/browser-reliability") / uuid.uuid4().hex
        if output.exists():
            parser.error("Reliability output already exists; choose a new directory")
        credentials = tuple(child_env.pop(key) for key in ("HABFLY_LOGIN_EMAIL", "HABFLY_LOGIN_PASSWORD"))
        print("Batch mode: visible Chromium with console progress; Ctrl-C aborts the entire batch.")
        try:
            report = run_reliability(options, output, runs=args.runs, credentials=credentials)
        finally:
            credentials = None
        print(
            f"Verified {report['passed_runs']}/{report['requested_runs']} runs; "
            f"{report['unique_stars']} distinct stars. Report: {output / 'report.json'}"
        )
        raise SystemExit(0 if report["all_requested_runs_passed"] else 1)
    if args.autonomous:
        print("TUI: p/space pauses; r/space resumes; a aborts; q quits. No n/y presses needed.")
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
