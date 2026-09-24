"""The experiment helpers must fail before loading data or starting an unapproved budget."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--variant", "structured-options", "--optimizer-updates", "801"],
        ["--variant", "topology-free-options", "--optimizer-updates", "800"],
        ["--variant", "structured", "--optimizer-updates", "800"],
        ["--variant", "semantic-options", "--optimizer-updates", "800", "--parent", "unused.pt"],
        ["--variant", "semantic-options"],
        ["--variant", "structured-options", "--parent", "unused.pt"],
    ],
)
def test_repair_budget_is_explicit_and_capped(tmp_path, arguments):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, "scripts/repair_distance.py", str(output), *arguments],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert not output.exists()


def test_audit_preserves_existing_output(tmp_path):
    output = tmp_path / "audit.json"
    output.write_text("preserve this")
    result = subprocess.run(
        [sys.executable, "scripts/inspect_distance.py", "missing-experiment", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0 and "fresh audit output" in result.stderr
    assert output.read_text() == "preserve this"


def test_documented_runner_has_offline_tripwires():
    # Runtime calls are blocked in the standalone processes, not global pytest state.
    for name in (
        "repair_distance.py",
        "inspect_distance.py",
        "train_measurement_identity.py",
        "train_source_request.py",
        "evaluate_distance_final.py",
        "train_distance_aliases.py",
        "verify_distance_demo.py",
    ):
        source = (Path("scripts") / name).read_text()
        assert "socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny" in source
        assert "SpreadsheetAdapter.__init__ = deny" in source


@pytest.mark.parametrize("runner", ["train_measurement_identity.py", "train_source_request.py"])
def test_identity_runner_preserves_existing_output_and_has_fixed_budget(tmp_path, runner):
    output = tmp_path / "identity"
    output.mkdir()
    for extra in ([], ["--optimizer-updates", "201"]):
        result = subprocess.run(
            [
                sys.executable,
                f"scripts/{runner}",
                str(output),
                "--parent",
                "missing.pt",
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 2
        assert not list(output.iterdir())


@pytest.mark.parametrize(
    "extra", [["--curriculum", "unknown"], ["--curriculum", "vocabulary-v1", "--optimizer-updates", "201"]]
)
def test_curriculum_cli_rejects_unknown_curricula_and_budget_extensions(tmp_path, extra):
    output = tmp_path / "never-created"
    result = subprocess.run(
        [sys.executable, "scripts/train_source_request.py", str(output), "--parent", "missing.pt", *extra],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and not output.exists()


@pytest.mark.parametrize("extra", [[], ["--epochs", "1"], ["--checkpoint", "other.pt"]])
def test_final_evaluation_preserves_output_and_rejects_training_or_selection(tmp_path, extra):
    output = tmp_path / "evaluation"
    output.mkdir()
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_distance_final.py", str(output), *extra],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and not list(output.iterdir())


@pytest.mark.parametrize(
    "extra",
    [
        ["--parent", "missing", "--updates", "1001"],
        ["--parent", "missing", "--updates", "0"],
        ["--evaluate", "--parent", "missing"],
        ["--evaluate", "--updates", "1000"],
        [],
    ],
)
def test_alias_runner_separates_training_and_frozen_evaluation(tmp_path, extra):
    output = tmp_path / "never-created"
    result = subprocess.run(
        [sys.executable, "scripts/train_distance_aliases.py", str(output), *extra],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and not output.exists()


@pytest.mark.parametrize(
    "runner,extra",
    [
        ("train_distance_aliases.py", ["--parent", "missing"]),
        ("verify_distance_demo.py", []),
    ],
)
def test_distance_readiness_runners_preserve_artifacts(tmp_path, runner, extra):
    result = subprocess.run(
        [sys.executable, f"scripts/{runner}", str(tmp_path), *extra],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2 and not list(tmp_path.iterdir())
