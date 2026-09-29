"""Typed authority boundary for explicit user-approved autosave assumptions.

These receipts prove only fresh visible state. No elapsed timer, banner, Save
click or invented persistence acknowledgement supplies their authority.
"""

import hashlib
import json
import re
from pathlib import PurePosixPath

from .browser import BrowserSafetyStop

MODE = "user_approved_autosave_visible_readback_v1"
ACKNOWLEDGEMENT_SOURCE = "user_approved_autosave_assumption_not_verified"


def autosave_flags():
    return {
        "mode": MODE,
        "strategy": "autosave",
        "authority": "visible_readback_only",
        "visible_readback_verified": True,
        "save_click_delivered": False,
        "save_acknowledgement_verified": False,
        "persistence_verified": False,
        "cross_session_persistence_verified": False,
        "scientific_verified": False,
        "correctness_verified": False,
        "task_completed": False,
        "project_completed": False,
        "automatic_retry": False,
        "browser_actions": 0,
    }


def autosave_workflow_flags():
    """Additive metadata only; leave workflow completion and its mode alone."""
    return {
        "save_strategy": "autosave",
        "save_authority": "visible_readback_only",
        "autosave_readback_mode": MODE,
        "persistence_verified": False,
    }


def autosave_manifest():
    policy = {
        "schema_version": 1,
        "mode": MODE,
        "user_approved": True,
        "authority": "fresh_visible_readback_only_not_persistence",
        "save_clicks": 0,
        "mandatory_banner": False,
        "elapsed_time_is_persistence_proof": False,
        "historical_failed_attempts_reinterpreted": False,
        "receipt_flags": autosave_flags(),
        "workflow_flags": autosave_workflow_flags(),
    }
    raw = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**policy, "sha256": hashlib.sha256(raw).hexdigest()}


def _relative(value):
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def validate_autosave_receipt(receipt, *, branch, star, output, extra_keys=()):
    """Validate exact common typed fields; each caller validates its sources.

    Source names are run-history-relative; this function never opens paths.
    Branch-only top-level fields must be explicitly enumerated by the reader.
    """
    expected = {"schema_version": 1, **autosave_flags()}
    required = set(expected) | {
        "branch",
        "star",
        "output",
        "source_sha256",
        "before_sha256",
        "after_sha256",
    }
    valid = (
        isinstance(receipt, dict)
        and set(receipt) == required | set(extra_keys)
        and all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items())
        and branch in {"no_planet", "positive", "terrestrial"}
        and receipt.get("branch") == branch
        and isinstance(star, str)
        and bool(star.strip())
        and isinstance(receipt.get("star"), str)
        and receipt["star"].casefold() == star.casefold()
        and _relative(output)
        and receipt.get("output") == output
        and isinstance(receipt.get("source_sha256"), dict)
        and bool(receipt["source_sha256"])
    )
    if valid:
        hashes = receipt["source_sha256"]
        valid = all(_relative(name) for name in hashes) and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in [*hashes.values(), receipt["before_sha256"], receipt["after_sha256"]]
        )
    if not valid:
        raise BrowserSafetyStop("invalid_autosave_visible_readback_receipt")
    return receipt


def validate_autosave_workflow_flags(receipt, *, enabled):
    """Forbid accidental autosave metadata on legacy explicit workflows."""
    flags = autosave_workflow_flags()
    if (
        type(enabled) is not bool
        or not isinstance(receipt, dict)
        or (
            any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in flags.items())
            if enabled
            else bool(set(receipt) & set(flags))
        )
    ):
        raise BrowserSafetyStop("invalid_autosave_workflow_authority")
