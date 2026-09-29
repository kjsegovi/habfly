"""Synthetic offline capture chains are not evidence of real pagination."""

import hashlib
import json
import socket
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import habfly.project_paginated_inventory as module
from habfly.browser_probe import save_probe
from habfly.browser_project_inventory import _capture_inventory
from habfly.browser_project_navigation import LIST_LABELS
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_paginated_inventory import (
    PaginatedInventoryError,
    merge_paginated_inventory,
    validate_paginated_inventory,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


def read(path):
    return json.loads(path.read_bytes())


def capture(names, start, total):
    end = start + len(names) - 1
    header = "Funding $50000 Data Quality 0% Scavenger Hunt 0/8 Observations Analyzed Data Star " + " ".join(
        LIST_LABELS["stellar"]
    )
    rows = [f"{name} 0.045 370 5.68E-10 1 1 UV 1 Main Sequence 1 1 1 Ga" for name in names]
    footer = f"viewing {start}-{end} of {total} total collected {total}"
    controls = [
        {
            "id": f"simulation-0:c{i}",
            "role": "button",
            "accessibility": "- button" + (" [disabled]" if disabled else ""),
            "enabled": not disabled,
            "actions": [],
            "protected": False,
        }
        for i, disabled in enumerate((start == 1, end == total))
    ]
    atoms = [
        {"text": header},
        *[{"text": row} for row in rows],
        *["button" + (" [disabled]" if not c["enabled"] else "") for c in controls],
        {"text": footer},
    ]
    return {
        "schema_version": 1,
        "mode": "read_only_browser_preflight",
        "outer_url": "http://localhost/activity",
        "outer_controls": [],
        "ignored_frame_urls": [],
        "allow_submission": False,
        "actions_executed": 0,
        "frames": [
            {
                "id": "simulation-0",
                "url": SIMULATION_URL,
                "text": "\n".join([header, *rows, footer]),
                "accessibility": yaml.safe_dump(atoms, sort_keys=False),
                "controls": controls,
            }
        ],
    }


def page(root, name, report, tick):
    directory = root / name
    checks = {}
    for offset, part in enumerate(("before", "after")):
        value = deepcopy(report)
        value["captured_at"] = (
            datetime(2026, 9, 26, tzinfo=UTC) + timedelta(seconds=tick + offset)
        ).isoformat()
        save_probe(value, directory / part)
        for item in ("observation", "manifest"):
            checks[f"{part}/{item}.json"] = sha(directory / part / f"{item}.json")
    counts, rows, _, _ = module._page_capture(value)
    receipt = {
        "schema_version": 1,
        "mode": module.PAGE_MODE,
        "section": "stellar",
        "viewing": counts,
        "rows": [{**r, "source_sha256": checks["after/observation.json"]} for r in rows],
        "source_sha256": checks,
    }
    write(directory / "page.json", receipt)
    return {
        "receipt": str((directory / "page.json").relative_to(root)),
        "sha256": sha(directory / "page.json"),
    }


def chain(root, sizes=(3, 3)):
    total = sum(sizes)
    names = [f"Star{i:02d}" for i in range(total)]
    reports, start = [], 1
    for size in sizes:
        reports.append(capture(names[start - 1 : start - 1 + size], start, total))
        start += size
    forward, revisit, tick = [], [], 0
    for i, report in enumerate(reports):
        forward.append(page(root, f"forward-{i}", report, tick))
        tick += 2
    for i, report in enumerate(reversed(reports)):
        revisit.append(page(root, f"revisit-{i}", report, tick))
        tick += 2
    anchor = page(root, "anchor", reports[0], tick)
    plan = {
        "schema_version": 1,
        "mode": module.PLAN_MODE,
        "forward": forward,
        "revisit": revisit,
        "assessment_anchor": anchor,
        "expected_stars": names,
    }
    write(root / "plan.json", plan)
    return plan


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("Network forbidden"))
    chain(tmp_path)
    return tmp_path


def validate(root):
    return validate_paginated_inventory(root, root / "plan.json", sha(root / "plan.json"))


def repin(root, name):
    """Rehash a coherently altered synthetic receipt, to test semantic rejection."""
    directory = root / name
    data = read(directory / "page.json")
    for part in ("before", "after"):
        manifest = read(directory / part / "manifest.json")
        manifest["observation_sha256"] = sha(directory / part / "observation.json")
        write(directory / part / "manifest.json", manifest)
        for item in ("observation", "manifest"):
            data["source_sha256"][f"{part}/{item}.json"] = sha(directory / part / f"{item}.json")
    for row in data["rows"]:
        row["source_sha256"] = data["source_sha256"]["after/observation.json"]
    write(directory / "page.json", data)
    plan = read(root / "plan.json")
    for ref in [*plan["forward"], *plan["revisit"], plan["assessment_anchor"]]:
        if ref["receipt"] == name + "/page.json":
            ref["sha256"] = sha(directory / "page.json")
    write(root / "plan.json", plan)


def test_valid_chain_has_separate_digests_and_never_claims_live_inventory(case):
    result = validate(case)
    assert result["recorded_page_chain_validated"] and result["recorded_complete_coverage_validated"]
    assert result["page_count"] == 2 and result["total_collected_in_recorded_chain"] == 6
    assert result["whole_collection_sha256"] != result["visible_assessment_anchor_sha256"]
    assert result["assessment_anchor"]["viewing"] == {"start": 1, "end": 3, "total": 6}
    for flag in (
        "live_pagination_verified",
        "pager_controls_grounded",
        "pager_transition_verified",
        "same_live_attempt_verified",
        "visible_exposure_verified",
        "collection_count_verified",
        "complete_visible_list_verified",
        "assessment_panel_verified",
        "journal_import_enabled",
        "scoring_input_enabled",
        "scientific_verified",
        "training_label",
        "task_completed",
        "project_completed",
        "submitted",
        "automatic_retry",
    ):
        assert result[flag] is False
    assert result["browser_actions"] == result["journal_writes"] == 0
    assert len(result["source_sha256"]) == 26
    assert all(sha(case / name) == value for name, value in result["source_sha256"].items())
    assert validate(case) == result


def test_merge_writes_only_new_distinct_receipt_and_preserves_inputs(case):
    before = {str(p): sha(p) for p in case.rglob("*.json")}
    result = merge_paginated_inventory(
        case, case / "merged", plan_path=case / "plan.json", plan_sha256=sha(case / "plan.json")
    )
    assert read(case / "merged/validated.json") == result
    assert not (case / "merged/confirmed.json").exists()
    assert all(sha(Path(name)) == value for name, value in before.items())
    with pytest.raises(PaginatedInventoryError, match="new_owned_output"):
        merge_paginated_inventory(
            case, case / "merged", plan_path=case / "plan.json", plan_sha256=sha(case / "plan.json")
        )


def test_collection_digest_does_not_depend_on_page_size_but_anchor_digest_does(case, tmp_path):
    other = tmp_path / "other"
    chain(other, (2, 2, 2))
    a, b = validate(case), validate(other)
    assert a["whole_collection_sha256"] == b["whole_collection_sha256"]
    assert a["visible_assessment_anchor_sha256"] != b["visible_assessment_anchor_sha256"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_forward",
        "missing_revisit",
        "reordered_forward",
        "reordered_revisit",
        "duplicate_forward",
        "duplicate_revisit",
        "reused_anchor",
        "different_anchor",
        "wrong_names",
        "duplicate_names",
        "wrong_schema",
        "boolean_schema",
        "extra_keys",
        "absolute_path",
        "parent_path",
        "bad_receipt_hash",
    ],
)
def test_malformed_changed_duplicate_missing_or_reordered_plan_stops(case, fault):
    plan = read(case / "plan.json")
    if fault == "missing_forward":
        plan["forward"].pop()
    elif fault == "missing_revisit":
        plan["revisit"].pop()
    elif fault.startswith("reordered_"):
        plan[fault.removeprefix("reordered_")].reverse()
    elif fault.startswith("duplicate_") and fault != "duplicate_names":
        values = plan[fault.removeprefix("duplicate_")]
        values[1] = values[0]
    elif fault == "reused_anchor":
        plan["assessment_anchor"] = plan["forward"][0]
    elif fault == "different_anchor":
        plan["assessment_anchor"] = page(
            case, "wrong-anchor", capture(["Star03", "Star04", "Star05"], 4, 6), 10
        )
    elif fault == "wrong_names":
        plan["expected_stars"][-1] = "Unknown"
    elif fault == "duplicate_names":
        plan["expected_stars"][1] = plan["expected_stars"][0].upper()
    elif fault in {"wrong_schema", "boolean_schema"}:
        plan["schema_version"] = 2 if fault == "wrong_schema" else True
    elif fault == "extra_keys":
        plan["live_pagination_verified"] = True
    elif fault == "absolute_path":
        plan["forward"][0]["receipt"] = str(case / plan["forward"][0]["receipt"])
    elif fault == "parent_path":
        plan["forward"][0]["receipt"] = "../" + case.name + "/forward-0/page.json"
    else:
        plan["forward"][0]["sha256"] = "z" * 64
    write(case / "plan.json", plan)
    with pytest.raises(PaginatedInventoryError):
        validate(case)


@pytest.mark.parametrize(
    "target",
    [
        "page.json",
        "before/manifest.json",
        "after/manifest.json",
        "before/observation.json",
        "after/observation.json",
    ],
)
def test_every_source_byte_hash_is_enforced(case, target):
    path = case / "forward-0" / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PaginatedInventoryError, match="source_hash_mismatch"):
        validate(case)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "unknown_frame",
        "wrong_section",
        "before_after_changed",
        "zero_range",
        "wrong_total",
        "text_ax_differ",
        "capture_time",
        "same_timestamp",
        "prior_timestamp",
        "changed_funding",
        "changed_outer",
        "changed_control",
        "pager_missing",
        "pager_state",
        "bool_schema",
    ],
)
def test_semantically_invalid_rehashed_capture_is_rejected(case, fault):
    if fault == "missing":
        (case / "forward-0/before/observation.json").unlink()
    else:
        for part in ("before", "after"):
            path = case / "forward-0" / part / "observation.json"
            data = read(path)
            frame = data["frames"][0]
            if fault == "unknown_frame":
                data["ignored_frame_urls"] = ["https://unexpected.invalid/"]
            elif fault == "wrong_section":
                frame["accessibility"] = '- text: "Planet details"'
            elif fault == "before_after_changed":
                if part == "after":
                    frame["text"] += " Changed answer"
            elif fault in {"zero_range", "wrong_total", "text_ax_differ"}:
                replacement = {
                    "zero_range": "0-3 of 6",
                    "wrong_total": "1-3 of 7",
                    "text_ax_differ": "2-3 of 6",
                }[fault]
                frame["text"] = frame["text"].replace("1-3 of 6", replacement)
                if fault != "text_ax_differ":
                    frame["accessibility"] = frame["accessibility"].replace("1-3 of 6", replacement)
            elif fault == "capture_time":
                data["captured_at"] = "not a timestamp"
            elif fault == "same_timestamp":
                data["captured_at"] = "2026-09-26T00:00:00+00:00"
            elif fault == "prior_timestamp":
                data["captured_at"] = (
                    "2026-09-26T00:01:0"
                    + str(part == "after").replace("False", "0").replace("True", "1")
                    + "+00:00"
                )
            elif fault == "changed_funding":
                frame["text"] = frame["text"].replace("$50000", "$49900")
                frame["accessibility"] = frame["accessibility"].replace("$50000", "$49900")
            elif fault == "changed_outer":
                data["outer_controls"] = [{"role": "button", "accessibility": '- button "Different"'}]
            elif fault == "changed_control":
                frame["controls"].append(
                    {"role": "button", "accessibility": '- button "Assess"', "enabled": True}
                )
            elif fault == "pager_missing":
                frame["controls"].pop()
            elif fault == "pager_state":
                frame["controls"][0]["enabled"] = True
            else:
                data["schema_version"] = True
            write(path, data)
        repin(case, "forward-0")
    with pytest.raises(PaginatedInventoryError):
        validate(case)


@pytest.mark.parametrize(
    "fault", ["duplicate_name", "gap", "overlap", "changed_total", "revisit_value", "revisit_row_order"]
)
def test_individually_valid_pages_do_not_allow_invalid_collection_chains(case, fault):
    plan = read(case / "plan.json")
    if fault == "duplicate_name":
        report = capture(["STAR00", "Star04", "Star05"], 4, 6)
    elif fault == "gap":
        report = capture(["Star04", "Star05"], 5, 6)
    elif fault == "overlap":
        report = capture(["Star02", "Star03", "Star04", "Star05"], 3, 6)
    elif fault == "changed_total":
        report = capture(["Star03", "Star04", "Star05"], 4, 7)
    elif fault == "revisit_row_order":
        report = capture(["Star04", "Star03", "Star05"], 4, 6)
    else:
        report = capture(["Star03", "Star04", "Star05"], 4, 6)
        frame = report["frames"][0]
        frame["text"] = frame["text"].replace("5.68E-10", "9.68E-10")
        frame["accessibility"] = frame["accessibility"].replace("5.68E-10", "9.68E-10")
    if fault.startswith("revisit"):
        plan["revisit"][0] = page(case, "replaced-revisit", report, 4)
    else:
        plan["forward"][1] = page(case, "replaced-forward", report, 2)
    write(case / "plan.json", plan)
    with pytest.raises(PaginatedInventoryError):
        validate(case)


def test_symlink_sources_and_output_outside_history_are_rejected(case, tmp_path):
    (case / "linked-page.json").symlink_to(case / "forward-0/page.json")
    plan = read(case / "plan.json")
    plan["forward"][0]["receipt"] = "linked-page.json"
    write(case / "plan.json", plan)
    with pytest.raises(PaginatedInventoryError):
        validate(case)
    with pytest.raises(PaginatedInventoryError):
        merge_paginated_inventory(
            case,
            case.parent / "outside-merged",
            plan_path=case / "plan.json",
            plan_sha256=sha(case / "plan.json"),
        )


def test_duplicate_json_keys_and_plan_hash_mismatch_rejected(case):
    path = case / "plan.json"
    old = sha(path)
    raw = path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    path.write_text(raw)
    with pytest.raises(PaginatedInventoryError, match="source_hash_mismatch"):
        validate_paginated_inventory(case, path, old)
    with pytest.raises(PaginatedInventoryError, match="duplicate_json_key"):
        validate(case)


def test_final_source_recheck_blocks_output_on_mutation(case, monkeypatch):
    original = module._Sources.unchanged

    def changed(book):
        path = case / "forward-0/page.json"
        path.write_bytes(path.read_bytes() + b"\n")
        original(book)

    monkeypatch.setattr(module._Sources, "unchanged", changed)
    with pytest.raises(PaginatedInventoryError):
        merge_paginated_inventory(
            case, case / "out", plan_path=case / "plan.json", plan_sha256=sha(case / "plan.json")
        )
    assert not (case / "out").exists()


def test_partial_capture_does_not_relax_existing_single_page_validator(case):
    report = read(case / "forward-0/after/observation.json")
    assert module._page_capture(report)[0] == {"start": 1, "end": 3, "total": 6}
    with pytest.raises(Exception, match="incomplete_visible_list"):
        _capture_inventory(report)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_hash",
        "extra_hash",
        "extra_flag",
        "row_hash",
        "row_order",
        "row_source",
        "boolean_range",
        "manifest_hash",
    ],
)
def test_rehashed_malformed_receipt_cannot_bypass_source_contract(case, fault):
    path = case / "forward-0/page.json"
    data = read(path)
    if fault == "missing_hash":
        del data["source_sha256"]["before/manifest.json"]
    elif fault == "extra_hash":
        data["source_sha256"]["other.json"] = "a" * 64
    elif fault == "extra_flag":
        data["live_pagination_verified"] = True
    elif fault == "row_hash":
        data["rows"][0]["row_accessibility_sha256"] = "a" * 64
    elif fault == "row_order":
        data["rows"].reverse()
    elif fault == "row_source":
        data["rows"][0]["source_sha256"] = data["source_sha256"]["before/observation.json"]
    elif fault == "boolean_range":
        data["viewing"]["start"] = True
    else:
        manifest_path = case / "forward-0/before/manifest.json"
        manifest = read(manifest_path)
        manifest["observation_sha256"] = "a" * 64
        write(manifest_path, manifest)
        data["source_sha256"]["before/manifest.json"] = sha(manifest_path)
    write(path, data)
    plan = read(case / "plan.json")
    plan["forward"][0]["sha256"] = sha(path)
    write(case / "plan.json", plan)
    with pytest.raises(PaginatedInventoryError):
        validate(case)


def test_prefix_only_coverage_with_matching_revisit_count_is_still_incomplete(case):
    plan = read(case / "plan.json")
    plan["forward"] = plan["forward"][:1]
    plan["revisit"] = plan["revisit"][1:]
    write(case / "plan.json", plan)
    with pytest.raises(PaginatedInventoryError, match="incomplete_collection_coverage"):
        validate(case)


def test_distinct_schema_is_rejected_by_existing_inventory_importer(case):
    from habfly.browser_no_planet_workflow import _Evidence
    from habfly.project_evidence import _inventory

    directory = case / "not-an-inventory"
    write(directory / "confirmed.json", validate(case))
    with pytest.raises(Exception, match="unsupported_inventory"):
        _inventory(_Evidence(case), directory)


def test_single_page_and_thirty_page_bounds_are_supported_only_as_recorded_consistency(tmp_path):
    for size in ((1,), (1,) * 30):
        root = tmp_path / str(len(size))
        chain(root, size)
        result = validate(root)
        assert result["total_collected_in_recorded_chain"] == sum(size)
        assert not result["live_pagination_verified"] and not result["collection_count_verified"]


def test_nonfinite_json_and_malformed_capture_have_structured_errors(case):
    path = case / "plan.json"
    path.write_text('{"schema_version":NaN}')
    with pytest.raises(PaginatedInventoryError, match="nonfinite_json"):
        validate(case)
    with pytest.raises(PaginatedInventoryError):
        merge_paginated_inventory(case, case / "out", plan_path=path, plan_sha256=sha(path))
    assert not (case / "out").exists()


def test_existing_real_seven_row_capture_is_readable_but_not_pagination_proof():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/full-stellar-probe/20260926-005/inventory-251/after/observation.json"
    )
    if not path.exists():
        pytest.skip("Optional immutable local visible capture is unavailable")
    before = sha(path)
    counts, rows, _boundary, anchor = module._page_capture(read(path))
    assert counts == {"start": 1, "end": 7, "total": 7} and len(rows) == 7
    assert len(anchor) == 64 and sha(path) == before
