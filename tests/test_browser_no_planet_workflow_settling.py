"""Pure diagnostic-source gates using public capture transcriptions.

These synthetic files exercise the new optional evidence helper with the real
planet mapper/projection/hash validation. They are not native Save or complete
workflow receipts; the existing _load_sources integration retains those gates.
"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from test_browser_planet import capture

import habfly.browser_no_planet_workflow as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import screen_identity
from habfly.browser_probe import save_probe


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=True))


def read(path):
    return json.loads(path.read_bytes())


def footer(report, present):
    report = deepcopy(report)
    frame = report["frames"][0]
    if present:
        frame["text"] = frame["text"].removesuffix("\nSave") + "\nData saved\nSave"
        atoms = yaml.safe_load(frame["accessibility"])
        atoms[-2]["text"] += " Data saved"
        frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)
    return report


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "save"
    before = capture("No")
    before["captured_at"] = "fixture-time"
    frame = before["frames"][0]
    frame["text"] += "\nSave"
    frame["accessibility"] += '- button "Save"\n'
    frame["controls"].append(
        {
            "id": "simulation-0:save",
            "role": "button",
            "enabled": True,
            "value": None,
            "accessibility": '- button "Save"',
        }
    )
    intent = {
        "mode": module.SAVE_MODE,
        "settle_reserved_notice": True,
        "reserved_notice_policy": module._SETTLE_POLICY,
        "reserved_phase_timeout_seconds": 20,
        "choice_sha256": "a" * 64,
    }

    def probe(stage, report, present, start):
        return {
            "schema_version": 1,
            "mode": "no_planet_preclick_footer_diagnostic",
            "stage": stage,
            "source": "ordinary_visible_footer_probe",
            "compared_capture_sha256": screen_identity(report),
            "compared_capture_is_simultaneous": False,
            "probe_started_monotonic_seconds": start,
            "probe_finished_monotonic_seconds": start + 0.05,
            "diagnostic_only": True,
            "fresh_save_acknowledgement_verified": False,
            "save_click_dispatched": False,
            "retry_authorized": False,
            "task_completed": False,
            "verdict": "present" if present else "absent",
            "notice_present": present,
            "visible_text": "Data saved" if present else None,
            "same_footer_and_exposure_verified": present,
        }

    def build(rounds=2, waits_per_round=1):
        directory.mkdir()
        for name in ("before", "read-guard/initial", "pre-reservation-observations/000"):
            save_probe(before, directory / name)
        save_probe(footer(before, True), directory / "after")
        for name in ("reserved.json", "confirmed.json"):
            write(directory / name, intent)
        write(directory / "read-guard/scope.json", {"fixture_only": True})
        write(
            directory / "acknowledgement.json",
            {
                "visible_text": "Data saved",
                "source": "fully_exposed_footer_text",
                "notice_was_already_present": False,
            },
        )
        write(directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 1})
        for stage, time in (("before-reservation", 97.0), ("settle-final-000", 98.0)):
            write(directory / "footer-probes" / (stage + ".json"), probe(stage, before, False, time))
        write(
            directory / "pre-reservation-settled.json",
            {
                "schema_version": 1,
                "scope": "bounded_read_only_fresh_no_save_precondition",
                "read_only_cycles": 1,
                "compared_capture_sha256": screen_identity(before),
                "choice_sha256": intent["choice_sha256"],
                "last_footer_probe": "footer-probes/settle-final-000.json",
                "notice_present": False,
                "save_enabled": True,
                "reservation_created": False,
                "save_click_dispatched": False,
                "fresh_save_acknowledgement_verified": False,
                "later_notice_excluded": False,
                "task_completed": False,
            },
        )
        time, wait = 100.0, 0
        for index in range(rounds):
            current = footer(before, index > 0)
            current["captured_at"] = f"fixture-{index}"
            save_probe(current, directory / "reserved-observations" / f"{index:03d}")
            stage = f"reserved-final-{index:03d}"
            write(
                directory / "footer-probes" / (stage + ".json"),
                probe(stage, current, index < rounds - 1, time),
            )
            time += 1
            if index < rounds - 1:
                for n in range(waits_per_round):
                    stage = f"reserved-wait-{wait:03d}"
                    write(
                        directory / "footer-probes" / (stage + ".json"),
                        probe(stage, current, n < waits_per_round - 1, time),
                    )
                    time, wait = time + 1, wait + 1
        save_probe(current, directory / "pre-dispatch-observation")
        write(
            directory / "reserved-notice-settled.json",
            {
                "schema_version": 1,
                "policy": module._SETTLE_POLICY,
                "full_revalidations": rounds,
                "waiting_footer_probes": wait,
                "last_capture": f"reserved-observations/{rounds - 1:03d}",
                "last_footer_probe": f"footer-probes/reserved-final-{rounds - 1:03d}.json",
                "compared_capture_sha256": screen_identity(current),
                "deadline_monotonic_seconds": 120.0,
                "notice_present": False,
                "reservation_retained": True,
                "save_click_dispatched": False,
                "fresh_save_acknowledgement_verified": False,
                "later_notice_excluded": False,
                "automatic_retry": False,
                "task_completed": False,
            },
        )
        return directory

    def validate(evidence=None, selected=None):
        evidence = evidence or module._Evidence(tmp_path)
        module._reserved_notice_evidence(
            evidence, directory, intent if selected is None else selected, before, module._planet(before)
        )
        return evidence

    return SimpleNamespace(
        root=tmp_path, directory=directory, before=before, intent=intent, build=build, validate=validate
    )


@pytest.mark.parametrize("rounds,waits", [(1, 1), (2, 1), (3, 2)])
def test_valid_sources_pin_every_file_and_leave_success_to_existing_save_gates(source, rounds, waits):
    source.build(rounds, waits)
    evidence = source.validate()
    assert set(evidence.hashes) == {
        str(p.relative_to(source.root)) for p in source.directory.rglob("*") if p.is_file()
    }
    assert source.directory in evidence.closed_trees
    evidence.unchanged()
    assert read(source.directory / "reserved-notice-settled.json")["task_completed"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("settle_reserved_notice", 1),
        ("settle_reserved_notice", False),
        ("settle_reserved_notice", "true"),
        ("reserved_phase_timeout_seconds", 20.0),
        ("reserved_phase_timeout_seconds", "20"),
        ("reserved_phase_timeout_seconds", 21),
        ("reserved_notice_policy", "different_policy"),
    ],
)
def test_confirmed_options_must_match_intent_in_type_and_value(source, key, value):
    source.build()
    path = source.directory / "confirmed.json"
    receipt = read(path)
    receipt[key] = value
    write(path, receipt)
    with pytest.raises(BrowserSafetyStop, match="reserved_notice_intent_mismatch"):
        source.validate()


@pytest.mark.parametrize("key", sorted(module._SETTLE_KEYS))
def test_confirmed_receipt_cannot_omit_selected_option(source, key):
    source.build()
    path = source.directory / "confirmed.json"
    receipt = read(path)
    receipt.pop(key)
    write(path, receipt)
    with pytest.raises(BrowserSafetyStop, match="reserved_notice_intent_mismatch"):
        source.validate()


@pytest.mark.parametrize("continuation", [{}, {"resumed": True}, {"reconciled": True}])
def test_option_comparison_leaves_legacy_and_continuations_unchanged(continuation):
    module._reserved_notice_options_match({"legacy_extra": True}, {}, **continuation)
    with pytest.raises(BrowserSafetyStop, match="reserved_notice_intent_mismatch"):
        module._reserved_notice_options_match({"settle_reserved_notice": True}, {}, **continuation)


@pytest.mark.parametrize(
    "key,value",
    [
        ("settle_reserved_notice", False),
        ("settle_reserved_notice", 1),
        ("reserved_notice_policy", "new_unapproved_policy"),
        ("reserved_phase_timeout_seconds", True),
        ("reserved_phase_timeout_seconds", 0),
        ("reserved_phase_timeout_seconds", 31),
        ("reserved_phase_timeout_seconds", float("nan")),
        ("reserved_phase_timeout_seconds", float("inf")),
        ("reserved_phase_timeout_seconds", "20"),
        ("resumed_same_intent", True),
        ("mode", "unapproved"),
    ],
)
def test_invalid_policy_fails_closed(source, key, value):
    source.build()
    source.intent[key] = value
    with pytest.raises(BrowserSafetyStop, match="invalid_reserved_notice_policy"):
        source.validate()


@pytest.mark.parametrize("key", sorted(module._SETTLE_KEYS))
def test_partial_policy_cannot_select_new_format(source, key):
    source.build()
    source.intent.pop(key)
    with pytest.raises(BrowserSafetyStop, match="invalid_reserved_notice_policy"):
        source.validate()


def test_legacy_missing_opt_in_stays_noop_but_does_not_adopt_new_format(source):
    source.directory.mkdir()
    evidence = source.validate(selected={})
    assert not evidence.hashes and not evidence.closed_trees
    write(source.directory / "reserved-notice-settled.json", {})
    with pytest.raises(BrowserSafetyStop, match="unselected_reserved_notice_evidence"):
        source.validate(selected={})


@pytest.mark.parametrize(
    "key,value",
    [
        ("full_revalidations", True),
        ("full_revalidations", 0),
        ("full_revalidations", 10**20),
        ("waiting_footer_probes", -1),
        ("waiting_footer_probes", 10**20),
        ("deadline_monotonic_seconds", float("nan")),
        ("deadline_monotonic_seconds", True),
        ("last_capture", "../elsewhere"),
        ("last_footer_probe", "footer-probes/reserved-final-000.json"),
        ("compared_capture_sha256", "0" * 64),
        ("policy", "unapproved"),
        ("schema_version", True),
        ("reservation_retained", False),
        ("notice_present", True),
        ("save_click_dispatched", True),
        ("fresh_save_acknowledgement_verified", True),
        ("later_notice_excluded", True),
        ("automatic_retry", True),
        ("task_completed", True),
        ("extra", "unexpected"),
    ],
)
def test_malformed_settled_receipt_never_authorizes_work(source, key, value):
    source.build()
    path = source.directory / "reserved-notice-settled.json"
    value_map = read(path)
    value_map[key] = value
    write(path, value_map)
    with pytest.raises(BrowserSafetyStop):
        source.validate()


@pytest.mark.parametrize(
    "key,value",
    [
        ("compared_capture_sha256", "0" * 64),
        ("stage", "reserved-final-000"),
        ("probe_started_monotonic_seconds", 90),
        ("probe_finished_monotonic_seconds", 120),
        ("probe_finished_monotonic_seconds", float("inf")),
        ("probe_started_monotonic_seconds", True),
        ("compared_capture_is_simultaneous", True),
        ("verdict", "present"),
        ("notice_present", True),
        ("visible_text", "Data saved"),
        ("same_footer_and_exposure_verified", True),
        ("diagnostic_only", False),
        ("fresh_save_acknowledgement_verified", True),
        ("save_click_dispatched", True),
        ("retry_authorized", True),
        ("task_completed", True),
    ],
)
def test_final_probe_absence_and_exact_capture_binding_required(source, key, value):
    source.build()
    path = source.directory / "footer-probes/reserved-final-001.json"
    record = read(path)
    record[key] = value
    write(path, record)
    with pytest.raises(BrowserSafetyStop):
        source.validate()


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "reorder",
        "symlink",
        "changed_capture",
        "late_newfile",
        "late_mutation",
        "late_symlink",
        "dispatch",
    ],
)
def test_closed_tree_hashes_ordinals_and_visible_projection(source, kind):
    source.build()
    evidence = source.validate() if kind.startswith("late_") else None
    path = source.directory / "footer-probes/reserved-wait-000.json"
    if kind == "missing":
        path.unlink()
    elif kind in {"extra", "late_newfile"}:
        write(source.directory / "footer-probes/reserved-wait-999.json", {})
    elif kind == "reorder":
        a = source.directory / "footer-probes/reserved-final-000.json"
        original = read(a)
        write(a, read(path))
        write(path, original)
    elif kind in {"symlink", "late_symlink"}:
        target = source.root / "untrusted.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif kind == "changed_capture":
        target = source.directory / "reserved-observations/000"
        report = read(target / "observation.json")
        report["frames"][0]["controls"][0]["value"] = "5001"
        report["frames"][0]["controls"][0]["accessibility"] = report["frames"][0]["controls"][0][
            "accessibility"
        ].replace("10000", "5001")
        report["frames"][0]["accessibility"] = report["frames"][0]["accessibility"].replace("10000", "5001")
        for leaf in target.iterdir():
            leaf.unlink()
        target.rmdir()
        save_probe(report, target)
    elif kind == "late_mutation":
        write(path, {**read(path), "diagnostic_only": False})
    elif kind == "dispatch":
        write(source.directory / "dispatch.json", {"kind": "CLICK", "visible_label": "Save", "max_clicks": 2})
    with pytest.raises(BrowserSafetyStop):
        evidence.unchanged() if evidence else source.validate()


def test_wait_must_end_absent_before_next_full_capture(source):
    source.build()
    path = source.directory / "footer-probes/reserved-wait-000.json"
    record = read(path)
    record.update(
        notice_present=True,
        verdict="present",
        visible_text="Data saved",
        same_footer_and_exposure_verified=True,
    )
    write(path, record)
    with pytest.raises(BrowserSafetyStop, match="missing_reserved_wait_probe"):
        source.validate()


def test_after_readback_cannot_change_scientific_values(source):
    source.build()
    path = source.directory / "after"
    changed = read(path / "observation.json")
    changed["frames"][0]["controls"][1]["value"] = "1"
    for leaf in path.iterdir():
        leaf.unlink()
    path.rmdir()
    save_probe(changed, path)
    with pytest.raises(BrowserSafetyStop):
        source.validate()
