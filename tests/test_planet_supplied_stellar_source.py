"""Offline source receipts, not native non-main workflow acceptance.

Stellar/class streams and native raw-copy captures use the real validators.
The upstream transit/presence/freshness seam is explicit: synthetic references
are not scientific proof or claims that the non-main browser branch is enabled.
"""

# ruff: noqa: F811

import hashlib
import socket
from copy import deepcopy

import pytest
import yaml
from test_browser_planet import capture as planet_capture
from test_browser_star_preflight import read, save_capture, write
from test_browser_stellar_sources import class_evidence, sources, visible_capture  # noqa: F401

import habfly.planet_supplied_stellar_source as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import committed_display


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planet(presence=None, *, observed=False):
    result = planet_capture(presence)
    result["outer_controls"] = []
    frame = result["frames"][0]
    frame["text"] = "ALTHINAGON\nOBSERVATIONS observe for has planet"
    frame["accessibility"] = frame["accessibility"].replace("Jyremis", "Althinagon")
    set_field(result, "observation_days", "5000" if observed else "")
    return result


def set_field(report, name, value):
    mapping = module._planet(report)
    ident = mapping["observation"]["values"]["browser_field_map"][name]["capture_target_id"]
    frame = report["frames"][0]
    numeric = [c for c in frame["controls"] if c["role"] == "textbox"]
    ordinal = next(i for i, c in enumerate(numeric) if c["id"] == ident)
    numeric[ordinal]["value"] = value
    replacement = {'textbox "0"': value} if value else 'textbox "0"'
    numeric[ordinal]["accessibility"] = yaml.safe_dump([replacement])
    atoms = yaml.safe_load(frame["accessibility"])
    indices = [
        i
        for i, item in enumerate(atoms)
        if (item if isinstance(item, str) else next(iter(item))).startswith("textbox ")
    ]
    atoms[indices[ordinal]] = replacement
    frame["accessibility"] = yaml.safe_dump(atoms, sort_keys=False)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Supplied source receipt must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def case(sources, monkeypatch):
    def create(selected_class="white_dwarf", clicks=1):
        item = sources(selected_class, clicks)
        item.navigation = item.history / "to-planet"
        before = visible_capture(False, populated=True, color=True)
        before["outer_controls"] = []
        after = planet()
        for name, report in (("before", before), ("pre-click", before), ("after", after)):
            save_capture(item.navigation / name, report)
        start, end = module.project_view(before), module.project_view(after)
        intent = {
            "mode": "bounded_project_navigation",
            "destination": "planet",
            "from": start,
            "expected_star": "Althinagon",
            "target_evidence": {
                "number": 2,
                "box": {"x": 10, "y": 10, "width": 24, "height": 24},
                "accessibility": "- img: 2",
            },
            "max_clicks": 1,
            "answer_writes": 0,
            "collection_clicks": 0,
            "save_clicks": 0,
            "assessment_clicks": 0,
            "submission_clicks": 0,
            "automatic_retry": False,
            "task_completed": False,
            "action_source": "deterministic_navigation",
        }
        write(item.navigation / "reserved.json", intent)
        write(
            item.navigation / "confirmed.json",
            {
                **intent,
                "to": end,
                "navigation_clicks": 1,
                "destination_verified": True,
                "same_star_verified": True,
                "suggested_required_text": module._required(after, end),
                "config_mutated": False,
            },
        )
        initial = planet("Yes", observed=True)
        measurements = {
            "line_shift": {"value": "0.001", "unit": "nm"},
            "brightness_drop": {"value": "1", "unit": "%"},
            "period_days": {"value": "100", "unit": "day"},
        }
        evidence = {"measurements": measurements}
        presence_dir = item.history / "presence"
        save_capture(presence_dir / "after", initial)
        presence = {"star": "ALTHINAGON", "evidence": evidence}
        write(presence_dir / "confirmed.json", presence)
        write(item.history / "reference/source.json", evidence)

        def valid_presence(owner, path, checksum):
            value = owner.json(path, checksum)
            capture, mapping, _ = owner.capture(path.parent / "after")
            return value, capture, mapping

        def valid_reload(owner, expected):
            assert owner.json(item.history / "reference/source.json") == expected

        def valid_fresh(owner, directory, recorded, expected):
            assert owner.json(directory / "evidence.json") == recorded
            valid_reload(owner, expected)

        # Explicit upstream source seam only. The tested module must invoke it;
        # current M/R, class/navigation, raw events and native-copy projections
        # are all handled by real code below.
        monkeypatch.setattr(module, "_presence", valid_presence)
        monkeypatch.setattr(module, "_reload", valid_reload)
        monkeypatch.setattr(module, "_fresh_evidence", valid_fresh)
        item.raw = item.history / "raw"
        previous = initial
        receipts, events = {}, []
        flags = module._flags_for(evidence)
        for i, name in enumerate(module.RAW, 1):
            mapping = module._planet(previous)
            field = mapping["observation"]["values"]["browser_field_map"][name]
            prefix = item.raw / "native-copies" / f"copy-{i:02d}"
            value, unit = measurements[name]["value"], measurements[name]["unit"]
            copy_intent = {
                "kind": "TYPE",
                "destination": name,
                "target": field["capture_target_id"],
                "value": value,
                "unit": unit,
                "action_source": "reference_diagnostic",
                "task_completed": False,
            }
            save_capture(prefix.with_name(prefix.name + "-before"), previous)
            previous = deepcopy(previous)
            set_field(previous, name, value)
            save_capture(prefix.with_name(prefix.name + "-after"), previous)
            receipt = {
                **copy_intent,
                "display": committed_display(value, value),
                "readback_verified": True,
                "correctness_verified": False,
            }
            write(prefix.with_name(prefix.name + "-reserved.json"), copy_intent)
            write(prefix.with_name(prefix.name + "-confirmed.json"), receipt)
            receipts[name] = receipt
            events.extend(
                {"kind": k, "payload": p, "provenance": flags["provenance"]}
                for k, p in (("action_proposed", copy_intent), ("action_result", receipt))
            )
        for i, event in enumerate(events):
            write(item.raw / f"native-event-{i:02d}.json", event)
        fresh = {"fixture": "fresh"}
        precopy = [{"fixture": f"precopy-{i}"} for i in range(1, 4)]
        for name, value in [("fresh", fresh), *[(f"precopy-{i}", v) for i, v in enumerate(precopy, 1)]]:
            write(item.raw / name / "evidence.json", value)
        raw_intent = {
            **flags,
            "schema_version": 1,
            "mode": flags["provenance"],
            "stage": "inputs",
            "star": "ALTHINAGON",
            "evidence": evidence,
            "fresh": fresh,
            "output": "raw",
            "presence_path": "presence/confirmed.json",
            "presence_sha256": sha(presence_dir / "confirmed.json"),
            "maximum_numeric_writes": 3,
            "max_seconds": 900,
            "destinations": list(module.RAW),
            "action_source": module._action_source(evidence),
        }
        write(item.raw / "reserved.json", raw_intent)
        book = module._Evidence(item.history)
        write(module._reservation(module._RasterEvidence(book), "ALTHINAGON", "inputs"), raw_intent)
        write(
            item.raw / "report.json",
            {
                **raw_intent,
                "raw_measurement_transport_verified": True,
                "numeric_writes": 3,
                "verified_fields": receipts,
                "precopy": precopy,
                "native_events": events,
                "saved": False,
                "assessed": False,
                "submitted": False,
            },
        )
        item.current = item.history / "current"
        current_sha = save_capture(item.current, previous)
        item.options = {
            "numeric_dir": item.numeric_dir,
            "color_dir": item.color_dir,
            "class_dir": item.class_dir,
            "navigation_dir": item.navigation,
            "raw_dir": item.raw,
            "current_capture_dir": item.current,
            "current_capture_sha256": current_sha,
            "expected_star": "Althinagon",
            "selected_class": selected_class,
            "expected_pack_hash": module.load_supplied_planet_pack().checksum,
            "expected_adapter_sha256": module.adapter_manifest()["sha256"],
        }
        return item

    return create


def build(item):
    return module.build_supplied_planet_stellar_inputs(item.history, **item.options)


def store(item, receipt):
    path = item.history / "receipt/confirmed.json"
    write(path, receipt)
    return path, sha(path)


def reload(item, path, checksum):
    return module.load_supplied_planet_stellar_inputs(
        item.history,
        path,
        expected_sha256=checksum,
        **{
            k: item.options[k]
            for k in ("expected_star", "selected_class", "expected_pack_hash", "expected_adapter_sha256")
        },
    )


@pytest.mark.parametrize("selected", sorted(module.CLASSES))
@pytest.mark.parametrize("clicks", [1, 2])
def test_receipt_keeps_actual_class_and_exact_readouts_without_fabricating_stellar_answers(
    case, selected, clicks
):
    item = case(selected, clicks)
    receipt = build(item)
    assert receipt["actual_class"] == selected
    assert receipt["inputs"]["stellar_mass"]["display_text"] == "2.368"
    assert receipt["inputs"]["stellar_radius"]["display_text"] == "1.961"
    assert set(receipt["stellar_source"]["bundle"]["readbacks"]) == {"distance", "luminosity", "temperature"}
    assert receipt["stellar_source"]["bundle"]["prefix"] is None
    assert all(receipt[k] == v and type(receipt[k]) is type(v) for k, v in module.FLAGS.items())
    assert receipt["pack_sha256"] == item.options["expected_pack_hash"]
    assert reload(item, *store(item, receipt)) == receipt


@pytest.mark.parametrize(
    "key,value",
    [
        ("expected_star", "Other"),
        ("selected_class", "red_giant"),
        ("selected_class", "main_sequence"),
        ("expected_pack_hash", "a" * 64),
        ("expected_adapter_sha256", "a" * 64),
        ("current_capture_sha256", "a" * 64),
        ("expected_pack_hash", True),
    ],
)
def test_expected_identity_is_explicit_and_never_repaired(case, key, value):
    item = case()
    item.options[key] = value
    with pytest.raises(BrowserSafetyStop):
        build(item)


@pytest.mark.parametrize("replacement", ["0", "-1", "nan", "Infinity", "", "2.369"])
def test_current_missing_invalid_or_changed_readout_rejects_even_with_new_capture_hash(case, replacement):
    item = case()
    capture = read(item.current / "observation.json")
    capture["frames"][0]["accessibility"] = capture["frames"][0]["accessibility"].replace(
        "2.368", replacement
    )
    item.options["current_capture_sha256"] = save_capture(item.current, capture)
    with pytest.raises(BrowserSafetyStop):
        build(item)


@pytest.mark.parametrize("mutation", ["wrong_unit", "star", "answer", "missing", "new_frame"])
def test_native_current_projection_is_not_just_mass_radius_matching(case, mutation):
    item = case()
    capture = read(item.current / "observation.json")
    frame = capture["frames"][0]
    if mutation == "wrong_unit":
        frame["accessibility"] = frame["accessibility"].replace("STAR RADIUS (Rs)", "STAR RADIUS (RE)")
    elif mutation == "star":
        frame["accessibility"] = frame["accessibility"].replace("Althinagon", "Other")
    elif mutation == "answer":
        set_field(capture, "period_days", "101")
    elif mutation == "missing":
        del frame["controls"][1]
    else:
        capture["ignored_frame_urls"] = ["http://localhost/unknown"]
    item.options["current_capture_sha256"] = save_capture(item.current, capture)
    with pytest.raises(BrowserSafetyStop):
        build(item)


@pytest.mark.parametrize(
    "target", ["navigation", "numeric", "class", "native", "reservation", "new_file", "symlink"]
)
def test_replay_rebuilds_sources_and_detects_mutation_or_addition(case, target):
    item = case()
    path, checksum = store(item, build(item))
    if target == "new_file":
        write(item.navigation / "unexpected.json", {})
    elif target == "symlink":
        leaf = item.current / "observation.json"
        original = item.history / "moved.json"
        leaf.rename(original)
        leaf.symlink_to(original)
    else:
        leaf = {
            "navigation": item.navigation / "confirmed.json",
            "numeric": item.numeric_dir / "manifest.json",
            "class": item.class_dir / "confirmed.json",
            "native": item.raw / "native-event-00.json",
            "reservation": item.raw / "reserved.json",
        }[target]
        value = read(leaf)
        value["changed"] = True
        write(leaf, value)
    with pytest.raises(BrowserSafetyStop):
        reload(item, path, checksum)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scientific_verified", True),
        ("browser_actions", False),
        ("actual_class", "main_sequence"),
        ("inputs", {}),
    ],
)
def test_hash_updating_receipt_cannot_promote_or_change_semantics(case, field, value):
    item = case()
    receipt = build(item)
    receipt[field] = value
    with pytest.raises(BrowserSafetyStop):
        reload(item, *store(item, receipt))


@pytest.mark.parametrize("helper", ["_presence", "_reload", "_fresh_evidence"])
def test_shared_reference_validation_failure_is_not_bypassed(case, monkeypatch, helper):
    item = case()

    def reject(*args, **kwargs):
        raise BrowserSafetyStop("reference_source_rejected")

    monkeypatch.setattr(module, helper, reject)
    with pytest.raises(BrowserSafetyStop, match="reference_source_rejected"):
        build(item)


def test_native_copy_readout_change_is_rejected_by_real_copy_chain(case):
    item = case()
    directory = item.raw / "native-copies/copy-02-after"
    report = read(directory / "observation.json")
    report["frames"][0]["accessibility"] = report["frames"][0]["accessibility"].replace("2.368", "9")
    save_capture(directory, report)
    with pytest.raises(BrowserSafetyStop, match="native_copy_capture_mismatch"):
        build(item)


@pytest.mark.parametrize(
    "change", [None, "star", "value", "unit", "display", "zero", "bool", "nan", "promotion"]
)
def test_current_mapping_leaf_check_is_exact_and_not_class_paint_authority(case, change):
    item = case()
    receipt = build(item)
    mapping = module._planet(read(item.current / "observation.json"))
    entry = mapping["observation"]["values"]["stellar_inputs"]["stellar_mass"]
    if change == "star":
        mapping["star_name"] = "Other"
    elif change == "value":
        entry["value"] = 9.0
    elif change == "unit":
        entry["unit"] = "ME"
    elif change == "display":
        entry["display_text"] = "2.3680"
    elif change == "zero":
        entry.update(value=0.0, display_text="0")
    elif change == "bool":
        entry.update(value=True, display_text="1")
    elif change == "nan":
        entry.update(value=float("nan"), display_text="nan")
    elif change == "promotion":
        receipt["scientific_verified"] = True
    assert module.matches_mapping(receipt, mapping) is (change is None)


@pytest.mark.parametrize("target", ["to-planet/pre-click", "to-planet/before"])
def test_class_link_requires_exact_completed_stellar_measurements(case, target):
    item = case()
    directory = item.history / target
    report = read(directory / "observation.json")
    report["frames"][0]["controls"][0]["value"] = "3"
    save_capture(directory, report)
    with pytest.raises(BrowserSafetyStop, match="stellar_readback_changed"):
        build(item)


def test_source_paths_cannot_escape_owned_history(case, tmp_path):
    item = case()
    outside = tmp_path / "outside"
    save_capture(outside, read(item.current / "observation.json"))
    item.options["current_capture_dir"] = outside
    with pytest.raises(BrowserSafetyStop, match="evidence_outside_history"):
        build(item)


@pytest.mark.parametrize(
    "change", ["unrelated_star", "own_modified", "own_deleted", "own_raw_added", "registry_stopped"]
)
def test_shared_claim_registry_allows_new_stars_without_weakening_own_evidence(case, change):
    item = case()
    receipt = build(item)
    path, checksum = store(item, receipt)
    owner = module._RasterEvidence(module._Evidence(item.history))
    claim = module._reservation(owner, "ALTHINAGON", "inputs")
    assert str(claim.parent.relative_to(item.history)) not in receipt["directory_trees"]
    assert str(claim.relative_to(item.history)) in receipt["source_sha256"]
    if change == "unrelated_star":
        write(module._reservation(owner, "Other Star", "inputs"), {"fixture": "unrelated"})
    elif change == "own_modified":
        write(claim, {**read(claim), "changed": True})
    elif change == "own_deleted":
        claim.unlink()
    elif change == "registry_stopped":
        write(claim.parent / "stopped.json", {"fixture": "uncertain registry"})
    else:
        write(item.raw / "native-copies/unexpected.json", {})
    if change == "unrelated_star":
        assert reload(item, path, checksum) == receipt
    else:
        with pytest.raises(BrowserSafetyStop):
            reload(item, path, checksum)
