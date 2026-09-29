"""Offline class provenance checks: synthetic records, no browser or training."""

import hashlib
import json
import socket
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_stellar import capture

import habfly.browser_star_preflight as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_stellar import map_stellar_capture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_bytes())


def save_capture(path, report):
    write(path / "observation.json", report)
    checksum = hashlib.sha256((path / "observation.json").read_bytes()).hexdigest()
    write(path / "manifest.json", {"observation_sha256": checksum})
    return checksum


def stellar(conditional=False):
    report = capture()
    report["ignored_frame_urls"] = []
    frame = report["frames"][0]
    for control in frame["controls"]:
        if control["role"] == "textbox":
            control["value"] = ""
    if conditional:
        prefix = "- combobox:\n  - option [selected]\n" + "".join(
            f'  - option "{p}"\n' for p in ("ka", "Ma", "Ga", "Ta")
        )
        fields = (
            '- text: mass (M\n- subscript: s\n- text: )\n- textbox "0"\n'
            '- text: radius (R\n- subscript: s\n- text: )\n- textbox "0"\n'
            '- text: lifetime (years)\n- textbox "0.00000"\n'
        )
        frame["accessibility"] = frame["accessibility"].replace(
            "- text: mass, radius and lifetime are only relevant for main sequence stars ",
            fields + prefix + "- text: ",
        )
        for index, placeholder in enumerate(("0", "0", "0.00000"), 7):
            frame["controls"].append(
                {
                    "id": f"simulation-0:c{index}",
                    "role": "textbox",
                    "enabled": True,
                    "accessibility": f'- textbox "{placeholder}"',
                    "value": "",
                }
            )
        frame["controls"].append(
            {
                "id": "simulation-0:c10",
                "role": "combobox",
                "enabled": True,
                "accessibility": prefix.strip(),
                "value": "",
            }
        )
    return report


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Class-source preflight must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def evidence(tmp_path):
    def create(clicks=1, selected_class="main_sequence"):
        history = tmp_path / f"history-{selected_class}-{clicks}"
        fresh, class_dir = history / "fresh", history / "class"
        painted = selected_class if clicks == 2 else None
        intermediate = "red_giant" if selected_class == "white_dwarf" else "white_dwarf"
        first = intermediate if clicks == 2 else selected_class
        claims = (
            {}
            if selected_class == "main_sequence"
            else {"scientific_verified": False, "training_label": False}
        )
        source_hash = save_capture(fresh / "stellar", stellar())
        write(
            fresh / "confirmed.json",
            {
                "star": "Althinagon",
                "fresh_blank_numeric_answers_verified": True,
                "class_selection_verified": False,
                "answer_writes": 0,
                "action_source": "deterministic_navigation",
                "painted_stellar_class": painted,
            },
        )
        write(
            fresh / "class-selection-reserved.json",
            {
                "star": "Althinagon",
                "source_capture_sha256": source_hash,
                "previous_unconfirmed_paint": painted,
                "selected_class": first,
                "action_source": "reference_diagnostic",
                "max_clicks": 1,
                "automatic_retry": False,
            },
        )
        write(
            class_dir / "scope.json",
            {
                "star": "Althinagon",
                "intended_class": selected_class,
                "fresh_star": str(fresh),
                "source_capture_sha256": source_hash,
                "max_class_clicks": clicks,
                "numeric_writes": 0,
                "clears_inherited_paint": clicks == 2,
                "intermediate_is_training_label": False,
                "learned_classification": False,
                "automatic_retry": False,
                **claims,
            },
        )
        write(
            class_dir / "confirmed.json",
            {
                "star": "Althinagon",
                "selected_class": selected_class,
                "class_clicks": clicks,
                "numeric_writes": 0,
                "readback_verified": True,
                "correctness_verified": False,
                "learned_classification": False,
                "intermediate_is_training_label": False,
                "action_source": "explicit_fresh_star_class_setup",
                "task_completed": False,
                **claims,
            },
        )
        save_capture(class_dir / "before", stellar())
        if clicks == 2:
            save_capture(class_dir / "intermediate", stellar())
        save_capture(class_dir / "after", stellar(selected_class == "main_sequence"))
        for index, chosen in enumerate([first, selected_class] if clicks == 2 else [first]):
            proposal = {
                "kind": "SELECT",
                "target": "stellar_class",
                "value": chosen,
                "action_source": "reference_diagnostic",
                "correctness_verified": False,
            }
            result = {
                "selected_class": chosen,
                "selection_source": "reference_diagnostic",
                "readback_verified": True,
                "correctness_verified": False,
                "rendering_sha256": ("a" if chosen == "main_sequence" else "b") * 64,
                "task_completed": False,
            }
            if index == 0:
                result.update(fresh_star_capture_sha256=source_hash, previous_unconfirmed_paint=painted)
            else:
                proposal.update(previous=intermediate, revision_reason="Explicit fixture synchronization")
                result.update(previous=proposal["previous"], revision_reason=proposal["revision_reason"])
            for offset, (event, payload) in enumerate(
                (("action_proposed", proposal), ("action_result", result))
            ):
                write(class_dir / f"event-{2 * index + offset}.json", {"event": event, "payload": payload})
        return SimpleNamespace(
            history=history,
            fresh=fresh,
            class_dir=class_dir,
            source_hash=source_hash,
            selected_class=selected_class,
        )

    return create


def validate(item, **kwargs):
    return module.validate_star_class_source(
        item.history,
        item.class_dir,
        kwargs.get("star", "Althinagon"),
        kwargs.get("selected_class", item.selected_class),
    )


@pytest.mark.parametrize("clicks", [1, 2])
@pytest.mark.parametrize("selected_class", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_verified_setup_is_read_only_and_hashes_every_source(evidence, clicks, selected_class):
    item = evidence(clicks, selected_class)
    before = {p: p.read_bytes() for p in item.history.rglob("*") if p.is_file()}
    result = validate(item, star="ALTHINAGON")
    assert result["star"] == "Althinagon" and result["class_clicks"] == clicks
    assert result["selected_class"] == selected_class
    assert result["class_rendering_sha256"] == ("a" if selected_class == "main_sequence" else "b") * 64
    assert result["source_capture_sha256"] == item.source_hash
    assert result["browser_actions"] == 0 and not result["classification_learned"]
    assert not result["correctness_verified"] and not result["current_class_paint_verified"]
    assert not result["scientific_verified"] and not result["training_label"]
    assert not result["task_completed"]
    assert result["source_hashes"] == {
        str(p.relative_to(item.history)): hashlib.sha256(raw).hexdigest() for p, raw in before.items()
    }
    assert before == {p: p.read_bytes() for p in item.history.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "path,key,value",
    [
        ("class/confirmed.json", "star", "Other Star"),
        ("class/confirmed.json", "selected_class", "red_giant"),
        ("class/confirmed.json", "readback_verified", False),
        ("class/confirmed.json", "class_clicks", True),
        ("class/confirmed.json", "class_clicks", 3),
        ("class/confirmed.json", "numeric_writes", False),
        ("class/confirmed.json", "learned_classification", True),
        ("class/confirmed.json", "correctness_verified", True),
        ("class/confirmed.json", "action_source", "explicit_blank_radio_recovery"),
        ("class/scope.json", "clears_inherited_paint", False),
        ("class/scope.json", "source_capture_sha256", "c" * 64),
        ("class/scope.json", "automatic_retry", True),
        ("fresh/confirmed.json", "painted_stellar_class", None),
        ("fresh/confirmed.json", "class_selection_verified", True),
        ("fresh/confirmed.json", "answer_writes", 1),
        ("fresh/class-selection-reserved.json", "automatic_retry", True),
        ("fresh/class-selection-reserved.json", "selected_class", "main_sequence"),
        ("fresh/class-selection-reserved.json", "source_capture_sha256", "d" * 64),
    ],
)
def test_wrong_incomplete_or_unsupported_provenance_rejected(evidence, path, key, value):
    item = evidence(2)
    target = item.history / path
    data = read(target)
    data[key] = value
    write(target, data)
    with pytest.raises(BrowserSafetyStop, match="star_class_preflight_"):
        validate(item)


@pytest.mark.parametrize(
    "index,key,value",
    [
        (0, "kind", "CLICK"),
        (0, "value", "main_sequence"),
        (1, "fresh_star_capture_sha256", "c" * 64),
        (1, "previous_unconfirmed_paint", None),
        (2, "previous", "red_giant"),
        (3, "readback_verified", False),
        (3, "selection_source", "checkpoint"),
        (3, "rendering_sha256", "not-a-hash"),
        (3, "revision_reason", "Different revision"),
    ],
)
def test_exact_reservation_and_class_event_chain_required(evidence, index, key, value):
    item = evidence(2)
    target = item.class_dir / f"event-{index}.json"
    data = read(target)
    data["payload"][key] = value
    write(target, data)
    with pytest.raises(BrowserSafetyStop, match="star_class_preflight_"):
        validate(item)


@pytest.mark.parametrize("directory", ["before", "intermediate", "after"])
@pytest.mark.parametrize("change", ["value", "measurement", "color", "unknown_frame"])
def test_all_class_captures_preserve_blank_fields_and_measurements(evidence, directory, change):
    item = evidence(2)
    target = item.class_dir / directory
    report = read(target / "observation.json")
    frame = report["frames"][0]
    if change == "value":
        frame["controls"][0]["value"] = "0"
    elif change == "measurement":
        frame["accessibility"] = frame["accessibility"].replace("0.045", "0.055")
    elif change == "color":
        frame["accessibility"] = frame["accessibility"].replace('option "UV"', 'option "UV" [selected]')
        control = next(c for c in frame["controls"] if c["role"] == "combobox")
        control["accessibility"] = control["accessibility"].replace('option "UV"', 'option "UV" [selected]')
    else:
        report["ignored_frame_urls"] = ["untrusted"]
    save_capture(target, report)
    with pytest.raises(BrowserSafetyStop, match="star_class_preflight_"):
        validate(item)


@pytest.mark.parametrize(
    "which", ["extra_event", "missing_event", "failed", "invalidated", "bad_capture_hash"]
)
def test_incomplete_extra_or_failed_evidence_stops(evidence, which):
    item = evidence()
    if which == "extra_event":
        write(item.class_dir / "event-2.json", {})
    elif which == "missing_event":
        (item.class_dir / "event-1.json").unlink()
    elif which in {"failed", "invalidated"}:
        write(item.class_dir / ("stopped.json" if which == "failed" else "invalidated.json"), {})
    else:
        write(item.class_dir / "after/manifest.json", {"observation_sha256": "f" * 64})
    with pytest.raises(BrowserSafetyStop, match="star_class_preflight_"):
        validate(item)


def test_mutation_during_validation_is_detected(evidence, monkeypatch):
    item = evidence()
    original = module._Evidence.read
    mutated = False

    def read_and_mutate(self, path):
        nonlocal mutated
        raw = original(self, path)
        if Path(path).name == "event-0.json" and not mutated:
            with (item.class_dir / "confirmed.json").open("a") as stream:
                stream.write(" ")
            mutated = True
        return raw

    monkeypatch.setattr(module._Evidence, "read", read_and_mutate)
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        validate(item)


@pytest.mark.parametrize("which", ["outside", "symlink", "malformed", "oversized"])
def test_unsafe_or_malformed_files_are_rejected_without_private_details(evidence, tmp_path, which):
    item = evidence()
    target = item.class_dir / "confirmed.json"
    if which == "outside":
        data = read(item.class_dir / "scope.json")
        data["fresh_star"] = str(tmp_path)
        write(item.class_dir / "scope.json", data)
    elif which == "symlink":
        moved = item.history / "copy.json"
        target.rename(moved)
        target.symlink_to(moved)
    elif which == "malformed":
        target.write_text("private invalid json")
    else:
        with target.open("wb") as stream:
            stream.truncate(32_000_001)
    with pytest.raises(BrowserSafetyStop) as caught:
        validate(item)
    assert "private" not in str(caught.value) and str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("selected", [None, "giant", "automatic", "Main Sequence", [], 1])
def test_only_explicit_native_classes_are_supported(evidence, selected):
    with pytest.raises(BrowserSafetyStop, match="unsupported_class"):
        validate(evidence(), selected_class=selected)


@pytest.mark.parametrize("selected_class", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_mapping_matches_only_the_same_visible_star_measurements_and_class(evidence, selected_class):
    item = evidence(selected_class=selected_class)
    source = validate(item)
    mapping = map_stellar_capture(
        stellar(selected_class == "main_sequence"),
        capture_sha256="a" * 64,
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )
    mapping["observation"]["values"]["selected_browser_class"] = selected_class
    assert module.matches_mapping(source, mapping)
    assert module.matches_mapping(source, mapping, class_rendering_sha256=source["class_rendering_sha256"])
    assert not module.matches_mapping(source, mapping, class_rendering_sha256="c" * 64)
    original = deepcopy(mapping)
    for field in mapping["observation"]["values"]["browser_field_map"].values():
        field["current_value"] = "1.234"
    assert module.matches_mapping(source, mapping)  # Color starts after numerical completion.
    for key, value in (
        ("selected_browser_class", "red_giant" if selected_class == "main_sequence" else "main_sequence"),
        ("star_name", "Other Star"),
        ("measurements", {}),
    ):
        altered = deepcopy(original)
        altered["observation"]["values"][key] = value
        assert not module.matches_mapping(source, altered)
    assert not module.matches_mapping(source, {})


@pytest.mark.parametrize("selected_class", ["red_giant", "supergiant", "white_dwarf"])
@pytest.mark.parametrize(
    "change",
    ["conditional", "same_intermediate", "wrong_previous", "wrong_paint", "missing_claim", "science_claim"],
)
def test_non_main_chain_cannot_bypass_branch_or_provenance_checks(evidence, selected_class, change):
    item = evidence(2, selected_class)
    if change == "conditional":
        save_capture(item.class_dir / "after", stellar(True))
    elif change == "same_intermediate":
        target = item.class_dir / "event-0.json"
        value = read(target)
        value["payload"]["value"] = selected_class
        write(target, value)
    elif change == "wrong_previous":
        target = item.class_dir / "event-2.json"
        value = read(target)
        value["payload"]["previous"] = "main_sequence"
        write(target, value)
    elif change == "wrong_paint":
        target = item.fresh / "confirmed.json"
        value = read(target)
        value["painted_stellar_class"] = None
        write(target, value)
    else:
        target = item.class_dir / "confirmed.json"
        value = read(target)
        if change == "missing_claim":
            value.pop("scientific_verified")
        else:
            value["scientific_verified"] = True
        write(target, value)
    with pytest.raises(BrowserSafetyStop, match="star_class_preflight_"):
        validate(item)


@pytest.mark.parametrize("selected_class", ["main_sequence", "red_giant", "supergiant", "white_dwarf"])
def test_mapping_rejects_wrong_conditional_branch(evidence, selected_class):
    source = validate(evidence(selected_class=selected_class))
    mapping = map_stellar_capture(
        stellar(selected_class != "main_sequence"),
        capture_sha256="a" * 64,
        allow_color_selection=True,
        allow_main_sequence_fields=True,
    )
    mapping["observation"]["values"]["selected_browser_class"] = selected_class
    assert not module.matches_mapping(source, mapping)


def test_actual_begollo_74_evidence_offline_when_available():
    history = Path(__file__).resolve().parents[1] / "experiments/full-stellar-probe/20260926-005"
    directory = history / "fresh-main-74"
    if not directory.is_dir():
        pytest.skip("Local saved Begollo class evidence is not distributed")
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = module.validate_star_class_source(history, directory, "Begollo", "main_sequence")
    assert len(result["source_hashes"]) == 16 and result["class_clicks"] == 2
    assert (
        result["class_rendering_sha256"] == "285abfeef9c2e4412d09b2c683eb47200fab0007df94e2d9b02ad1263081046e"
    )
    assert result["measurements"]["browser_parallax"]["value"] == 0.113
    assert result["measurements"]["browser_wavelength"]["value"] == 347
    assert result["measurements"]["browser_flux"]["value"] == 5.65e-9
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
