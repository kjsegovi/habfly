"""Offline synthetic stellar histories; no browser, model or production writes."""

import hashlib
import json
import socket
from copy import deepcopy

import pytest
from test_browser_star_preflight import evidence as class_evidence  # noqa: F401
from test_browser_star_preflight import read, save_capture, stellar, write

import habfly.browser_no_planet_workflow as legacy
import habfly.browser_stellar_sources as module
from habfly.browser import BrowserSafetyStop
from habfly.contracts import RuntimeEvent


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stellar source validation must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def stream(directory, manifest, payloads):
    events = [("hello", {"synthetic_fixture": True}), *payloads, ("episode_summary", manifest)]
    raw = "".join(
        RuntimeEvent(event=kind, sequence=i, run_id="offline-fixture", payload=payload).model_dump_json()
        + "\n"
        for i, (kind, payload) in enumerate(events)
    )
    (directory / "events.jsonl").write_text(raw)
    write(
        directory / "manifest.json", {**manifest, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    )


def visible_capture(main, *, populated=False, color=False):
    report = stellar(main)
    frame = report["frames"][0]
    if main:
        frame["accessibility"] = frame["accessibility"].replace("- option [selected]", "- option")
        frame["accessibility"] = frame["accessibility"].replace('- option "Ga"', '- option "Ga" [selected]')
        prefix = frame["controls"][-1]
        prefix["value"] = "Ga"
        prefix["accessibility"] = prefix["accessibility"].replace("- option [selected]", "- option")
        prefix["accessibility"] = prefix["accessibility"].replace('- option "Ga"', '- option "Ga" [selected]')
    for control in frame["controls"]:
        if control["role"] == "textbox":
            control["value"] = "1" if populated else ""
    if color:
        frame["accessibility"] = frame["accessibility"].replace('- option "UV"', '- option "UV" [selected]')
        choice = frame["controls"][3]
        choice["value"] = "UV"
        choice["accessibility"] = choice["accessibility"].replace('- option "UV"', '- option "UV" [selected]')
    return report


@pytest.fixture
def sources(class_evidence):  # noqa: F811 - imported fixture
    def create(selected_class="main_sequence", clicks=1):
        item = class_evidence(clicks, selected_class)
        main = selected_class == "main_sequence"
        item.numeric_dir, item.color_dir = item.history / "numeric", item.history / "color"
        save_capture(item.numeric_dir / "capture", visible_capture(main))
        save_capture(item.color_dir / "capture", visible_capture(main, populated=True))
        units = (
            dict(module.UNITS)
            if main
            else {n: module.UNITS[n] for n in ("distance", "luminosity", "temperature")}
        )
        if main:
            units["lifetime"] = "Ga"
        readbacks = {
            name: {
                "exact_copied": "1",
                "display_value": "1",
                "unit": unit,
                "exact_input_verified": True,
                "commit_key": "Tab",
                **module.committed_display("1", "1"),
            }
            for name, unit in units.items()
        }
        observation = module._stellar(visible_capture(main, populated=True))["observation"]
        numeric = {
            "full_stellar_numeric_transport_verified": True,
            "outcome": "full_stellar_numeric_transport_verified",
            "learned_policy": True,
            "checkpoint_unchanged": True,
            "optimizer_updates": 0,
            "write_attempts": len(units),
            "verified_fields": list(units),
            "numeric_readbacks": readbacks,
            "provenance": {
                "selected_class": selected_class,
                "lifetime_prefix": "Ga" if main else None,
                "optimizer_updates": 0,
                "checkpoint_sha256": "a" * 64,
                "graph_hash": "b" * 64,
                "knowledge_pack_hash": "c" * 64,
            },
        }
        payloads = [("action_proposed", {"action_source": "frozen_lifetime_checkpoint"})]
        payloads += [
            (
                "action_result",
                {
                    "numeric_copy_verified": True,
                    "numeric_readback": receipt,
                    "action": {"kind": "TYPE", "value": "1"},
                    "observation": observation,
                },
            )
            for receipt in readbacks.values()
        ]
        stream(item.numeric_dir, numeric, payloads)
        color = {
            "color_transport_verified": True,
            "outcome": "color_transport_verified",
            "write_attempts": 1,
            "receipt": {"selected_color": "UV", "readback_verified": True},
            "provenance": {
                "color_gate_passed": True,
                "optimizer_updates": 0,
                "color_checkpoint_sha256": "d" * 64,
                "color_reference_hash": "e" * 64,
            },
        }
        stream(
            item.color_dir,
            color,
            [
                ("action_proposed", {"action_source": "checkpoint"}),
                (
                    "action_result",
                    {
                        "color_transport_verified": True,
                        "receipt": color["receipt"],
                        "observation": module._stellar(visible_capture(main, populated=True, color=True))[
                            "observation"
                        ],
                    },
                ),
            ],
        )
        return item

    return create


def load(item, **kwargs):
    return module.read_stellar_sources(
        item.history,
        numeric_dir=item.numeric_dir,
        color_dir=item.color_dir,
        class_dir=item.class_dir,
        **kwargs,
    )


def rewrite(item, component, change):
    directory = item.history / component
    manifest = read(directory / "manifest.json")
    manifest.pop("events_sha256")
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    payloads = [(e["event"], e["payload"]) for e in events[1:-1]]
    change(manifest, payloads)
    stream(directory, manifest, payloads)


@pytest.mark.parametrize("selected_class", module.CLASSES)
@pytest.mark.parametrize("clicks", [1, 2])
def test_all_classes_exact_fields_and_legacy_prefix_hash_parity(sources, selected_class, clicks, monkeypatch):
    item = sources(selected_class, clicks)
    before = {p: p.read_bytes() for p in item.history.rglob("*") if p.is_file()}
    result = load(item, expected_star="ALTHINAGON")
    assert result["mode"] == module.MODE
    core = result["bundle"]
    assert core["class"] == selected_class
    assert core["prefix"] == ("Ga" if selected_class == "main_sequence" else None)
    assert len(core["readbacks"]) == (6 if selected_class == "main_sequence" else 3)
    assert set(core) == {
        "star",
        "class",
        "prefix",
        "readbacks",
        "color",
        "measurements",
        "class_rendering_sha256",
    }
    assert result["numeric_provenance"]["selected_class"] == selected_class
    assert result["color_provenance"]["color_gate_passed"] is True
    assert result["class_provenance"]["classification_learned"] is False
    assert result["browser_actions"] == 0
    for flag in ("current_class_paint_verified", "scientific_verified", "task_completed"):
        assert result[flag] is False

    class EndOfStellarPrefix(Exception):
        pass

    class PrefixBook(legacy._Evidence):
        def json(self, path):
            # The legacy prefix has no Save reads until this exact call.
            if path == item.history / "save" / "confirmed.json":
                raise EndOfStellarPrefix
            return super().json(path)

    (item.history / "save").mkdir()
    (item.history / "choice").mkdir()
    old = PrefixBook(item.history)
    bundles = []
    old_match = legacy._values_match

    def remember_bundle(mapping, bundle, **kwargs):
        bundles.append(bundle)
        return old_match(mapping, bundle, **kwargs)

    monkeypatch.setattr(legacy, "_values_match", remember_bundle)
    with pytest.raises(EndOfStellarPrefix):
        legacy._load_sources(
            old,
            numeric_dir=item.numeric_dir,
            color_dir=item.color_dir,
            class_dir=item.class_dir,
            choice_dir=item.history / "choice",
            save_dir=item.history / "save",
        )
    assert result["source_sha256"] == old.hashes
    assert core == bundles[-1]
    assert result["source_sha256"] == {
        str(p.relative_to(item.history)): hashlib.sha256(raw).hexdigest() for p, raw in before.items()
    }
    assert before == {p: p.read_bytes() for p in item.history.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "component,key,value",
    [
        ("numeric", "write_attempts", 6.0),
        ("numeric", "write_attempts", True),
        ("numeric", "optimizer_updates", False),
        ("numeric", "learned_policy", 1),
        ("numeric", "checkpoint_unchanged", 1),
        ("numeric", "full_stellar_numeric_transport_verified", 1),
        ("color", "write_attempts", True),
        ("color", "write_attempts", 1.0),
        ("color", "write_attempts", 2),
        ("color", "color_transport_verified", 1),
    ],
)
def test_typed_counts_and_proof_flags(sources, component, key, value):
    item = sources()
    rewrite(item, component, lambda manifest, payloads: manifest.update({key: value}))
    with pytest.raises(BrowserSafetyStop):
        load(item)


@pytest.mark.parametrize("value", [False, True, 1, 0.0, None, "missing"])
def test_color_optimizer_provenance_is_explicit_typed_zero(sources, value):
    item = sources()

    def mutate(manifest, payloads):
        if value == "missing":
            manifest["provenance"].pop("optimizer_updates")
        else:
            manifest["provenance"]["optimizer_updates"] = value

    rewrite(item, "color", mutate)
    with pytest.raises(BrowserSafetyStop, match="unexpected_browser_optimizer_updates"):
        load(item)


@pytest.mark.parametrize(
    "change", ["extra_field", "missing_field", "duplicate_verified", "prefix", "unknown_class"]
)
@pytest.mark.parametrize("selected_class", ["main_sequence", "white_dwarf"])
def test_applicability_cannot_be_invented_or_dropped(sources, selected_class, change):
    item = sources(selected_class)

    def mutate(manifest, payloads):
        if change == "extra_field":
            manifest["numeric_readbacks"]["invented"] = deepcopy(manifest["numeric_readbacks"]["distance"])
        elif change == "missing_field":
            del manifest["numeric_readbacks"]["distance"]
        elif change == "duplicate_verified":
            manifest["verified_fields"].append("distance")
        elif change == "prefix":
            manifest["provenance"]["lifetime_prefix"] = None if selected_class == "main_sequence" else "Ga"
        else:
            manifest["provenance"]["selected_class"] = "giant"

    rewrite(item, "numeric", mutate)
    with pytest.raises(BrowserSafetyStop):
        load(item)


@pytest.mark.parametrize(
    "change", ["unit", "display", "action", "star", "missing_copy", "duplicate_copy", "flag_alias"]
)
def test_exact_copy_event_not_repaired(sources, change):
    item = sources("red_giant")

    def mutate(manifest, payloads):
        result = payloads[1][1]
        if change == "unit":
            result["numeric_readback"]["unit"] = "pc"
        elif change == "display":
            result["numeric_readback"]["display_value"] = "2"
        elif change == "action":
            result["action"]["value"] = "2"
        elif change == "star":
            result["observation"]["values"]["star_name"] = "Other"
        elif change == "missing_copy":
            payloads.pop()
        elif change == "duplicate_copy":
            payloads.append(deepcopy(payloads[1]))
        else:
            result["numeric_readback"]["exact_input_verified"] = 1

    rewrite(item, "numeric", mutate)
    with pytest.raises(BrowserSafetyStop):
        load(item)


@pytest.mark.parametrize(
    "change",
    ["wrong_star", "changed_value", "changed_color", "receipt_alias", "wrong_class", "bad_hash", "failed"],
)
def test_color_class_and_source_mutation_stops(sources, change):
    item = sources("supergiant")
    if change in {"wrong_star", "changed_value", "changed_color", "receipt_alias"}:

        def mutate(manifest, payloads):
            result = payloads[1][1]
            values = result["observation"]["values"]
            if change == "wrong_star":
                values["star_name"] = "Other"
            elif change == "changed_value":
                values["browser_field_map"]["distance"]["current_value"] = "2"
            elif change == "changed_color":
                values["color"]["selected"] = "Red"
            else:
                result["receipt"]["readback_verified"] = 1

        rewrite(item, "color", mutate)
    elif change == "wrong_class":
        value = read(item.class_dir / "confirmed.json")
        value["selected_class"] = "main_sequence"
        write(item.class_dir / "confirmed.json", value)
    elif change == "bad_hash":
        (item.numeric_dir / "capture/observation.json").write_text("{}")
    else:
        write(item.class_dir / "invalidated.json", {})
    with pytest.raises(BrowserSafetyStop):
        load(item)


def test_wrapper_expected_star_and_existing_book_isolation(sources):
    item = sources()
    with pytest.raises(BrowserSafetyStop, match="star_mismatch"):
        load(item, expected_star="Other")
    with pytest.raises(BrowserSafetyStop, match="invalid_expected_star"):
        load(item, expected_star=True)
    book = legacy._Evidence(item.history)
    write(item.history / "unrelated.json", {"not_a_stellar_source": True})
    book.json(item.history / "unrelated.json")
    result = module.load_stellar_sources(
        book, numeric_dir=item.numeric_dir, color_dir=item.color_dir, class_dir=item.class_dir
    )
    assert "unrelated.json" not in result["source_sha256"]
    assert "unrelated.json" in book.hashes
    result["bundle"]["readbacks"]["distance"]["exact_copied"] = "999"
    assert load(item)["bundle"]["readbacks"]["distance"]["exact_copied"] == "1"
    (item.color_dir / "events.jsonl").write_text("tampered")
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        book.unchanged()


@pytest.mark.parametrize("kind", ["outside", "symlink", "malformed", "missing", "summary_alias"])
def test_owned_source_guards(sources, tmp_path, kind):
    item = sources()
    if kind == "outside":
        item.numeric_dir = tmp_path
    elif kind == "symlink":
        link = item.history / "linked-numeric"
        link.symlink_to(item.numeric_dir, target_is_directory=True)
        item.numeric_dir = link
    elif kind == "malformed":
        (item.numeric_dir / "manifest.json").write_text('"not a manifest"')
    elif kind == "missing":
        (item.numeric_dir / "manifest.json").unlink()
    else:
        value = read(item.color_dir / "manifest.json")
        value["write_attempts"] = 1.0
        write(item.color_dir / "manifest.json", value)
    with pytest.raises(BrowserSafetyStop):
        load(item)
