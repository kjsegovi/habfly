"""Fresh safe-parser parity, without browsers, inference or cached evidence."""

import copy
import json

import pytest
import yaml
from test_browser_stellar import HASH, SNAPSHOT, capture

import habfly.browser_stellar as module


def outcome(snapshot, loader, monkeypatch):
    monkeypatch.setattr(module, "_ACCESSIBILITY_LOADER", loader)
    try:
        return ("value", json.dumps(module._atoms(snapshot), ensure_ascii=True, sort_keys=True, default=repr))
    except module.StellarMappingError as exc:
        return ("error", str(exc))


@pytest.mark.parametrize(
    "snapshot",
    [
        SNAPSHOT,
        '- textbox "0"',
        '- textbox "0": ""',
        '- textbox "0": "0"',
        '- text: "parallax (\\") 0.045 peak λ 370 nm"',
        '- text: "99.99% Δ λ — ★"',
        '- combobox:\n  - option "Red" [selected]\n  - option "Blue"',
        '- checkbox "I am ready to submit project." [checked]',
        '- button "Save" [disabled]',
        "- text: |\n    first\n    second",
        "- text: >\n    first\n    second",
        '- text: "first\\nsecond\\tend"',
        '- text: "\\u0085\\u2028\\u2029"',
        "- text: 1e-09",
        "- text: 01",
        "- text: yes",
        "- text: .nan",
        "- text: .inf",
        "- text: null",
        "- text: 2026-09-28",
        "- text: 1\n  text: 2",
        "- &x text",
        "- *x",
        "- !thing text",
        "- !!str text",
        "%TAG !e! tag:example.com,2000:\n---\n- !e!thing text",
        '- !!python/object/apply:os.system ["never execute"]',
        "- text: &x [*x]",
        "- [unsupported, sequence]",
        "- {text: value, other: value}",
        "- {1: text}",
        "text: not-a-list",
        "[]",
        "",
        None,
        1,
        ["text"],
        '- text: "unterminated',
        "- text:\n\t- tab",
        '- text: "\\q"',
        '- text: "\x00"',
        '- text: "\ud800"',
        "---\n- text\n---\n- other",
        "- text\n" * 1001,
        "- text: " + "x" * 100001,
        *["- text: " + "[" * depth + "0" + "]" * depth for depth in (10, 64, 128, 256, 1024)],
    ],
)
def test_c_safe_loader_matches_original_values_and_rejection_codes(snapshot, monkeypatch):
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("Optional installed LibYAML unavailable; SafeLoader fallback remains")
    assert outcome(snapshot, yaml.CSafeLoader, monkeypatch) == outcome(snapshot, yaml.SafeLoader, monkeypatch)


@pytest.mark.parametrize("value", [None, "", "0", "1.234E-9"])
def test_complete_stellar_map_and_source_are_unchanged(value, monkeypatch):
    if not hasattr(yaml, "CSafeLoader"):
        pytest.skip("Optional installed LibYAML unavailable")
    report = capture()
    if value is not None:
        for item in report["frames"][0]["controls"][:3]:
            item["value"] = value
    original = copy.deepcopy(report)
    monkeypatch.setattr(module, "_ACCESSIBILITY_LOADER", yaml.SafeLoader)
    expected = module.map_stellar_capture(report, capture_sha256=HASH)
    monkeypatch.setattr(module, "_ACCESSIBILITY_LOADER", yaml.CSafeLoader)
    assert module.map_stellar_capture(report, capture_sha256=HASH) == expected
    assert report == original


def test_every_call_reparses_current_bytes_and_preserves_alias_tag_gate(monkeypatch):
    calls = []
    parse, load = yaml.parse, yaml.load

    def observed_parse(value, **kwargs):
        calls.append(("parse", value, kwargs.get("Loader")))
        return parse(value, **kwargs)

    def observed_load(value, **kwargs):
        calls.append(("load", value, kwargs["Loader"]))
        return load(value, **kwargs)

    monkeypatch.setattr(yaml, "parse", observed_parse)
    monkeypatch.setattr(yaml, "load", observed_load)
    for value in ("- text: first", "- text: first", "- text: changed"):
        module._atoms(value)
    assert [item[0] for item in calls] == ["parse", "load"] * 3
    assert all(item[2] is None for item in calls if item[0] == "parse")
    assert all(item[2] is module._ACCESSIBILITY_LOADER for item in calls if item[0] == "load")
    with pytest.raises(module.StellarMappingError, match="unsupported_accessibility_yaml"):
        module._atoms('- !!python/object/apply:os.system ["never execute"]')
    assert calls[-1][0] == "parse"  # Unsafe YAML never reaches construction.


@pytest.mark.parametrize("error", [yaml.YAMLError("fixture"), UnicodeError("fixture"), RecursionError()])
def test_optional_constructor_errors_fall_back_to_original_safe_outcome(monkeypatch, error):
    original = yaml.load
    marker, calls = object(), []
    monkeypatch.setattr(module, "_ACCESSIBILITY_LOADER", marker)

    def load(value, Loader):
        calls.append(Loader)
        if Loader is marker:
            raise error
        return original(value, Loader=Loader)

    monkeypatch.setattr(yaml, "load", load)
    assert module._atoms('- text: "visible"') == [("text", "visible")]
    assert calls == [marker, yaml.SafeLoader]


def test_deep_input_uses_original_constructor_instead_of_widening_acceptance(monkeypatch):
    original, calls = yaml.load, []

    def load(value, Loader):
        calls.append(Loader)
        return original(value, Loader=Loader)

    monkeypatch.setattr(yaml, "load", load)
    module._atoms("- text: " + "[" * 64 + "0" + "]" * 64)
    assert calls == [yaml.SafeLoader]
