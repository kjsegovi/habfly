"""Pure initial-selection provenance tests; all page reads are injected."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_browser_project_next_star_steps import fresh_capture, starfield_png

import habfly.browser_next_star as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_project_next_star_steps import _fresh
from habfly.browser_setup import BrowserSetup, SetupStop


@pytest.fixture
def initial(tmp_path, monkeypatch):
    source = tmp_path / "setup"
    source.mkdir()
    png = starfield_png()
    (source / "setup-starfield.png").write_bytes(png)
    report = fresh_capture("Althinagon")
    page = SimpleNamespace(frames=[SimpleNamespace(url=module.SIMULATION_URL)])
    page.context = SimpleNamespace(pages=[page])
    setup = object.__new__(BrowserSetup)  # Never constructs routing, credentials or a browser.
    setup.page, setup.config, setup.output = page, object(), source
    setup.stage, setup.closed = "stellar_screen_ready", True
    setup.star_clicked = setup.view_clicked = True
    setup.starfield_anchor, setup.excluded_star_points = (0.75, 0.4), ()
    setup.star_point = module.visible_star_point(png, anchor=setup.starfield_anchor)
    setup.ready_signature = module.screen_identity(report)
    calls = []
    hook = SimpleNamespace(on_read=None)

    def inspect(_page, _config):
        calls.append("read")
        if hook.on_read:
            hook.on_read(len(calls))
        return deepcopy(report)

    handle = SimpleNamespace(evaluate=lambda *_: True)
    monkeypatch.setattr(module, "inspect_page", inspect)
    monkeypatch.setattr(module, "read_class_choices", lambda _: ({"selected": None}, {"class": handle}))
    return SimpleNamespace(root=tmp_path, setup=setup, png=png, calls=calls, hook=hook)


@pytest.mark.parametrize("anchor", [(0.4, 0.55), (0.75, 0.4)])
def test_new_capture_pins_actual_anchor_and_exact_public_png(initial, anchor):
    setup = initial.setup
    setup.starfield_anchor = anchor
    setup.star_point = module.visible_star_point(initial.png, anchor=anchor)
    output = initial.root / "initial"
    receipt = module.capture_initial_setup_star(setup, output)
    scope = json.loads((output / "scope.json").read_bytes())
    assert scope["selection_schema_version"] == 1 and scope["starfield_anchor"] == list(anchor)
    assert scope["excluded_points"] == [] and scope["starfield_image"] == "setup-starfield.png"
    assert (output / "setup-starfield.png").read_bytes() == initial.png
    assert (setup.output / "setup-starfield.png").read_bytes() == initial.png
    assert initial.calls == ["read", "read"]
    assert not receipt["class_selection_verified"] and not receipt["task_completed"]
    entry = {
        "path": str(output / "confirmed.json"),
        "sha256": module.hashlib.sha256((output / "confirmed.json").read_bytes()).hexdigest(),
    }
    book = _Evidence(initial.root)
    validated, _, _ = _fresh(book, entry)
    assert validated == receipt
    assert "initial/setup-starfield.png" in book.hashes
    assert module.validate_initial_selection(scope, initial.png, receipt["selected_point"]) == anchor


@pytest.mark.parametrize("change", ["point", "anchor", "png", "exclusions"])
def test_provenance_changes_during_read_only_capture_fail(initial, change):
    def hook(count):
        if count != 2:
            return
        if change == "point":
            initial.setup.star_point = {**initial.setup.star_point, "x": 1}
        elif change == "anchor":
            initial.setup.starfield_anchor = (0.4, 0.55)
        elif change == "png":
            (initial.setup.output / "setup-starfield.png").write_bytes(b"changed")
        else:
            initial.setup.excluded_star_points = (initial.setup.star_point,)

    initial.hook.on_read = hook
    with pytest.raises(BrowserSafetyStop, match="changed_during_capture"):
        module.capture_initial_setup_star(initial.setup, initial.root / "initial")
    assert not (initial.root / "initial/confirmed.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("selection_schema_version", True),
        ("selection_schema_version", 2),
        ("starfield_anchor", [float("nan"), 0.4]),
        ("starfield_anchor", [0.75]),
        ("excluded_points", None),
        ("excluded_points", [{}]),
        ("starfield_image", "../setup/setup-starfield.png"),
        ("starfield_sha256", "f" * 64),
    ],
)
def test_unknown_or_malformed_provenance_is_not_repaired(initial, key, value):
    output = initial.root / "initial"
    receipt = module.capture_initial_setup_star(initial.setup, output)
    scope = json.loads((output / "scope.json").read_bytes())
    scope[key] = value
    with pytest.raises((BrowserSafetyStop, SetupStop)):
        module.validate_initial_selection(scope, initial.png, receipt["selected_point"])


def test_old_default_only_receipt_remains_supported_but_no_anchor_retrofit(initial):
    png = initial.png
    legacy = {"starfield_sha256": module.hashlib.sha256(png).hexdigest()}
    default = module.visible_star_point(png)
    other = module.visible_star_point(png, anchor=(0.75, 0.4))
    assert default != other
    assert module.validate_initial_selection(legacy, png, default) == (0.4, 0.55)
    with pytest.raises(BrowserSafetyStop, match="selection_source_changed"):
        module.validate_initial_selection(legacy, png, other)
    with pytest.raises(BrowserSafetyStop, match="unversioned"):
        module.validate_initial_selection({**legacy, "starfield_anchor": [0.75, 0.4]}, png, other)


def test_new_receipt_rejects_substituted_external_png_path(initial):
    output = initial.root / "initial"
    module.capture_initial_setup_star(initial.setup, output)
    entry = {
        "path": str(output / "confirmed.json"),
        "sha256": module.hashlib.sha256((output / "confirmed.json").read_bytes()).hexdigest(),
        "starfield_path": str(initial.setup.output / "setup-starfield.png"),
    }
    with pytest.raises(BrowserSafetyStop, match="versioned_initial_starfield_path_mismatch"):
        _fresh(_Evidence(initial.root), entry)


def test_source_provenance_checks_never_change_existing_files(initial):
    output = initial.root / "initial"
    receipt = module.capture_initial_setup_star(initial.setup, output)
    previous = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    scope = json.loads((output / "scope.json").read_bytes())
    assert module.validate_initial_selection(scope, initial.png, receipt["selected_point"]) == (0.75, 0.4)
    assert {path: path.read_bytes() for path in previous} == previous
