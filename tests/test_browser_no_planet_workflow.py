"""Read-only workflow tests on intercepted local fixtures, never the live preview.

The numeric/color streams below are synthetic transport fixtures, not learned
checkpoints or performance evidence. Native class/No/Save adapters produce the
other receipts. Navigation is replaced by a two-screen fixture seam; navigation
itself has its own dual-tab integration suite.
"""
# ruff: noqa: F811

import hashlib
import json
from copy import deepcopy

import pytest
from test_browser_full_stellar import full_html
from test_browser_no_planet_save import (  # noqa: F401
    apply_revisit,
    consumed_predispatched,
    no_save_page,
    predispatched,
    revisit_no_save_page,
    save,
)
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_window_choice import window_page  # noqa: F401

import habfly.browser_no_planet_workflow as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_class_recovery import synchronize_fresh_stellar_class
from habfly.browser_classification import read_class_choices
from habfly.browser_no_planet_save import (
    reconcile_no_planet_autosave,
    resume_no_planet_save,
    save_no_planet_work,
)
from habfly.browser_numeric import committed_display, digest
from habfly.browser_probe import inspect_page, save_probe
from habfly.contracts import RuntimeEvent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_bytes())


def stream(directory, manifest, payloads):
    events = [("hello", {"synthetic_fixture": True}), *payloads, ("episode_summary", manifest)]
    raw = "".join(
        RuntimeEvent(event=kind, sequence=i, run_id="synthetic-test-only", payload=payload).model_dump_json()
        + "\n"
        for i, (kind, payload) in enumerate(events)
    )
    (directory / "events.jsonl").write_text(raw)
    write(
        directory / "manifest.json", {**manifest, "events_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    )


def snapshot(frame):
    # Fixture setup only: serialize current test controls before switching pages.
    return frame.locator("body").evaluate("""e=>{
      const clone=e.cloneNode(true), old=e.querySelectorAll('input,select');
      [...clone.querySelectorAll('input,select')].forEach((x,i)=>{
        if(x.tagName==='INPUT')x.setAttribute('value',old[i].value);
        else [...x.options].forEach(o=>o.toggleAttribute('selected',o.value===old[i].value));
      });return clone.innerHTML;
    }""")


def stellar_sources(page, frame, history, *, selected_class="main_sequence", inherited_paint=False):
    star = read(history / "choice/confirmed.json")["star"]
    frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", full_html().replace("Althinagon", star))
    frame.evaluate("""()=>window.choose=(e,main)=>{
      document.querySelectorAll('.choice label').forEach(x=>x.classList.remove('selected'));
      e.classList.add('selected');document.querySelector('#conditional').style.display=main?'block':'none';
      document.querySelector('#warning').style.display=main?'none':'block';
    }""")
    if inherited_paint:
        frame.locator(".choice label").nth(module.CLASSES.index(selected_class)).evaluate(
            "e=>e.classList.add('selected')"
        )
    fresh = history / "fresh"
    save_probe(inspect_page(page, config()), fresh / "stellar")
    write(
        fresh / "confirmed.json",
        {
            "star": star,
            "fresh_blank_numeric_answers_verified": True,
            "class_selection_verified": False,
            "answer_writes": 0,
            "action_source": "deterministic_navigation",
            "painted_stellar_class": selected_class if inherited_paint else None,
        },
    )
    synchronize_fresh_stellar_class(
        page, config(), history / "class", fresh_star=fresh, selected_class=selected_class
    )
    main = selected_class == "main_sequence"
    if main:
        frame.locator("#prefix").select_option("Ga")
    save_probe(inspect_page(page, config()), history / "numeric/capture")
    readbacks = {}
    units = (
        {**module.UNITS, "lifetime": "Ga"}
        if main
        else {key: module.UNITS[key] for key in ("distance", "luminosity", "temperature")}
    )
    for name, unit in units.items():
        frame.locator("#" + name).fill("1")
        readbacks[name] = {
            "exact_copied": "1",
            "display_value": "1",
            "unit": unit,
            "exact_input_verified": True,
            "commit_key": "Tab",
            **committed_display("1", "1"),
        }
    final_numeric = module._stellar(inspect_page(page, config()))["observation"]
    numeric = {
        "full_stellar_numeric_transport_verified": True,
        "outcome": "full_stellar_numeric_transport_verified",
        "learned_policy": True,
        "checkpoint_unchanged": True,
        "optimizer_updates": 0,
        "write_attempts": len(units),
        "verified_fields": list(readbacks),
        "numeric_readbacks": readbacks,
        "provenance": {
            "selected_class": selected_class,
            "lifetime_prefix": "Ga" if main else None,
            "fixture_only": True,
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
                "observation": final_numeric,
            },
        )
        for receipt in readbacks.values()
    ]
    stream(history / "numeric", numeric, payloads)
    save_probe(inspect_page(page, config()), history / "color/capture")
    frame.get_by_role("combobox").first.select_option("UV")
    observation = module._stellar(inspect_page(page, config()))["observation"]
    color = {
        "color_transport_verified": True,
        "outcome": "color_transport_verified",
        "write_attempts": 1,
        "receipt": {"selected_color": "UV", "readback_verified": True},
        "provenance": {
            "color_gate_passed": True,
            "optimizer_updates": 0,
            "fixture_only": True,
            "color_checkpoint_sha256": "d" * 64,
            "color_reference_hash": "e" * 64,
        },
    }
    stream(
        history / "color",
        color,
        [
            ("action_proposed", {"action_source": "checkpoint"}),
            (
                "action_result",
                {"color_transport_verified": True, "receipt": color["receipt"], "observation": observation},
            ),
        ],
    )
    return snapshot(frame)


@pytest.fixture
def sources(no_save_page, tmp_path):
    page, frame, _ = no_save_page
    save(no_save_page, tmp_path)
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    return page, frame, {"stellar": stellar, "planet": planet}


def load(history, **overrides):
    options = {
        name + "_dir": history / directory
        for name, directory in (
            ("numeric", "numeric"),
            ("color", "color"),
            ("class", "class"),
            ("choice", "choice"),
            ("save", "save"),
        )
    }
    options.update(overrides)
    return module._load_sources(module._Evidence(history), **options)


def run(page, history, **overrides):
    options = {name + "_dir": history / name for name in ("numeric", "color", "class", "choice", "save")}
    options.update(overrides)
    return module.verify_no_planet_workflow(
        page, config(), history / "workflow", run_history=history, **options
    )


def navigation_seam(monkeypatch, frame, screens, callback=None):
    calls = []

    def navigate(page, config, output, destination, expected_star=None):
        calls.append(destination)
        frame.locator("body").evaluate("(e,h)=>e.innerHTML=h", screens[destination])
        if callback:
            callback(destination)
        return {
            "same_star_verified": True,
            "destination_verified": True,
            "suggested_required_text": [],
            "navigation_clicks": 1,
        }

    monkeypatch.setattr(module, "navigate_project", navigate)
    return calls


def test_fixture_evidence_and_read_only_workflow(sources, tmp_path, monkeypatch):
    page, frame, screens = sources
    # Preserve the original Main-only receipt format, which predates these
    # explicit false flags. Non-main sources are never allowed this omission.
    for filename in ("confirmed.json", "scope.json"):
        path = tmp_path / "class" / filename
        original = read(path)
        original.pop("scientific_verified")
        original.pop("training_label")
        write(path, original)
    bundle = load(tmp_path)
    assert bundle["star"] == "JYREMIS" and bundle["save_click_delivered"]
    assert digest(read_class_choices(frame)[0]) == bundle["class_rendering_sha256"]
    calls = navigation_seam(monkeypatch, frame, screens)
    receipt = run(page, tmp_path)
    assert calls == ["stellar", "planet"]
    assert receipt["task_completed"] and receipt["authority"] == "visible_workflow_readback"
    assert receipt["planet"]["outcome"] == "no_planet"
    assert receipt["planet"]["provenance"] == "reference_prediction"
    assert receipt["habitability"]["branch_applicability_verified"]
    assert not receipt["habitability"]["transport_verified"]
    assert receipt["source_save_click_delivered"]
    assert receipt["save_acknowledgement_source"] == "explicit_save_fresh_visible_footer"
    assert receipt["answer_writes"] == receipt["save_clicks"] == receipt["habitability_writes"] == 0
    for key in (
        "scientific_verified",
        "correctness_verified",
        "project_completed",
        "submitted",
        "browser_acceptance_passed",
        "training_label",
        "cross_session_persistence_verified",
    ):
        assert receipt[key] is False
    assert receipt["source_sha256"] and set(receipt["current_screen_sha256"]) == {"stellar", "planet"}
    assert not page.get_by_role("checkbox").is_checked()


def test_fresh_reserved_notice_policy_keeps_native_save_and_workflow_compatible(
    no_save_page, tmp_path, monkeypatch
):
    page, frame, _ = no_save_page
    saved = save(no_save_page, tmp_path, settle_reserved_notice=True)
    assert saved["settle_reserved_notice"] is True
    assert saved["reserved_notice_policy"] == "bounded_read_only_pre_dispatch_settle_v1"
    assert saved["reserved_phase_timeout_seconds"] == 20
    assert frame.evaluate("window.fixtureSaves") == 1
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    calls = navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    receipt = run(page, tmp_path)
    assert calls == ["stellar", "planet"]
    assert receipt["task_completed"] is True
    assert receipt["source_save_click_delivered"] is True
    assert receipt["save_acknowledgement_source"] == "explicit_save_fresh_visible_footer"
    assert receipt["answer_writes"] == receipt["save_clicks"] == 0
    assert receipt["project_completed"] is False
    assert receipt["correctness_verified"] is False
    assert receipt["scientific_verified"] is False
    assert not page.get_by_role("checkbox").is_checked()


@pytest.mark.parametrize("selected_class", ["red_giant", "supergiant", "white_dwarf"])
@pytest.mark.parametrize("inherited_paint", [False, True])
def test_non_main_native_chain_completes_and_imports_without_conditional_claims(
    no_save_page, tmp_path, monkeypatch, selected_class, inherited_paint
):
    from test_project_evidence import inventory

    from habfly.project_evidence import import_verified_no_planet
    from habfly.project_progress import ProjectJournal

    page, frame, _ = no_save_page
    save(no_save_page, tmp_path)
    planet = snapshot(frame)
    stellar = stellar_sources(
        page, frame, tmp_path, selected_class=selected_class, inherited_paint=inherited_paint
    )
    calls = navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    bundle = load(tmp_path)
    assert bundle["class"] == selected_class and bundle["prefix"] is None
    assert set(bundle["readbacks"]) == {"distance", "luminosity", "temperature"}
    receipt = run(page, tmp_path)
    assert calls == ["stellar", "planet"] and receipt["task_completed"]
    assert receipt["classification"] == selected_class
    assert receipt["numeric_provenance"]["lifetime_prefix"] is None
    assert set(receipt["stellar_fields"]) == {"distance", "luminosity", "temperature"}
    assert receipt["habitability"]["outcome"] == "not_applicable"
    assert receipt["habitability"]["applicability_reason"] == "no_planet"
    for key in (
        "hidden_values_inspected",
        "hidden_values_blank_verified",
        "scientific_verified",
        "training_label",
    ):
        assert receipt[key] is False
    assert not receipt["habitability"]["transport_verified"]
    assert receipt["habitability"]["branch_applicability_verified"]
    assert read(tmp_path / "class/confirmed.json")["class_clicks"] == (2 if inherited_paint else 1)
    inv = inventory(tmp_path, [receipt["star"]], "current", classes={receipt["star"]: selected_class})
    journal = ProjectJournal(tmp_path, project_id="fixture-only", attempt_id="non-main").create()
    result = import_verified_no_planet(journal, tmp_path, inv, tmp_path / "workflow")
    state = journal.load().reduce()
    star = state.stars[result["star_id"]]
    assert result["progress"]["verified"] == 1 and star.task_completed
    assert star.stellar_classification.decision.value == selected_class
    assert star.stellar_classification.decision.provenance == "reference_prediction"
    assert star.stellar_numeric.decision.provenance == "learned_prediction"
    assert star.habitability.branch_applicability_verified and not star.habitability.transport_verified
    assert not state.reservations and not state.receipts and not result["progress"]["project_completed"]
    before = journal.path.read_bytes()
    assert import_verified_no_planet(journal, tmp_path, inv, tmp_path / "workflow")["idempotent"]
    assert journal.path.read_bytes() == before


def test_corrupt_evidence_never_navigates(sources, tmp_path, monkeypatch):
    page, frame, screens = sources
    calls = navigation_seam(monkeypatch, frame, screens)
    for index, name in enumerate(
        (
            "numeric/events.jsonl",
            "color/manifest.json",
            "class/after/observation.json",
            "fresh/stellar/observation.json",
            "choice/preselect-progress/chart.png",
            "save/acknowledgement.json",
            "save/before/manifest.json",
        )
    ):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(b"{}")
        with pytest.raises((BrowserSafetyStop, ValueError, KeyError)):
            load(tmp_path)
        with pytest.raises(BrowserSafetyStop):
            module.verify_no_planet_workflow(
                page,
                config(),
                tmp_path / f"workflow-{index}",
                run_history=tmp_path,
                **{key + "_dir": tmp_path / key for key in ("numeric", "color", "class", "choice", "save")},
            )
        path.write_bytes(original)
        assert calls == []
    assert load(tmp_path)["star"] == "JYREMIS"


def test_forged_semantics_rejected_even_with_rehashed_events(sources, tmp_path):
    mutations = [
        ("numeric", "learned_policy", False),
        ("numeric", "write_attempts", 5),
        ("numeric", "checkpoint_unchanged", False),
        ("numeric", "optimizer_updates", True),
        ("color", "color_transport_verified", False),
        ("color", "write_attempts", 2),
    ]
    for directory, key, bad in mutations:
        manifest_path, events_path = (
            tmp_path / directory / "manifest.json",
            tmp_path / directory / "events.jsonl",
        )
        old_manifest, old_events = manifest_path.read_bytes(), events_path.read_bytes()
        manifest = read(manifest_path)
        manifest.pop("events_sha256")
        manifest[key] = bad
        events = [json.loads(line) for line in old_events.splitlines()]
        stream(tmp_path / directory, manifest, [(e["event"], e["payload"]) for e in events[1:-1]])
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path)
        manifest_path.write_bytes(old_manifest)
        events_path.write_bytes(old_events)
    for path, key, bad in [
        (tmp_path / "class/confirmed.json", "learned_classification", True),
        (tmp_path / "save/confirmed.json", "cross_session_persistence_verified", True),
        (tmp_path / "save/confirmed.json", "has_planet", "Yes"),
    ]:
        raw = path.read_bytes()
        data = read(path)
        data[key] = bad
        write(path, data)
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path)
        path.write_bytes(raw)


@pytest.mark.parametrize(
    "mutation", ["number", "unit", "color", "class", "planet_zero", "outer", "source_changed"]
)
def test_current_mismatch_stops_without_repair(sources, tmp_path, monkeypatch, mutation):
    page, frame, screens = sources

    def change(section):
        if section == "stellar":
            if mutation == "number":
                frame.locator("#distance").fill("2")
            elif mutation == "unit":
                frame.locator("#prefix").select_option("Ma")
            elif mutation == "color":
                frame.get_by_role("combobox").first.select_option("red")
            elif mutation == "class":
                frame.locator(".choice label").first.evaluate("e=>e.classList.remove('selected')")
            elif mutation == "source_changed":
                path = tmp_path / "numeric/events.jsonl"
                path.write_bytes(path.read_bytes() + b" ")
        elif mutation == "planet_zero":
            frame.locator("#period_days").fill("0")
        elif mutation == "outer":
            page.get_by_role("checkbox").check()

    calls = navigation_seam(monkeypatch, frame, screens, change)
    with pytest.raises(BrowserSafetyStop):
        run(page, tmp_path)
    assert not (tmp_path / "workflow/confirmed.json").exists()
    stopped = read(tmp_path / "workflow/stopped.json")
    assert stopped["answer_writes"] == 0 and not stopped["task_completed"] and not stopped["automatic_retry"]
    assert calls in (["stellar"], ["stellar", "planet"])


@pytest.mark.parametrize("continued", [False, True])
def test_no_click_reconciled_or_same_intent_acknowledgement_is_distinct(
    no_save_page, tmp_path, monkeypatch, continued
):
    import habfly.browser_no_planet_save as save_module

    page, frame, options = no_save_page
    original = save_module.persist_json

    def interrupt(path, value):
        original(path, value)
        if path.name == "reserved.json":
            frame.locator("#save-notice").evaluate("e=>e.textContent='Data saved'")

    monkeypatch.setattr(save_module, "persist_json", interrupt)
    with pytest.raises(BrowserSafetyStop, match="stale_acknowledgement"):
        save_no_planet_work(page, config(), tmp_path / "failed-save", **options)
    monkeypatch.setattr(save_module, "persist_json", original)
    if continued:
        frame.locator("#save-notice").evaluate("e=>e.textContent=''")
        resume_no_planet_save(page, config(), tmp_path / "failed-save", run_history=tmp_path)
        save_dir = tmp_path / "failed-save/resume"
    else:
        reconcile_no_planet_autosave(
            page, config(), tmp_path / "failed-save", tmp_path / "save", run_history=tmp_path
        )
        save_dir = tmp_path / "save"
    assert frame.evaluate("window.fixtureSaves") == int(continued)
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    calls = navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    receipt = run(page, tmp_path, save_dir=save_dir)
    assert calls == ["stellar", "planet"] and receipt["task_completed"]
    assert receipt["source_save_click_delivered"] is continued
    assert receipt["source_save_resumed_same_intent"] is continued
    assert receipt["save_acknowledgement_source"] == (
        "explicit_save_fresh_visible_footer" if continued else "visible_footer_no_explicit_save_dispatch"
    )
    assert read(tmp_path / "failed-save/stopped.json")["save_may_have_occurred"] is False
    if continued:
        path = save_dir / "dispatch.json"
        original = path.read_bytes()
        write(path, {"kind": "CLICK"})
        with pytest.raises(BrowserSafetyStop, match="unconfirmed_same_intent_continuation"):
            load(tmp_path, save_dir=save_dir)
        path.write_bytes(original)


def test_evidence_owned_paths_and_change_detection(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    evidence = module._Evidence(history)
    write(history / "source.json", {})
    evidence.read(history / "source.json")
    write(history / "source.json", {"changed": True})
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        evidence.unchanged()
    with pytest.raises(BrowserSafetyStop, match="outside_history"):
        evidence.read(tmp_path / "outside.json")
    (history / "link.json").symlink_to(history / "source.json")
    with pytest.raises(BrowserSafetyStop, match="symlink"):
        evidence.read(history / "link.json")
    directory = history / "completed"
    directory.mkdir()
    stable = module._Evidence(history)
    stable.clean(directory)
    write(directory / "invalidated.json", {})
    with pytest.raises(BrowserSafetyStop, match="failed_evidence"):
        stable.unchanged()


def test_revisited_no_click_workflow_recomputes_view_exception_and_preserves_source(
    revisit_no_save_page, tmp_path, monkeypatch
):
    page, frame, _ = revisit_no_save_page
    predispatched(revisit_no_save_page, tmp_path, monkeypatch)
    apply_revisit(frame)
    review = tmp_path / "review"
    reconciliation = reconcile_no_planet_autosave(
        page, config(), tmp_path / "save", review, run_history=tmp_path, timeout_seconds=0
    )
    assert len(reconciliation["historical_view_equivalence"]["differences"]) == 3
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    receipt = run(page, tmp_path, save_dir=review)
    assert receipt["task_completed"] and not receipt["source_save_click_delivered"]
    assert not receipt["scientific_verified"] and not receipt["project_completed"]
    assert not receipt["habitability"]["transport_verified"]
    assert receipt["habitability"]["branch_applicability_verified"]
    for change in ("missing", "changed", "invented_science", "numeric_boolean"):
        changed = deepcopy(reconciliation)
        if change == "missing":
            changed.pop("historical_view_equivalence")
        elif change == "changed":
            changed["historical_view_equivalence"]["differences"].pop()
        elif change == "invented_science":
            changed["historical_view_equivalence"]["scientific_verified"] = True
        else:
            changed["historical_view_equivalence"]["scientific_verified"] = 0
        # Mutating both summaries cannot bypass recomputation from original captures.
        write(review / "confirmed.json", changed)
        disposition = read(review / "disposition.json")
        disposition.pop("historical_view_equivalence", None)
        if "historical_view_equivalence" in changed:
            disposition["historical_view_equivalence"] = changed["historical_view_equivalence"]
        write(review / "disposition.json", disposition)
        with pytest.raises(BrowserSafetyStop, match="view_equivalence_mismatch"):
            load(tmp_path, save_dir=review)


def test_reconciled_consumed_continuation_requires_all_no_dispatch_evidence(
    no_save_page, tmp_path, monkeypatch
):
    page, frame, _ = no_save_page
    consumed_predispatched(no_save_page, tmp_path, monkeypatch)
    reconcile_no_planet_autosave(
        page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
    )
    planet = snapshot(frame)
    stellar = stellar_sources(page, frame, tmp_path)
    calls = navigation_seam(monkeypatch, frame, {"stellar": stellar, "planet": planet})
    receipt = run(page, tmp_path, save_dir=tmp_path / "review")
    assert calls == ["stellar", "planet"] and receipt["task_completed"]
    assert not receipt["source_save_click_delivered"]
    assert not receipt["source_save_resumed_same_intent"]
    assert receipt["source_save_continuation_stopped_predispatch"]
    assert receipt["save_acknowledgement_source"] == "visible_footer_no_explicit_save_dispatch"
    assert not (tmp_path / "save/dispatch.json").exists()
    assert not (tmp_path / "save/resume/dispatch.json").exists()
    expected = read(tmp_path / "review/confirmed.json")
    for mutation in ("omitted", "missing_hash", "numeric_false", "native_click"):
        changed = deepcopy(expected)
        if mutation == "omitted":
            del changed["continuation_disposition"]
        elif mutation == "missing_hash":
            changed["continuation_disposition"]["source_sha256"].pop("save/resume/stopped.json")
        elif mutation == "numeric_false":
            changed["continuation_disposition"]["save_click_dispatched"] = 0
        else:
            changed["save_click_delivered"] = True
        write(tmp_path / "review/confirmed.json", changed)
        with pytest.raises(BrowserSafetyStop):
            load(tmp_path, save_dir=tmp_path / "review")
    write(tmp_path / "review/confirmed.json", expected)
    evidence = module._Evidence(tmp_path)
    module._load_sources(
        evidence,
        **{key + "_dir": tmp_path / key for key in ("numeric", "color", "class", "choice")},
        save_dir=tmp_path / "review",
    )
    write(tmp_path / "save/resume/dispatch.json", {"kind": "CLICK"})
    with pytest.raises(BrowserSafetyStop, match="continuation_disposition"):
        load(tmp_path, save_dir=tmp_path / "review")
    # Sources already consumed by the verifier remain hash-monitored too.
    path = tmp_path / "save/resume/stopped.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(BrowserSafetyStop, match="evidence_changed"):
        evidence.unchanged()


def test_legacy_reconciliation_without_continuation_stays_compatible(no_save_page, tmp_path, monkeypatch):
    from test_browser_no_planet_save import predispatched

    page, frame, _ = no_save_page
    predispatched(no_save_page, tmp_path, monkeypatch)
    reconcile_no_planet_autosave(
        page, config(), tmp_path / "save", tmp_path / "review", run_history=tmp_path, timeout_seconds=0
    )
    stellar_sources(page, frame, tmp_path)
    # Old receipts did not have this optional key. They remain valid only when
    # there genuinely is no continuation claim or directory to account for.
    for filename in ("confirmed.json", "disposition.json"):
        value = read(tmp_path / "review" / filename)
        value.pop("continuation_disposition")
        value.pop("notice_wait_seconds")
        write(tmp_path / "review" / filename, value)
    assert not load(tmp_path, save_dir=tmp_path / "review")["save_continuation_stopped_predispatch"]
    write(tmp_path / "save/resume-reserved.json", {"incomplete": True})
    with pytest.raises(BrowserSafetyStop, match="continuation_disposition"):
        load(tmp_path, save_dir=tmp_path / "review")


def test_non_main_prefix_none_and_reference_visibility_are_not_zero():
    mapping = {
        "star_name": "Example",
        "observation": {
            "values": {
                "measurements": {},
                "browser_field_map": {"distance": {"current_value": "1", "unit": "ly", "value_known": True}},
                "conditional_fields_visible": False,
                "lifetime_prefix": None,
                "color": {"selected": "red"},
            }
        },
    }
    bundle = {
        "star": "Example",
        "measurements": {},
        "readbacks": {"distance": {"display_value": "1", "unit": "ly"}},
        "class": "red_giant",
        "prefix": None,
        "color": "red",
    }
    module._values_match(mapping, bundle)
    changed = deepcopy(mapping)
    changed["observation"]["values"]["browser_field_map"]["distance"]["current_value"] = "0"
    with pytest.raises(BrowserSafetyStop, match="readback_changed"):
        module._values_match(changed, bundle)
