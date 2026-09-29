"""Injected/pure gates only; synthetic native evidence is not live acceptance."""

import hashlib
import json
import socket
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_browser_numeric import OUTER, config
from test_project_paginated_inventory import capture

import habfly.browser_project_paginated_inventory_steps as module
from habfly.browser import BrowserSafetyStop
from habfly.browser_numeric import screen_identity
from habfly.browser_stellar import SIMULATION_URL
from habfly.project_paginated_inventory import PaginatedInventoryError


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def list_footer(report, *, notice):
    """Synthetic version of the paired text/AX footer observed in probe340."""
    value = deepcopy(report)
    frame = value["frames"][0]
    counts = module._parse(value)[0]
    footer = (
        f"viewing {counts['start']}-{counts['end']} of {counts['total']} total collected {counts['total']}"
    )
    rendered = (
        f" \n \nVIEWING\n{counts['start']}-{counts['end']} OF {counts['total']}"
        f"\n \nTOTAL COLLECTED\n{counts['total']}\nSave"
    )
    frame["text"] = frame["text"].replace(footer, ("Data saved\n" if notice else "") + rendered)
    frame["accessibility"] = frame["accessibility"].rstrip() + '\n- button "Save"'
    if notice:
        frame["accessibility"] = frame["accessibility"].replace(
            "\n- button", "\n- text: Data saved\n- button", 1
        )
    return value


def pair(counts):
    # Footer text really moves the pair horizontally at page transitions.
    left = 323.3359375 if counts["start"] == 1 else 318.8828125
    return [
        {
            "index": i,
            "target_id": f"simulation-0:c{i}",
            "box": {"x": left + 34.453125 * i, "y": 555, "width": 30, "height": 30},
            "enabled": enabled,
            "native_button": True,
            "exposed": True,
        }
        for i, enabled in enumerate((counts["start"] > 1, counts["end"] < counts["total"]))
    ]


@pytest.fixture
def subject(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("Network forbidden"))
    state = SimpleNamespace(
        names=[f"Star{i:02d}" for i in range(21)],
        index=0,
        clock=0,
        reads=0,
        callbacks={},
        events=[],
        clicks=[],
        native_generation=0,
        handles=None,
        cancelled=False,
        read_hook=None,
        click_hook=None,
        unknown_modal=False,
        changed_total=False,
        row_overrides={},
        overlay=False,
        dispatch_error=False,
    )
    main = object()
    frame = SimpleNamespace(url=SIMULATION_URL, parent_frame=main)
    page = SimpleNamespace(url=OUTER, main_frame=main, frames=[main, frame])
    page.context = SimpleNamespace(pages=[page])
    page.on = lambda kind, callback: state.callbacks.update({kind: callback})
    page.remove_listener = lambda kind, _callback: state.callbacks.pop(kind)
    page.context.on = lambda kind, callback: state.callbacks.update({"context-" + kind: callback})
    page.context.remove_listener = lambda kind, _callback: state.callbacks.pop("context-" + kind)
    page.wait_for_timeout = lambda _: None

    class Handle:
        def __init__(self, index):
            self.index, self.generation = index, state.native_generation

        def evaluate(self, script, other):
            assert script == "(a,b)=>a.isConnected&&a===b"
            return self is other and self.generation == state.native_generation

        def click(self, *, timeout):
            assert 0 < timeout <= 3000
            owner = state.owner
            assert (owner.output / "transitions" / f"{owner.attempts:02d}" / "reserved.json").exists()
            state.clicks.append(self.index)
            if state.click_hook:
                state.click_hook()
            if state.dispatch_error:
                raise RuntimeError("PRIVATE COOKIE / SESSION ERROR")
            state.index += 1 if self.index == 1 else -1
            state.native_generation += 1
            state.handles = None

    def current():
        names = state.names[state.index * 10 : state.index * 10 + 10]
        report = capture(names, state.index * 10 + 1, len(state.names))
        if state.changed_total:
            report["frames"][0]["text"] = report["frames"][0]["text"].replace("collected 21", "collected 22")
        for old, new in state.row_overrides.items():
            for field in ("text", "accessibility"):
                report["frames"][0][field] = report["frames"][0][field].replace(old, new)
        report["captured_at"] = (
            datetime(2026, 9, 26, tzinfo=UTC) + timedelta(seconds=state.reads)
        ).isoformat()
        return report

    def bind(_frame, report, counts):
        if state.overlay:
            raise BrowserSafetyStop("paginated_inventory_live_pager_occluded")
        controls = pair(counts)
        module._geometry(controls, counts)
        if state.handles is None:
            state.handles = [Handle(i) for i in range(2 + counts["end"] - counts["start"] + 1)]
        return controls, state.handles[:2]

    def injected_read(owner):
        # Explicit native-transport seam: real capture parser/source validator,
        # but deterministic page/geometry/handles instead of a browser launch.
        owner._context()
        state.reads += 1
        if state.read_hook:
            state.read_hook(state.reads)
        if state.unknown_modal:
            raise BrowserSafetyStop("unexpected_modal")
        report = current()
        counts, rows, boundary, row_sha = module._parse(report)
        if owner._boundary is None:
            owner._boundary = boundary
        module._require(
            boundary == owner._boundary and counts["total"] == len(owner.scope["expected_stars"]),
            "collection_or_boundary_changed",
        )
        controls, _ = bind(frame, report, counts)
        owner._frame = frame
        return {
            "report": report,
            "counts": counts,
            "rows": [
                {**r, "visible_name_box": {"x": 392, "y": 320 + i * 25, "width": 80, "height": 16}}
                for i, r in enumerate(rows)
            ],
            "pager": controls,
            "handles": state.handles,
            "row_sha": row_sha,
        }

    monkeypatch.setattr(module.PaginatedInventorySteps, "_read", injected_read)
    monkeypatch.setattr(module, "_pager", bind)
    return SimpleNamespace(root=tmp_path, state=state, page=page, current=current, Handle=Handle)


def create(subject, **options):
    state = subject.state
    owner = module.PaginatedInventorySteps(
        subject.page,
        config(),
        subject.root / options.pop("output", "inventory"),
        run_history=subject.root,
        expected_stars=options.pop("expected_stars", state.names),
        emit=options.pop("emit", state.events.append),
        cancelled=lambda: state.cancelled,
        _clock=lambda: state.clock,
        **options,
    )
    state.owner = owner
    return owner


def complete(subject, **options):
    owner = create(subject, **options)
    for _ in range(8):
        if owner.finished:
            break
        owner.advance()
    assert owner.status == "completed", owner.state()
    return owner


@pytest.mark.parametrize("total", [1, 9, 10, 11, 20, 21, 30])
def test_complete_forward_reverse_and_anchor_owned_proof(subject, total):
    subject.state.names = [f"Star{i:02d}" for i in range(total)]
    owner = complete(subject)
    pages = math_ceil_pages = (total + 9) // 10
    assert owner.advances == 2 * math_ceil_pages + 1
    assert subject.state.clicks == [1] * (pages - 1) + [0] * (pages - 1)
    assert subject.state.index == 0 and not subject.state.callbacks
    receipt, rows, anchor = module.load_live_paginated_inventory(subject.root, owner.output)
    assert receipt == owner.report
    assert [r["name"] for r in rows] == subject.state.names
    assert set(rows[0]) == {"name", "text"}
    assert module.parse_inventory_page(anchor)[0] == {"start": 1, "end": min(10, total), "total": total}
    assert receipt["anchor_capture"] == "inventory/anchor-01/after"
    assert receipt["complete_visible_list_verified"] is False
    assert receipt["complete_collection_traversal_verified"] and receipt["live_pagination_verified"]
    assert receipt["whole_collection_sha256"] != receipt["visible_assessment_anchor_sha256"]
    assert all(receipt[k] == 0 for k in module.ZERO)
    assert all(receipt[k] is False for k in module.FALSE)
    for path, expected in receipt["source_sha256"].items():
        assert hashlib.sha256((subject.root / path).read_bytes()).hexdigest() == expected
    assert "inventory/anchor-01/after" in receipt["validated_directories"]
    assert not read(owner.output / "consistency.json")["live_pagination_verified"]
    saved_events = [json.loads(line) for line in (owner.output / "events.jsonl").read_text().splitlines()]
    assert saved_events == subject.state.events
    count = len(subject.state.clicks)
    owner.advance()
    owner.resume()
    owner.tick()
    owner.close()
    assert len(subject.state.clicks) == count


def test_constructor_is_page_free_and_each_advance_has_at_most_one_click(subject):
    owner = create(subject)
    assert subject.state.reads == 0 and subject.state.clicks == []
    assert not owner.tick()["finished"] and subject.state.reads == 0
    for _ in range(7):
        before = len(subject.state.clicks)
        owner.advance()
        assert len(subject.state.clicks) - before <= 1
    assert owner.finished and owner.report["total_collected"] == 21


@pytest.mark.parametrize("effect", ["fades", "persists", "cancel", "popup", "deadline", "changed_rows"])
def test_autosave_settle_is_read_only_bounded_and_context_checked(subject, monkeypatch, effect):
    owner = create(subject, max_seconds=1 if effect == "deadline" else 240)
    before = list_footer(subject.current(), notice=True)
    after = list_footer(subject.current(), notice=False)
    if effect == "changed_rows":
        after["frames"][0]["text"] = after["frames"][0]["text"].replace("0.045", "0.046")
    original = deepcopy(before)
    captures, waits = [], []

    def inspect(*_):
        result = before if not waits or effect in {"persists", "deadline"} else after
        captures.append(result)
        return result

    def wait(ms):
        if not ms:
            return
        waits.append(ms)
        subject.state.clock += ms / 1000
        if effect == "cancel":
            subject.state.cancelled = True
        elif effect == "popup":
            subject.state.callbacks["context-page"](None)

    monkeypatch.setattr(module, "inspect_page", inspect)
    subject.page.wait_for_timeout = wait
    if effect in {"fades", "changed_rows"}:
        result = owner._settled_report()
        assert result == after and waits == [100]
        assert screen_identity(before) != screen_identity(result)  # No identity relaxation.
        if effect == "changed_rows":
            assert "0.046" in result["frames"][0]["text"]  # Never repaired.
    else:
        with pytest.raises(BrowserSafetyStop):
            owner._settled_report()
        assert 0 < sum(waits) <= 3001
    assert before == original and not subject.state.clicks
    assert not (owner.output / "forward-01").exists()
    owner.close()


@pytest.mark.parametrize(
    "change",
    [
        "unpaired_text",
        "unpaired_ax",
        "misplaced",
        "message",
        "duplicate",
        "row",
        "score",
        "funding",
        "control",
        "outer",
    ],
)
def test_autosave_recognition_never_changes_evidence_or_accepts_unpaired_notice(change):
    baseline = list_footer(capture(["Star01"], 1, 1), notice=False)
    notice = list_footer(capture(["Star01"], 1, 1), notice=True)
    original = deepcopy(notice)
    assert module.paired_footer_autosave(notice)
    assert not module.paired_footer_autosave(baseline)
    assert notice == original
    frame = notice["frames"][0]
    if change == "unpaired_text":
        frame["text"] = baseline["frames"][0]["text"]
    elif change == "unpaired_ax":
        frame["accessibility"] = baseline["frames"][0]["accessibility"]
    elif change == "misplaced":
        frame["text"] = "Data saved\n" + frame["text"].replace("Data saved\n", "")
    elif change == "message":
        frame["text"] = frame["text"].replace("Data saved", "Save failed")
        frame["accessibility"] = frame["accessibility"].replace("Data saved", "Save failed")
    elif change == "duplicate":
        frame["text"] += "\nData saved"
    elif change in {"row", "score", "funding"}:
        old, new = {
            "row": ("0.045", "0.046"),
            "score": ("Quality 0%", "Quality 1%"),
            "funding": ("$50000", "$49900"),
        }[change]
        for field in ("text", "accessibility"):
            frame[field] = frame[field].replace(old, new)
    elif change == "control":
        frame["controls"][0]["enabled"] = True
    else:
        notice["outer_url"] = "http://localhost/elsewhere"
    assert module.screen_identity(notice) != module.screen_identity(baseline)
    if change in {"unpaired_text", "unpaired_ax", "misplaced", "message", "duplicate"}:
        assert not module.paired_footer_autosave(notice)


@pytest.mark.parametrize("start", [1, 2])
def test_mid_page_start_never_normalizes_or_clicks(subject, start):
    subject.state.index = start
    owner = create(subject)
    owner.advance()
    assert owner.finished and not subject.state.clicks
    assert owner.failure == "paginated_inventory_live_unexpected_page_transition"


@pytest.mark.parametrize("value", [[], ["A"], ["Star01", "star01"], ["Star01", " Star02"], ["S00"] * 31])
def test_expected_names_are_exact_bounded_set_without_journal(subject, value):
    with pytest.raises(BrowserSafetyStop, match="expected_stars"):
        create(subject, expected_stars=value)
    assert subject.state.reads == 0


def test_same_count_wrong_expected_name_fails_complete_proof(subject):
    expected = [*subject.state.names[:-1], "Wrong"]
    owner = create(subject, expected_stars=expected)
    for _ in range(7):
        owner.advance()
    assert owner.finished and owner.status == "stopped"
    assert not (owner.output / "confirmed.json").exists()


@pytest.mark.parametrize("moment", ["last_page", "after_return"])
def test_noncurrent_first_page_mutation_is_detected_on_reverse(subject, moment):
    owner = create(subject)
    for _ in range(3):
        owner.advance()
    assert subject.state.index == 2
    if moment == "after_return":
        owner.advance()  # recapture last page
        owner.advance()  # return to page2
    subject.state.row_overrides["Star00 0.045"] = "Star00 0.046"
    for _ in range(5):
        owner.advance()
    assert owner.finished and owner.status == "stopped"
    assert not (owner.output / "confirmed.json").exists()


@pytest.mark.parametrize("effect", ["overlay", "replace", "abort", "callback_error", "modal", "navigation"])
def test_predispatch_revalidation_prevents_wrong_native_click(subject, effect):
    def callback(event):
        subject.state.events.append(event)
        if event["event"] != "action_proposed":
            return
        if effect == "overlay":
            subject.state.overlay = True
        elif effect == "replace":
            subject.state.native_generation += 1
            subject.state.handles = None
        elif effect == "abort":
            subject.state.owner.abort()
        elif effect == "callback_error":
            raise RuntimeError("PRIVATE CREDENTIALS")
        elif effect == "modal":
            subject.state.unknown_modal = True
        else:
            subject.page.url = "http://localhost/other"

    owner = create(subject, emit=callback)
    owner.advance()
    owner.advance()
    assert owner.finished and subject.state.clicks == []
    assert not (owner.output / "confirmed.json").exists()
    assert "PRIVATE" not in (owner.output / "events.jsonl").read_text()


def test_uncertain_click_is_reserved_and_no_automatic_retry(subject):
    owner = create(subject)
    owner.advance()
    subject.state.dispatch_error = True
    owner.advance()
    assert owner.finished and owner.state()["navigation_outcome_uncertain"]
    assert (owner.output / "transitions/01/reserved.json").exists()
    assert not (owner.output / "transitions/01/confirmed.json").exists()
    owner.advance()
    assert subject.state.clicks == [1]


@pytest.mark.parametrize("budget", [{"max_clicks": 1}, {"max_advances": 2}, {"max_seconds": 1}])
def test_fixed_budgets_never_expand(subject, budget):
    owner = create(subject, **budget)
    owner.advance()
    if "max_seconds" in budget:
        owner.pause()
        subject.state.clock = 1
    for _ in range(8):
        owner.advance()
    assert owner.finished and owner.status == "stopped"
    assert not (owner.output / "confirmed.json").exists()


@pytest.mark.parametrize("advance", [0, 1, 2, 3, 4, 5, 6])
def test_abort_each_boundary_is_sticky(subject, advance):
    owner = create(subject)
    for _ in range(advance):
        owner.advance()
    clicks = list(subject.state.clicks)
    owner.abort()
    for _ in range(3):
        owner.advance()
        owner.resume()
        owner.tick()
    assert owner.finished and owner.status == "aborted" and subject.state.clicks == clicks


@pytest.mark.parametrize(
    "what", ["receipt_flag", "receipt_text", "footer", "native", "transition", "event", "stopped"]
)
def test_loader_recomputes_exact_proof_and_rejects_mutation(subject, what):
    owner = complete(subject)
    if what in {"receipt_flag", "receipt_text"}:
        path = owner.output / "confirmed.json"
        data = read(path)
        if what == "receipt_flag":
            data["atomic_server_snapshot_verified"] = True
        else:
            data["rows"][0]["text"] += " forged"
        write(path, data)
    elif what == "footer":
        path = owner.output / "anchor-01/after/observation.json"
        data = read(path)
        data["frames"][0]["text"] += " viewing 1-10 of 21"
        write(path, data)
    elif what == "native":
        path = owner.output / "forward-01/native.json"
        data = read(path)
        data["pager"][0]["box"]["x"] += 10
        write(path, data)
    elif what == "transition":
        path = owner.output / "transitions/01/reserved.json"
        data = read(path)
        data["direction"] = "previous"
        write(path, data)
    elif what == "event":
        with (owner.output / "events.jsonl").open("a") as stream:
            stream.write("{}\n")
    else:
        write(owner.output / "revisit-02/stopped.json", {"failure": True})
    with pytest.raises((BrowserSafetyStop, ValueError)):
        module.load_live_paginated_inventory(subject.root, owner.output)


@pytest.mark.parametrize(
    "change", ["range", "missing_row", "blank_malformed", "negative", "unknown_class", "duplicate"]
)
def test_public_page_parser_rejects_malformed_or_missing_evidence(change):
    value = capture([f"Star{i:02d}" for i in range(10)], 1, 11)
    frame = value["frames"][0]
    if change == "range":
        frame["text"] = frame["text"].replace("1-10", "1-9")
    elif change == "missing_row":
        frame["accessibility"] = frame["accessibility"].replace("Star00 0.045", "Missing")
    else:
        replacement = {
            "blank_malformed": ("1 1 UV 1 Main Sequence 1 1 1 Ga", "- -"),
            "negative": ("0.045 370", "-0.045 370"),
            "unknown_class": ("Main Sequence", "Unknown Class"),
            "duplicate": ("Star01", "Star00"),
        }[change]
        frame["accessibility"] = frame["accessibility"].replace(*replacement)
    with pytest.raises((BrowserSafetyStop, PaginatedInventoryError)):
        module.parse_inventory_page(value)


@pytest.mark.parametrize(
    "change", ["wrong_size", "wrong_y", "wrong_gap", "reordered", "enabled", "third", "tag", "covered"]
)
def test_exact_observed_footer_pair_geometry_is_not_general_button_authority(change):
    counts = {"start": 1, "end": 10, "total": 11}
    controls = pair(counts)
    if change == "wrong_size":
        controls[0]["box"]["width"] = 40
    elif change == "wrong_y":
        controls[0]["box"]["y"] -= 100
    elif change == "wrong_gap":
        controls[1]["box"]["x"] += 3
    elif change == "reordered":
        controls.reverse()
    elif change == "enabled":
        controls[0]["enabled"] = True
    elif change == "third":
        controls.append(deepcopy(controls[1]))
    elif change == "tag":
        controls[0]["native_button"] = False
    else:
        controls[0]["exposed"] = False
    with pytest.raises(BrowserSafetyStop):
        module._geometry(controls, counts)


def test_actual_grounded_forward_and_backward_capture_bytes_parse_unchanged():
    root = Path(__file__).resolve().parents[1] / "experiments/full-stellar-probe/20260926-005"
    if not (root / "pager-transition-335/after/observation.json").exists():
        pytest.skip("Optional local read-only development capture is not distributed")
    sources = [
        root / d / "observation.json"
        for d in (
            "list-pager-332/capture",
            "pager-transition-333/after",
            "list-pager-334/capture",
            "pager-transition-335/after",
        )
    ]
    before = {path: path.read_bytes() for path in sources}
    parsed = [module.parse_inventory_page(json.loads(raw)) for raw in before.values()]
    assert [p[0] for p in parsed] == [
        {"start": 1, "end": 10, "total": 11},
        {"start": 11, "end": 11, "total": 11},
        {"start": 11, "end": 11, "total": 11},
        {"start": 1, "end": 10, "total": 11},
    ]
    assert parsed[0] == parsed[3] and parsed[1] == parsed[2]
    assert parsed[1][1][0]["name"] == "Gamor" and parsed[1][1][0]["text"].endswith("- - - - - - - -")
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_live_parser_preserves_legacy_partial_page_rejection():
    from habfly.browser_project_inventory import _capture_inventory

    report = capture([f"Star{i:02d}" for i in range(10)], 1, 11)
    assert module.parse_inventory_page(report)[0]["total"] == 11
    with pytest.raises(BrowserSafetyStop, match="incomplete_visible_list"):
        _capture_inventory(report)


def test_returned_anchor_is_real_first_page_not_concatenated_rows(subject):
    owner = complete(subject)
    _, _, anchor = module.load_live_paginated_inventory(subject.root, owner.output)
    assert screen_identity(anchor) == screen_identity(read(owner.output / "anchor-01/after/observation.json"))
    assert "Star20" not in anchor["frames"][0]["text"]


@pytest.mark.parametrize("effect", ["abort", "source_mutation", "modal", "detached"])
def test_after_reservation_guards_leave_uncertainty_without_native_dispatch(subject, monkeypatch, effect):
    owner = create(subject)
    owner.advance()
    original = module.persist_json

    def persist(path, payload):
        original(path, payload)
        if path.name != "reserved.json":
            return
        if effect == "abort":
            owner.abort()
        elif effect == "source_mutation":
            (owner.output / "forward-01/after/observation.json").write_text("{}")
        elif effect == "modal":
            subject.state.callbacks["dialog"](object())
        else:
            subject.state.native_generation += 1
            subject.state.handles = None

    monkeypatch.setattr(module, "persist_json", persist)
    owner.advance()
    assert owner.finished and not subject.state.clicks
    assert (owner.output / "transitions/01/reserved.json").exists()
    assert owner.state()["navigation_outcome_uncertain"]
    assert not (owner.output / "confirmed.json").exists()


def test_duplicate_name_across_distinct_pages_never_certifies_collection(subject):
    subject.state.row_overrides["Star20"] = "Star00"
    owner = create(subject)
    for _ in range(8):
        owner.advance()
    assert owner.status == "stopped" and not (owner.output / "confirmed.json").exists()


def test_final_callback_abort_cannot_publish_confirmed_inventory(subject):
    def callback(event):
        if event["event"] == "episode_summary" and event["payload"].get("status") == "completed":
            subject.state.owner.abort()

    owner = create(subject, emit=callback)
    for _ in range(8):
        owner.advance()
    assert owner.status == "aborted" and not (owner.output / "confirmed.json").exists()


def test_reentrant_event_callback_cannot_dispatch(subject):
    def callback(event):
        if event["event"] == "action_proposed":
            subject.state.owner.advance()

    owner = create(subject, emit=callback)
    owner.advance()
    owner.advance()
    assert owner.finished and not subject.state.clicks


@pytest.mark.parametrize("change", ["mode", "live_flag", "hash_map", "whole_digest", "transition_target"])
def test_self_consistent_rehashed_declarations_still_recompute_proof(subject, change):
    owner = complete(subject)
    path = owner.output / "confirmed.json"
    receipt = read(path)
    if change == "mode":
        receipt["mode"] = "offline_paginated_inventory_consistency"
    elif change == "live_flag":
        receipt["live_pagination_verified"] = False
    elif change == "hash_map":
        receipt["source_sha256"].pop("inventory/anchor-01/after/manifest.json")
    elif change == "whole_digest":
        receipt["whole_collection_sha256"] = "f" * 64
    else:
        transition = subject.root / receipt["transitions"][0]["path"]
        altered = read(transition)
        altered["target"]["box"]["x"] += 10
        write(transition, altered)
        receipt["transitions"][0]["sha256"] = hashlib.sha256(transition.read_bytes()).hexdigest()
    write(path, receipt)
    with pytest.raises((BrowserSafetyStop, ValueError)):
        module.load_live_paginated_inventory(subject.root, owner.output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_clicks", True),
        ("max_clicks", 5),
        ("max_advances", 9),
        ("max_advances", 1.0),
        ("max_seconds", 0),
        ("max_seconds", 601),
        ("max_seconds", float("nan")),
        ("max_seconds", float("inf")),
    ],
)
def test_declared_limits_reject_invalid_types_and_overruns_before_page_calls(subject, key, value):
    with pytest.raises(BrowserSafetyStop):
        create(subject, **{key: value})
    assert subject.state.reads == 0 and not subject.state.clicks


@pytest.mark.parametrize("change", [None, "named", "third", "outer_overlay", "tag", "size", "reordered"])
def test_native_binder_requires_exact_visible_unnamed_exposed_pair(monkeypatch, change):
    counts = {"start": 1, "end": 10, "total": 11}
    report = capture([f"Star{i:02d}" for i in range(10)], 1, 11)
    controls = pair(counts)
    buttons = []
    for p in controls:
        snapshot = "- button" + (" [disabled]" if not p["enabled"] else "")
        button = SimpleNamespace(
            is_visible=lambda: True,
            aria_snapshot=lambda s=snapshot: s,
            is_enabled=lambda enabled=p["enabled"]: enabled,
            evaluate=lambda _script, _kind, box=p["box"]: box,
        )
        button.element_handle = lambda b=button, **_: b
        buttons.append(button)
    viewport = {"width": 950, "height": 600}
    if change == "named":
        buttons[1].aria_snapshot = lambda: '- button "Next"'
    elif change == "third":
        buttons.append(buttons[0])
    elif change == "tag":
        buttons[0].evaluate = lambda *_: None
    elif change == "size":
        viewport["width"] = 949
    elif change == "reordered":
        # Capture/native disagreement must not be repaired by choosing any enabled button.
        report["frames"][0]["controls"].reverse()
    frame = SimpleNamespace(
        locator=lambda _: SimpleNamespace(evaluate=lambda _: viewport),
        get_by_role=lambda _: SimpleNamespace(all=lambda: buttons),
    )
    monkeypatch.setattr(module, "_outer_exposed", lambda *_: change != "outer_overlay")
    if change is None:
        metadata, handles = module._pager(frame, report, counts)
        assert metadata == controls and handles == buttons
    else:
        with pytest.raises(BrowserSafetyStop):
            module._pager(frame, report, counts)
