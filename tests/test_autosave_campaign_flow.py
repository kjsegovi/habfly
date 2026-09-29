"""Offline autosave completion across inventory, campaign and next-star gates.

Reuses the declared synthetic learned-source/owner/UI seams, not real course
acceptance. Inventory admission, canonical imports, campaign validation and
next-star completion validation execute unchanged. No browser or model runs.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_browser_project_campaign_steps as fixture
from test_project_evidence import read, sha, write

import habfly.browser_project_finalize_steps as finalizer
import habfly.browser_project_inventory_steps as inventory_steps
import habfly.browser_project_next_star_steps as next_star
from habfly.browser import BrowserSafetyStop
from habfly.browser_autosave import ACKNOWLEDGEMENT_SOURCE, autosave_flags, autosave_workflow_flags
from habfly.browser_no_planet_workflow import _Evidence
from habfly.browser_numeric import screen_identity
from habfly.project_progress import ProgressError

campaign_rig = fixture.rig


def drive(rig):
    """Same bounded test driver, counting abort cleanup separately from work."""
    component = rig.c
    if component.status == "idle":
        component.start(paused=True)
    for _ in range(70):
        if component.finished:
            return component.state()
        if component.phase == "awaiting_class_source":
            component.provide_class(
                class_dir="explicit-class", selected_class="main_sequence", lifetime_prefix="Ga"
            )
        elif component.phase == "awaiting_planet_class":
            component.provide_planet_class(
                name="terrestrial" if component.owner.branch == "terrestrial" else "gas_giant",
                rationale="supplied reference",
            )
        elif component.phase == "awaiting_gases":
            component.provide_gases(
                gases=["CO2"], rationale="supplied spectral comparison", supplied_greenhouse_increment=10
            )
        elif component.phase == "awaiting_habitability":
            component.provide_habitability(choice="not_habitable", rationale="supplied chamber readback")
        else:
            before = len(rig.flags.calls)
            component.step()
            assert len([call for call in rig.flags.calls[before:] if not call.endswith(".abort")]) <= 1
    pytest.fail("unbounded scheduler")


@pytest.fixture
def autosave_campaign(request, monkeypatch):
    observations = SimpleNamespace(workflows=[], admissions=[], transitions=[], mutation=None)

    def creator(original, branch):
        def create(history, star, suffix, registry, **kwargs):
            directory = original(history, star, suffix, registry, **kwargs)
            hashes, bundle = registry[str(history / f"source-{suffix}/numeric")]
            path = history / f"source-{suffix}/save/confirmed.json"
            capture = bundle["habitability_capture"] if branch == "terrestrial" else bundle["save_capture"]
            proof = {
                "schema_version": 1,
                **autosave_flags(),
                "branch": branch,
                "star": star,
                "output": str(path.parent.relative_to(history)),
                "source_sha256": {
                    name: value for name, value in hashes.items() if name != str(path.relative_to(history))
                },
                "before_sha256": screen_identity(capture),
                "after_sha256": screen_identity(capture),
            }
            if branch == "no_planet":
                choice = history / f"source-{suffix}/choice/confirmed.json"
                proof.update(choice_path=str(choice.relative_to(history)), choice_sha256=sha(choice))
                bundle.update(save_click_delivered=False, acknowledgement_source=ACKNOWLEDGEMENT_SOURCE)
            write(path, proof)
            hashes[str(path.relative_to(history))] = sha(path)
            bundle.update(autosave_workflow_flags())
            receipt = read(directory / "confirmed.json")
            receipt.update(
                autosave_workflow_flags(),
                source_sha256=hashes,
                save_acknowledgement_verified=False,
                source_save_click_delivered=False,
                save_acknowledgement_source=ACKNOWLEDGEMENT_SOURCE,
            )
            if observations.mutation is not None:
                observations.mutation(receipt, branch)
            write(directory / "confirmed.json", receipt)
            # This is the actual pre-inventory admission that stopped the live
            # No branch; it must also accept typed positive/terrestrial sources.
            admitted, _ = inventory_steps._workflow(
                _Evidence(history), directory, sha(directory / "confirmed.json"), star
            )
            observations.admissions.append(deepcopy(admitted))
            observations.workflows.append(directory)
            return directory

        return create

    for name, branch in (
        ("no_workflow", "no_planet"),
        ("positive_workflow", "positive"),
        ("terrestrial_workflow", "terrestrial"),
    ):
        monkeypatch.setattr(fixture, name, creator(getattr(fixture, name), branch))
    rig = request.getfixturevalue("campaign_rig")
    # The upstream fixture is a declared source-validation seam. Add exactly the
    # clean-directory/tree guards returned by the real new readback loaders.
    for helper in (fixture.no_evidence, fixture.positive_evidence, fixture.terrestrial_evidence):
        original = helper._load_sources

        def sources(book, _original=original, **directories):
            bundle = _original(book, **directories)
            directory = book.clean(directories["save_dir"])
            book.closed_trees[directory] = book.tree(directory)
            return bundle

        monkeypatch.setattr(helper, "_load_sources", sources)
    transition = rig.c._transition_factory

    def checked_transition(*args, **kwargs):
        state, previous, _ = next_star._completed_owner(
            _Evidence(rig.root), kwargs["journal"], kwargs["completed_owner_dir"], kwargs["expected_stars"]
        )
        observations.transitions.append((previous.name, state.report()["verified"]))
        return transition(*args, **kwargs)

    rig.c._transition_factory = checked_transition
    return rig, observations


def test_three_autosave_branches_pass_inventory_campaign_and_next_star(autosave_campaign):
    rig, seen = autosave_campaign
    result = drive(rig)
    assert result["status"] == "handoff", result
    assert result["phase"] == "awaiting_assessment"
    assert result["verified_stars"] == 3 and result["target_workflows_verified"] is True
    assert result["task_completed"] is result["project_completed"] is False
    assert len(seen.admissions) == 3
    assert seen.transitions == [("Althinagon", 1), ("Beta", 2)]
    assert all(
        item["save_strategy"] == "autosave"
        and item["save_acknowledgement_verified"] is False
        and item["source_save_click_delivered"] is False
        and item["persistence_verified"] is False
        for item in seen.admissions
    )
    state = rig.journal.load().reduce()
    assert not state.reservations and not state.receipts and not state.pending
    assert all(star.task_completed for star in state.stars.values())
    # Third branch can pass the same real previous-owner gate too. This is
    # read-only; the bounded three-star campaign never selects a fourth star.
    book = _Evidence(rig.root)
    _, previous, _ = next_star._completed_owner(
        book, rig.journal, rig.flags.owners[-1].output, ["Althinagon", "Beta", "Gamma"]
    )
    assert previous.name == "Gamma"
    assert rig.root / "source-2/save" in book.clean_directories
    assert rig.root / "source-2/save" in book.closed_trees


@pytest.mark.parametrize("branch", ["no_planet", "positive", "terrestrial"])
@pytest.mark.parametrize("change", ["ack", "alias", "mode"])
def test_invalid_autosave_authority_stops_before_inventory_import_or_next_star(
    autosave_campaign, branch, change
):
    rig, seen = autosave_campaign

    def mutation(receipt, current):
        if current == branch:
            key, value = {
                "ack": ("save_acknowledgement_verified", True),
                "alias": ("persistence_verified", 0),
                "mode": ("autosave_readback_mode", "unknown"),
            }[change]
            receipt[key] = value

    seen.mutation = mutation
    result = drive(rig)
    assert result["status"] == "stopped"
    expected = {"no_planet": 0, "positive": 1, "terrestrial": 2}[branch]
    assert result["verified_stars"] == expected
    assert len(seen.admissions) == expected
    assert rig.journal.load().reduce().report()["verified"] == expected
    assert not result["project_completed"]


@pytest.mark.parametrize("filename", ["dispatch.json", "stopped.json"])
def test_next_star_retains_autosave_source_closure_after_validation(autosave_campaign, filename):
    rig, _ = autosave_campaign
    assert drive(rig)["phase"] == "awaiting_assessment"
    book = _Evidence(rig.root)
    next_star._completed_owner(
        book, rig.journal, rig.flags.owners[-1].output, ["Althinagon", "Beta", "Gamma"]
    )
    write(rig.root / "source-2/save" / filename, {"unexpected_late_fixture": True})
    with pytest.raises((BrowserSafetyStop, ProgressError)):
        book.unchanged()


def test_finalizer_cached_evidence_checks_closed_readback_tree(tmp_path):
    directory = tmp_path / "readback"
    write(directory / "confirmed.json", {"fixture_only": True})
    book = finalizer._PinnedEvidence(tmp_path)
    book.clean(directory)
    book.read(directory / "confirmed.json")
    book.closed_trees[directory] = book.tree(directory)
    book.unchanged()
    write(directory / "dispatch.json", {"unexpected_late_fixture": True})
    with pytest.raises(BrowserSafetyStop, match="tree_changed"):
        book.unchanged()
