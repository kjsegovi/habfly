"""Class transport fixtures contain no grading or classification oracle."""
# ruff: noqa: F811

import json

import pytest
from test_browser_numeric import chromium, config, page  # noqa: F401
from test_browser_planet_numeric import planet_page  # noqa: F401

from habfly.browser import BrowserSafetyStop
from habfly.browser_planet_classification import CLASSES, select_planet_class
from habfly.browser_planet_numeric import ANSWER_UNITS


@pytest.fixture
def classified_page(planet_page):
    page, frame = planet_page
    for name in ANSWER_UNITS:
        frame.locator("#" + name).fill("1")
    frame.locator("body").evaluate("""e=>{
        const style=document.createElement('style');style.textContent='.choice.selected label{border-color:white}.choice.selected label::after{opacity:1;background:white}';e.append(style);
        document.querySelectorAll('.choice label').forEach(label=>label.onclick=()=>label.parentElement.classList.add('selected'));
    }""")
    return page, frame


@pytest.mark.parametrize("name", CLASSES)
def test_explicit_class_selection_preserves_all_numbers_and_external_controls(
    classified_page, tmp_path, name
):
    page, frame = classified_page
    receipt = select_planet_class(page, config(), tmp_path / "class", name, source="reference_diagnostic")
    assert receipt["readback_verified"] and not receipt["correctness_verified"]
    assert not receipt["task_completed"] and receipt["numeric_writes"] == 0
    assert all(frame.locator("#" + k).input_value() == "1" for k in ANSWER_UNITS)
    assert not page.get_by_role("checkbox").is_checked()
    with pytest.raises(BrowserSafetyStop, match="already_selected"):
        select_planet_class(page, config(), tmp_path / "second", name, source="checkpoint")


@pytest.mark.parametrize("mutation", ["field", "dialog", "no_selection", "two_selected"])
def test_unexpected_class_effect_is_preserved_as_failure(classified_page, tmp_path, mutation):
    page, frame = classified_page
    code = {
        "field": "document.querySelector('#planet_mass').value='999'",
        "dialog": "confirm('private confirmation')",
        "no_selection": "e.parentElement.classList.remove('selected')",
        "two_selected": "document.querySelectorAll('.choice').forEach(c=>c.classList.add('selected'))",
    }[mutation]
    frame.locator(".choice label").first.evaluate(
        "(e,code)=>{const old=e.onclick;e.onclick=()=>{old();new Function('e',code)(e)}}", code
    )
    with pytest.raises(BrowserSafetyStop):
        select_planet_class(page, config(), tmp_path / "class", "gas_giant", source="checkpoint")
    assert not (tmp_path / "class/confirmed.json").exists()
    failure = (tmp_path / "class/stopped.json").read_text()
    assert json.loads(failure)["write_may_have_occurred"] and "private confirmation" not in failure


def test_missing_measurements_or_invalid_class_never_click(classified_page, tmp_path):
    page, frame = classified_page
    frame.locator("#period_days").fill("")
    with pytest.raises(BrowserSafetyStop):
        select_planet_class(page, config(), tmp_path / "class", "gas_giant", source="checkpoint")
    assert not json.loads((tmp_path / "class/stopped.json").read_text())["write_may_have_occurred"]
    with pytest.raises(BrowserSafetyStop, match="invalid_planet_class"):
        select_planet_class(page, config(), tmp_path / "invalid", "invented", source="checkpoint")


def revision_setup(frame):
    frame.locator(".choice label").first.click()
    frame.locator("body").evaluate("""e=>{
        document.querySelectorAll('.choice label').forEach(label=>label.onclick=()=>{
            document.querySelectorAll('.choice').forEach(c=>c.classList.remove('selected'));
            label.parentElement.classList.add('selected');
        });
    }""")


def test_explicit_revision_swaps_only_known_painted_circles(classified_page, tmp_path):
    page, frame = classified_page
    revision_setup(frame)
    receipt = select_planet_class(
        page,
        config(),
        tmp_path / "revision",
        "ice_giant",
        source="reference_diagnostic",
        expected_previous="gas_giant",
        revision_reason="Bounded course-feedback diagnostic",
    )
    assert receipt["previous"] == "gas_giant" and receipt["readback_verified"]
    assert not receipt["correctness_verified"] and receipt["numeric_writes"] == 0
    assert all(frame.locator("#" + k).input_value() == "1" for k in ANSWER_UNITS)
    with pytest.raises(BrowserSafetyStop, match="revision_precondition"):
        select_planet_class(
            page,
            config(),
            tmp_path / "repeat",
            "ice_giant",
            source="reference_diagnostic",
            expected_previous="gas_giant",
            revision_reason="Must not repeat",
        )
    assert not json.loads((tmp_path / "repeat/stopped.json").read_text())["write_may_have_occurred"]


@pytest.mark.parametrize("case", ["wrong_previous", "changed_number", "old_still_selected"])
def test_revision_preserves_failures_and_never_corrects_other_fields(classified_page, tmp_path, case):
    page, frame = classified_page
    revision_setup(frame)
    if case == "changed_number":
        frame.locator(".choice label").nth(1).evaluate("""e=>{
            const old=e.onclick;e.onclick=()=>{old();document.querySelector('#planet_mass').value='999'};
        }""")
    elif case == "old_still_selected":
        frame.locator(".choice label").nth(1).evaluate(
            "e=>e.onclick=()=>e.parentElement.classList.add('selected')"
        )
    with pytest.raises(BrowserSafetyStop):
        select_planet_class(
            page,
            config(),
            tmp_path / "revision",
            "ice_giant",
            source="reference_diagnostic",
            expected_previous="terrestrial" if case == "wrong_previous" else "gas_giant",
            revision_reason="Fixture diagnostic",
        )
    failure = json.loads((tmp_path / "revision/stopped.json").read_text())
    assert failure["write_may_have_occurred"] == (case != "wrong_previous")
    assert not (tmp_path / "revision/confirmed.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_previous": "gas_giant"},
        {"revision_reason": "No previous selection"},
        {"expected_previous": "ice_giant", "revision_reason": "Same class"},
        {"expected_previous": "other", "revision_reason": "Invalid class"},
    ],
)
def test_revision_requires_explicit_previous_and_reason(classified_page, tmp_path, kwargs):
    page, _ = classified_page
    with pytest.raises(BrowserSafetyStop, match="invalid_planet_class_revision"):
        select_planet_class(
            page,
            config(),
            tmp_path / "revision",
            "ice_giant",
            source="reference_diagnostic",
            **kwargs,
        )
    assert not (tmp_path / "revision").exists()


@pytest.mark.parametrize("change_axis", [False, True])
def test_disappearing_flux_tooltip_keeps_axis_changes_detectable(classified_page, tmp_path, change_axis):
    page, frame = classified_page
    # Same accessible chart shape as the course: the dynamic tooltip is part
    # of the image's accessible name, not a form control or a measurement input.
    frame.locator("svg").evaluate("""e=>{
        const tip=document.createElementNS('http://www.w3.org/2000/svg','text');
        tip.setAttribute('y','29');tip.textContent='Brightness: 100% , Day: 40';
        e.append(tip);
    }""")
    frame.locator(".choice label").first.evaluate(
        """(e,change)=>{
        const old=e.onclick;e.onclick=()=>{
            old();document.querySelector('svg text:last-child').remove();
            if(change)document.querySelector('svg text').textContent='Normalized Flux Days Observed 7 8 9';
        };
    }""",
        change_axis,
    )
    if change_axis:
        with pytest.raises(BrowserSafetyStop, match="unexpected_planet_class_side_effect"):
            select_planet_class(page, config(), tmp_path / "class", "gas_giant", source="checkpoint")
        assert not (tmp_path / "class/confirmed.json").exists()
    else:
        receipt = select_planet_class(page, config(), tmp_path / "class", "gas_giant", source="checkpoint")
        assert receipt["readback_verified"]
