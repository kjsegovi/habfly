"""Exact layout-specific footer normalization, independent of browser fixtures."""

from copy import deepcopy

import pytest

from habfly.browser_full_stellar import full_stellar_status_projection
from habfly.browser_numeric import screen_identity
from habfly.browser_stellar import SIMULATION_URL

WARNING = "mass, radius and lifetime are only relevant for main sequence stars "
CHOICES = "main sequence red giant supergiant white dwarf 1 Rs"


def capture(warning, *, text_notice=False, ax_notice=False):
    return {
        "frames": [
            {
                "url": SIMULATION_URL,
                "text": "Fixture\n0.055\n1 Rs\n" + ("Data saved\n" if text_notice else "") + "Save",
                "accessibility": '- text: Fixture\n- textbox "0"\n- text: '
                + warning
                + CHOICES
                + (" Data saved" if ax_notice else "")
                + '\n- button "Save"',
                "controls": [
                    {"role": "textbox", "accessibility": '- textbox "0"', "value": "", "enabled": True},
                    {"role": "button", "accessibility": '- button "Save"', "enabled": True},
                ],
            }
        ]
    }


@pytest.mark.parametrize("warning", ["", WARNING])
@pytest.mark.parametrize("text_notice,ax_notice", [(True, False), (False, True), (True, True)])
def test_exact_notice_is_independent_between_reads_and_does_not_modify_evidence(
    warning, text_notice, ax_notice
):
    before = capture(warning)
    notice = capture(warning, text_notice=text_notice, ax_notice=ax_notice)
    raw = deepcopy(notice)
    assert screen_identity(full_stellar_status_projection(before)) == screen_identity(
        full_stellar_status_projection(notice)
    )
    assert notice == raw


@pytest.mark.parametrize(
    "change", ["answer", "measurement", "other_notice", "duplicate_notice", "warning", "frame"]
)
def test_other_changes_still_differ(change):
    before = capture(WARNING)
    changed = capture(WARNING, ax_notice=True)
    frame = changed["frames"][0]
    if change == "answer":
        frame["controls"][0]["value"] = "1"
    elif change == "measurement":
        frame["text"] = frame["text"].replace("0.055", "0.056")
    elif change == "other_notice":
        frame["accessibility"] = frame["accessibility"].replace("1 Rs Data saved", "1 Rs Error")
    elif change == "duplicate_notice":
        frame["accessibility"] = "- text: Data saved elsewhere\n" + frame["accessibility"]
    elif change == "warning":
        frame["accessibility"] = frame["accessibility"].replace(WARNING, "Changed applicability warning ")
    else:
        frame["url"] += "?changed=1"
    assert screen_identity(full_stellar_status_projection(before)) != screen_identity(
        full_stellar_status_projection(changed)
    )


def test_reference_class_setup_uses_same_footer_only_comparison():
    from habfly.browser_class_setup_steps import _stable_screen

    before = capture(WARNING)
    after = capture(WARNING, text_notice=True, ax_notice=True)
    original = deepcopy(after)
    assert _stable_screen(before) == _stable_screen(after)
    assert after == original
    after["frames"][0]["controls"][0]["value"] = "2"
    assert _stable_screen(before) != _stable_screen(after)
