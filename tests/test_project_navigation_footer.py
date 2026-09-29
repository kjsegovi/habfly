"""Only the observed autosave footer may change during read-only navigation."""

from copy import deepcopy

import pytest

from habfly.browser_numeric import screen_identity
from habfly.browser_project_navigation import navigation_status_projection
from habfly.browser_stellar import SIMULATION_URL

VIEW = {"surface": "detail", "section": "planet", "star": "Grim"}


def capture(*, text_notice=False, ax_notice=False, busy=False):
    return {
        "outer_controls": [{"role": "checkbox", "value": False}],
        "frames": [
            {"url": "https://other.invalid/reference", "text": "Reference Data saved"},
            {
                "url": SIMULATION_URL,
                "text": "GRIM\nNormalized Flux\nBrightness: 100% , Day: 217\n"
                "STAR MASS (Ms)\n2.179\nSTAR RADIUS (Rs)\n1.831\nORBIT (years)\n0.000"
                + ("\nData saved" if text_notice else "")
                + "\nSave",
                "accessibility": '- text: Grim\n- textbox "0": "5000"\n'
                "- text: STAR MASS (Ms) 2.179 STAR RADIUS (Rs) 1.831 ORBIT (years) 0.000"
                + (" Data saved" if ax_notice else "")
                + '\n- button "Save"'
                + (" [disabled]" if busy else ""),
                "controls": [
                    {"role": "textbox", "value": "5000", "accessibility": '- textbox "0": "5000"'},
                    {
                        "role": "button",
                        "enabled": not busy,
                        "accessibility": '- button "Save"' + (" [disabled]" if busy else ""),
                    },
                ],
            },
        ],
    }


@pytest.mark.parametrize("text_notice", [False, True])
@pytest.mark.parametrize("ax_notice", [False, True])
@pytest.mark.parametrize("busy", [False, True])
def test_exact_footer_and_independently_sampled_notice_are_transient(text_notice, ax_notice, busy):
    original = capture(text_notice=text_notice, ax_notice=ax_notice, busy=busy)
    untouched = deepcopy(original)
    assert navigation_status_projection(original, VIEW) == capture()
    assert original == untouched


@pytest.mark.parametrize(
    "mutation",
    ["star", "measurement", "orbit", "chart", "tooltip", "input", "outside", "other_frame", "button"],
)
def test_non_footer_changes_remain_visible_to_guard(mutation):
    changed = capture(text_notice=True, ax_notice=True)
    frame = changed["frames"][1]
    if mutation in {"star", "measurement", "orbit", "chart", "tooltip"}:
        old, new = {
            "star": ("GRIM", "Other"),
            "measurement": ("2.179", "2.180"),
            "orbit": ("0.000", "1.000"),
            "chart": ("Normalized Flux", "Changed Axis"),
            "tooltip": ("Day: 217", "Day: 218"),
        }[mutation]
        frame["text"] = frame["text"].replace(old, new)
    elif mutation == "input":
        frame["controls"][0]["value"] = "9999"
    elif mutation == "outside":
        changed["outer_controls"][0]["value"] = True
    elif mutation == "other_frame":
        changed["frames"][0]["text"] = "Reference"
    else:
        frame["controls"].append({"role": "button", "enabled": False, "accessibility": '- button "List"'})
    assert screen_identity(navigation_status_projection(changed, VIEW)) != screen_identity(capture())


@pytest.mark.parametrize("location", ["middle", "duplicate", "unknown_footer", "other_surface"])
def test_unknown_or_relocated_notice_is_not_erased(location):
    changed = capture(text_notice=True, ax_notice=True)
    frame = changed["frames"][1]
    view = VIEW
    if location == "middle":
        frame["text"] = frame["text"].replace("\nData saved\nSave", "\nSave")
        frame["text"] = frame["text"].replace("GRIM\n", "GRIM\nData saved\n")
    elif location == "duplicate":
        frame["text"] = "Data saved\n" + frame["text"]
    elif location == "unknown_footer":
        frame["accessibility"] = frame["accessibility"].replace("ORBIT (years)", "UNKNOWN")
    else:
        view = {"surface": "detail", "section": "habitability", "star": "Grim"}
    result = navigation_status_projection(changed, view)
    assert result != capture()
    assert "Data saved" in str(result["frames"][1])
