"""Pure frame identity/visibility guards; no browser or hidden content access."""

from types import SimpleNamespace

import pytest
from test_browser_numeric import config

import habfly.browser_probe as probe
from habfly.browser import BrowserSafetyStop
from habfly.browser_project_submission_steps import BrowserProjectSubmissionSteps


@pytest.fixture
def surface(monkeypatch):
    class Frame:
        def __init__(self, url, parent=None, visible=True):
            self.url, self.parent_frame, self.visible = url, parent, visible

        def frame_element(self):
            return SimpleNamespace(is_visible=lambda: self.visible)

    cfg = config()
    main = Frame(cfg.url)
    known = [Frame(rule.url, main) for rule in cfg.frames for _ in range(rule.count)]
    hidden = Frame("about:blank", main, False)
    # The nested iframe itself is visible, but its ancestor is hidden.
    nested = Frame("https://unconfigured.invalid/hidden", hidden)
    page = SimpleNamespace(
        main_frame=main,
        frames=[main, *known, hidden, nested],
        on=lambda *_: None,
        context=SimpleNamespace(on=lambda *_: None),
    )
    owner = SimpleNamespace(page=page, config=cfg, _watching=False, _dialogs=[], _popups=[])
    BrowserProjectSubmissionSteps._watch(owner)

    def context():
        if owner._frames != page.frames:
            raise BrowserSafetyStop("context_changed")

    owner._context = context
    checked = []
    monkeypatch.setattr(probe, "_check_auth_and_modals", checked.append)
    return SimpleNamespace(
        owner=owner,
        page=page,
        known=known,
        main=main,
        hidden=hidden,
        nested=nested,
        checked=checked,
        frame=Frame,
    )


def test_baseline_hidden_frame_content_is_not_read(surface):
    item = surface
    BrowserProjectSubmissionSteps._feedback_context(item.owner)
    assert item.checked == [item.main, *item.known]
    assert item.hidden not in item.checked and item.nested not in item.checked
    assert item.owner._frames == item.page.frames


@pytest.mark.parametrize("mutation", ["revealed", "new", "replaced", "known_hidden"])
def test_visibility_and_native_frame_set_cannot_change(surface, mutation):
    item = surface
    if mutation == "revealed":
        item.hidden.visible = True
    elif mutation == "new":
        item.page.frames.append(item.frame("about:blank", item.main, False))
    elif mutation == "replaced":
        item.page.frames[-1] = item.frame(item.nested.url, item.hidden)
    elif mutation == "known_hidden":
        item.known[0].visible = False
    with pytest.raises(BrowserSafetyStop, match="context_changed|diagnostic_visible_frames_changed"):
        BrowserProjectSubmissionSteps._feedback_context(item.owner)
    assert not item.checked


@pytest.mark.parametrize("url", ["about:blank", "https://unconfigured.invalid/visible"])
def test_unknown_visible_frame_even_in_initial_baseline_is_rejected(surface, url):
    item = surface
    item.page.frames.append(item.frame(url, item.main))
    item.owner._watching = False
    BrowserProjectSubmissionSteps._watch(item.owner)
    with pytest.raises(BrowserSafetyStop, match="diagnostic_unknown_frame"):
        BrowserProjectSubmissionSteps._feedback_context(item.owner)
    assert not item.checked


def test_hidden_frame_revealed_during_auth_reads_is_rejected(surface, monkeypatch):
    item = surface

    def read(frame):
        item.checked.append(frame)
        item.hidden.visible = True

    monkeypatch.setattr(probe, "_check_auth_and_modals", read)
    with pytest.raises(BrowserSafetyStop, match="diagnostic_visible_frames_changed"):
        BrowserProjectSubmissionSteps._feedback_context(item.owner)
    assert item.hidden not in item.checked and item.nested not in item.checked


def test_known_visible_frame_navigating_outside_expected_identity_is_rejected(surface):
    item = surface
    item.known[0].url = "https://unconfigured.invalid/visible"
    with pytest.raises(BrowserSafetyStop, match="diagnostic_unknown_frame"):
        BrowserProjectSubmissionSteps._feedback_context(item.owner)
    assert not item.checked
