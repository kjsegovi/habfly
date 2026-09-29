"""Fit ordering from biological features, not a physical threshold lookup."""

import pytest
import torch

from habfly.color_reference import ColorReference
from habfly.model.color_head import OrdinalColorHead
from habfly.training.color_ordering import initialize_ordered_readout, ordering_diagnostic, projection_hash


def curved_features():
    axis = torch.linspace(-4.25, 4.25, 72)
    # Neither of these dimensions is monotone, but their combination can be.
    features = torch.stack(
        [axis + 4 * axis.sin(), axis - 4 * axis.sin()] + [torch.sin(axis * i) for i in range(1, 15)], 1
    )
    labels = torch.bucketize(axis, torch.linspace(-3.5, 3.5, 8))
    wavelengths = (axis.double() / 4 + 6).exp()
    return features, labels, wavelengths


def test_train_only_fit_removes_foldback_including_red_and_infrared(monkeypatch):
    torch.set_num_threads(1)
    features, labels, values = curved_features()
    head = OrdinalColorHead(16)
    head.fit_normalization(features)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.weight[0, 0] = 1
        head.score.bias.zero_()
    assert ordering_diagnostic(head, features, values)["score_decreases"] > 0
    before = {n: t.clone() for n, t in head.named_buffers()}
    monkeypatch.setattr(ColorReference, "private_label", lambda *a: pytest.fail("Used a reference oracle"))
    monkeypatch.setattr(ColorReference, "guard", lambda *a: pytest.fail("Used physical thresholds"))
    metadata = initialize_ordered_readout(head, features, labels, values)
    assert metadata["closed_form_fits"] == 1 and metadata["fit_examples"] == 72
    assert metadata["ordering"]["score_decreases"] == 0
    assert torch.equal(head(features).argmax(-1), labels)
    assert all(torch.equal(before[n], t) for n, t in head.named_buffers())
    score_hash = projection_hash(head)
    assert all(not p.requires_grad for p in head.score.parameters())
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.003)
    loss = head.loss(features, labels)[0]
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad
    )
    optimizer.step()
    assert projection_hash(head) == score_hash
    assert ordering_diagnostic(head, features, values)["score_decreases"] == 0
    assert (head(features).argmax(-1)[labels >= 7] >= 7).all()


def test_fit_is_deterministic_and_handles_missing_smoke_classes_without_new_examples():
    x, y, v = curved_features()
    select = torch.tensor([0, 9, 18, 35])
    x, y, v = x[select], y[select], v[select]
    heads = [OrdinalColorHead(16) for _ in range(2)]
    reports = []
    for head in heads:
        head.fit_normalization(x)
        reports.append(initialize_ordered_readout(head, x, y, v))
        assert (head.cutpoints().diff() > 0).all() and torch.isfinite(head(x)).all()
    assert reports[0] == reports[1] and reports[0]["fit_examples"] == 4
    assert all(torch.equal(t, heads[1].state_dict()[n]) for n, t in heads[0].state_dict().items())


@pytest.mark.parametrize(
    "fault", ["nan_features", "zero_wavelength", "single_class", "reversed_labels", "flat_features"]
)
def test_invalid_or_conflicting_training_data_fails_without_lookup(fault):
    x, y, v = curved_features()
    head = OrdinalColorHead(16)
    head.fit_normalization(x)
    if fault == "nan_features":
        x[0, 0] = float("nan")
    elif fault == "zero_wavelength":
        v[0] = 0
    elif fault == "single_class":
        y.zero_()
    elif fault == "reversed_labels":
        y = 8 - y
    else:
        x.zero_()
    with pytest.raises(ValueError):
        initialize_ordered_readout(head, x, y, v)
