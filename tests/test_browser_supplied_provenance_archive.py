"""Production archive joined to the strict recorded-trajectory gate reader.

Uses only the existing TEST_SEED synthetic fixture and fake checkpoint bytes;
never reserved evaluation cases, original held cases, models or a browser.
"""

import socket

from test_supplied_input_transfer import test_complete_reader_and_owned_relocation as synthetic_gate

import habfly.browser_supplied_provenance as module
import habfly.training.supplied_input_transfer as transfer


def test_archive_revalidates_actual_recorded_gate_with_unavailable_originals(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network allowed in the synthetic archive integration")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    original = transfer.require_supplied_input_transfer_gate
    history = tmp_path / "owned-history"
    history.mkdir()
    saved = {}

    def observed(*args, **kwargs):
        validated = original(*args, **kwargs)
        if kwargs.get("artifact_paths") is None:
            assert not saved
            saved["gate"] = validated
            saved["link"] = module.archive_supplied_input_transfer_gate(
                history, validated, task="temperature"
            )
        return validated

    monkeypatch.setattr(transfer, "require_supplied_input_transfer_gate", observed)
    synthetic_gate(tmp_path, monkeypatch)
    # The fixture has now removed/moved original checkpoint, metadata and gate
    # files, and made generator/environment/model APIs raise if called.
    book = module._Evidence(history)
    result = module.load_archived_supplied_input_transfer_gate(
        book, saved["link"], task="temperature", **module._expected(saved["gate"]["identity"])
    )
    assert result["scores"]["episodes"] == result["scores"]["completed"] == 100
    assert result["historical_provenance_verified"] is True
    assert result["current_sources_verified"] is False
    assert result["native_browser_enabled"] is False
    assert len(book.hashes) > 204
    book.unchanged()
