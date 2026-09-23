"""Restricted Google Sheets cell transport. No formula evaluation lives here.

Only the dedicated copy's three numeric input cells can be written. A single
worker owns a local advisory lock; operators must also prevent cross-host use.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import Contract

SOURCE_ID = "1xr6vbzvlCZRRLpHrgXhzTOL1_tt1rH0Qq4mL-Q-XCPA"
INPUTS = {"flux": "A2", "parallax": "B2", "wavelength": "L2"}
OUTPUTS = {
    "distance": "C2",
    "luminosity": "F2",
    "temperature": "M2",
    "mass": "G2",
    "radius": "H2",
    "lifetime": "I2",
}
UNITS = {
    "flux": "W/m2",
    "parallax": "arcsec",
    "wavelength": "nm",
    "distance": "ly",
    "luminosity": "Lsun",
    "temperature": "K",
    "mass": "Msun",
    "radius": "Rsun",
    "lifetime": "yr",
}
FORMULA_CELLS = ("C2", "D2", "E2", "F2", "G2", "H2", "I2", "M2")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SpreadsheetError(RuntimeError):
    """Sanitized operational failure, safe for trajectories and stdout."""


class SpreadsheetConfig(Contract):
    spreadsheet_id: str
    worksheet: Literal["Sheet1"] = "Sheet1"
    credentials_path: Path = Field(repr=False, exclude=True)
    inputs: dict[str, str] = Field(default_factory=lambda: dict(INPUTS))
    outputs: dict[str, str] = Field(default_factory=lambda: dict(OUTPUTS))
    expected_formula_fingerprint: str | None = None
    expected_headers: dict[str, str] = Field(default_factory=dict)
    requests_per_minute: int = Field(default=40, ge=1, le=50)
    retries: int = Field(default=3, ge=0, le=5)
    request_timeout: float = Field(default=20, gt=0, le=60)
    exclusive_worker: Literal[True] = True

    @model_validator(mode="after")
    def boundary(self):
        if self.spreadsheet_id == SOURCE_ID:
            raise ValueError("The original spreadsheet is never an execution target")
        if not re.fullmatch(r"[A-Za-z0-9_-]{20,200}", self.spreadsheet_id):
            raise ValueError("Use the working copy's spreadsheet ID, not a URL")
        if self.inputs != INPUTS or self.outputs != OUTPUTS:
            raise ValueError("The stellar pilot permits only the fixed Sheet1 row-2 mapping")
        if self.expected_formula_fingerprint and not re.fullmatch(
            r"[a-f0-9]{64}", self.expected_formula_fingerprint
        ):
            raise ValueError("Invalid formula fingerprint")
        return self

    def content_identity(self):
        return {
            "task": "stellar",
            "calculation_mode": "google_sheets",
            "version": 1,
            "source_spreadsheet_id": SOURCE_ID,
            "spreadsheet_id": self.spreadsheet_id,
            "worksheet": self.worksheet,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "units": UNITS,
            "formula_fingerprint": self.expected_formula_fingerprint,
            "headers": self.expected_headers,
        }


def load_spreadsheet_config(path):
    return SpreadsheetConfig.model_validate_json(Path(path).read_text())


class GoogleSheetsTransport:
    def __init__(self, config: SpreadsheetConfig):
        # Do not include path names, token material, response bodies, or exception
        # strings in errors. The protocol boundary displays exception messages.
        path = config.credentials_path.expanduser().resolve()
        repository = Path(__file__).resolve().parents[2]
        if path.is_relative_to(repository):
            raise SpreadsheetError("credentials_must_be_outside_repository")
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise SpreadsheetError("credentials_require_private_file_permissions")
        try:
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2.service_account import Credentials

            credentials = Credentials.from_service_account_file(
                str(path), scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            self.session = AuthorizedSession(
                credentials, refresh_timeout=config.request_timeout, max_refresh_attempts=1
            )
        except Exception:  # noqa: BLE001 - credential libraries must not leak secrets at the protocol boundary
            raise SpreadsheetError("service_account_credentials_unavailable") from None
        self.config = config
        self.base = f"https://sheets.googleapis.com/v4/spreadsheets/{config.spreadsheet_id}"
        self.last_request = 0.0

    def request(self, method, suffix="", **kwargs):
        for attempt in range(self.config.retries + 1):
            remaining = 60 / self.config.requests_per_minute - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
            self.last_request = time.monotonic()
            try:
                response = self.session.request(
                    method,
                    self.base + suffix,
                    timeout=self.config.request_timeout,
                    max_allowed_time=self.config.request_timeout * 2,
                    **kwargs,
                )
                status = response.status_code
                if status in (401, 403):
                    raise SpreadsheetError("spreadsheet_access_denied")
                if status < 400:
                    return response.json()
                if status != 429 and status < 500:
                    raise SpreadsheetError(f"spreadsheet_http_{status}")
            except SpreadsheetError:
                raise
            except Exception:  # noqa: BLE001, S110 - transport exceptions can contain tokens; retry without logging them
                pass
            if attempt < self.config.retries:
                time.sleep(min(8, 2**attempt) + random.random() / 4)
        raise SpreadsheetError("spreadsheet_retries_exhausted")

    def snapshot(self):
        data = self.request(
            "GET",
            params={
                "ranges": "'Sheet1'!A1:M2",
                "includeGridData": "true",
                "fields": "spreadsheetId,sheets(properties(title),data(rowData(values(userEnteredValue,effectiveValue,formattedValue))))",
            },
        )
        try:
            if (
                data["spreadsheetId"] != self.config.spreadsheet_id
                or data["sheets"][0]["properties"]["title"] != "Sheet1"
            ):
                raise ValueError()
            rows = data["sheets"][0]["data"][0].get("rowData", [])
            cells = {}
            for row in range(2):
                values = rows[row].get("values", []) if row < len(rows) else []
                for col in range(13):
                    cell = values[col] if col < len(values) else {}
                    entered, effective = cell.get("userEnteredValue", {}), cell.get("effectiveValue", {})
                    cells[f"{chr(65 + col)}{row + 1}"] = {
                        "formula": entered.get("formulaValue"),
                        "value": effective.get(
                            "numberValue", effective.get("stringValue", effective.get("boolValue"))
                        ),
                        "error": bool(effective.get("errorValue")),
                        "entered": entered.get(
                            "numberValue", entered.get("stringValue", entered.get("boolValue"))
                        ),
                    }
            return cells
        except (KeyError, IndexError, TypeError, ValueError):
            raise SpreadsheetError("invalid_spreadsheet_response") from None

    def write(self, values):
        if not values or not set(values) <= set(INPUTS.values()):
            raise SpreadsheetError("write_outside_input_allowlist")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values.values()
        ):
            raise SpreadsheetError("numeric_inputs_required")
        self.request(
            "POST",
            "/values:batchUpdate",
            json={
                "valueInputOption": "RAW",
                "data": [
                    {"range": f"'Sheet1'!{cell}", "values": [[value]]} for cell, value in values.items()
                ],
            },
        )

    def clear(self, cells):
        if not cells or not set(cells) <= set(INPUTS.values()):
            raise SpreadsheetError("clear_outside_input_allowlist")
        self.request("POST", "/values:batchClear", json={"ranges": [f"'Sheet1'!{c}" for c in cells]})

    def close(self):
        self.session.close()


class SpreadsheetAdapter:
    def __init__(self, config, transport=None):
        self.config = config
        self.transport = transport or GoogleSheetsTransport(config)
        self.lock = None
        self.results = {}
        self.generation = 0
        self.verified = False

    def inspect(self):
        cells = self.transport.snapshot()
        formulas = {c: cells[c]["formula"] for c in FORMULA_CELLS}
        headers = {c: cells[c[:-1] + "1"]["value"] for c in (*INPUTS.values(), *OUTPUTS.values())}
        if not all(isinstance(v, str) and v.strip() for v in headers.values()):
            raise SpreadsheetError("missing_stellar_headers")
        if not all(isinstance(v, str) and v.startswith("=") for v in formulas.values()):
            raise SpreadsheetError("missing_stellar_formulas")
        if any(cells[c]["formula"] for c in INPUTS.values()):
            raise SpreadsheetError("input_cell_contains_formula")
        return {
            "spreadsheet_id": self.config.spreadsheet_id,
            "headers": headers,
            "formulas": formulas,
            "formula_fingerprint": digest(formulas),
            "cells": cells,
        }

    def checked_snapshot(self):
        snapshot = self.inspect()
        if not self.config.expected_formula_fingerprint or not self.config.expected_headers:
            raise SpreadsheetError("pin_inspected_fingerprint_and_headers_before_writing")
        if snapshot["formula_fingerprint"] != self.config.expected_formula_fingerprint:
            raise SpreadsheetError("spreadsheet_formulas_changed")
        if snapshot["headers"] != self.config.expected_headers:
            raise SpreadsheetError("spreadsheet_headers_changed")
        return snapshot["cells"]

    def acquire(self):
        if self.lock is not None:
            return
        # Advisory host lock cannot exclude another machine: the runbook requires
        # exclusive operator ownership of this copy across all hosts.
        directory = Path(tempfile.gettempdir()) / f"habfly-sheet-locks-{os.getuid()}"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / (digest(self.config.spreadsheet_id) + ".lock")
        lock = path.open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise SpreadsheetError("spreadsheet_worker_already_running") from None
        self.lock = lock

    def invalidate(self):
        self.generation += 1
        self.results = {}

    def reset(self):
        self.acquire()
        self.invalidate()
        self.checked_snapshot()
        self.transport.clear(list(INPUTS.values()))
        cells = self.checked_snapshot()
        if any(cells[c]["entered"] is not None for c in INPUTS.values()):
            raise SpreadsheetError("spreadsheet_reset_readback_failed")

    def calculate(self, inputs):
        self.acquire()
        self.invalidate()
        if set(inputs) != set(INPUTS) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in inputs.values()
        ):
            raise SpreadsheetError("positive_complete_stellar_inputs_required")
        self.checked_snapshot()
        self.transport.write({INPUTS[k]: v for k, v in inputs.items()})
        previous = None
        # Read actual effective cell values twice, bound to the current input
        # tuple. All pilot formulas are ordinary intra-sheet arithmetic.
        for _ in range(4):
            cells = self.checked_snapshot()
            if any(cells[c]["entered"] != inputs[k] for k, c in INPUTS.items()):
                raise SpreadsheetError("spreadsheet_input_readback_mismatch")
            current = {k: cells[c]["value"] for k, c in OUTPUTS.items()}
            valid = all(
                not cells[c]["error"]
                and isinstance(current[k], (float, int))
                and not isinstance(current[k], bool)
                and math.isfinite(current[k])
                for k, c in OUTPUTS.items()
            )
            if valid and previous == current:
                self.results = current
                return dict(current)
            previous = current if valid else None
        raise SpreadsheetError("spreadsheet_outputs_unavailable_or_unstable")

    def verify(self):
        self.acquire()
        before = self.checked_snapshot()
        saved = {c: before[c]["entered"] for c in INPUTS.values()}
        if any(
            v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v))
            for v in saved.values()
        ):
            raise SpreadsheetError("verification_requires_blank_or_numeric_inputs")
        try:
            values = self.calculate({"flux": 5.15e-13, "parallax": 0.032, "wavelength": 212.0})
            expected = {
                "distance": 101.875,
                "luminosity": 0.015708042022455068,
                "temperature": 13668.7193396226,
                "mass": 0.3052153006226406,
                "radius": 0.02256635219411464,
                "lifetime": 194305121024.2034,
            }
            if any(not math.isclose(values[k], v, rel_tol=1e-6, abs_tol=1e-9) for k, v in expected.items()):
                raise SpreadsheetError("spreadsheet_golden_case_mismatch")
            # A distinct second input catches a transport returning old results.
            second = self.calculate({"flux": 1.03e-12, "parallax": 0.064, "wavelength": 424.0})
            if any(
                not math.isclose(second[k], expected[k] / 2, rel_tol=1e-6, abs_tol=1e-9)
                for k in ("distance", "luminosity", "temperature")
            ):
                raise SpreadsheetError("spreadsheet_recalculation_failed")
            self.verified = True
            return {
                "verified": True,
                "cases": 2,
                "formula_fingerprint": self.config.expected_formula_fingerprint,
            }
        finally:
            self.invalidate()
            # Never restore through a changed header/formula mapping.
            self.checked_snapshot()
            numeric = {c: v for c, v in saved.items() if v is not None}
            blanks = [c for c, v in saved.items() if v is None]
            if numeric:
                self.transport.write(numeric)
            if blanks:
                self.transport.clear(blanks)
            after = self.checked_snapshot()
            if any(after[c]["entered"] != v for c, v in saved.items()):
                raise SpreadsheetError("spreadsheet_restore_failed")

    def close(self):
        self.invalidate()
        self.transport.close()
        if self.lock:
            self.lock.close()
            self.lock = None
