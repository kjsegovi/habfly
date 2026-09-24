# Captured stellar screen → offline field mapping

Checkpoint: 2026-09-24. This is a deterministic perception/field-mapping check,
**not a learned policy rollout or browser completion**. No browser action or
training was performed while inspecting and mapping the saved capture.

## Verified capture

The user's separate managed-Chromium preview contains **ALTHINAGON**, not the
CRABILTIA star in the Firefox session. Those sessions must not be conflated.
The source capture is `experiments/browser-preflight-001/observation.json` and its
manifest hash verifies:

`99d9f460e95a2f8af20be802758bf95d77cb421856539dec7a867a34c7a726eb`

The simulation frame is HabWorlds star project v1.5.2. UPDATE SCORE and SUBMIT
PROJECT are separate sibling frames with the same widget URL; their controls
and the outer submission checkbox are recorded as protected. The probe records
zero actions. The user manually selected/collected the star as setup; that is
not learned behavior. This older capture did not record setup mode explicitly.

| Visible measurement | Parsed number | Unit | Evidence |
| --- | --- | --- | --- |
| Parallax | 0.045 | arcsec | Visible `parallax (\")` label |
| Peak wavelength | 370 | nm | Visible label |
| Flux | 5.68E-10 | W/m² | Visible Flux number; unit from the previously inspected stellar table for this version |

Metallicity 0.21 is visible but is not substituted for any of these measurements.
The raw spellings and unit provenance remain in the normalized observation.

| Destination | Capture-local control ID | Label evidence | Normalized unit |
| --- | --- | --- | --- |
| Distance | `simulation-0:c3` | distance (ly) | ly |
| Luminosity | `simulation-0:c4` | luminosity (L + subscript s) | Lsun |
| Temperature | `simulation-0:c5` | temperature (K) | K |

All three textboxes have accessible name `"0"`. **That is not proof of their
current value.** This capture did not explicitly record native `input_value`, so
the mapper marks all three current values unknown. Future probes record the
visible native value separately; empty string and the number zero stay distinct.
No recapture is needed to retain this existing mapping evidence.

The mapper uses surrounding visible AX labels, not the repeated textbox names.
It associates these with the capture's role-ordered inventory and explicitly
marks those bindings **not live verified**. They cannot be used as saved/live
selectors. Duplicate/missing fields, incompatible labels/units, conflicting star
identity, stale/tampered captures, unsupported layouts and unexpected conditional
fields fail instead of falling back to guessed positions.

## What was added

- `habfly browser map-stellar CAPTURE_DIR OUTPUT_DIR`: hash-checks a saved capture
  and produces a normalized version-1 `Observation`, field map and manifest.
  It runs offline without a browser, model, spreadsheet or calculator.
- `plan_numeric_copy`: validates a **caller-selected** tool result, destination,
  unit and capture identity; preserves the exact float representation. It emits
  a non-executable copy intent, not an `Action`, selector or browser write. It
  does not choose the operation/destination or correct a wrong numeric answer.
- The probe distinguishes actual native input values from accessible names.
- Future manual setup can explicitly use `--new-test-session`, which permits the
  user to select/collect one star in an independent managed session. The probe
  itself remains read-only. Default existing-state mode still forbids replacing
  a missing star. Neither mode copies Firefox's profile/authentication state.

The normalized observation has `star_class: null`, no grading answers, no
actionable controls and `policy_ready: false`. Classification words on the page
are available choices, not a supplied classification. The current six-calculation
checkpoint must **not** receive a fabricated class to make it run. Main-sequence
mass/radius/lifetime remain unmapped while their actual controls are not visible.

## Reproduce offline

Already performed against the captured evidence; this does not reopen the browser:

```sh
.venv/bin/python scripts/habfly_offline.py browser map-stellar \
  experiments/browser-preflight-001 experiments/browser-stellar-map-002
```

Artifacts: `mapping.json` and `manifest.json`. The manifest contains hashes of the
source capture and mapping. Choose a new output directory if rerunning; existing
evidence is never overwritten. The offline wrapper denies Python network
connections and Sheets initialization. Tests additionally make calculator use fail.

## Remaining gate

The next bridge checkpoint now has a separate [paused numeric diagnostic](browser-numeric-test.md).
It binds current visible fields and verifies exact numeric readback, rejecting
changed stars/controls and keeping Save, assessment, score updates and submission
unavailable. Its synthetic fixture tests are not live HabWorlds acceptance. The
offline mapping command remains non-executable, and real-screen verification is
the next manual gate.

Color selection and **four-way** classification need their own learning gate;
the existing local training supplies class and groups giants together. The
H-R image and color endpoint ambiguities remain as documented in the preflight
guide. No new epochs, extra held-out cases, PPO, class assumptions, or grading
tolerances were introduced here.

## Validation results

- Full Python regression suite: **402 passed, 9 existing skips**.
- Mapper-specific tests: **27 passed**, including missing/zero/negative values,
  scientific notation, shuffled field labels, duplicate/foreign controls,
  incompatible units, unknown input values, stale capture hashes, and exact
  copying without silently correcting a wrong number.
- The final mapping was generated with the offline wrapper into
  `experiments/browser-stellar-map-002`. Its SHA-256 is
  `aa17b4ec6aefe2a40d6bb5c5fda5fd0c1ea1c2b79c5f42823a0c9ec5abf50a75`.
  The earlier `-001` mapping is retained as intermediate evidence.
- Changed Python files pass Ruff/format checks; `git diff --check` passes.
- The source capture hash is unchanged. The promoted lifetime checkpoint remains
  `e383ce87cc94e04cf5147f6e3a09ad33caf0578c79b2e85ce27b196527d52c07`.
- No live UI changes, policy inference, extra training or final-test episodes.
