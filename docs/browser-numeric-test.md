# Paused numeric browser diagnostic

This checkpoint tests **field identity and exact numeric transport**, not learned
inference, a completed star, course correctness, or saved project progress.
The human selects calculations and bindings. No training/checkpoint is loaded.
The existing local learned TUI is unchanged; live browser policy execution remains
disabled. Use this terminal diagnostic before connecting that policy to HabWorlds.

## Scope and safeguards

- One isolated Chromium context for sign-in, manual setup, capture and stepping.
  It does not share Firefox's cookies or attempt state. Leave Firefox unchanged.
- Manually select/collect any **one** star in this independent test session and
  open its stellar data. Do not choose a class or color. The current mapper
  intentionally accepts only the captured three-field, unclassified screen.
- Only distance (ly), luminosity (Lsun), and temperature (K) may be filled, at most
  once each. Each write requires terminal confirmation and includes one Tab to
  move focus off that field and check its committed display. No buttons, Enter, Save,
  assessment, score updates, or submission are executed. The site's own input
  handlers/autosave may still change this test attempt.
- Current rendered AX labels and independent visible DOM text adjacency must
  agree. Native input handles are bound anew, not loaded from a saved capture.
  Human pauses are followed by current URL/frame/control/value checks.
- Missing/ambiguous labels, changed star or measurements, replaced nodes, modals,
  auth loss, popups, unknown visible frames, and stale values stop execution.
- The initial readback must preserve the exact copied float string. After a
  controlled Tab, the target may display an equivalent numeric spelling or
  mathematically consistent nearest-decimal rounding retaining at least four
  significant digits. Both conventional half-even/half-up tie choices are allowed.
  This is an explicit **transport acceptance policy**, not an inferred universal
  course formatter or grading tolerance. It uses no relative-error epsilon.
  Truncation, coarser rounding, changes before exact verification, and arbitrary
  changed numbers still stop. The original tool result remains full precision.
  Only the target's native/AX value is exempt from the exact snapshot comparison
  during that controlled transition; labels, roles, enabled state, measurements,
  other fields and protected controls must remain unchanged. Later changes still
  stop. This does not establish persistence across Save/reload or delayed updates.
- Maximum three writes, six calculations, and 15 minutes after initial capture.
  Time limits are checked on resuming a human pause; they do not interrupt stdin.
  Failures stop, retain evidence, and never retry or automatically undo a write.
- The explicit `unclassified_common_only` calculation scope uses the existing
  knowledge pack and constants for these three formulas. It does not manufacture
  a giant/main-sequence class; mass/radius/lifetime are prohibited. Existing
  supplied-class training behavior and the pack hash are unchanged.

## Run from the HabFly repository

Optional offline configuration check (no browser, network, or artifact writes):

```sh
.venv/bin/python -m habfly browser test-numeric configs/browser_probe.example.json experiments/browser-numeric-001 --check
```

Run the test with the **plain URL** copied from the browser address bar, not a
Markdown `[label](url)` link. These are two separate terminal commands: first
`read`, then paste only the URL and press Enter; then run Python.

```sh
read -r HABFLY_PREVIEW_URL
```

```sh
.venv/bin/python -m habfly browser test-numeric configs/browser_probe.example.json experiments/browser-numeric-001 --new-test-session --url "$HABFLY_PREVIEW_URL"
```

If sign-in sends Chromium to the dashboard, paste the **original URL** back into
that Chromium tab. Do not create a new preview sequence. Manually select/collect
one star, open View Star Data, and close help panels. Confirm capture in the
terminal only when its stellar tab is visible. A different star from the saved
ALTHINAGON capture is fine. Keep this Chromium process open throughout.

For a straightforward transport check, answer the prompts in this order. The
source/result identifiers appear in the terminal; copy them exactly.

| Calculation | Input prompts and source IDs | Answer destination |
| --- | --- | --- |
| `distance` | parallax: `browser_parallax` | `distance` |
| `luminosity` | flux: `browser_flux`; distance: `result:1` | `luminosity` |
| `temperature` | wavelength: `browser_wavelength` | `temperature` |

Confirm each displayed number/unit/destination with `y` only if it matches your
intent. The diagnostic does not correct wrong bindings or destinations. Input
units must match; invalid selections stop the run. Enter `stop` at a calculation
prompt, decline any write, or press Ctrl-C to end. A cancelled or failed run is
not task success. After three verified writes the browser stays open for visual
inspection until you press Enter in the terminal. Do not Save or score it.

## Expected artifacts and handoff

`experiments/browser-numeric-001/` contains:

- `capture/observation.json` and `capture/manifest.json`: initial read-only evidence.
- `events.jsonl`: version-1 observations, operator-selected calculation bindings,
  results, proposed TYPE actions, write attempts, readbacks, and any stop reason.
- `manifest.json`: event hash, knowledge-pack hash, scope, write attempts, verified
  fields, outcome, display policy, and separate exact-copy/committed-display receipts.
  `task_completed` and `browser_acceptance_passed` remain false.
- `screen-change-001.json` (on a snapshot mismatch): complete before/after
  student-visible snapshots, their hashes, and a bounded difference list. The
  manifest and event stream reference and hash this file. Terminal output shows
  the first five changed JSON paths; long text is excerpted near the change.
  The complete snapshots remain available even if that summary is truncated.
  No additional browser reads are made to produce this evidence, and unsafe
  navigation/authentication stops do not trigger a new page capture.

Successful transport reports `numeric_transport_verified` with all three fields
verified. A strict stop is useful evidence of a layout/formatting mismatch; do not
keep rerunning it blindly. Share the output directory and stop reason for inspection.
Existing output directories cannot be overwritten. No login state, screenshots,
browser traces, session URL queries, or credentials are recorded. Artifacts do
contain student-visible course information: keep them local.

Offline replay (no browser or credentials required):

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/browser-numeric-001/events.jsonl
```

This command prints the preserved event stream. It does not execute its actions.
Live TUI integration, class/color selection, conditional fields, persistence and
one-star completion remain separate gates. No additional training is needed for
this mechanical test.

## First manual run and diagnostic update

The user's `browser-numeric-001` run selected HYASTAH. Distance was copied and
read back as `108.66666666666666 ly`, then the next calculation stopped with
`stale_numeric_observation`. There was one write attempt, no luminosity or
temperature write, and no automated Save/score/submission. The old log did not
retain the mismatching snapshot, so its exact cause is **unknown**.

The follow-up adds evidence only: snapshot equality, field/URL guards, exact
readback, per-write confirmation, and action limits are unchanged. A future
snapshot mismatch still stops. Captured timestamps and setup-mode metadata are
the only excluded fields, exactly as before. No status banner or other visible
change is automatically treated as harmless. The next manual run must use a new
output directory, `experiments/browser-numeric-002`, preserving the first run.

The original event-stream SHA-256 remains
`7adaa27af9220a28293b1c6b830b9a5e0940404780921b07a5cb8a39b32c0ff7`;
the initial capture remains
`f15d5f299ef47bbd4ba2513d005997c10ea4158d6711e19ff602dfcada779565`.
The diagnostic update passes 31 focused tests, including a disappearing visible
status after the first write, value/star changes, changes during binding,
blank-versus-zero/missing values, bounded summaries with complete evidence,
artifact hashes, no extra browser reads, and old/new event replay.
Full Python regressions after the diagnostic-only update: **433 passed, 9 existing skips**.
Ruff/format and whitespace checks pass. The update did not run a live browser
session or alter the original failed run or promoted checkpoint.

## Controlled-commit fix after the second manual run

The user's `browser-numeric-002` run selected JENA. Distance read back exactly as
`141.7391304347826`. Filling luminosity with `10.922720945303972` then exposed a
distance display of `141.7`, producing `unrelated_field_changed`. This establishes
the observed value transition, not a universal formatting rule or course tolerance.
Its preserved event SHA-256 is
`e16af79da894cdd354055ad72d3f7345cfd20c6bb26103788d0283eb1e0a532e`.

Each confirmed fill now has two reads: exact value while focused, followed by
Tab and a committed-display check. Both values, the commit key, and the acceptance
policy are recorded in events, observations and the manifest. Subsequent tool
bindings still use the original result, never the rounded browser value. The
same process checks the final temperature field rather than leaving it focused
and prematurely declaring success. Scope remains numeric transport only.

The next manual run must use `experiments/browser-numeric-003`. Both previous
runs are preserved. No live UI or learned checkpoint was changed by the fix.

Validation after the fix: **468 Python tests passed, 9 existing skips**;
**28 Rust tests passed**, including the Python bridge. The 66 focused numeric
tests include a complete three-field JENA fixture with four-significant-digit
blur formatting, full-precision result reuse, malformed/wrong/truncated/coarse
display rejection, altered neighboring fields, and immediate versus committed
readback. Both old and new traces replay offline; Ruff/format and whitespace
checks pass. These fixture results are not a new live HabWorlds acceptance run.

## Initial implementation validation (2026-09-24)

- Full Python regressions: **427 passed, 9 existing skips**. After the final
  late-stop reporting guard, all **26 numeric diagnostic tests passed** again.
- Rust: **27 standard tests plus the Python subprocess/replay integration test
  passed** (28 total). No Rust source changes were needed for this checkpoint.
- All new browser requests were intercepted and fulfilled from disposable HTML
  fixtures. Tests exercise shuffled field order, independent visible labels,
  replaced inputs, stale values/stars, navigation, unknown frames, native dialogs,
  popups, readback mismatch, wrong units, conditional-operation rejection,
  cancellation, interruption, limits, protected controls and preserved failures.
- Offline configuration checking and replay passed with Python network access
  and Sheets initialization disabled. Changed Python files pass Ruff and format
  checks; `git diff --check` passes.
- Original capture SHA-256 remains
  `99d9f460e95a2f8af20be802758bf95d77cb421856539dec7a867a34c7a726eb`.
  Promoted lifetime checkpoint SHA-256 remains
  `e383ce87cc94e04cf5147f6e3a09ad33caf0578c79b2e85ce27b196527d52c07`.
- **No live preview changes, learned browser inference, training runs, or fresh
  held-out learning cases were performed.** Real DOM binding, formatting and
  readback still require the manual test above.
