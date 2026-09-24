# Supervised learned browser diagnostic

This is a transfer check of the frozen `temperature-source-003` checkpoint on
the real 2,000-node graph, not a new training run or full HabWorlds attempt.
The supervised Devaem run (`c9dbdac18d8d491bb3336d309832162e`) completed automatic
setup, all 31 learned decisions, and all three verified copies. Autonomous
execution remains a separate live reliability gate. No classification is supplied or inferred.

## Run

From the repository root, first validate locally (no browser or network needed):

```sh
.venv/bin/python scripts/browser_tui.py --check
```

Set `HABFLY_PREVIEW_URL` to the existing preview's **plain URL**, quoted in the
shell, then run:

```sh
.venv/bin/python scripts/browser_tui.py
```

This starts a fresh visible Chromium context beside the TUI. It cannot reuse the
Firefox attempt, and does not open, reset, or change Firefox. In Chromium, sign
in if needed and manually select one star, open its Stellar detail screen, and
close help. The site may autosave ordinary UI changes.

### Optional automatic setup

To script login, the four introductory screens, selecting one visible star, and
opening its Stellar screen, use:

```sh
.venv/bin/python scripts/browser_tui.py --auto-setup
```

Enter the local author email and password at the terminal prompts. The password
prompt is hidden. Credentials are not saved in configuration, source, command
arguments, traces, screenshots, or datasets. Do not put them in the profile or
paste them into commands. For an existing secure process environment, the launcher
also accepts `HABFLY_LOGIN_EMAIL` and `HABFLY_LOGIN_PASSWORD`; Python consumes these
before launching Playwright, and diagnostic driver logging is disabled.

This is **scripted setup, not learned behavior**. It opens a new Chromium context,
submits login at most once to the same local origin, closes the cookie notice,
advances only the recognized introduction, and clicks one rendered star dot. It
uses a tightly scoped image check for the canvas-drawn “View Star Data” link when
no semantic control exists. No star catalog or hidden application state is read;
star selection does not use measurements. Unknown/ambiguous layouts stop, not retry.
The setup deadline is 90 seconds. Within it, starfield rendering gets up to 30
seconds: sample every half-second and require a matching visible candidate on two
successive samples before clicking. Blank/transient images cause read-only waiting,
not a reload or a star click. Detection includes compact, high-contrast dim dots
at the actual CSS pixel scale. An unconfirmed star selection stops after eight seconds.
Only the simulation iframe crops used for those two clicks are saved, never the
login page. Selecting a star collects it in this new session; the site may autosave.

After strict screen validation, the TUI becomes **ready and paused** automatically;
skip **b** and press **n** to begin learned decisions. Every numeric write still
requires **y**. **a** or **q** stops setup. The default launcher without
`--auto-setup` retains manual setup. No Firefox state, checkpoint, or training data
is changed. `--auto-setup --check` validates offline without credential prompts.

Readiness now requires a recorded star-open transition, exposed editable fields,
loaded companion frames, and stable observations. Mounted transparent, covered,
or off-screen detail labels cannot mark the star map ready. On a setup failure,
the TUI shows the specific safe reason and leaves Chromium open for inspection;
no more model or setup actions run. Press **q** to close it. Login-page screenshots,
driver exception text and credentials are never captured as error evidence.

The live login/introduction/star/detail route and supervised end-to-end numeric
transport passed on 2026-09-24. Readiness uses exposed editable fields, strict
student-visible DOM/AX mapping and two stable observations. The exact paired
"Data saved" footer notice is excluded from screen identity comparisons; raw
evidence is preserved, and other screen/answer changes still stop execution.

- **b**: capture this ready screen. No writes or model actions yet.
- **n**: one learned tool decision. Calculation, input, result, destination, and
  unit selections use the same semantic local controls used during training.
- A copy proposal pauses with the full exact number, unit, destination, previous
  field value, and the planned Tab commit. **y** confirms only that copy. **n**
  and resume cannot approve it. **a** aborts without approving.
- **v** cycles observation, controls, and neural views. Arrows scroll.
- **q** exits. Browser and runtime are cleaned up.

Only distance, luminosity, and temperature can be written, once each. The local
unit selections acknowledge fixed visible field units; they do not manipulate a
browser unit dropdown. The final learned "Check stellar analysis" control checks
transport receipts only, never a HabWorlds assessment. Results stay full precision
even when the site displays rounded values. There is no Save, score update,
classification, submission, sheet access, expert repair, or hidden grading data.

Browser probabilities are explicitly **uncalibrated**; local calibration is not
evidence of browser confidence. Neural activity comes from the frozen policy.
Maximums: 64 learned decisions, 6 calculations, 3 writes, 15 minutes after capture;
repeated unchanged tool actions stop early. Any unapproved visible screen change, stale
target, modal, navigation escape, invalid units, or failed readback stops the run.
A failed write may have changed a field: there is no retry, rollback, or reset.

## Explicit autonomous mode and reliability batches

Default behavior above stays supervised. The new flag explicitly authorizes
automatic entry of only distance, luminosity and temperature into one star:

```sh
.venv/bin/python scripts/browser_tui.py --autonomous --check
.venv/bin/python scripts/browser_tui.py --autonomous
```

`--autonomous` includes automatic setup and starts running; no `n` or `y` is
required. In the TUI, **p/space** pauses, **r/space** resumes, **a** aborts and
**q** quits. Pause/abort are handled between actions, not midway through an
in-flight browser call. Proposing a copy and executing it are separate ticks;
pausing before execution preserves the pending proposal without writing.
While paused, **n** performs one decision and **y** can explicitly approve a
pending copy. Such an approval is logged as human, not autonomous.

For repeat testing, the same launcher supports console batch mode (not the TUI):

```sh
.venv/bin/python scripts/browser_tui.py --autonomous --runs 3 --output experiments/browser-reliability-001
```

Enter credentials once at the secure prompts. Chromium remains visible; sessions
run sequentially and credentials are never written to artifacts. **Ctrl-C** stops
the whole batch. Counts must be 1–10 and the output directory must be new. Omitting
`--output` creates a unique directory under `experiments/browser-reliability`.
The first failure/abort ends the batch, without retry or replacement cases.
Each run retains the normal action/time limits; the batch also checks a 1,050-second
per-run deadline between operations, including model loading. Frozen model/graph/pack
identities are rechecked before subsequent browsers start. No optimizer runs.

The output contains:

- `report.json`: requested/attempted/passed runs, distinct stars and measurement
  cases, failure reasons, decisions, writes, elapsed time and peak process memory.
- `runs/<run-id>.jsonl`: complete version-1 runtime trace, replayable offline.
- `runs/<run-id>/`: raw captures, numeric copy receipts, failure evidence and manifest.

Only hashed, verified manifests with three autonomous-copy receipts count as a pass.
Duplicate stars are reported, not silently replaced; an all-pass batch need not
meet the distinct-star target. These are fresh **browser contexts**, not newly
provisioned learner accounts, and do not establish independent full-project
acceptance or persistence. Calibration remains `browser_transfer_not_calibrated`.
Save, assessment, score update, spending, deletion and submission stay unavailable.

Protocol version remains 1. `browser_execution: autonomous` is an explicit runtime
option, valid only for one-star `browser_numeric` with automatic setup. A configured
profile cannot silently enable this launcher mode without the `--autonomous` flag.
Manifests distinguish `autonomous_browser_numeric_transfer` from supervised runs;
per-copy receipts distinguish autonomous opt-in from human approval. Replay never
launches or controls a browser, regardless of recorded execution mode.

## Evidence and replay

Each run creates a new ID beneath `experiments/browser-learned`:

- `<id>.jsonl`: model observations, decisions, real hidden-state summaries and
  lifecycle, including pending confirmation and exact/committed readbacks.
- `<id>/events.jsonl`: lower-level calculated bindings and fill/Tab receipts.
- `<id>/capture/`: student-visible pre-write evidence.
- `<id>/manifest.json`: checkpoint/graph/pack hashes, verified fields, write count,
  outcome, and `optimizer_updates: 0`; `task_completed` and browser acceptance
  remain false even when `numeric_transport_passed` is true.
- `<id>/screen-change-*.json`: visible before/after mismatch evidence on a safety stop.
- With automatic setup: `setup-starfield.png` and, if required,
  `setup-star-link.png` are iframe-only visual evidence; the setup journal records
  coordinates, image hashes and `action_source: deterministic_setup`, with no
  learned actions or rewards attributed to setup.
- If the map is still loading, `setup-starfield-loading.png` preserves the first
  empty/transient frame; `setup-starfield-rejected.png` preserves a terminal
  detection/timeout failure. Only the simulation iframe is included.

Replay does not launch Chromium or require network access:

```sh
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --python .venv/bin/python --replay experiments/browser-learned/REPLACE_WITH_RUN_ID.jsonl
```

Protocol version stays 1. Manual ready and approval use `step` payloads
`{"browser_ready":true}` and `{"approve_copy":true}`. Empty `step` performs a
normal learned decision. Each approval is consumed once and revalidates the
live screen immediately before writing. Historical simulator and numeric logs
remain readable. No checkpoint, graph, or previous experiment is overwritten.
Automatic setup uses the optional start field `browser_setup: automatic` under
the same protocol version; the default remains `manual`.
