# Supervised four-field browser checkpoint

This combines the frozen `temperature-source-003` numeric model and gated
`color-pilot-006` color model on the real 2,000-node graph. No training or new
learning claims are made. The coordinator is scripted; measurement, calculation,
binding, destination, unit and color decisions still come from the existing
models. Calculation and exact copying remain deterministic tools.

## Preflight (offline, no credentials or browser)

```sh
.venv/bin/python scripts/browser_four_field_tui.py --check
```

Both original promotion gates must pass and graph hashes must match. The default
profile is supervised, paused, local and one-star. Autonomous batches require
the explicit flags described below; a profile cannot silently opt into one.
The existing standalone numeric and color commands remain unchanged.

## One supervised live check

```sh
.venv/bin/python scripts/browser_four_field_tui.py
```

Paste the plain preview URL when prompted (not a Markdown link). Enter local
credentials at the prompts; the password is hidden and never added to artifacts.
An existing `HABFLY_PREVIEW_URL` environment variable takes precedence, so unset it
first if it contains a stale or malformed URL. The launcher starts fresh Chromium,
logs in and picks a visible star. Firefox and its preview remain untouched.
Use `--manual-setup` if you prefer to reach the Stellar tab yourself, then press `b`.

Keep focus in the TUI while using these keys:

1. `n` advances each numeric policy decision. At each pending browser copy,
   review the destination/value and press `y` once. There are three numeric writes.
2. Once all numeric readbacks pass, `n` rechecks the same star, visible measurements,
   frame, native control identities and field contents before switching to color.
   This scripted handoff performs no write and clears the recurrent model state.
3. `n` selects the color source, `n` proposes the color, `y` approves that one native
   dropdown selection, and `n` checks its receipt.
4. Chromium stays open at the end for inspection. `q` closes it; `a` aborts early.

Only distance, luminosity, temperature and the color dropdown may be changed.
No Save, assessment, score update, submission, classification, mass, radius,
lifetime or planet controls are used. The site's normal autosave may still occur.
In supervised mode each write requires its own approval. Repeated `n` cannot approve pending writes.
Stop/abort never retries, rolls back, or clears an already-written field.

The numeric and color stages retain their independent limits (64/8 actions and
900/120 seconds). The combined wall-clock cap is 1,200 seconds. Use a new invocation
after a safety stop; do not reinterpret a partial run as a pass.

## Evidence and scope

Artifacts are placed under `experiments/browser-four-field/`:

- `<run-id>.jsonl`: version-1 runtime stream, with active policy stage and neural
  activity. Browser confidence remains explicitly uncalibrated.
- `<run-id>/manifest.json`: both frozen checkpoint identities, linked component
  manifest hashes, handoff evidence, three numeric receipts and one color receipt.
- `<run-id>/numeric/` and `<run-id>/color/`: independent transport journals,
  visible captures and failure evidence.

`four_field_transport_verified` means four allowed writes with verified readbacks,
not that the course graded the answers correct. `task_completed` and
`browser_acceptance_passed` remain false. Earlier numeric receipts are retained
even when color fails; the overall result then remains false.

Replay is offline and cannot write to a browser:

```sh
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --python .venv/bin/python --replay experiments/browser-four-field/RUN-ID.jsonl
```

The first supervised live attempt (Issai) passed all four readbacks and the
post-run local reference checks. The next gate is a separately authorized,
bounded combined reliability batch.

## Explicit autonomous reliability batch

```sh
.venv/bin/python scripts/browser_four_field_tui.py --autonomous --runs 10 --check
.venv/bin/python scripts/browser_four_field_tui.py --autonomous --runs 10
```

The first command is offline. The second prompts for the URL/login and runs up
to ten fresh Chromium sessions sequentially, with at most four native writes
per run and no `n`/`y` presses. The default supervised command is unchanged.
Use `--output NEW-DIRECTORY` to choose a new artifact location. Never reuse an
existing output directory. Ctrl-C aborts the batch; no failed run is retried.

Both models, the graph, knowledge pack, color reference and promotion evidence
are pinned before launch and rechecked before each browser opens. Automatic
setup occurs only once per session; the numeric-to-color handoff reuses that
same page and its native control identities. Proposals, writes and the handoff
occupy separate scheduler ticks, so pausing or aborting cancels the next action.

The batch closes each browser before it computes post-run reference answers.
A pass requires all four verified readbacks, correct numeric values/units and
color source/selection, preserved numeric fields after color, valid action and
artifact records, and zero runtime errors. References never correct or guide
the policy. The first failed accuracy, safety, setup, cleanup, or evidence gate
ends the batch. Each run has a 1,260-second outer limit; component limits remain.

`experiments/browser-four-field-reliability/<batch-id>/report.json` records per-run
results and failure categories, timing, Python peak memory (excluding Chromium),
distinct stars/measurement cases, and covered color bands. Raw runtime and
component journals remain under `runs/` for offline replay. Repeated stars count
as repeated coverage; there are no extra attempts to replace them. These are
fresh browser sessions, not fresh provisioned accounts or course acceptance.

## Developer verification — 2026-09-25

- Python regression: 720 passed, 9 skipped; Rust: 41 passed, including the
  Python-subprocess integration test.
- Real frozen models on an intercepted Chromium fixture: 31 numeric decisions,
  read-only handoff, 3 color decisions, 4 separately approved native writes.
  Numeric values remained unchanged through color selection. Runtime and all
  component journals replayed exactly with no model or browser connection.
- Failure coverage includes changed stars/measurements/values, replaced native
  controls, changes during handoff, aborts in both stages, time limits, nonblank
  initial fields, and a color change that unexpectedly alters a numeric answer.
- Both standalone preflights and the new combined preflight passed offline;
  the two promoted checkpoint hashes remain unchanged. No live HabWorlds run
  was performed for this implementation gate.

## Autonomous combined gate — 2026-09-25

The separately authorized batch is saved at
`experiments/browser-four-field-reliability/20260925-001/report.json`.

- 10/10 runs passed on 10 distinct stars and 10 distinct measurement cases.
- 340 learned decisions; 30 numeric copies, their 30 Tab commits, and 10 color
  selections. Zero invalid actions, runtime errors, retries or model updates.
- All 30 numeric values matched an independent evaluation of the transcribed
  worksheet equations. All 10 colors matched the versioned local reference.
  Numeric readbacks stayed unchanged through every color selection.
- Color coverage: IR 5, UV 1, Green 1, Violet 2, Red 1. This combined batch does
  not establish live coverage of Blue, Cyan, Yellow or Orange; earlier standalone
  and supervised evidence remains separate.
- Sum of per-run elapsed times: 700.616 seconds (mean 70.062 seconds). Python peak
  RSS: 305,954,816 bytes; this excludes Chromium memory.
- All 40 event streams replayed exactly offline with network connections and
  policy loading disabled: 1,930 runtime events, 180 coordinator events, 340
  numeric events and 70 color events. Component/artifact and checkpoint hashes
  matched. JSON artifacts contained no supplied login credentials or session query.
- Before the live batch: 168 targeted Python regression tests and 42 Rust tests
  passed. Both supervised and explicit autonomous preflights passed.

This verifies the bounded four-field workflow, not classification, HabWorlds
grading, saving a completed star, fresh-account acceptance or project submission.
The next expansion needs a separately grounded stellar-classification checkpoint
before conditional mass/radius/lifetime reconstruction can be enabled.
