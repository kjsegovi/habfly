# Peak-wavelength color checkpoint

This is a separate, local **learned color decision**. It is not perceived stellar
color, four-way stellar classification, or course grading. Existing numeric
browser profiles/checkpoints remain unchanged, and their three-write limit is
not expanded.

## Reference and ambiguity

`src/habfly/packs/stellar_color.json` preserves the visible v1.5.2 reference,
units and inspection provenance. Its full content is hashed independently of
the established six-formula knowledge pack. The thirteen independent golden
cases cover interiors and uniquely assigned endpoints.

The shared endpoints **450, 475, 570, 590 and 620 nm** are ambiguous. The open
interval **494 < wavelength < 495 nm** is uncovered. These cases stop; they
are not assigned a label or counted as successful abstentions in learning scores.
Resolving them requires authoritative lesson clarification. The browser helper
also refuses values outside the pilot's 100–1800 nm domain.

```sh
.venv/bin/python scripts/train_color.py validate
```

## Agent contract and model

The same `SELECT`/`CLICK` contract exposes shuffled measurements, shuffled color
options and a local check control. The learner selects the measurement, selects
a band, then checks. Distractors include a reference star, a temperature, and an
incompatible-unit wavelength. Wrong but valid choices are neither repaired nor
replaced with expert answers. Changing the measurement clears the pending color;
reset clears all state. An episode is limited to eight actions.

`structured_color_v1` is a new, explicitly identified encoding. The frozen
temperature policy supplies initial compatible weights, including the learned
measurement pointer. A three-feature sensory projection adds selected-wavelength
log magnitude and public selection/answer occupancy. A nine-option head reads
only the biological efferent state. No band-membership feature or reference
oracle supplies the prediction. Character input and the fixed signed sparse
topology are retained. No graph-specific neuron embeddings are added.

The reference oracle is used only to create private expert/grade labels. At
inference a guard can reject ambiguous/domain-invalid inputs; it cannot return a
label. Provenance distinguishes this learned head from deterministic setup,
arithmetic tools and categorical transport.

## Bounded offline sequence

Run from the repository root; all output directories must be new. The script
denies Python network connects and Google Sheets adapter initialization. Neither
credentials nor a running browser are needed.

```sh
# Engineering smoke: four cases, two epochs, eight full-sequence updates.
.venv/bin/python scripts/train_color.py train experiments/color-smoke-002 --profile smoke

# Next learning run: 64 cases, five epochs, 320 full-sequence updates.
.venv/bin/python scripts/train_color.py train experiments/color-pilot-001 --profile pilot
```

Both use CPU, one thread, hidden size 16, seed 0 and the real 2,000-node graph.
No PPO, automatic retries or budget increases. One hundred deterministic expert
cases must pass before training. Demonstrations are reused across epochs.
Train, calibration, development, manual, expert-validation and final-test numeric
values/templates are disjoint. The pilot has 16 calibration and 16 development
cases. Final-test policy evaluation is a separate command and remains sealed
unless at least 15/16 development cases pass with no invalid actions or reference
errors:

```sh
.venv/bin/python scripts/train_color.py test experiments/color-pilot-001
```

The final gate requires at least 90/100 tasks, zero invalid actions/reference
errors, and at least 80% completion in every band. An already-run final directory
cannot be reused. A failed gate does not switch to an expert or promote the model.

Artifacts include `dataset-manifest.json`, private split files,
`demonstrations.json`, `status.json`, `training/checkpoint.pt` and its manifest,
`report.json`, and per-case JSONL traces plus reports under `train-rollouts/`,
`development-rollouts/`, and (only after explicit evaluation) `final/`.
Reports separate source accuracy, color accuracy, task completion, invalid
actions and recoverable reference errors. All failures remain replayable.
Option calibration is teacher-forced and color-only; it does not calibrate
browser confidence or action/target heads.

## Local TUI and replay

The TUI can inspect an experimental checkpoint without promoting it. It starts
paused and shows the selected measurement, wavelength, reference bands,
ambiguities, selected color and errors. `n` steps, space resumes, `q` quits.

```sh
.venv/bin/python scripts/color_tui.py experiments/color-pilot-001 --check
.venv/bin/python scripts/color_tui.py experiments/color-pilot-001
.venv/bin/python -m habfly replay experiments/color-pilot-001/development-rollouts/color-9800000.jsonl
```

Manual seeds are 10000000–10000099. Replay keeps protocol version 1 and does not
load a checkpoint, reference, credentials, browser or network connection.

## Browser boundary

`browser_color.py` supplies a fixture-tested, one-selection native adapter and
the same visible source/color/check interface. Its factory rejects checkpoints
that have not passed the separate color final gate. It validates the current
visible label, nine-option inventory, exact element identity, selected-option
readback and all unrelated screen state. Existing color selections, stale
observations, modals, navigation changes, replaced/disabled controls and
unrelated mutations stop the attempt. No retries or rollback follow a write.

The only browser mutation it exposes is one explicit color selection. Numeric
entry methods are disabled. Save, scoring, spending, deletion, classification
and submission are not exposed. Transport verification does not establish that
the chosen color was correct. Automatic composition with the numeric batch and
a live color acceptance run are **not enabled by this checkpoint**; they follow
the learned-color gate. The existing three-field browser command is unchanged.

## Recorded smoke result

`experiments/color-smoke-001`: eight finite optimizer updates, reload verified,
100/100 scripted expert cases, **1/4 learned training tasks and 0/2 learned
development tasks**. Loss decreased from 6.9317 to 0.6233, but this is not learned
completion. The smoke is intentionally too small to cover all nine classes;
the pilot is still an explicit next run. No final-test or live color run was made.

## Color-head refinement

The first pilot, `experiments/color-pilot-001`, completed **14/64 training** and
**3/16 development** tasks. Input selection was 100%, with zero invalid actions
or reference errors. The workflow took three actions on every case, but the
classifier predicted only Violet or IR. A low sequence-average loss was not a
passing color-learning result.

`ordinal_v2` replaces only the color readout. It reads the frozen biological
effector pool, standardized using training features only, and learns a scalar
projection plus eight ordered latent cutpoints. The ordering of the nine bands
is an explicit architectural prior. No wavelength thresholds, band-membership
features, private labels or reference lookup are used at inference. Cutpoints
start equally spaced in arbitrary latent units, not nanometers. Normalization
buffers are stored in the checkpoint. The original `linear_v1` checkpoints and
numeric model configurations remain loadable.

The objective is color negative log likelihood plus 0.25 times ordinal binary
cross entropy. It is no longer diluted by already-solved source/check decisions.
Only the head trains (AdamW, learning rate 0.03, clip norm 1); the core, source
pointer, action and target heads are frozen and hash-checked. Cached training
features are the actual frozen network outputs, not a numeric bypass. Full
closed-loop training/development rollouts still use the entire network.

```sh
.venv/bin/python scripts/train_color.py refine experiments/color-pilot-001 experiments/color-pilot-002
```

This is one **additional 320-update** comparison: 64 existing cases, five epochs,
CPU, one thread, hidden 16, seed 0. Together with the parent, the lineage has
**640 color-training updates**. Split and demonstration files are copied
byte-for-byte; no new cases or final-test data are added. Reports retain parent
hashes, the frozen-workflow hash, trainable parameter names and loss components.
The command refuses an already-refined parent, an existing output directory or
a parent whose final test was opened. It does not open a browser or promote a
checkpoint. The existing final-test and browser gates still apply.

Recorded comparison, `experiments/color-pilot-002`: **51/64 training (79.7%)**
and **11/16 development (68.75%)**, versus 14/64 and 3/16 in its parent. All 80
rollouts used exactly three actions, selected the correct input, and had zero
invalid actions, reference errors or infrastructure failures. The five failed
development cases chose adjacent bands (Violet → UV, Cyan → Blue twice,
Green → Yellow, Yellow → Orange). Cyan passed 0/2 development cases. All nine
bands now have successful training examples, but this remains a failed
development gate, not browser readiness.

All 320 additional losses and gradients were finite; checkpoint reload and the
unchanged non-color-head state hash were verified. The run took 7.04 seconds
(peak process RSS 351,322,112 bytes). The two loss objectives differ, so their
raw loss values are not directly comparable. No additional run, final test or
live color write was performed. The numeric checkpoint remains unchanged.
