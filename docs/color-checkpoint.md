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

## Lower-rate continuation and checkpoint selection

`stabilize` continues an `ordinal_v2` refinement at learning rate **0.003**,
preserving the parent optimizer's momentum and step counters. It trains only
the existing color head; neither normalization nor the biological core is
refitted. The original split and demonstration files are copied byte-for-byte.

```sh
.venv/bin/python scripts/train_color.py stabilize experiments/color-pilot-002 experiments/color-pilot-003
```

The pilot cap is still **64 cases × five epochs = 320 additional updates**,
CPU, one thread, hidden 16, seed 0. It neither extends itself nor creates new
boundary examples. A previously stabilized parent, opened final test, or
existing output directory is rejected.

The starting point (epoch 0) and all five trained epochs are saved under
`epochs/epoch-NNN/`, with reloadable checkpoints, optimizer state, fixed-weight
losses, and full training/development rollout traces. These end-of-epoch losses
evaluate one frozen model over an entire split; unlike the online training
loss, they are not an average of different intermediate models.

Selection is fixed in advance: minimize development errors, maximize completed
development tasks, then minimize teacher-forced development color NLL; exact
ties keep the earlier epoch. Training and calibration metrics do not select the
checkpoint. The selected head **and matching optimizer state** are restored
before the single calibration pass and final local train/development report.
Development scores are now explicitly model-selection scores, not independent
final-test results. The separate 100-case final test remains sealed.

For this lineage, the report counts **960 attempted color-training updates**
(320 original + 320 ordinal refinement + 320 continuation), even if an earlier
epoch is selected. Separate fields record the selected checkpoint's retained
updates and the discarded continuation updates. Loading validates candidate
hashes, selection, frozen-workflow identity and update accounting. Browser
color writes remain disabled until the existing separate acceptance gates pass.

Recorded run, `experiments/color-pilot-003`: development completion by epoch was
**11, 12, 13, 14, 14, 14 out of 16** (including the baseline). Epoch 3 was selected
because its development color NLL, 0.3505, beat epochs 4 and 5 at equal completion.
The restored checkpoint completes **58/64 training (90.625%)** and **14/16
development (87.5%)**, with 100% input selection and zero invalid actions,
reference errors or infrastructure failures. All selected-model rollouts use
three actions. The remaining development errors are Green → Yellow and
Red → Orange. This does **not** pass the 15/16 gate.

The run used all 320 authorized updates; the selected checkpoint retains 192
of them, with 128 later updates discarded. Total color-training work is 960
updates; the retained model lineage is 832. Fixed-weight training loss improved
from 0.4856 at the baseline to 0.2153 at the selected epoch. All losses/gradients
were finite; frozen weights, normalization, split hashes, the numeric checkpoint
and matching selected optimizer state were preserved. Runtime was 35.73 seconds
with peak process RSS 342,081,536 bytes. All 560 candidate/selected-model traces
replayed offline. No new training cases, final-test run, or live browser attempt
was made.

## Boundary-focused curriculum

The `boundaries` command creates a new dataset and experiment from the selected
stabilized checkpoint. For the pilot it mixes **32 unchanged original cases and
32 independently generated boundary cases**, balanced to seven or eight cases
per color. The sampling rule uses the versioned reference, not the development
mistakes: two samples on each side of all eight transitions, at random positive
offsets of 1–10% of the adjacent band span (span capped at 50 nm). The Cyan/Green
gap uses the separate printed edges at 494 and 495 nm. No ambiguous endpoint or
uncovered gap is assigned an invented label. Sampling metadata and expected
answers are never exposed in policy observations.

```sh
.venv/bin/python scripts/train_color.py boundaries experiments/color-pilot-003 experiments/color-pilot-004
```

This is **five epochs / 320 additional updates** at 0.003, CPU, one thread,
hidden 16, seed 0. Only the color head trains. Its parent's selected optimizer
state is resumed; normalization, the connectome and all other learned weights
stay frozen. The four-case engineering smoke path uses two retained/two new
cases and does not claim coverage of all bands. The pilot is the full 32/32 mix.

Calibration and development files are byte-identical to the parent. The full
original training set is preserved byte-for-byte as `regression.json` and is
evaluated at the baseline, every epoch, and on the restored selected checkpoint.
The mixed curriculum has new demonstrations and hashes. Manifests record the
recipe, parent checkpoint/content identity, original dataset hashes, selected
optimizer lineage, new-case count and original-case regression baseline.

Checkpoint selection remains development-only, including the epoch-0 candidate.
Passing the 15/16 development gate is not enough if original-case completion
falls below the parent's baseline or introduces invalid actions/reference or
infrastructure errors. Regression is a promotion guard, not a checkpoint
selection score. The separate final test remains sealed. When it is explicitly
run later, split validation includes **all 96 historical and current training
cases**, not only the current 64. The command rejects an already-boundary-trained
parent, an opened final test, or an existing destination. No retries, new budget,
browser writes, or final-test evaluation happen automatically.

Recorded comparison, `experiments/color-pilot-004`: **no development improvement**.
Development completion across baseline/epochs 1–5 was **14, 13, 14, 14, 14, 14**
out of 16. At equal completion, every trained candidate had worse development
color NLL than the baseline. Epoch 0 was retained, including its optimizer state.
The selected model remains **58/64 on the original cases** and **14/16 on
development**, and achieves **47/64 on the new mixed curriculum**. The mixed
curriculum is a different dataset; its score is not a regression from 58/64.
Input selection is 100%, with zero invalid actions, reference errors or
infrastructure failures in all three selected-model evaluations.

All 320 additional updates were attempted and discarded from the selected
weights: total attempted color-training work is **1,280 updates**, while the
retained lineage remains **832**. The run took 58.36 seconds with peak process
RSS 350,601,216 bytes. Frozen weights/normalization, dataset identities and
checkpoint reload were verified, and all **1,008 traces replayed offline**.
The original-case regression guard passed, but the 15/16 development gate did
not; no final-test or browser run was made.

A read-only diagnostic on the 32 new training cases then kept the selected
wavelength and weights fixed and multiplied only nonselected measurement values
by 1.37. With teacher-forced source selection, **2/32 color predictions changed**
(17/32 correct before, 19/32 after; maximum latent-score shift 0.10025). This
demonstrates sensitivity to irrelevant numeric context, not proof that it
explains all errors. The probe performed zero optimizer updates and did not
evaluate final-test cases.

## Selected-measurement graph path

`selected_graph_v1` isolates color inference from irrelevant measurements and
workflow history. The selected wavelength's existing raw log-magnitude feature
enters sensory neurons through the frozen color projection, then runs from a
fresh zero state through the **same sparse connectome, shared recurrent cell,
and efferent pooling**. The color readout gets only those efferent features;
there is no direct numeric-to-answer shortcut or reference-band lookup. Selecting
the wrong valid measurement uses that measurement, without automatic repair.
Missing, nonfinite, nonpositive or wrong-unit selections fail explicitly.

The original workflow pass, recurrent state and source/action/target heads are
unchanged. Old checkpoints default to `workflow_v1`. Calibration, cached training
features and live option selection share the same versioned input path. On color
actions, version-1 telemetry shows the isolated graph activity, labels its path,
and preserves the workflow activity separately. The TUI displays the path label.

```sh
.venv/bin/python scripts/train_color.py isolate experiments/color-pilot-004 experiments/color-pilot-005
```

This comparison is already run; choose a fresh destination only for a separately
authorized repeat. It copies the parent's training, original-case regression,
calibration, development and demonstration files byte-for-byte. No new cases,
network calls or final-test data are used. Only the 26-parameter color head
trains: five epochs, 320 updates, learning rate 0.003, CPU, one thread, hidden 16,
seed 0. Because the representation changed, training-only normalization is refit
once and frozen, and optimizer moments are reset. Learned head parameters are
warm-started. Manifests explicitly record this distinction and the new input mode.

Baseline plus every epoch is saved, with development-only checkpoint selection
and the unchanged original-case regression guard. Changing the input mode is not
silently treated as a compatible continuation of the old readout distribution.

Recorded result, `experiments/color-pilot-005`: the interference gate passed,
but the learning gates **failed**. Both before and after training, all **64/64**
existing training cases had bit-identical color logits when distractor numbers,
measurement/control order, wording, previous answers and supplied workflow pooled
state changed. Unit tests also exercise actual different recurrent histories,
arbitrary selected IDs, option permutations and finite biological gradients.

Development completion across baseline/epochs 1–5 was **6, 5, 10, 11, 11, 11 /16**;
epoch 5 was selected by lowest development NLL among tied candidates. Selected
scores were **38/64 mixed training**, **45/64 original cases**, and **11/16
development**, versus the parent's 47/64, 58/64, 14/16. Source selection stayed 100%,
with zero invalid actions, reference errors or infrastructure failures. This is
not an accuracy improvement and the checkpoint is not promoted. The new feature
distribution substantially changes the inherited readout; isolation alone does
not solve its learning problem.

All 320 updates were retained: 1,600 cumulative attempted color updates and 1,152
in the selected lineage. Fixed-weight mixed training loss fell from 9.3103 to 3.1987.
Losses and gradients were finite, checkpoint reload succeeded, and the frozen
workflow hash stayed unchanged. Runtime was 61.72 seconds with peak process RSS
355,074,048 bytes. Calibration used 16 cases (NLL 1.0546, ECE 0.3030), and does not claim
calibrated action or browser confidence. All 1,008 candidate/selected traces
replayed offline, including 1,008 isolated color-activity events. The frozen
numeric checkpoint hash and its browser preflight remain unchanged; no final
test, manual-case evaluation or live HabWorlds attempt was made.

An additional frozen-weight, training-only diagnostic found that the learned
scalar rank decreases at 15 of 63 adjacent sorted wavelengths. Red ranks overlap
Orange, and IR ranks turn back into lower-color ranges; neither Red nor IR has
a correct training prediction. This is evidence about the current learned
projection, not proof that the biological features cannot support an accurate
readout. The diagnostic used no optimizer updates or held-out cases.

Implementation verification: **627 Python tests passed, 9 skipped**; **36 Rust
checks passed**, including the explicitly enabled Python subprocess bridge. Ruff,
Rust formatting, diff whitespace checks, both local preflights, checkpoint/dataset
identity checks and offline replay passed. These are engineering checks, not
evidence that the color learning gate passed.

Next: investigate which biological features preserve wavelength ordering on
training cases, then propose a bounded readout initialization/training comparison. Do not
expand its budget or promote it merely because invariance and code tests pass.

## Training-only wavelength-ordering readout

The `order` command keeps `selected_graph_v1`, the biological topology/core,
workflow heads, and train-only normalization unchanged. It changes **how the
existing ordinal readout is trained**, not the inference interface:

1. Fit a linear combination of the 16 efferent features to the selected
   training log wavelengths, centered/scaled using training data only. One
   double-precision ridge solve includes both the 64 training levels and their
   63 adjacent secants, teaching value and ordering together. Ridge weight is
   fixed at 0.001 and secant weight at 1; there is no hyperparameter search.
2. Initialize latent cutpoints from the nearest labeled training examples on
   either side of each class transition. Missing engineering-smoke classes use
   interpolated/extrapolated latent positions, not new numeric examples or
   invented physical color rules. Initial softness uses minimum cutpoint spacing
   divided by four, within the existing head's limits.
3. Freeze the learned scalar projection. Train only the eight ordered cutpoints
   and one softness parameter with the unchanged 320-update pilot budget.

Inference still passes the actual selected measurement through the biological
graph; the color head consumes only efferent activity. Neither raw wavelength
nor a reference threshold table is fed directly to the readout at inference.
Fitting an initializer is supervised training: an epoch-0 candidate is **not an
untrained baseline**, even though it has zero additional optimizer updates.
The fit count and method are recorded separately from gradient-update counts.

```sh
.venv/bin/python scripts/train_color.py order experiments/color-pilot-005 experiments/color-pilot-006
```

Training/demonstration/calibration/development/original-regression files are
copied unchanged. The command requires a single isolated parent and rejects an
existing destination, a recursive ordering run, or an opened final test. Every
epoch and its optimizer state are archived. The original-case promotion guard
keeps the stronger inherited **58/64** floor, rather than lowering it to the
failed immediate parent's 45/64. Development remains checkpoint-selection data,
not an independent final test. The separate 100-case final test is never opened
by this command.

The first real-graph initialization preflight used training levels alone and
was rejected before any optimizer updates: seven of 63 adjacent scores reversed,
including an overlapping Blue/Cyan transition. Its failure is preserved in
`experiments/color-ordering-preflight-001/report.json`. The revised initializer
adds training secants to the same solve; it must pass the unchanged zero-reversal
training guard. This is a second initialization recipe, not an extra 320-update
run. Passing that guard is not proof of global monotonicity or final accuracy.

Recorded comparison, `experiments/color-pilot-006`: **the local development and
original-case gates passed**. The selected initialized readout completes **64/64
mixed training**, **64/64 original cases**, and **16/16 development**. Source and
color selection are both 100%, with zero invalid actions, reference errors or
infrastructure failures. Red and IR each pass 7/7 examples in each training set
and 1/1 in development. The two training sets overlap; do not add their counts
and describe them as 128 independent examples.

Epoch 0, after the supervised initialization fit, was selected. Development
completion across baseline/epochs 1–5 was **16, 15, 15, 15, 15, 15 /16**. All
320 authorized gradient updates were attempted and discarded from the selected
weights. The retained initializer is genuinely learned from training examples,
not an untrained model or a reference-band lookup. Historical attempted color
optimizer work totals 1,920 updates; the selected parent optimizer lineage
remains 1,152, with the new supervised closed-form fit recorded separately.

The selected training score has **zero reversals across 63 adjacent wavelength
pairs**, versus 15 in its parent. All 64 counterfactual context checks retain
bit-identical logits. Frozen graph/workflow parameters, normalization and scalar
projection checks pass, as does checkpoint reload. The run took 61.32 seconds
with peak process RSS 356,204,544 bytes. All **1,008 traces / 17,136 events** replay
offline. Training, calibration, development, regression and demonstration files
are byte-identical to the parent. The numeric checkpoint and browser preflight
remain unchanged.

Calibration uses the separate 16-example calibration set (temperature reaches
the configured 0.1 lower bound; fitted NLL and ECE are approximately 0.00000143).
These are small-sample fitted calibration results, not independent evidence of
browser reliability or generally perfect confidence. Action/browser confidence
remains uncalibrated. No final-test, manual-case evaluation or browser attempt
was made by this experiment.

Next: the unchanged, separately invoked 100-case final evaluation. It is now
eligible, not already passed. Only after that gate should color move toward
manual TUI and browser acceptance. No further training budget was added.

Verification for this change: **635 Python tests passed, 9 skipped**; **36 Rust
checks passed**, including the explicitly enabled Python subprocess bridge.
Ruff, Rust formatting, diff whitespace, numeric/color TUI preflights, immutable
parent/dataset checks, checkpoint reload and offline replay all passed.

## Supervised browser color handoff

The separate `color-pilot-006/final/report.json` now records **100/100 unseen
local cases**, all nine bands passing, and zero invalid actions/reference errors.
The immutable training report still says its own run did not open the final test.
The manual local TUI also completed its three-decision case. Neither result is
live browser acceptance.

The browser color launcher checks that final gate and exact checkpoint identity
before opening Chromium. It reuses scripted login/introduction/visible-star
setup, then pauses for learned decisions and one explicit human approval:

```sh
.venv/bin/python scripts/browser_color_tui.py --check
.venv/bin/python scripts/browser_color_tui.py
```

The second command prompts for the existing preview's **plain URL** if
`HABFLY_PREVIEW_URL` is unset, then email and a hidden password unless the secure
login environment is already present. No credentials are stored in artifacts.
Firefox is untouched; Chromium opens a fresh session. For manual setup instead,
add `--manual-setup` and press **b** after reaching a fresh star's Stellar tab.

After automatic setup finishes, keep focus in the terminal:

1. **n**: select a visible measurement with the learned policy.
2. **n**: propose a color. Review its source, wavelength, unit and label.
3. **y**: approve that single native color selection; visible readback is checked.
4. **n**: finish the local receipt check (not a site assessment button).
5. Inspect Chromium, then **q** to close it. **a** aborts an active run.

Only one color write is allowed; no numeric answers, class selection, Save,
scoring or submission. The site's normal autosave may still run. Browser
probabilities remain uncalibrated. Changed pages, selections, controls or local
proposal state stop the run without retries/rollback. A preselected color is
rejected. Chromium stays open after a transport completion/failure for inspection.

Artifacts live in `experiments/browser-color/<run-id>.jsonl` (runtime replay)
and `experiments/browser-color/<run-id>/` (visible capture, setup evidence,
transport events and manifest). A successful receipt means only that the
approved label was selected and read back: `task_completed` and
`browser_acceptance_passed` stay false. The numeric workflow remains separate.

Replay needs neither browser nor credentials:

```sh
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --python .venv/bin/python --replay experiments/browser-color/<run-id>.jsonl
```

Developer verification uses intercepted disposable fixtures, including the
actual gated checkpoint. The next user-owned gate is one supervised live run;
inspect its saved receipt before considering any autonomous color reliability run.

Verification: full Python regression **657 passed, 9 skipped**; the final
color-specific rerun **23 passed** (including a subsequently added manual-secret
scrubbing test); Rust **37 passed**, including the explicitly enabled subprocess
bridge. Both checkpoint preflights, changed-file lint/formatting and diff checks
pass. Repository-wide Ruff still reports four existing issues in the untouched
`scripts/build_neuron_table.py` and `scripts/inspect_connectome.py`. No live
HabWorlds run or additional training was performed for this handoff.

## Bounded autonomous color reliability

The supervised live run `c07a560cdff94f74a3a4c670cd92b67f` selected the current
star Gannost's **379 nm** measurement, proposed **UV**, and read back exactly one
approved selection. A separate post-run check agrees with the local reference;
HabWorlds grading was not invoked. Its extra step requests while approval was
pending/after completion were harmless but emitted confusing errors. Those
requests now display guidance and cannot advance a pending or completed run.

The default launcher remains supervised. An explicit pair of flags enables
the separate console reliability batch (not the interactive TUI):

```sh
.venv/bin/python scripts/browser_color_tui.py --autonomous --runs 10 --check
.venv/bin/python scripts/browser_color_tui.py --autonomous --runs 10
```

The first command is offline validation with no browser or credentials. The
second prompts for the plain preview URL/email/hidden password as needed,
then runs **up to ten** sequential fresh Chromium contexts. No `n`/`y` presses
are needed. Each run has one native color-write allowance, eight learned-action
steps, the existing 90-second setup/120-second color limits, and a 240-second
runner deadline. **Ctrl-C** aborts the batch. A failed run, incorrect color,
changed identity, evidence failure, or failed cleanup stops it without retries.
No numeric answers, class selection, Save, scoring, submission or training is
enabled; the site's normal autosave may still run. Fresh browser contexts are
not fresh accounts or valid full-project resets.

Each model proposal and native write occurs on a separate tick. Runtime
pause/abort can cancel a pending selection; a paused runtime never writes
automatically. Explicit autonomous authorization is recorded in each receipt.
Supervised receipts and old replay logs remain supported.

Output defaults to a new `experiments/browser-color-reliability/<batch-id>/`
directory; `--output NEW_DIRECTORY` selects a different unused location. The
runner saves partial progress after every attempt. `report.json` separates
verified readbacks, source selections, post-run local-reference matches,
policy failures and safety/infrastructure failures. It reports observed bands,
unique stars/measurement cases, elapsed time, and Python peak RSS (excluding
Chromium). Repeated stars are reported, not silently replaced with extra runs.
Ten successful runs of the same star do not prove ten distinct-star coverage.

Private reference checking happens **after browser closure**, against the saved
visible measurement and learned selection. It never supplies an action,
corrects a wrong color, or changes the transport receipt's
`correctness_verified:false`. Browser/course acceptance remains false even if
the batch passes. The frozen checkpoint, graph, color reference, and final
report identities are pinned before the first run and rechecked before every
browser session. Traces remain replayable offline through the existing TUI.

Developer tests cover a ten-run orchestration fixture, two actual-checkpoint
runs in distinct intercepted Chromium contexts (UV/Red), supervised defaults,
explicit CLI opt-in, pause/abort/stale/timeout guards, bad evidence, wrong
colors without repair, identity drift, credential cleanup, and numeric
regressions. No live reliability batch is started by these checks.

Next: the user runs the live batch above; inspect its report and failed traces
before combining color with the three-field numeric workflow.

Verification for this update: **689 Python tests passed, 9 skipped**; **39 Rust
checks passed**, including the Python subprocess bridge. The final color batch
fixture rerun passed all **16 tests**, including real-checkpoint UV/Red runs in
separate intercepted contexts. Supervised/autonomous color and numeric
preflights pass; the original 41-event live supervised trace replays exactly
offline. Changed-file Ruff, formatting and diff checks pass. No live batch or
additional training was performed.
