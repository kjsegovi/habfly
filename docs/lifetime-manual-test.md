# Learned stellar lifetime checkpoint

This checkpoint adds lifetime to the local distance → luminosity → temperature →
mass → radius workflow. All six calculations are required for a **supplied
main-sequence class**. Giants and white dwarfs require only distance, luminosity
and temperature. Classification learning, planets, habitability and actual
HabWorlds execution remain deferred. No spreadsheet or browser is modified.

The unchanged knowledge pack preserves spreadsheet `Sheet1!I2`:
`10000000000 * mass ** (-2.5)`, with mass in `Msun` and lifetime in **`yr`**,
not billions of years. Independent golden cases are 1 Msun → 10,000,000,000 yr,
4 Msun → 312,500,000 yr, and 0.25 Msun → 320,000,000,000 yr.
The learned policy selects the calculation, mass result, answer destination and
unit. The deterministic tool calculates and copies the number exactly; it does
not correct wrong selections.

## Manual trial

The launcher requires passed development and a separate frozen 100-case final
evaluation. From `/Users/kjsegovi/Projects/habfly`:

```sh
.venv/bin/python scripts/lifetime_tui.py --check
.venv/bin/python scripts/lifetime_tui.py
```

Starts paused: **n** single-steps, **Space** resumes/pauses, **v** changes panels,
and **q** quits. A 150 × 45 terminal is convenient. The full observation view
includes the supplied class, required fields, reference card, result bindings,
answers and units.

- Seed `8500000`: main sequence, **60 actions**. Expect distance `ly`, luminosity
  `Lsun`, temperature `K`, mass `Msun`, radius `Rsun`, and lifetime `yr`.
- Seed `8500001`: white dwarf, **31 actions**, no mass, radius or lifetime.
- Seed `8500003`: giant, the same **31-action** skip path.

Default main-sequence answers:

| Field | Exact tool value | Unit |
| --- | ---: | --- |
| Distance | 146.25278384996008 | ly |
| Luminosity | 0.013004209325470098 | Lsun |
| Temperature | 4895.27853547378 | K |
| Mass | 0.2891790610897379 | Msun |
| Radius | 0.16008223812249658 | Rsun |
| Lifetime | 222373428366.2679 | yr |

```sh
.venv/bin/python scripts/lifetime_tui.py --seed 8500001
.venv/bin/python scripts/lifetime_tui.py --seed 8500003
```

For main sequence, lifetime must bind the calculated mass `r4`, not luminosity
`r2`, radius `r5`, or a reference-star reading. The new result `r6` must be copied
to lifetime and paired with `yr` without division by a billion. Completion is
`task_completed`, not reaching the 64-action safety cap. All six calculations
remain selectable; skipping is a learned choice, not a runtime override. Supplied
class and required fields mean this is not independent classification learning.

No training, spreadsheet, credentials or browser is started. Traces are saved
under `experiments/manual-lifetime/`. Replay works offline:

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/manual-lifetime/YOUR_RUN.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --replay experiments/manual-lifetime/YOUR_RUN.jsonl
```

## Encoding and compatibility

The previous structured unit vocabulary contained `Gyr`, but the knowledge pack
and visible unit dropdown use `yr`. New `structured_tool_v6` preserves v5's public
task-state features and dimensions while representing `yr` in that last unit
slot. v4/v5 retain their original vocabulary. Migration copies the radius weights
and zeros only the new `yr` projection column so it starts neutral; it does not
change the parent tensor or convert a value. Tests check unchanged pre-learning
forward outputs and dropdown scores on radius observations.

No neurons, edges, recurrent layers, hidden dimensions or parameters are added.
The character/source-language encoders stay frozen. Old checkpoint files remain
intact. Their confidence is conservatively marked uncalibrated after model-source
changes; old calibration is not silently reused. The new model receives separate
calibration. Action and control-target confidence are not probabilities of
dropdown-value correctness or whole-task success. Neural telemetry is recurrent
hidden-state RMS, not biological firing rates.

## Bounded experiment

Parent: `experiments/radius-001/training/checkpoint.pt`. Real 2,000-node /
132,365-edge graph, hidden size 16, CPU, one thread, seed 0, no PPO. Eight training
cases cover four main-sequence stars, two white dwarfs and two giants, with all
classes represented under each training template. The initial cap is 200
quantity/unit recognition updates plus 400 full-workflow updates. Recognition
includes radius as a possible distractor when selecting mass.

Calibration and development each use 16 separate cases. Final evaluation uses
100 different cases (50/25/25 by class) and four held-out complete instruction
templates. Split seeds, IDs, inputs and templates are checked disjoint. Private
grading answers are never observations. Measurements are randomized independently
of supplied class; these cases test an assigned calculation workflow, not
physically consistent H-R classification.

```sh
.venv/bin/python scripts/train_lifetime.py train --output experiments/lifetime-001 --updates 400
.venv/bin/python scripts/train_lifetime.py train --output experiments/lifetime-002 --parent experiments/lifetime-001/training/checkpoint.pt --recognition-updates 0 --updates 200
.venv/bin/python scripts/train_lifetime.py train --output experiments/lifetime-003 --parent experiments/lifetime-001/training/checkpoint.pt --recognition-updates 0 --updates 200 --train-components options
.venv/bin/python scripts/train_lifetime.py evaluate --output experiments/lifetime-003
.venv/bin/python scripts/verify_distance_demo.py experiments/lifetime-tui-verification-001 --profile configs/lifetime_tui.json
.venv/bin/python scripts/verify_mass_branch.py experiments/lifetime-counterfactual-001 --profile configs/lifetime_tui.json
```

Use fresh output names when reproducing. Consumed final numeric cases are
rejected; reusing cases or instruction templates is rehearsal, not fresh
acceptance. Expected artifacts include split manifests, private cases, expert
demonstrations, recognition data, per-update losses, reloadable checkpoint,
seen/development rollouts, a frozen final report, manual runtime/replay checks
and paired-class diagnostics. Network and Sheets-adapter tripwires are installed
for these runs; this is not an operating-system network sandbox.

## Verification status

Focused tests cover independent golden values, applicability, missing/zero/
invalid inputs, wrong valid bindings, stale results, exact copying, reset
isolation, finite gradients, learned year selection, checkpoint migration/reload,
old/new observation encodings, split separation, runtime boundaries and TUI
rendering. The runner requires 100 deterministic expert cases before training.
Passing code/expert checks is not learned task completion.

Regression results: **348 Python passed**, nine intentionally inapplicable
parameterizations skipped; **28 Rust/bridge passed**. Ruff, Cargo formatting and
Clippy passed. All five previous checkpoint files retain their original hashes,
and their launchers still pass readiness checks. All **12/12 radius runtime
regressions** passed pause/step/resume/save with exact offline replay:
`experiments/radius-regression-lifetime-001/report.json`.

The initial `lifetime-001` run completed **5/8 training and 8/16 development**
tasks. Every giant/white-dwarf case completed; the main-sequence development cases
reselected radius at step 51 instead of switching to lifetime, eventually reaching
the step limit. The final test was **not opened**. On recorded expert histories,
action and control-target selection were perfect; the only three wrong training
dropdown selections were radius versus lifetime, with logit margins of roughly
0.02–0.05. Subsequent expert-history choices, including mass binding and `yr`,
were correct. Frozen evidence: `experiments/lifetime-001/option-diagnostic.json`.
Those histories are diagnostic, not autonomous completion.

`lifetime-002` continued from that checkpoint for **200 additional workflow
updates** on the same eight cases. Recognition is frozen; skipping recognition
requires a perfect frozen check. No new cases, rule overrides or PPO are added.
This regressed to **4/8 training and 8/16 development** completions: main-sequence
cases now made the wrong action/control choice immediately after temperature.
Evidence is preserved in `experiments/lifetime-002/branch-diagnostic.json`; its
final test was **not opened**. Progress previously printed only the last episode
loss, which happened to be an easier giant case. Progress now also reports the
full latest cycle's mean and per-class means.

`lifetime-003` returns to the better **001** checkpoint for a capped **200-update
dropdown-only refinement** on the same eight cases. Only `option_projection` and
`option_query` can train. The recurrent core, action/control heads, measurement
recognition and language encoders remain frozen. Every demonstration is still
processed through its full recurrent history; no expert stage, hidden answer or
action override is introduced at inference. Tests and an emitted frozen-component
audit ensure other parameters cannot change. It passed **8/8 training and 16/16
development tasks**, with perfect recorded expert-history action, control and
dropdown selection. The unused free-text answer decoder's zero exact-match score
is not a tool-copy score; numeric task answers were all correct.

Direct checkpoint comparison confirms that **only the two dropdown weight
tensors changed from 001** (`experiments/lifetime-003/frozen-components.json`).
Its 200 updates and in-process verification took 107.62 seconds with 624,607,232
bytes peak RSS. All losses were finite; the last full cycle mean was 0.07915
(main sequence 0.14921, white dwarf 0.00916, giant 0.00900). These total losses
include fixed components during head-only fitting, so completion and selection
accuracy, not a near-zero scalar, determine the learning gate.

The retained lineage adds 200 recognition + 400 workflow + 200 dropdown-only
updates to the radius parent. The failed 002 comparison separately used 200
workflow updates and is not a parent of 003. All experiment artifacts are kept.

The frozen final evaluation passed **100/100 new held-out tasks**:

| Supplied class | Completed | Required path |
| --- | ---: | --- |
| Main sequence | 50/50 | Six results, 60 actions; lifetime from calculated mass, in years |
| White dwarf | 25/25 | Three results, 31 actions; no mass, radius or lifetime |
| Giant | 25/25 | Three results, 31 actions; no mass, radius or lifetime |

Calculation selection, input binding, numeric answer and unit accuracy were all
100%. Invalid actions, tool errors, API failures and infrastructure failures were
zero. Every applicable case reused the calculated mass for lifetime. Final
evaluation took 93.16 seconds with 281,903,104 bytes peak process RSS; it made zero
optimizer updates and left all model tensors/checkpoint bytes unchanged.
Seeds `8400000`–`8400099` and their four complete templates are now **consumed**;
future tuning needs a new versioned holdout for fresh acceptance claims.

Promoted checkpoint: `experiments/lifetime-003/training/checkpoint.pt`.
SHA-256: `e383ce87cc94e04cf5147f6e3a09ad33caf0578c79b2e85ce27b196527d52c07`.
Final report: `experiments/lifetime-003/final/report.json`.
Action/target calibration used 16 separate episodes / 728 expert-history decisions,
with ECE zero on that small set. This does not measure dropdown-value or whole-task
confidence, nor arbitrary instruction understanding.

All **12/12 manual runtime cases** passed pause/step/resume/save with matching
checkpoint actions, real neural activity and exact offline replay:
`experiments/lifetime-tui-verification-001/report.json`. Another **12/12 paired
class diagnostics** passed: four identical measurement sets/instructions/IDs/
shuffle seeds each ran as main sequence, white dwarf and giant. Only supplied
class, required fields and private grading references changed, and the frozen
model selected the correct 60/31-action path:
`experiments/lifetime-counterfactual-001/report.json`. These are manual-case
diagnostics, not additional fresh acceptance tests.

An interactive Rust TUI run passed on seed `8500000`: paused single-step, resume,
60-action completion and full observation view showing `mass ← r4`, `r6 → lifetime`,
all six exact answers, and lifetime `222373428366.2679 yr` without unit conversion.
Quit returned exit code 0, preserving one successful summary and 60 real neural
activity events. Offline replay matched the saved event stream exactly. Evidence:
`experiments/lifetime-tui-verification-001/interactive.json`.
