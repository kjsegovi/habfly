# Learned distance, luminosity and temperature

This is a local, tool-assisted three-calculation checkpoint, not a HabWorlds
browser attempt. It preserves the working distance and two-calculation models.
The calculator's spreadsheet-derived equations, constants and units are unchanged.
The network learns selections and binding; the tool performs arithmetic and exact
copying. No spreadsheet, credentials or network is required.

## Manual trial

From `/Users/kjsegovi/Projects/habfly`:

```sh
.venv/bin/python scripts/temperature_tui.py --check
.venv/bin/python scripts/temperature_tui.py
```

The profile must pass its saved held-out promotion gate before it can launch. It
starts paused: **n** steps, **Space** resumes/pauses, **v** cycles views, **q** quits.
Use the full observation view to inspect measurements, bindings, all three results,
answers and units. Around 150 columns × 45 rows is comfortable; arrows scroll.

The expected workflow takes **31 learned actions**:

1. Select distance, bind current-star parallax, calculate, copy, and choose `ly`.
2. Select luminosity, bind current-star flux and calculated distance `r1`,
   calculate, copy, and choose `Lsun`.
3. Select temperature, bind current-star peak wavelength in `nm`, calculate,
   copy, and choose `K`.
4. Check all three answers; expect `task_completed: true` and no tool error.

The 64-action limit is a safety stop, not success. All six calculations and
reference-star distractors remain available. No expert supplies inference actions.

Default manual seed `5500000` has these expected outputs:

| Field | Value | Unit |
| --- | ---: | --- |
| Distance | 50.49916409941928 | ly |
| Luminosity | 0.016427934512163192 | Lsun |
| Temperature | 2643.53305309766 | K |

Other manual seeds are `5500000`–`5500011`:

```sh
.venv/bin/python scripts/temperature_tui.py --seed 5500007
```

Traces save automatically to `experiments/manual-temperature/`. Use the path
shown in the runtime to replay without retraining or credentials:

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/manual-temperature/YOUR_RUN.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --replay experiments/manual-temperature/YOUR_RUN.jsonl
```

## Experiment history and boundaries

All runs used the real 2,000-node / 132,365-edge graph, hidden size 16, CPU,
one thread and seed 0. No PPO or graph/model architecture change was introduced.

| Experiment | Bounded work | Development result | Final test |
| --- | --- | --- | --- |
| `temperature-001` | 200 quantity/unit recognition + 400 full-workflow updates | 16/16 | **0/100**, failed and retained |
| `temperature-source-001` | 800 source-language-only updates | 4/16 | Not opened |
| `temperature-source-002` | 1,000 source-language-only updates; clause composition | 12/16 | Not opened |
| `temperature-source-003` | 500 source-language-only updates; punctuation variation | 16/16 | **100/100**, frozen model |

The first failure was source-language generalization, not arithmetic or the absence
of HabWorlds. With weights/candidates fixed, replacing only the failed wording
(“not those of the reference star”) with development wording changed the selected
source from reference to current. Evidence is preserved in
`experiments/temperature-001/failure-diagnostic.json` and its failed trajectories.

The repair used balanced pairs requesting **either** current or reference star,
with identical candidates and graph context within each pair. Systematic clause
order and punctuation variation prevents solving the training task by always
choosing current, or the first/last mentioned star. The final repair passed all
200 paired development decisions, all recorded language rehearsal, 8/8 new
workflows and 8/8 original workflows. Workflow, quantity/unit parameters and
parent checkpoint files were checked unchanged during these source-only updates.

Failed final wording is explicitly marked consumed rehearsal. Failed numeric cases
and trajectories were **not** used for repair training. The new final set uses
`5400000`–`5400099` and four new instruction templates, separate from training,
calibration, development and manual cases. Both final sets must be treated as
consumed once evaluated; do not reuse them as fresh gates for future tuning.

Selected checkpoint: `experiments/temperature-source-003/training/checkpoint.pt`.
SHA-256: `dfdabff4308e2b743684a186b67fc191adfe14d29925134369237c11d25d65b6`.
The final report is `experiments/temperature-source-003/final/report.json`.

All 100 new final tasks took 31 actions, correctly reused the distance result and
selected current-star flux and wavelength, and copied all three outputs exactly.
Calculation selection, input binding, numeric answer and unit accuracy were all
100%; invalid actions, tool errors, API errors and infrastructure failures were
all zero. The regression suite passed **271 Python + 24 Rust/bridge tests**.
The frozen final evaluation took 59.38 seconds and peaked at 271,286,272 bytes
process RSS; checkpoint bytes and model tensors remained unchanged. All 12 manual
runtime cases passed pause/step/resume/save and exact offline replay verification.
The original distance and luminosity checkpoints each retained 12/12 manual
runtime passes, with their checkpoint files unchanged.

Runtime evidence: `experiments/temperature-tui-verification-001/report.json`.
An interactive Rust TUI check on seed `5500007` also passed: paused single-step,
resume, focused observation with all three answers/units, and quit after completion.
Its trace retained one successful summary and 31 neural-activity events, with no
errors or checkpoint changes; evidence is in
`experiments/temperature-tui-verification-001/interactive.json`.
Earlier-checkpoint regressions: `experiments/distance-regression-temperature-001/`
and `experiments/luminosity-regression-temperature-001/`. Runs used socket and
Sheets-adapter tripwires, not an operating-system network sandbox.

Action-kind and control-target probabilities are calibrated on 16 separate
episodes / 496 expert-history decisions. They do **not** measure dropdown-value
correctness or whole-task success. The initial failed run could choose the right
control confidently while selecting the wrong measurement. Neural activity is
real recurrent-state RMS, not biological firing rate.

This establishes only known-vocabulary, local three-calculation behavior. Arbitrary
instructions, classification learning, mass/radius/lifetime, planets, habitability,
connectome superiority over baselines and live HabWorlds completion remain unproven.

## Reproduce bounded runs

Use fresh output names; existing experiments are never overwritten. These commands
do not start a live HabWorlds attempt. Reproducing used cases is rehearsal, not a
new final acceptance test.

```sh
.venv/bin/python scripts/train_temperature.py train --output experiments/temp-repro-001 --updates 400
.venv/bin/python scripts/train_temperature_source.py train --output experiments/temp-repro-source-001 --parent experiments/temp-repro-001/training/checkpoint.pt --context-dataset experiments/temp-repro-001 --updates 800
.venv/bin/python scripts/train_temperature_source.py train --output experiments/temp-repro-source-002 --parent experiments/temp-repro-source-001/training/checkpoint.pt --context-dataset experiments/temp-repro-001 --composition --updates 1000
.venv/bin/python scripts/train_temperature_source.py train --output experiments/temp-repro-source-003 --parent experiments/temp-repro-source-002/training/checkpoint.pt --context-dataset experiments/temp-repro-001 --composition --punctuation --updates 500
```

Final evaluation is separately invoked and rejects failed development gates,
changed checkpoint/content identities, already-opened final directories and reused
numeric seeds found in neighboring final manifests. A future fresh evaluation
requires a deliberately versioned new numeric and instruction-template holdout.
