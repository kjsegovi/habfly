# Learned stellar radius checkpoint

This checkpoint extends distance → luminosity → temperature → mass with radius
**only for supplied main-sequence classifications**. Giants and white dwarfs
require distance, luminosity and temperature only. Classification learning,
lifetime, planets, habitability and live HabWorlds remain deferred. The original
spreadsheet and previous checkpoints are unchanged.

The unchanged knowledge pack preserves spreadsheet `Sheet1!H2`:
`sqrt(luminosity) / (temperature / 5800)^2`, with inputs `Lsun` and `K` and output
`Rsun`. Independent checks include (1 Lsun, 5800 K) → 1 Rsun,
(16 Lsun, 11600 K) → 1 Rsun, and (4 Lsun, 2900 K) → 8 Rsun.
The policy chooses the operation, both input bindings, result, destination and
unit. The deterministic tool performs arithmetic and exact copying; it does not
repair wrong choices.

## Manual trial

The launcher requires a separate frozen 100-case evaluation to pass before
starting. From `/Users/kjsegovi/Projects/habfly`:

```sh
.venv/bin/python scripts/radius_tui.py --check
.venv/bin/python scripts/radius_tui.py
```

Starts paused: **n** single-steps, **Space** resumes/pauses, **v** changes panels,
and **q** quits. A 150 × 45 terminal is convenient. The observation panel shows
the supplied class, required fields, reference card, bindings, results and units.

- Seed `7500000`: main sequence; expect **51 actions** and five answers:
  distance `ly`, luminosity `Lsun`, temperature `K`, mass `Msun`, radius `Rsun`.
- Seed `7500001`: white dwarf; expect **31 actions** and three answers, with
  neither mass nor radius calculated.
- Seed `7500003`: giant; expect the same **31-action** skip path.

Default main-sequence answers:

| Field | Exact tool value | Unit |
| --- | ---: | --- |
| Distance | 42.27555570310214 | ly |
| Luminosity | 0.0396422654254617 | Lsun |
| Temperature | 2003.4191514655408 | K |
| Mass | 0.3976251511007067 | Msun |
| Radius | 1.6687511482537318 | Rsun |

```sh
.venv/bin/python scripts/radius_tui.py --seed 7500001
.venv/bin/python scripts/radius_tui.py --seed 7500003
```

For main sequence, radius must use the earlier luminosity `r2` **and** temperature
`r3`, not the mass result `r4` or reference-star measurements. Check that `r5` is
copied to radius and paired with `Rsun`. Completion is `task_completed`, not
reaching the 64-action safety cap. All six calculations remain selectable; the
runtime does not override the learned choices. Public required fields and class
are provided, so success is not evidence of independent classification.

No training, browser, spreadsheet or credentials are used by the demo. Traces
save automatically under `experiments/manual-radius/`. Replay is offline:

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/manual-radius/YOUR_RUN.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --replay experiments/manual-radius/YOUR_RUN.jsonl
```

## Bounded experiment and artifacts

The starting checkpoint is `experiments/mass-003/training/checkpoint.pt`. The real
graph has 2,000 nodes and 132,365 edges. Hidden size 16, CPU, one thread, seed 0,
no PPO. No model architecture or graph changes are needed: the v5 observation
encoding already includes radius, its units and the public required-field flags.

Eight training cases cover four main-sequence stars, two white dwarfs and two
giants; every training template covers all three classes. The cap is 200
quantity/unit recognition updates and 400 full-workflow updates. Source-language
parameters stay frozen. Calibration and development each use 16 separate cases;
final evaluation uses 100 unseen cases (50/25/25 by class) and four held-out
complete instruction templates. Split seeds, IDs, inputs and templates are
checked disjoint. Grading answers stay outside observations. Measurements are
randomized independently of class: these are assigned tool-use tasks, not
physically consistent H-R classification examples.

```sh
.venv/bin/python scripts/train_radius.py train --output experiments/radius-001 --updates 400
.venv/bin/python scripts/train_radius.py evaluate --output experiments/radius-001
.venv/bin/python scripts/verify_distance_demo.py experiments/radius-tui-verification-001 --profile configs/radius_tui.json
.venv/bin/python scripts/verify_mass_branch.py experiments/radius-counterfactual-001 --profile configs/radius_tui.json
```

Use fresh output directories when reproducing. Previously consumed final cases
are rejected; reusing cases or templates is rehearsal, not fresh acceptance.
Expected artifacts include split manifests, private cases, expert trajectories,
recognition data, per-update losses, checkpoint, seen/development rollouts,
separate frozen final report, manual runtime/replay checks and class diagnostics.
Local runs install network and Sheets-adapter tripwires; this is not an
operating-system network sandbox.

Action and control-target confidence are calibrated on separate expert histories,
not dropdown-value correctness or whole-task success. Neural activity shows real
recurrent hidden-state RMS, not biological firing rates.

## Verification status

The new regression tests cover independent radius golden cases, applicability,
missing/zero/invalid inputs, both result dependencies, wrong valid bindings,
invalidation/reset isolation, exact copying, finite gradients, checkpoint reload,
split separation, runtime boundaries and TUI rendering. The scripted expert
completes 100 deterministic cases before training. Passing these code/expert
checks is not evidence of learned completion.

The first bounded run, `experiments/radius-001`, passed **8/8 training and 16/16
development tasks**, then **100/100 fresh held-out tasks** with frozen weights:

| Supplied class | Completed | Required path |
| --- | ---: | --- |
| Main sequence | 50/50 | Five results, 51 actions, radius from luminosity and temperature |
| White dwarf | 25/25 | Three results, 31 actions, no mass or radius |
| Giant | 25/25 | Three results, 31 actions, no mass or radius |

Calculation selection, input binding, numeric answer and unit accuracy were all
100%. Invalid actions, tool errors, API failures and infrastructure failures were
zero. Both radius dependencies were reused correctly in all 50 applicable cases.
Final evaluation took 85.21 seconds with 275,283,968 bytes peak process RSS. There
were zero optimizer updates; model tensors and checkpoint bytes stayed unchanged.
Seeds `7400000`–`7400099` and their four templates are now **consumed**. Future
tuning needs a new versioned holdout for fresh acceptance claims.

Checkpoint: `experiments/radius-001/training/checkpoint.pt`.
SHA-256: `0c472155a0d816317d7cb66a6d2e1c6656302de470d03c88d0d4143b3e9e3529`.
Final report: `experiments/radius-001/final/report.json`.

Training used exactly 200 recognition + 400 workflow updates, with no retries or
budget increases. All losses were finite; workflow loss decreased from 10.4524 to
0.01572. Recognition was 98/98 on its separate development cases, and recorded
expert-history action/target/value decisions were all correct on train and
development. The unused free-text answer decoder's zero exact-match score is not
the tool-copy metric: final numeric answers were all correct. Training plus its
in-process validation took 271.62 seconds with 626,737,152 bytes peak RSS. Reload
preserved model tensors.

Action and control-target calibration used 16 episodes / 656 expert-history
decisions. ECE was approximately 0.00000131 and 0 respectively on this small set;
this does not establish calibration on arbitrary instructions or dropdown values.
Source-language parameters, architecture and graph remain unchanged from the
mass parent, as recorded in `experiments/radius-001/frozen-components.json`.

Regression tests: **321 Python passed**, five intentionally inapplicable
parameterizations skipped; **27 Rust/bridge passed**. Ruff, Cargo formatting,
Clippy and patch whitespace checks passed. The original distance, luminosity,
temperature and mass checkpoint hashes are unchanged, and all four old launchers
still pass their readiness checks. All **12/12 mass runtime regressions** passed
with exact offline replay (`experiments/mass-regression-radius-001/report.json`).

All **12/12 radius manual runtime cases** passed pause/step/resume/save, emitted
real checkpoint neural activity and replayed exactly offline
(`experiments/radius-tui-verification-001/report.json`). Another **12/12 paired
class diagnostics** passed: four identical sets of measurements, instructions,
IDs and shuffle seeds each ran as main sequence, white dwarf and giant. Only
supplied class, visible required fields and private grading references changed;
the frozen policy followed the appropriate 51/31-action path
(`experiments/radius-counterfactual-001/report.json`). These are manual-case
diagnostics, not additional fresh acceptance cases.

An interactive Rust TUI run also passed on seed `7500000`: paused single-step,
resume, 51-action completion, full observation view with both radius bindings,
all five exact answers/units, and `r5 → radius`. Quit returned exit code 0 and
preserved one successful summary and 51 real neural-activity events; replay
matched the saved events exactly. Evidence:
`experiments/radius-tui-verification-001/interactive.json`.
