# Learned stellar mass checkpoint

This checkpoint extends the local distance → luminosity → temperature workflow
with mass **only for supplied main-sequence classifications**. It does not learn
classification, radius, lifetime, planet analysis, or live HabWorlds interaction.
Previous distance, luminosity and temperature checkpoints are preserved.

The existing knowledge pack is unchanged: spreadsheet `Sheet1!G2` is
`luminosity^(1/3.5)` with input `Lsun` and output `Msun`. An independent golden
check uses 128 Lsun → 4 Msun. The learned policy chooses the operation, binds the
previous luminosity result, selects the mass result and destination, and selects
units. The tool performs the arithmetic and exact copying.

## Promotion and manual trial

The mass launcher refuses to run unless a separate frozen 100-case evaluation
passes. A main-sequence case has four required answers; giant and white-dwarf cases
have three. Their supplied classification and required fields are visible, and
the calculation dropdown still offers all six operations. Skipping mass is a
learned workflow choice, not a hidden runtime action override. This does not prove
independent classification or applicability reasoning when required fields are
absent.

From `/Users/kjsegovi/Projects/habfly`:

```sh
.venv/bin/python scripts/mass_tui.py --check
.venv/bin/python scripts/mass_tui.py
```

Starts paused: **n** single-steps, **Space** resumes/pauses, **v** changes panels,
and **q** quits. The full observation view includes supplied class, required
fields, bindings, results, answers and units. A 150 × 45 terminal is convenient.

- Seed `6500000`: main sequence; expect **40 actions** and distance in `ly`,
  luminosity in `Lsun`, temperature in `K`, and mass in `Msun`.
- Seed `6500001`: white dwarf; expect **31 actions**, no mass calculation.
- Seed `6500003`: giant; expect **31 actions**, no mass calculation.

Default main-sequence answers:

| Field | Exact tool value | Unit |
| --- | ---: | --- |
| Distance | 47.81837326235952 | ly |
| Luminosity | 0.21876686379107962 | Lsun |
| Temperature | 2438.0840815812094 | K |
| Mass | 0.6477736790802984 | Msun |

```sh
.venv/bin/python scripts/mass_tui.py --seed 6500001
.venv/bin/python scripts/mass_tui.py --seed 6500003
```

For main sequence, verify that the mass input is the calculated luminosity `r2`,
not distance, temperature, or a reference-star reading. The new result `r4` should
be copied exactly to mass and paired with `Msun`. Completion is `task_completed`,
not reaching the 64-action safety cap. No spreadsheet, credentials, browser or
training is started by these commands.

Traces are saved automatically under `experiments/manual-mass/`. Replay is offline:

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/manual-mass/YOUR_RUN.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --replay experiments/manual-mass/YOUR_RUN.jsonl
```

## Bounded experiment

The first two runs start from the promoted temperature checkpoint. Real 2,000-node /
132,365-edge graph, hidden size 16, seed 0, CPU, one thread, no PPO. Eight training cases cover four main
sequence, two white dwarfs and two giants. Each training template covers all
classes, with shuffled controls and reference-star distractors.
Numbers are randomized independently of the supplied class: these cases test
tool use under an assignment, not physically consistent H-R classification.

The cap is 200 quantity/unit recognition updates plus 400 complete workflow
updates. Source-language parameters remain frozen. Calibration and development
each use 16 separate cases. Final evaluation uses 100 different cases:
50 main sequence, 25 white dwarf, 25 giant, with four held-out instruction
templates. Seeds, numeric inputs, case IDs and complete instruction templates are
checked disjoint across splits. Generated grading answers are never observations.

```sh
.venv/bin/python scripts/train_mass.py train --output experiments/mass-002 --updates 400
.venv/bin/python scripts/train_mass.py train --output experiments/mass-003 --parent experiments/mass-002/training/checkpoint.pt --recognition-updates 0 --updates 200
.venv/bin/python scripts/train_mass.py evaluate --output experiments/mass-003
.venv/bin/python scripts/verify_distance_demo.py experiments/mass-tui-verification-001 --profile configs/mass_tui.json
.venv/bin/python scripts/verify_mass_branch.py experiments/mass-counterfactual-001
```

Use fresh output names when reproducing; previous outputs are never overwritten.
Final evaluation rejects already-consumed test seeds. Reusing earlier cases or
templates is rehearsal, not a fresh acceptance gate.

Expected artifacts include manifests, private cases, expert demonstrations,
recognition records, per-update losses, teacher-forced scores, development and
seen-case trajectories, a reloadable checkpoint, frozen final report, and runtime
verification with exact offline replay. Training, development and final scores
are reported separately; passing code tests is not learned task completion.

Action-kind and target confidence are calibrated on separate expert histories.
They are not probabilities of dropdown-value correctness or whole-task success.
Neural telemetry is real recurrent hidden-state activity, not a biological firing
rate. Experiments use network and Sheets-adapter tripwires; these are not an
operating-system network sandbox.

## Current verification

The initial `mass-001` run completed 4/8 training and 8/16 development cases:
all giants and white dwarfs, but no main-sequence cases. It checked completion
immediately after temperature. Its final test was **not opened**. Evidence is
preserved in `experiments/mass-001/branch-diagnostic.json`: with frozen weights
and the same measurements/history, changing only public class/required fields
still produced Check in both branches. The v4 explicit state vector did not
include either field, leaving the distinction in pooled character/control input.

`mass-002` uses `structured_tool_v5`, adding public required-field and supplied-class
flags at the sensory input. It adds 192 projection weights; the fixed biological
topology, hidden size, shared recurrent core, action heads, calculation formulas
and source-language parameters are unchanged in structure. New columns start at
zero, and migration is tested to preserve v4 outputs before learning. There are no
new applicability rules, action overrides, or expert stage hints in this encoding.

Old checkpoints retain their v4 encoding and files. Their confidence is
conservatively marked uncalibrated after model-source changes; no old calibration
is silently carried over. The new checkpoint gets its own calibration.

The second run still completed 8/16 development cases, but reached 100% action-kind
and target accuracy on recorded expert histories. The only remaining supervised
error was choosing temperature instead of mass in the Calculation dropdown at
step 31. Its frozen training-history scores are saved in
`experiments/mass-002/option-diagnostic.json`. Its final test was **not opened**.
`mass-003` continues from this checkpoint for 200 additional workflow updates on
the same eight cases. No new recognition updates are run; skipping them requires
a perfect frozen recognition check. No additional cases or PPO are introduced.

The third run passed **8/8 training and 16/16 development** tasks, then completed
**100/100 fresh held-out tasks** with frozen weights:

| Supplied class | Completed | Required path |
| --- | ---: | --- |
| Main sequence | 50/50 | Four results, 40 actions, mass from luminosity |
| White dwarf | 25/25 | Three results, 31 actions, no mass |
| Giant | 25/25 | Three results, 31 actions, no mass |

Calculation selection, input binding, numeric answer and unit accuracy were all
100%. Invalid actions, tool errors, API failures and infrastructure failures were
zero. Final evaluation took 71.45 seconds with 275,808,256 bytes peak process RSS.
There were zero optimizer updates, and model tensors/checkpoint bytes remained
unchanged. Seeds `6400000`–`6400099` and their four templates are now **consumed**;
future tuning requires a new versioned holdout for fresh acceptance claims.

Checkpoint: `experiments/mass-003/training/checkpoint.pt`.
SHA-256: `bf5e52fcb6ffb9014301b43a2ac9718af0d75fcd41938253ceab51e18de90c90`.
Final report: `experiments/mass-003/final/report.json`.
The retained successful lineage adds 200 recognition + 600 workflow updates to
the temperature parent; the failed v4 comparison separately used 200 + 400.
All losses were finite and checkpoint reloading preserved tensors. The unused
free-text answer decoder's exact-match score is not a tool-copy metric; numeric
answer accuracy above comes from completed task fields.

Regression tests: **298 Python passed** (two non-mass parameterizations of a
mass-specific guard skipped), **26 Rust/bridge passed**. Ruff, Cargo formatting,
Clippy and patch whitespace checks passed. All 12 previous temperature runtime
cases still complete, pause/step/resume/save, and replay exactly:
`experiments/temperature-regression-mass-002/report.json`. Distance, luminosity
and temperature checkpoint file hashes remain unchanged. Old launchers still
pass their promotion checks.

The new action-kind and control-target calibration used 16 separate episodes /
568 expert-history decisions. Both calibration ECE values are zero on this small
set; this does not establish calibration for arbitrary instructions or dropdown
values. Source-language tensors stayed identical to the temperature parent;
recognition tensors stayed unchanged in the continuation, as recorded in
`experiments/mass-003/frozen-components.json`.

All **12/12 manual runtime cases** passed pause/step/resume/save and exact offline
replay (`experiments/mass-tui-verification-001/report.json`). Another **12/12 paired
class diagnostics** passed: four identical measurement sets/instructions each ran
as main sequence, white dwarf and giant. Only supplied class, visible required
fields and private grading references changed. The policy followed the correct
40/31-action branch without updates, confirming the choice was not tied to those
numbers (`experiments/mass-counterfactual-001/report.json`). These are manual-case
diagnostics, not additional fresh acceptance cases.

The first interactive run exposed a Rust JSON parsing precision issue: the
Python luminosity `0.21876686379107962` displayed as `0.2187668637910796` (one
binary64 ULP lower). Python tool execution and exact copying were unaffected.
Enabling the already-installed JSON library's `float_roundtrip` feature fixes
display/replay parsing. A protocol regression test failed before the fix and now
checks the exact floating-point bits and serialization round trip. No model,
knowledge pack, checkpoint or training data changed for this fix.

The corrected interactive Rust TUI check passed on seed `6500000`: paused
single-step, resume, 40-action completion, focused observation with supplied
class, all four exact results/units, `luminosity ← r2`, and `r4 → mass`. Quitting
preserved its single successful summary and 40 real neural-activity events.
Evidence: `experiments/mass-tui-verification-001/interactive.json`.
