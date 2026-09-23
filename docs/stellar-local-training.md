# Local-tool stellar training

This is the default stellar path. All commands below run locally/offline with
installed Python/Rust dependencies and the existing biological 2,000-node graph.
They do not open HabWorlds, contact Google, load credentials, or modify a workbook.
The current preview is preserved as an actual attempt, not a resettable fixture.

[Recorded smoke results](stellar-local-verification.md): the implementation and
expert gates pass, but the two-epoch checkpoint completed 0/100 unseen tasks. The
smoke `001` artifacts already exist; replay them or use new output names to rerun.
The larger pilot in section 6 has since been executed and completed **0/16 development
cases** without executing a calculation. Hold the final 100-case learned-policy
evaluation; use the [bounded distance diagnostic](distance-diagnostic.md) next.

For an explicit offline tripwire, replace `.venv/bin/python -m habfly` in any
command with `.venv/bin/python scripts/habfly_offline.py`. This rejects Python
socket connections and Sheets-adapter initialization, and is used for the recorded
smoke experiment. It is not an OS sandbox for untrusted code.

## What the network learns

The network selects an operation, reads its reference card, selects an input
parameter and visible source, binds it, and executes. It then selects a result
and answer destination, copies the number exactly, and chooses units. All are
ordinary SELECT/CLICK controls on the `calculation` or `task` surface. The same
biological core, character tokenizer, and nine action kinds are retained.

The tool does arithmetic only. It never selects, binds, repairs, rounds for the
learner, or substitutes a private grading result. A valid same-unit distractor
produces its actual wrong answer. Bad units, missing/zero/negative/nonfinite inputs,
invalid domains, or inapplicable operations return recoverable tool errors.

Star class is supplied. Distance, luminosity, and temperature are always required;
mass, radius, and lifetime are required only for main-sequence stars. Classification,
planet analysis, habitability, and actual course acceptance remain pending.

## 1. Inspect and validate the knowledge pack

```sh
.venv/bin/python -m habfly knowledge inspect
.venv/bin/python -m habfly knowledge validate
.venv/bin/python -m habfly data validate data/processed/graphs-v2/graph-2000
```

The pack is `src/habfly/packs/stellar_knowledge.json`, separate from the synthetic
Mini-HabWorlds ContentPack. `--pack PATH` selects another compatible pack. The SHA256
covers every formula, constant, unit, rule, reference, and provenance field.

| Operation | Inputs | Spreadsheet equation/conversion | Result |
| --- | --- | --- | --- |
| distance | parallax, arcsec | 3.26 / parallax | ly |
| luminosity | flux, W/m2; distance, ly | distance × 9460500000000000 → metres; flux × 4π × metres² → W; divide by 3.827e26 | Lsun |
| temperature | wavelength, nm | 2897768.5 / wavelength | K |
| mass | luminosity, Lsun | luminosity^(1/3.5) | Msun |
| radius | luminosity, Lsun; temperature, K | sqrt(luminosity) / (temperature/5800)² | Rsun |
| lifetime | mass, Msun | 1e10 × mass^(-2.5) | yr |

Constants deliberately retain the spreadsheet values rather than substituting
newer astronomical constants. Source cells and original expressions are stored
in each reference card. The arithmetic parser accepts only bounded arithmetic
and declared names; there is no Python `eval`, function call, external execution,
or model-authored expression. Supplied expressions must come from the validated pack.

Expected checkpoint: `verified: true`, nine independent fixed golden cases,
pack hash, and valid graph. Goldens test numerical transcription, not HabWorlds
grading fidelity. Private grading references are generated only after this gate.

## 2. Collect smoke demonstrations

```sh
.venv/bin/python -m habfly data demonstrations experiments/stellar-local-data-smoke-001 \
  --task stellar --profile configs/stellar_local_smoke.yaml
```

The scripted expert operates exactly the visible controls available to the learner.
The collector requires 100/100 deterministic expert cases before publishing the
dataset, then records four training, two calibration, two development, and 100
final-test cases. Seeds, numeric tuples, and instruction templates are separated.
Each episode has shuffled controls, distractor measurements, and a 128-action limit.

Artifacts: `manifest.json`, `status.json`, `{split}.json`, `{split}-report.json`,
and per-case `{split}-{index}.events.jsonl`. Case files contain private grading
references; the policy receives only normalized observations, never case records.
Events contain selected reference cards, bindings, actual results, exact copies,
feedback, and completion state, not grading answers. Backend, pack hash, graph hash,
split identities and `local_tool_assisted` mode are recorded in provenance.

Expected checkpoint: `expert_gate_passed: true`, complete split reports, no invalid
expert actions. This is **scripted**, not learned success. Incomplete collections
retain progress/failure traces but are not usable datasets. Output directories are
never overwritten; use new run names and `--dataset PATH` when rerunning.

## 3. Train the bounded smoke checkpoint

```sh
.venv/bin/python -m habfly train policy experiments/stellar-local-smoke-001 \
  --task stellar --profile configs/stellar_local_smoke.yaml
```

Exactly four training episodes, two epochs, CPU, one thread, hidden size 16, seed 0.
BC reuses recorded steps; it does not execute the calculator or use the network.
No PPO, automatic retries with larger budgets, or checkpoint reuse from Sheets.
Artifacts: `checkpoint.pt`, its manifest, `report.json`, and train/validation
trajectory records. Expected checkpoint: finite losses/gradients, reloadable
checkpoint, and measured held-out imitation/calibration. A finite loss does not
establish learned task completion.

## 4. Evaluate closed-loop on unseen cases

```sh
.venv/bin/python -m habfly evaluate model \
  --task stellar --profile configs/stellar_local_smoke.yaml \
  --checkpoint experiments/stellar-local-smoke-001/checkpoint.pt \
  --split test --runs 100 --seed 300000 --output experiments/stellar-local-eval-smoke-001
```

This uses actual tool executions chosen by the policy, not teacher-forced actions.
Report: completion, calculation selection, input binding, result/destination copy,
unit selection, numeric-field accuracy, invalid actions, recoverable tool errors,
infrastructure/API failures, steps/reward, calibration, elapsed time, and process
peak resident memory. Selection accuracy counts an operation as correct when it is
applicable and required for the supplied star; it does not assess optimal ordering.
Numeric-field accuracy is separate from exact copying: copying a wrong result is
still exact, but fails grading. Calibration measures action/target correctness on
held-out expert contexts, not probability of whole-task success.

Gate: at least 90/100 unseen tasks completed and zero invalid actions/infrastructure
failures. `STOP` and step-limit termination are failures, not `task_completed`.
All failures are preserved as `.trajectory.json` and `.events.jsonl`; `report.json`
and `episodes.json` retain scores. Do not tune against this test set. Any later
development beyond the predefined pilot should reserve a fresh final-test split.

## 5. Replay or inspect a live local episode

```sh
.venv/bin/python -m habfly replay experiments/stellar-local-eval-smoke-001/300000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/stellar-local-eval-smoke-001/300000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --autostart \
  --start-payload '{"task":"stellar","calculation_backend":"local","policy":"checkpoint","seed":300000,"graph":"data/processed/graphs-v2/graph-2000","checkpoint":"experiments/stellar-local-smoke-001/checkpoint.pt","dataset":"experiments/stellar-local-data-smoke-001"}'
```

Press `v` for the focused observation view: operation description, input requirements,
bindings, result IDs/units, copies, errors, measurements and answers. Normal panels
show selected controls, action/target confidence, reward and real neural activity.
The protocol remains JSONL version 1; `Observation.calculation` is optional, and old
Sheets/Mini-HabWorlds replays still work. Replays execute neither formulas nor APIs.
Reset clears all result IDs. Changed bindings invalidate the pending result and
dependent history; unrelated completed results remain available. At most 16 result
records are retained per episode to bound observation budgets. Reference cards and
measurements receive independent character budgets so later fields remain visible.

## 6. Larger pilot: explicit next run, not automatic

```sh
.venv/bin/python -m habfly data demonstrations experiments/stellar-local-data-pilot-001 \
  --task stellar --profile configs/stellar_local_pilot.yaml
.venv/bin/python -m habfly train policy experiments/stellar-local-pilot-001 \
  --task stellar --profile configs/stellar_local_pilot.yaml
.venv/bin/python -m habfly evaluate model \
  --task stellar --profile configs/stellar_local_pilot.yaml \
  --checkpoint experiments/stellar-local-pilot-001/checkpoint.pt \
  --split development --runs 16 --seed 200000 --output experiments/stellar-local-dev-pilot-001
.venv/bin/python -m habfly evaluate model \
  --task stellar --profile configs/stellar_local_pilot.yaml \
  --checkpoint experiments/stellar-local-pilot-001/checkpoint.pt \
  --split test --runs 100 --seed 300000 --output experiments/stellar-local-eval-pilot-001
```

Pilot budget: 64 training episodes, five epochs, 16 calibration and 16 development,
100 final-test, same graph/CPU/thread/hidden/seed. The pilot is predefined, not a
response to test-set tuning. These commands do not authorize or start a browser run.

## Optional Google Sheets compatibility

`calculation_backend: google_sheets` selects the retained guarded adapter. Older
runtime settings with `spreadsheet_config` and no backend still resolve to Sheets;
otherwise new stellar settings default to local. Explicit local mode ignores any
credential path. See [the separate guide](stellar-training.md). Neither backend
silently falls back to the other. Dataset and checkpoint content identities must
match exactly; changing formulas or backend requires new demonstrations/checkpoints.
