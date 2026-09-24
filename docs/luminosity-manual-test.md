# Learned distance → luminosity checkpoint

This local checkpoint is ready for a manual TUI trial. It uses the real 2,000-node,
132,365-edge graph, hidden size 16, CPU and one thread. It does not open HabWorlds,
Google Sheets, a browser, or a network connection. The prior distance checkpoint
and its launcher are unchanged.

## Run it

From `/Users/kjsegovi/Projects/habfly`:

```sh
.venv/bin/python scripts/luminosity_tui.py --check
.venv/bin/python scripts/luminosity_tui.py
```

The check must say `ready`; the launcher refuses missing, modified or failed-gate
artifacts. It starts **paused**. Use a terminal around 150 columns × 45 rows.

- **n**: one learned action; **Space**: resume/pause.
- **v**: overview → neurons → controls → full observation; arrows scroll.
- **q**: exit. Exiting after completion preserves the successful summary.

Watch the policy select distance, bind current-star parallax, calculate and copy
distance, then select luminosity. It must bind **the current star's flux** and its
own **`r1` distance result**, calculate `r2`, copy it to luminosity, select `Lsun`,
and check completion. The expected workflow takes 22 actions, with a 64-action
safety limit. All six calculation choices and reference-star distractors remain
visible; no expert chooses the next action during checkpoint execution.

Default manual seed `3500000` should finish with:

| Answer | Value | Unit |
| --- | ---: | --- |
| Distance | 46.21574034965014 | ly |
| Luminosity | 46.94995023503717 | Lsun |

Look for `task_completed: true`, both answers, and no tool errors. Other manual
cases are `3500000`–`3500011`, e.g.:

```sh
.venv/bin/python scripts/luminosity_tui.py --seed 3500007
```

Every run saves its JSONL trace under `experiments/manual-luminosity/`. Replay the
path shown by the runtime, without training or credentials:

```sh
.venv/bin/python scripts/habfly_offline.py replay experiments/manual-luminosity/YOUR_RUN.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- --replay experiments/manual-luminosity/YOUR_RUN.jsonl
```

## Verified results

Promoted checkpoint: `experiments/luminosity-002/training/checkpoint.pt`.
SHA-256: `241b645fe5412421fc905b6bd11062b29548fead8812674888866da86e89870c`.

| Gate | Actual result |
| --- | --- |
| Independent knowledge-pack golden checks | 9 passed; includes distance, luminosity and conversions |
| Scripted expert validation | 100/100 chains, 22 actions each |
| Recorded training workflows | 8/8 completed |
| Development workflows | 16/16 completed |
| Frozen final numeric cases | **100/100 completed**, 22 actions each |
| Actual distance-result reuse + current-star flux | 100/100 |
| Calculation selection, binding, numeric answers, units | 100% each on final cases |
| Invalid actions / recoverable tool errors / infrastructure failures | 0 / 0 / 0 |
| Manual runtime pause/step/resume/save and exact offline replay | 12/12 passed |
| Preserved distance checkpoint regression | 12/12 passed |
| Python / Rust regressions (including Python subprocess bridge) | 246 / 23 passed |

The interactive Rust TUI also completed manual seed `3500007` in 22 actions.
Its focused observation showed `r1` bound as distance, current-star `m3` as flux,
both answers and units, and no tool error. Quitting kept one successful summary.
Evidence is in `experiments/luminosity-tui-verification-001/report.json` and
`interactive.json`; the old distance regression is in
`experiments/distance-regression-luminosity-001/report.json`.

The final test took 40.69 seconds with process peak RSS 272,039,936 bytes. Training
and evaluation used socket/Sheets tripwires. These are application-level checks,
not an operating-system network sandbox. The frozen test did zero optimizer
updates, verified unchanged tensors and checkpoint bytes, and preserved every
trajectory, including the failures of the first run.

Action and target probabilities were temperature-calibrated separately on 16
calibration episodes (352 expert-history decisions; measured ECE 0 on this small,
fully correct set). They are **not** calibrated probabilities of whole-task
success or reliable estimates for other tasks. SELECT-option probabilities and
free-form text answers are not claimed as calibrated. Neural activity is real
hidden-state RMS, not a biological firing rate.

The task is narrow: supplied star class, familiar quantity/control vocabulary,
held-out instruction templates, new numeric cases and shuffled controls. It does
not establish general calculation planning, arbitrary instructions, classification,
all six stellar calculations, habitability, browser completion, or a connectome
advantage over baselines. Numeric arithmetic and exact copying are tool-assisted.

## Reproducible bounded experiments

`luminosity-001` preserved an initial failure: 400 workflow updates, 0/16 development
completions and 51.7% teacher-forced exact action accuracy. Mixed source selection
worked, but visible control identities were still confused.

`luminosity-002` continued those weights with an opt-in learned control-label
projection for 800 additional workflow updates. It reached 100% exact action
accuracy on both training and development histories and passed closed-loop gates.
Each invocation also used 200 small quantity/unit recognition updates; the source
language head stayed frozen. Character features were memoized only while their
encoder was frozen, with equivalence tests. New state and control features are
public field identities, not hidden grades, expert stages or next-action hints;
state features enter sensory neurons and decisions still use graph propagation.

The original failed run can be reconstructed with `--control-encoding characters`.
Use fresh output names for any rerun:

```sh
.venv/bin/python scripts/train_luminosity.py train --output experiments/luminosity-repro-001 --control-encoding characters --updates 400
.venv/bin/python scripts/train_luminosity.py train --output experiments/luminosity-repro-002 --parent experiments/luminosity-repro-001/training/checkpoint.pt --updates 800
```

The original final cases (`3400000`–`3400099`) are now **consumed**, not a fresh
gate for future tuning. Evaluation is a separate command and rejects previously
consumed numeric cases in neighboring experiment manifests:

```sh
# Only for a new experiment with a passed development gate and unused test cases:
.venv/bin/python scripts/train_luminosity.py evaluate --output experiments/NEW_EXPERIMENT
```

No training is needed for the manual trial. Reports and cases live beside each
checkpoint; private grading references are never placed in policy observations.
Use `--offset` deliberately when designing a new numeric holdout, and introduce
new held-out wording before claiming another fresh instruction-template gate.
