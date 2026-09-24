# Learned distance TUI: ready for a local manual trial

This checkpoint completed **100/100 newly held-out distance tasks**, each in ten
actions, with **zero invalid actions, tool errors, API failures or infrastructure
failures**. It runs the real 2,000-node biological graph with hidden size 16 on one
CPU thread. The calculation tool does the arithmetic; the learned policy chooses
the operation, measurement, binding, result, destination and unit.

This is **distance only** in a local simulator. It does not establish arbitrary
English understanding, correction of earlier mistakes, the other five stellar
calculations, classification, planets, habitability or HabWorlds/browser completion.
Do not open a live account for this trial. No spreadsheet, credentials, network,
training, data collection or submission is started by the launch command.

## Launch

From Terminal on this Mac:

```sh
cd /Users/kjsegovi/Projects/habfly
.venv/bin/python scripts/distance_tui.py
```

Use a terminal around **150 columns × 40 rows** for the overview. Smaller terminals
work with focused panels. The launcher validates the selected checkpoint, knowledge
pack, dataset hashes and passed gates, then starts the Rust TUI **paused**. It uses
Cargo's locked/offline mode; dependencies must already be installed. The default
profile is `configs/distance_tui.json`.

To check the setup without opening the TUI:

```sh
.venv/bin/python scripts/distance_tui.py --check
```

Expected: `status: ready`, `scope: distance_only`, `paused: true`, seed `2400000`.

## What to try

1. Confirm the header says **paused**, **policy checkpoint**, **Stage distance**,
   **seed 2400000**, and **browser not_connected**. Before the first step, neural
   activity is unavailable because no forward pass has run yet.
2. Press **n** once per decision. Watch calculation `distance`, input `parallax`,
   a measurement labeled **current star**, binding, calculation, result selection,
   destination, exact copy, unit **ly**, and final check. The control order is
   shuffled; measurement IDs are not fixed meanings.
3. Press **Space** to resume automatically (0.8 seconds between steps); press it
   again to pause. Use **v** to cycle overview → neurons → controls → full
   observation. **Up/Down** scroll controls, or the full observation in that view.
4. Expect **completed**, **step 10**, `task_completed: true`, and “Stellar task
   completed.” The latter is the environment's generic completion message; only
   its distance field is required here.
5. Press **q** to exit. Quitting an already completed run preserves its successful
   trace. **a** aborts an active run; **s** starts the configured case again.

The first case's current-star parallax is **0.028500955 arcsec**; the copied result
is **114.38213210750304 ly**. The reference-star parallax is a distractor, not an
alternative answer. You can inspect all values and the selected binding in the
full observation panel. The cyan marker tracks the last selected control across
observation-local ID changes when its visible identity remains unambiguous.

Confidence is explicitly **uncalibrated**. A displayed 99% policy probability is
not a measured 99% success guarantee. Neural activity is hidden-state RMS from the
checkpoint's actual recurrent state, not a biological firing rate or a scripted
expert's observer network.

Try another verified demo case (seeds **2400000 through 2400011**):

```sh
.venv/bin/python scripts/distance_tui.py --seed 2400007
```

These twelve demo cases have now been checked through the live runtime and are
not an additional untouched evaluation set. Each completed in ten actions with
pause, single-step, resume, save and offline replay verified.

## Traces and replay

Every live session automatically writes a new JSONL trace under
`experiments/manual-distance/`. Press **t** for an additional copy at the TUI's
default `experiments/traces/tui-session.jsonl`; an existing copy is never overwritten.
Use a fresh `--trace PATH` with the direct Rust command if another copy is desired.

Replay the saved verification run without loading weights or contacting anything:

```sh
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/distance-tui-verification-001/2400000.events.jsonl
```

For text-only replay:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-tui-verification-001/2400000.events.jsonl
```

If the UI crashes, stays paused after **Space**, selects reference-star parallax,
or fails to complete: keep the trace and report the seed, last action and visible
message. Do not retrain or delete artifacts to reset the UI.

## Evidence and boundaries

- Selected checkpoint: `experiments/distance-aliases-002/training/checkpoint.pt`.
  SHA-256: `e8a157c1a020245f50348f922113bb9b685335b11d501df1d1731fd864ff10a1`.
- Development/training report: `experiments/distance-aliases-002/report.json`.
  New paired source recognition: **324/324**, both sides **162/162 pairs**;
  previously consumed rehearsal: **936/936**. Quantity/unit recognition: **100%**.
  New development workflows: **36/36**; original workflows: **4/4**.
- Frozen final report: `experiments/distance-aliases-002/final/report.json`.
  **100/100**, ten actions each, **16.84 seconds**, peak process RSS **271,712,256
  bytes**. Seeds 2300000–2300099, separate numeric cases and complete instruction
  templates. The final set is now consumed; it must not become a repeatedly tuned
  “unseen” checkpoint-selection gate.
- Runtime verification: `experiments/distance-tui-verification-001/report.json`.
  **12/12**, real checkpoint actions and neural activity, exact offline replay.
- Training used two explicitly capped **1,000-update** source-only experiments
  (12 decisions per update), CPU/one thread/seed 0, no PPO. The first solved new
  development but forgot the original instruction; it is retained as a failed
  gate in `distance-aliases-001`. The second included the omitted full instruction
  in contrastive rehearsal. Every non-source parameter and parent checkpoint
  stayed unchanged; checkpoints reloaded identically. No training or checkpoint
  selection occurred after the final 100-case evaluation.

The alias mapping is learned from supervised examples, not a runtime source-name
parser or expert fallback. Correct grading answers remain outside observations.
Earlier failed runs, including the original **0/8** final distance evaluation,
remain intact. This result supports a manual distance demo, not wider deployment.

Final implementation checks: **225 Python tests and 22 Rust tests passed**, including
the real Python subprocess bridge. Ruff, Python/Rust formatting and whitespace
checks passed. All 100 final traces validate as successful 44-event, ten-action
episodes; all split hashes and the checkpoint file remain unchanged. An actual
interactive Rust session also completed seed 2400007, and its trace retained one
successful summary after quitting:
`experiments/manual-distance/4037d814d93d4fbcba59d75fe019523d.jsonl`.
