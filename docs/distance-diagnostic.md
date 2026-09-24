# Bounded distance diagnostic

Current manual checkpoint: [learned distance TUI guide](distance-manual-test.md).
It passes a separately sealed 100-case distance evaluation. The commands and
failed early experiments below are historical research context, not the demo launch path.

Follow-up: [training investigation and repair experiments](distance-training-repair.md)
records the completed v2 failure, explicit experimental model modes, and the
separately approved optimizer-budget comparison. The default diagnostic below
retains its 20-epoch cap.

The stellar pilot completed training but achieved **0/16 development completions**.
It attempted bindings without executing calculations. This diagnostic asks a narrower
question: can the same network learn one complete distance workflow on four cases,
and, only after fitting those cases, transfer it to unseen measurements?

This is a new experiment, not a resumed pilot or an acceptance run. It does not change
the neural architecture, tokenizer, calculator equations, or existing pilot artifacts.
The original `distance-diagnostic-001` run achieved **55% training exact-action
accuracy and 0/4 seen-case completions**; the unseen policy check was held back.
The patched **v2 experiment was trained on the real graph and failed its fit gate**;
later source-recognition investigations are recorded in the linked repair log.
The original artifacts remain intact. Automated small-graph training and gradient
checks alone are code validation, not evidence of learned distance competence.

## Run one command

From Terminal:

```sh
cd /Users/kjsegovi/Projects/habfly
.venv/bin/python scripts/habfly_offline.py diagnose distance \
  experiments/distance-diagnostic-002 \
  --profile configs/distance_diagnostic.yaml
```

This collects its own demonstrations, trains one fresh checkpoint, reloads it,
scores the individual decisions, and checks closed-loop behavior. The offline
wrapper rejects Python socket connections and Sheets initialization. Neither
HabWorlds nor a spreadsheet is opened. Dependencies and graph are already local.

Use a fresh output name if that folder already contains artifacts; it will not
overwrite or resume a prior run. If an error occurs, stop and inspect `status.json`.
Do not rerun with more epochs automatically. No separate collection, training, or
evaluation command is needed for this diagnostic.

## Explicit budget

| Setting | Default and upper bound |
| --- | --- |
| Graph | Existing biological 2,000-node `graphs-v2/graph-2000` |
| Network | Unchanged; hidden size 16 |
| Execution | CPU, one thread, seed 0 |
| Training | Four cases, 20 epochs; **80 episode-level optimizer updates maximum** |
| Supervision | Ten decisions per episode; **800 supervised decisions maximum** |
| Gradient history | All ten actions, with fixed weights until the episode update |
| Calibration | Two separate cases |
| Development imitation check | Two separate cases |
| Closed-loop training-case check | Four seen cases, at most 32 actions each |
| Unseen check | Eight cases, at most 32 actions each, only after the fit gate |

Each expert trajectory contains ten actions. Lower counts/epochs can be configured
for testing; values above the table's caps are rejected. There are no resume,
checkpoint-transfer, PPO, retry, or budget-escalation options. Epoch count is not
a wall-clock guarantee. Progress prints `collecting`, `training`, and evaluation
stages; training may be quiet until the fixed 20 epochs finish.

## What changed in v2

- Distance training retains recurrent gradients across each ten-action episode.
  It averages the ten decision losses, clips gradients, then updates once. State
  resets between episodes. Other existing training profiles retain their one-step
  sequence default; this patch does not silently increase their memory budgets.
- Every visible control is independently encoded before pooling into the recurrent
  input. Long structured controls are split into bounded sections; an oversized
  indivisible field fails explicitly rather than being silently truncated. The
  target head uses the same budget protection. No neuron embeddings, parameters,
  topology, hidden dimensions, or tokenizer vocabulary were added.
- Supervised action loss uses inference's legal-action mask. Target loss uses the
  legal targets for the **expert action kind**, not the model's predicted kind.
  Disabled, wrong-kind, and padded targets are excluded; invalid expert labels fail
  explicitly. Calibration and inference share the same mask helpers.

The four numeric training cases, seed, epochs, optimizer/learning rate, auxiliary
loss weights, and held-out gates are unchanged. **This is not an equal-optimizer-
update comparison**: v1 used 800 per-decision updates; v2 uses 80 episode updates
with the same 800 supervised decision exposures. Longer gradient history uses more
memory. Three fixes changed together, so any improvement is not a single-factor
causal attribution. No success is assumed and no extra epochs are authorized.

The v2 content identity records the sequence length, per-control encoding, and
loss-mask contract, preventing reuse of a v1 diagnostic checkpoint as v2. Existing
checkpoint files and replay logs are preserved. Replaying recorded events remains
offline; rerunning old weights through the changed encoder is not a reproduction
of the old policy's behavior.

## What is simplified—and what is not

Only **distance** is a required answer. The observation still offers six calculation
choices, current-star and reference-star measurements, irrelevant flux/wavelength,
all unit choices, and shuffled controls. The expert follows the existing visible
SELECT/CLICK interface. It does not give the learner stage hints or repair choices.

The private grader and diagnostic stage labels never enter observations, model
inputs, or rewards. The same restricted calculator returns actual selected-input
results; a valid but wrong parallax remains wrong. Numbers are copied exactly.
The single answer destination simplifies destination selection; passing this test
does not establish multi-field destination selection skill.

Cases use new seeds starting at 600000/700000/800000/900000 rather than reusing the
stellar pilot's training/development/final-test seeds. Numeric parallaxes and case
IDs are checked for overlap across diagnostic splits; each split has its own
instruction template. Unseen expert records are collected and reserved, but are
not supplied to optimization, calibration, or development evaluation.

## Two complementary scores

1. **Teacher-forced decisions:** the model receives the expert's observation at
   each step, with recurrent state carried through the episode. Its own decoded
   action is scored for kind, target, selected value, and exact match. No expert
   action is injected into the model. This shows which decisions it can reproduce
   when earlier steps have succeeded.
2. **Closed-loop behavior:** the model's actions actually change the environment.
   The report records correct milestones reached, the first milestone not reached,
   repeated-action counts, completion, and distinct policy/tool/infrastructure
   errors. Milestones are independent observations, not a forced action order.

The ten scored decisions are calculation selection, input-parameter selection,
source selection, binding, execution, result selection, destination selection,
exact copying, unit selection, and completion check. Correct execution/copy scores
require the current star's correctly bound parallax and correct distance—not merely
an execute click or a copied number. No-attempt stage scores remain distinguishable
from attempted-but-incorrect actions.

The **training-fit gate** requires at least 95% exact teacher-forced action accuracy
on training cases, all four training cases completed closed-loop, and zero invalid
actions, tool errors, or infrastructure/API failures. If it fails, the command stops
without running the unseen policy check. It does not add epochs or change settings.

## Reading the outcome

| `outcome` | Meaning and next discussion |
| --- | --- |
| `training_imitation_failed` | The bounded run could not reproduce the training decisions. Inspect per-stage labels, loss, representation, and optimization before scaling. |
| `training_rollout_failed` | Decisions on expert observations fitted, but the free-running workflow failed. Inspect first divergence and repeated actions. |
| `unseen_generalization_failed` | Training-fit gate passed, but transfer to eight new cases did not meet the same 95%/all-completed/error-free diagnostic criterion. |
| `distance_diagnostic_passed` | Both diagnostic gates passed. This is distance-only evidence, not stellar-project or browser acceptance. |
| `infrastructure_failure` | A runtime/IO failure occurred during a rollout; do not interpret it as a learning failure. |

A failure narrows the next investigation; it does not prove a particular root cause.
`stellar_acceptance_gate_passed` is always false. Keep the original 100-case stellar
acceptance evaluation on hold. A completed command can legitimately report a failed
learning gate. Errors after collection begins leave `status.json` with `stage: failed`;
preflight failures (such as a missing graph) can occur before an output folder is created.

## Expected artifacts

- `manifest.json`: configuration, real graph hash/node count, knowledge-pack hash,
  split identities/hashes, expert verification, training contract, optimizer-step
  cap, and supervised-decision cap.
- `train.json`, `calibration.json`, `development.json`, `test.json`: private case
  records and expert trajectories, with separate identities from full stellar runs.
- `training/checkpoint.pt` and its manifest: fresh, reloadable diagnostic checkpoint.
  It rejects ordinary stellar content identities; do not use it in a full stellar run.
- `training/report.json`: finite losses, calibration and imitation results,
  sequence length, actual optimizer updates, and actual supervised decisions.
- `seen/`: training-case closed-loop reports, trajectories, and version-1 JSONL replays.
- `unseen/`: created only if the fit gate passes; unseen closed-loop evidence.
- `report.json`: final outcome, per-stage teacher-forced and closed-loop scores,
  gate decisions, exact budget, elapsed time and process peak memory.
- `status.json`: progress or explicit failure status.

For example, replay the first seen-case rollout without a model or calculator:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-diagnostic-002/seen/600000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/distance-diagnostic-002/seen/600000.events.jsonl
```

The existing TUI displays the distance instruction, calculation inputs/results,
actual actions, reward, and neuron activity. Additional diagnostic stage scores are
in the report, not hidden hints on the policy's observation surface.

After the command finishes, share `report.json` or tell Codex it finished. Inspect
the outcome before authorizing another experiment.

## Implementation validation

- Full Python regression: **162 passed**, including 13 new sequence/encoding/mask
  checks and the 12 distance diagnostic tests.
- Rust regression: **19 regular tests plus the Python subprocess bridge passed**.
- Ruff, touched-file formatting checks, and `git diff --check` passed.
- 100 distance expert cases complete in ten actions each. Wrong parallax, stage
  attribution, fit-gated unseen evaluation, runtime-failure classification, replay,
  new-checkpoint compatibility, and refusal to overwrite artifacts are covered.
- A one-epoch, 16-node synthetic-graph test verifies finite training loss and actual
  checkpoint reload. It is a test fixture, not the user-facing 2,000-node experiment.
- New regression checks cover late controls, long-control splitting, shuffled
  ordering, action-conditioned loss masks, invalid expert labels, backward credit
  through earlier inputs, fixed weights within sequences, partial chunks, episode
  resets, and exact update/decision accounting.
- No new real-graph training experiment is started by implementation validation.
- A single ten-action forward/backward check on the real 2,000-node graph (hidden
  size 16, CPU, one thread, seed 0) produced finite loss and gradients without any
  optimizer update or weight change. It took about 1.43 seconds with a process
  peak RSS of 660 MB. This is a gradient-feasibility check, not a training score or
  a full-run memory/time guarantee. Network and Sheets access were blocked.
