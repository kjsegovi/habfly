# Local stellar checkpoint — 2026-09-23

Historical smoke snapshot: the pilot was unrun at the time of this report. It has
since completed with 0/16 development completions. The next investigation is the
[distance-only diagnostic](distance-diagnostic.md); the smoke results below are preserved.

Implementation and bounded experiment evidence. This is **local_tool_assisted**
learning, not autonomous arithmetic and not a HabWorlds acceptance attempt.
No Google spreadsheet, Excel workbook, credential file, or current browser preview
was used or modified by this experiment. The larger pilot has not been run.

## Implementation checks

- Nine independent fixed numerical goldens pass, preserving all six spreadsheet
  operations and intermediate luminosity conversions. Solar-normalized cases are
  also checked. Units and main-sequence applicability are enforced.
- 100 deterministic local expert cases complete through the visible SELECT/CLICK
  interface. Tests cover wrong bindings, no automatic correction, exact copying,
  stale results, reset isolation, units, missing/zero inputs, restricted arithmetic,
  token budgets, finite gradients, backend/checkpoint compatibility, and replay.
- Local tests block socket connections and Sheets initialization. BC tests additionally
  fail if any calculation is executed during training. Replay is tested with both
  backends unavailable.
- Python final full regression: **137 passed** in 64.49 seconds, including local
  browser fixtures, old Mini-HabWorlds/Sheets behavior, and the real-graph check.
- Rust: 19 regular tests plus the installed Python-runtime bridge test passed;
  Clippy with warnings denied and formatting passed. The local-tool observation
  renderer is covered alongside legacy Sheets/replay views.
- Real biological 2,000-node graph, hidden size 16: calculation observations change
  outputs; forward values and backward gradients are finite.
- Ruff and `git diff --check` pass. Existing unrelated/untracked work is preserved;
  no commit or push was performed.

## Actual smoke run

All CLI commands used `scripts/habfly_offline.py`, which rejects Python socket
connections and Google Sheets adapter initialization. The script is a tripwire,
not an operating-system security sandbox. Dependencies were already installed.

| Setting | Actual value |
| --- | --- |
| Graph | Real MaleCNS 2,000 nodes, 132,365 edges |
| Graph fingerprint | `580eaa081ccf57e466343352a9a61b16a4ec8f025055c0797afe0cca7a228bfd` |
| Knowledge pack fingerprint | `a556ff2abff8311d7c77db7575f3d4e88f39d14e45933f0bce664cf80a22c872` |
| Training | Four episodes, two epochs, CPU, one thread, hidden size 16, seed 0 |
| Held-out partitions | Two calibration, two development, 100 final-test cases |
| Episode action cap | 128 |
| Collector expert gate | 100/100 completed, zero invalid actions |
| Dataset collection | 8.33 seconds |
| Training epoch losses | 8.935669 → 6.869628 |
| Training elapsed | 65.02 seconds |
| Training process peak RSS | 1,367,867,392 bytes (about 1.27 GiB) |
| Development exact-action accuracy | 10.99% over 91 teacher-forced steps |
| Development action-kind accuracy | 64.84% |
| Development target accuracy | 18.68% |
| Development value exact accuracy | 13.19% |
| Reload | Checkpoint loaded successfully for closed-loop evaluation |

Calibration is temperature scaling on separate expert trajectories: action
temperature 1.51005, NLL 0.67378, ECE 0.06327 (91 contexts); target temperature
1.32132, NLL 2.00270, ECE 0.09319 (59 targeted contexts). These are action/target
probabilities under held-out expert contexts, not calibrated task-success probabilities.
The short-answer head's generic imitation score is not stellar numeric accuracy;
stellar answers are exact tool-result copies and are scored separately in evaluation.

## Actual closed-loop result: learning gate failed

The reloaded checkpoint completed **0/100 unseen tasks**. All 100 episodes reached
the 128-action limit: 12,800 actions total. There were zero invalid actions,
recoverable tool errors, API failures, or infrastructure failures. This is a
policy-learning failure, not successful task completion or a calculator failure.

| Evaluation metric | Actual result |
| --- | --- |
| Applicable/required operation selections | 3,926 / 3,972 (98.84%) |
| Input bindings | Zero attempted; accuracy not meaningfully measured |
| Result-to-answer copies | Zero attempted; accuracy not meaningfully measured |
| Unit selections | Zero attempted; accuracy not meaningfully measured |
| Correct numeric fields | 0 / 462 (0%) |
| Correct final units | 0 / 462 (0%) |
| Completed tasks | 0 / 100 (0%) |
| Invalid actions / tool errors / infrastructure failures | 0 / 0 / 0 |
| Evaluation elapsed | 234.43 seconds |
| Evaluation process peak RSS | 1,236,254,720 bytes (about 1.15 GiB) |
| Expansion gate | Failed; required at least 90/100 completed, zero invalid actions |

The operation-selection metric counts repeated selections of an applicable required
operation; it is **not** calculation execution or workflow accuracy. The report's
zero-attempt ratios are numerically zero, with attempt counters preserved. For
example, seed 300000 repeatedly selected the `distance` answer destination and
`luminosity` input parameter, never bound a source, and produced no tool result.
No test-driven tuning, extra epochs, or larger experiment followed this failure.

The offline CLI replay of seed 300000 passed and emitted all 516 version-1 events.
Failure trajectories and neural activity are retained for inspection. The correct
next learning checkpoint is a deliberately launched pilot, not a live browser attempt.

## Artifacts and next checkpoint

- Dataset and expert traces: `experiments/stellar-local-data-smoke-001/`.
- Training report and reloadable checkpoint: `experiments/stellar-local-smoke-001/`.
- Closed-loop report and failure replays: `experiments/stellar-local-eval-smoke-001/`.
- [Sequential commands and expected outputs](stellar-local-training.md).

The predefined larger pilot remains an explicit next run: 64 episodes, five epochs,
16 calibration and 16 development cases. No automatic budget increase, PPO, browser
attempt, classification, planet, or habitability expansion was performed. Preserve
this smoke checkpoint and its failures; code correctness and scripted expert success
do not establish learned competence or the 90/100 expansion gate.
