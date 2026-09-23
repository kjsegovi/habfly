# Implementation verification — 2026-09-22

This records engineering evidence. Learning/generalization and real HabWorlds acceptance are separate gates.

Final checks: **82 Python tests passed** (including 25 Chromium fixture cases), **17 Rust tests passed**, and the separately enabled live Rust/Python integration test passed. Ruff, Clippy with warnings denied, Rust formatting, Git whitespace checks, dependency lock verification, and wheel packaging passed.

## Real data

- Canonical neurons: 166,700.
- Raw connection rows: 151,856,684.
- Eligible directed edges: 25,582,938.
- Nested graph sizes: 2,000; 5,000; 10,000; 30,000.
- Each selected node is reachable from a sensory input and can reach an output using nonzero signed edges.
- Corrected role-balanced build: 60.61 seconds, 1,557,676,032 bytes peak RSS.
- Artifacts: `data/processed/graphs-v2/graph-{size}`. Earlier selection is retained as `graphs-v1` and is not used by the profiles.

## Numerical model evidence

`data/processed/graphs-v2/model-smoke.json` contains source hashes, graph hashes, timings, and an exact provenance supplement for the immutable graph build.

- 100 finite, deterministic repeated forwards on the real 2,000-node graph.
- Changed instructions change the action outputs.
- Real 30,000-node no-gradient inference passes.
- Small-graph finite, nonzero gradients reach character embeddings, sensory projection, and shared recurrent cell.
- Severing the graph eliminates instruction influence on the output; changing edge signs changes output activity.
- Same model parameter count across graph sizes: 48,784 at hidden width 32.

Reproduce the real graph/model check:

```sh
.venv/bin/python tests/test_graph_model_integration.py --real-graphs data/processed/graphs-v2 --report experiments/real-graph-smoke.json
```

## Simulator, browser, and terminal

- The scripted synthetic expert completes 1, 3, and 30 stars in 15, 43, and 421 steps respectively for the checked seed.
- Gymnasium's environment checker passes on MiniHabWorlds.
- Chromium fixtures verify visible-only extraction, stale targets, fields/options, hover/zoom/drag, pixel fallback, navigation, popups, modal/authentication stops, time/step/repetition limits, and final-submission gating.
- Rust tests cover JSONL parsing, bounds, subprocess behavior, rendering, scoped confidence, and terminal layouts.
- A separate live Rust-to-Python bridge test covers start, pause, step, resume, save, replay, and abort.
- An actual PTY run verified start, single-step, clean exit, and terminal restoration.
- The Python wheel includes the lesson JSON and all modules. Dependency lock validation succeeds.

Full suite command: `.venv/bin/python -m pytest -q`. Browser tests require a launch-capable environment; the first sandboxed full run could not launch Chromium because macOS denied its bootstrap port. The same suite succeeded when run with the needed process permission. This was an environment restriction, not a skipped browser check.

## Remaining acceptance work

Smoke training reports and checkpoints are under `experiments/verification-model-final/`. Their low-budget scores must not be interpreted as evidence of learned HabWorlds completion. Experiments preserve source/lock hashes, dirty-worktree status, samples/trajectories, calibration scope, and measured gates.

The fresh `policy-current/checkpoint.pt` smoke run used one epoch and two expert episodes. It reached 20% exact held-out action imitation. Its subsequent closed-loop test completed **0/3** one-star episodes; all three reached their 90-step limit with no invalid control actions. The report and trajectories are in `experiments/verification-model-final/closed-loop/`. Recognition, grounding, arithmetic, and all three-seed baseline smoke runs likewise reported unmet learning gates. These are functional training checks, with no claim of solved task behavior.

The exact HabWorlds content pack, faithful simulator mapping, local activity configuration, fresh-account provisioning, chart-image training, and learned 1→3→30-star browser runs remain pending. No real HabWorlds account or course record was accessed during implementation.
