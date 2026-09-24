# HabFly

A local connectome-constrained learning experiment, with a Python/PyTorch engine and a Rust terminal interface. Develop on the Mac, train larger graphs on the Framework, and watch decisions and recurrent neuron activity beside a Playwright-controlled browser.

The default runnable demo uses an explicitly **synthetic star-analysis environment**. Its scripted expert completes one, three, or 30 stars. A separate **learned distance-only checkpoint** now passes 100/100 held-out local tasks and is ready for a [manual TUI trial](docs/distance-manual-test.md). Neither demo establishes learned completion of the actual HabWorlds Project.

## Try the learned local checkpoints

The newest checkpoint adds **main-sequence radius** to distance, luminosity,
temperature and mass. It passed **100/100 new held-out local tasks**, reusing both
luminosity and temperature for radius. Supplied giant/white-dwarf classes skip
mass and radius; classification itself is not learned. Start the paused TUI with:

```sh
.venv/bin/python scripts/radius_tui.py
```

See the [radius manual guide](docs/radius-manual-test.md) for the 51-action
five-calculation path, 31-action skip path, exact answers and offline replay.

The preserved mass checkpoint adds **main-sequence mass** to distance, luminosity and
temperature. It passed **100/100 new held-out local tasks**: 50 main-sequence
stars calculate mass from luminosity; 25 white dwarfs and 25 giants skip it.
Classification is supplied, not learned. Start the paused TUI with:

```sh
.venv/bin/python scripts/mass_tui.py
```

See the [mass manual guide](docs/mass-manual-test.md) for the 40-action mass path,
31-action skip path, expected values, preserved experiments and replay.

The preserved three-calculation checkpoint adds **temperature** to distance and luminosity. After a
source-language repair, it passed **100/100 new held-out local tasks**. Start the
paused three-calculation TUI with:

```sh
.venv/bin/python scripts/temperature_tui.py
```

See the [three-calculation manual guide](docs/temperature-manual-test.md) for
expected answers, preserved failed experiments, scope and replay.

The preserved **distance → luminosity** checkpoint passes **100/100** held-out
local tasks, including reuse of the calculated distance and selection of the
current star's flux. Start its paused TUI with:

```sh
.venv/bin/python scripts/luminosity_tui.py
```

See the [two-calculation manual guide](docs/luminosity-manual-test.md) for expected
answers, scope, verified scores and replay. This remains local tool-assisted
learning, not a live HabWorlds attempt.

With this machine's existing graph, dependencies and verified checkpoint:

```sh
.venv/bin/python scripts/distance_tui.py
```

It starts paused. **n** steps, **Space** resumes, **v** cycles panels, **q** exits.
No browser, spreadsheet or training is started. See the [manual guide and actual
scores](docs/distance-manual-test.md) for scope, expected results and replay.

## Run the terminal demo

From this repository, with Python 3.11+ and Rust installed:

```sh
uv sync --frozen
cargo run --manifest-path tui/Cargo.toml --locked -- --autostart
```

The TUI starts `.venv/bin/python -u -m habfly runtime --jsonl`. It shows the current observation, visible options, chosen action and target, confidence when available, rewards, progress, neuron body IDs and metadata, population activity, and the event timeline.

Press **Space** to pause/resume, **n** to single-step while paused, **a** to abort, **t** to save the trace, and **q** to exit. Use arrow keys to scroll controls. A 150×45 terminal shows the full layout.

```sh
cargo run --manifest-path tui/Cargo.toml --locked -- --autostart --start-payload '{"policy":"expert","stars":3,"seed":12}'
cargo run --manifest-path tui/Cargo.toml --locked -- --replay experiments/runs/YOUR_RUN.jsonl
```

The default policy is a **scripted expert**. Its neuron panel shows a real, untrained observer network responding to the observations; those neural activations do not select the expert's actions. A checkpoint run labels its actions as learned. Missing confidence is shown as unknown, and uncalibrated probabilities are labeled explicitly. Neural activity is hidden-state RMS, not a biological firing rate.

To inspect actual MaleCNS body IDs and population activity using the graph already built on this machine:

```sh
cargo run --manifest-path tui/Cargo.toml --locked -- --autostart --start-payload '{"graph":"data/processed/graphs-v2/graph-2000"}'
```

Press **v** to cycle focused observation/neuron views on a smaller terminal.

## Connectome checkpoint

The original canonical neuron table and scripts remain available. Raw data and generated arrays stay outside Git.

```sh
.venv/bin/python scripts/inspect_connectome.py
.venv/bin/python -m habfly data inspect
.venv/bin/python -m habfly data build --output data/processed/my-graphs
.venv/bin/python -m habfly data validate data/processed/my-graphs/graph-2000
```

The builder streams the 151,856,684 source edges twice, retains edges joining eligible canonical neurons, and uses disk-backed arrays. It selects nested graphs using role-balanced, complete directed paths. Every selected neuron must be reachable from a sensory input and able to reach an output through nonzero weights. Source hashes, ordering, selection policy, array hashes, and validation results are recorded in `graph_manifest.json`. Existing artifact directories are never overwritten.

The verified local build is `data/processed/graphs-v2/`:

| Nodes | Edges | Nonzero signed edges |
| ---: | ---: | ---: |
| 2,000 | 132,365 | 128,416 |
| 5,000 | 539,545 | 527,852 |
| 10,000 | 1,421,939 | 1,398,736 |
| 30,000 | 4,420,405 | 4,375,287 |

The eligible edge count is **25,582,938**. The build and validation took approximately 61 seconds with a peak resident footprint of 1.45 GiB on this Mac. An initial `graphs-v1` build is retained, but `graphs-v2` corrects its output-heavy selection and is the configured graph family.

Weights use the canonical table's provisional transmitter sign and raw synapse count, normalized by incoming absolute signed mass. Unknown polarity remains zero. Role proportions are soft selection targets; connectivity takes precedence. The current masks combine sensory/visual inputs and descending/efferent readout. Morphology and raw EM data are unnecessary for this graph pipeline.

Source: [Janelia MaleCNS v1.0 downloads](https://male-cns.janelia.org/download/), attributed under the release's CC-BY terms. HabFly is an artificial recurrent network constrained by connectome wiring.

## Model and training checkpoints

Start with [Local-tool stellar training](docs/stellar-local-training.md), the
recommended offline path. A versioned knowledge pack preserves the spreadsheet
equations, constants, units, applicability, and provenance. The network learns
which calculations, inputs, results, destinations, and units to select; a bounded
deterministic tool performs arithmetic. No spreadsheet app, credentials, network,
vector database, or downloaded model is required after dependencies are installed.
The [bounded smoke report](docs/stellar-local-verification.md) records passing
tool/expert gates but **0/100 learned task completions**. The later pilot completed
**0/16 development cases**. Subsequent [distance-only investigations](docs/distance-diagnostic.md)
now produce a checkpoint completing **100/100 new distance cases**. This does not
change those earlier full-task failures. Use the [learned distance TUI guide](docs/distance-manual-test.md)
for the distance checkpoint, or the [radius TUI guide](docs/radius-manual-test.md)
for the latest conditional five-calculation checkpoint. Full-task training and
browser acceptance remain later gates.

[Google Sheets stellar training](docs/stellar-training.md) remains an optional,
explicit backend with separate datasets/checkpoints and no local fallback. Neither
path operates the current HabWorlds attempt or modifies the original spreadsheet.

The model has a character tokenizer, local text encoder, sensory projection, fixed sparse signed graph, shared gated neuron update, and action, target, value-string, answer-string, pointer, and reward-value heads. Learned parameters do not depend on neuron count. Graph changes require explicit checkpoint transfer and invalidate confidence calibration.

Start with a short numerical check:

```sh
.venv/bin/python -m habfly evaluate model
.venv/bin/python -m habfly evaluate model --graph data/processed/graphs-v2/graph-2000
```

Train into a **new output directory** for each experiment:

```sh
.venv/bin/python -m habfly train recognize experiments/recognize-01 --profile configs/mac_smoke.yaml
.venv/bin/python -m habfly train ground experiments/ground-01 --profile configs/mac_smoke.yaml --checkpoint experiments/recognize-01/checkpoint.pt
.venv/bin/python -m habfly train arithmetic experiments/arithmetic-01 --profile configs/mac_smoke.yaml --checkpoint experiments/ground-01/checkpoint.pt
.venv/bin/python -m habfly train policy experiments/policy-01 --profile configs/mac_smoke.yaml --episodes 8 --checkpoint experiments/arithmetic-01/checkpoint.pt
```

`mac_smoke.yaml` uses a 32-node synthetic graph for fast engineering checks. `mac_dev.yaml` uses the real 2,000-node graph; `framework_full.yaml` uses 30,000 nodes. All defaults are CPU based. Use `--transfer-graph` explicitly when continuing learned weights on a different graph.

Each training output contains a measured report and checkpoint with tokenizer, graph/content provenance, optimizer state, stage, seed, and calibration. Smoke-run reports are engineering checks, not evidence that the curriculum gates have passed. The first one-epoch recognition smoke run correctly reported `gate_passed: false`.

```sh
.venv/bin/python -m habfly evaluate model --checkpoint experiments/policy-01/checkpoint.pt --stars 1 --runs 3 --output experiments/policy-01-evaluation
.venv/bin/python -m habfly evaluate baseline experiments/baselines-01 --stage ground --seeds 0,1,2 --epochs 3 --examples 96
```

Baseline evaluation trains biological, degree/sign-matched randomized, and topology-free models with matched trainable parameter counts and training settings. Arithmetic keeps held-out single-digit operand pairs separate and reports two-digit inputs as evaluation-only OOD cases. Calibration examples are separate from reported test examples.

Reinforcement learning is deliberately gated by fresh expert and behavioral-cloning completion runs:

```sh
.venv/bin/python -m habfly train policy experiments/ppo-01 --method ppo --checkpoint experiments/policy-01/checkpoint.pt --profile configs/mac_smoke.yaml
```

PPO must refuse a checkpoint that cannot already complete the prerequisite episodes. See the emitted gate result before increasing training budgets. Local training does not use a hosted model or an API key.

## HabWorlds integration gate

The [HabWorlds Project reference](https://kb.inspark.education/habworlds-project) describes lightcurve zoom/pan/hover, star analysis, and final submission. Its reset behavior preserves data, so real clean attempts require a newly provisioned account.

`src/habfly/packs/synthetic.json` contains a versioned test pack with simplified formulas. Before real lesson training, replace it with a verified content pack and map the actual visible lesson states, controls, formulas, units, charts, and completion indicator. The generic simulator's screen sequence and generated parameters must be adapted and validated against those observations. The bundled pack is not that verification.

The browser adapter operates through ordinary UI actions, reads visible text/control semantics, and uses observation-local target IDs. Charts use visible tooltips first and chart crops if structured values are unavailable. A small local vision encoder is included, but pixel inference requires a checkpoint trained with chart images. The adapter stops on stale observations, authentication loss, unrecognized modals, popups, out-of-bound navigation, repeated no-progress actions, or limits. Submission is disabled by default.

1. Run a local HabWorlds/Torus instance and provision a dedicated fresh account manually.
2. Authenticate outside the agent run; keep Playwright storage state in an ignored `*.storage-state.json` file.
3. Copy `configs/browser.example.json`, specify the exact local activity path and its visible labels, and supply the verified content pack.
4. Train/evaluate the matching policy in the simulator. Run one star first, then three, then the full 30-star workflow. Completion is the final goal, not a perfect score.
5. Enable `allow_submission` explicitly only when testing submission on that disposable account.

```sh
.venv/bin/python -m habfly evaluate browser configs/browser.local.json \
  experiments/verified-policy/checkpoint.pt \
  data/processed/graphs-v2/graph-2000 \
  configs/verified-content.json
```

The browser integration has been tested against local synthetic HTML fixtures; it has not been tested against the actual HabWorlds lesson. Browser runs save observations, actions, chart crops, failure screenshots, and Playwright traces. Keep those local because they may contain course data.

## Runtime and verification

The Rust TUI uses version-1 JSONL over a subprocess's stdin/stdout. Python logs belong on stderr. Commands are `start`, `pause`, `resume`, `step`, `abort`, `save_trace`, and `replay`; events include observations, actions, results, neural activity, state, errors, and episode summaries. See [the TUI guide](tui/README.md) and [architecture](docs/architecture.md).

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src/habfly tests
cargo test --manifest-path tui/Cargo.toml --locked
cargo test --manifest-path tui/Cargo.toml --locked -- --ignored
cargo clippy --manifest-path tui/Cargo.toml --locked --all-targets -- -D warnings
cargo fmt --manifest-path tui/Cargo.toml --check
```

Browser tests need Chromium: `.venv/bin/python -m playwright install chromium`. They run against disposable local fixtures. Some sandboxed environments require permission to launch Chromium or bind a local fixture server; the remaining Python tests can still run independently.

To move to the Framework, reproduce `uv.lock` and `tui/Cargo.lock`, transfer graph artifact directories and checkpoints, and validate graphs on arrival. The engine fails on incompatible graph/tokenizer/content identities unless an explicit graph transfer is requested. No credentials are included in the portable artifacts.
