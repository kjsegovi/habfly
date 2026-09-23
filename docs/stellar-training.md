# Spreadsheet-assisted stellar training

This is the **optional Google Sheets backend**. The recommended starting point is
[offline local-tool training](stellar-local-training.md). Existing Google profiles
explicitly specify `calculation_backend: google_sheets`; their commands remain supported.

HabFly learns to select measurements, place them in your spreadsheet, read the
appropriate results, copy numbers exactly, and choose units. Google Sheets does
the arithmetic. There is no local formula-engine fallback. The current HabWorlds
preview and original spreadsheet are never execution targets for this pilot.

## 1. One-time setup

The user supplied [this dedicated working copy](https://docs.google.com/spreadsheets/d/14o_7Dim-0dAghnvMIpn_PZfSmDO8CSO6Vmu9Y_PcqVo/edit).
Its ID is prefilled in `configs/spreadsheet.example.json`. Live access, headers,
and formulas still require verification; the code does not assume they are valid.

1. Enable **Google Sheets API** in your Google Cloud project.
2. Create a dedicated service account without project roles or domain-wide delegation.
3. Download its JSON key to a private location **outside this repository**. On
   macOS/Linux use `chmod 600` on that file. Never paste its contents into chat,
   logs, Git, or experiment directories.
4. Share **only the working copy** with the service account's email as Editor.
   Do not share the original or a containing Drive folder.
5. Copy `configs/spreadsheet.example.json` to ignored
   `configs/spreadsheet.local.json`. Set `credentials_path` to the key's absolute
   path. Leave the fingerprint null and headers empty until inspection below.

See [Google's service-account setup and file-sharing guidance](https://developers.google.com/workspace/guides/create-credentials).
The Sheets scope is used for this dedicated service account, not your personal
Google login. File permissions come from sharing. The three-cell restriction is
enforced by HabFly, not OAuth.

```sh
uv sync --frozen
.venv/bin/python -m habfly spreadsheet inspect configs/spreadsheet.local.json
```

`inspect` is read-only. Check its copy ID, headers, and formulas against the
workbook. Copy `formula_fingerprint` into `expected_formula_fingerprint` and
the complete `headers` object into `expected_headers` in your local config.
Never automatically repin unexpected formula changes.

**Checkpoint:** configuration is pinned, credentials remain private, no writes yet.

## 2. Verify recalculation

```sh
.venv/bin/python -m habfly spreadsheet verify configs/spreadsheet.local.json
```

This **writes two golden input sets to the copy** and restores its previous
numeric/blank inputs. It checks six stellar formulas and changed-input
recalculation. It never rewrites formulas. If access disappears, restoration may
fail explicitly; do not assume restoration succeeded after an error.

Permitted writes: `Sheet1!A2` (flux W/m²), `B2` (parallax arcseconds), `L2`
(peak wavelength nm). Outputs: C2 (distance ly), F2 (luminosity Lsun), M2
(temperature K), G2 (mass Msun), H2 (radius Rsun), I2 (lifetime years).
Intermediate formulas D2/E2 are also fingerprinted. Other cells stay unchanged.
Errors in unrelated planet columns do not block this stellar-only pilot.

Only one machine may use the copy at a time. The local advisory lock excludes
other processes on this host, **not** a simultaneous Framework worker. Keep
`exclusive_worker: true` only while honoring that operational requirement. Avoid
manual edits while an adapter run is active.

**Checkpoint:** `verified: true`, `cases: 2`. This is spreadsheet verification,
not a learned policy or course acceptance result.

## 3. Collect demonstrations

```sh
.venv/bin/python -m habfly data demonstrations experiments/stellar-data-smoke-001 \
  --task stellar --profile configs/stellar_smoke.yaml
```

The network-dependent collector first runs a 100-case scripted expert gate.
It then collects four training, two calibration, two development, and 100 final-test
episodes. The splits have distinct numeric tuples, seeds, and instruction templates.
Star class is supplied; classification learning is deferred. Grading references
come from correctly populated spreadsheet runs and are never policy observations.

Each policy step may select wrong measurements; the adapter never fixes the
choice or substitutes an expert result. Values are written with `RAW` and read as
effective numbers without display rounding. Missing/error/nonfinite outputs,
changed formulas, inconsistent readback, or lost access stop the run.

Collection is intentionally rate-limited to 40 total requests/minute. It may
take substantially longer than offline training. See Google's
[value API](https://developers.google.com/workspace/sheets/api/guides/values) and
[quota guidance](https://developers.google.com/workspace/sheets/api/limits).

Artifacts: manifest, per-split cases/trajectories, expert reports, and per-case
`.events.jsonl` replays. Incomplete collections do not publish a usable manifest.
Use a new output directory after interruption; v1 has no partial-collection resume.
Reset clears only A2/B2/L2. Collection leaves the final episode's inputs in the
copy; unlike `verify`, it is not a restore operation.

**Checkpoint:** `expert_gate_passed: true` and complete split reports. This is
scripted expert performance, not learned performance.

## 4. Train offline

```sh
.venv/bin/python -m habfly train policy experiments/stellar-smoke-001 \
  --profile configs/stellar_smoke.yaml
```

Training does not load credentials or contact Google. It reuses demonstrations
for two epochs on the real 2,000-node graph, hidden size 16, one CPU thread, seed 0.
Outputs include loss history, development imitation scores, separate calibration,
`checkpoint.pt`, and a manifest pinning graph, content, formula fingerprint,
cell mappings, and dataset splits. Existing experiments are never overwritten.

For the pilot, collect a fresh dataset and train five epochs:

```sh
.venv/bin/python -m habfly data demonstrations experiments/stellar-data-pilot-001 \
  --task stellar --profile configs/stellar_pilot.yaml
.venv/bin/python -m habfly train policy experiments/stellar-pilot-001 \
  --profile configs/stellar_pilot.yaml
```

The pilot uses 64 training, 16 calibration, 16 development, and 100 final-test
episodes after its own 100-case expert gate. No PPO, automatic budget increases,
or checkpoint transfer is enabled for stellar v1.

**Checkpoint:** finite loss, reloadable checkpoint, inspected development scores.
This does not establish closed-loop success.

## 5. Evaluate live and replay offline

```sh
.venv/bin/python -m habfly evaluate model \
  --checkpoint experiments/stellar-pilot-001/checkpoint.pt \
  --profile configs/stellar_pilot.yaml --task stellar \
  --runs 100 --seed 300000 --output experiments/stellar-eval-001
```

For development use `--split development --seed 200000 --runs 16` and a fresh
output directory. Smoke evaluation uses the smoke profile/checkpoint instead.
Seeds must exist in the collected dataset; add the collection seed to these
offsets when using a nonzero profile seed. Never tune repeatedly on the final
test and continue calling it unseen.

Evaluation uses the real sheet for the learned policy's chosen inputs. It records
completion, steps, rewards, input/output selection, numeric answers, units,
invalid actions, API failures, calibration, timing, and memory. API failures stop
the run separately from policy mistakes. Unattempted episodes are not successes.

**Gate:** at least 90/100 final-test episodes completed, zero invalid actions,
zero API failures. Smaller evaluations cannot pass. Simulator grading uses
`rel_tol=1e-6`, `abs_tol=1e-9`; these are not verified HabWorlds tolerances.
Stellar completion is not project submission or one complete real star.

```sh
.venv/bin/python -m habfly replay experiments/stellar-eval-001/300000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked -- \
  --replay experiments/stellar-eval-001/300000.events.jsonl
```

Replay needs no credentials or network. For one live spreadsheet-connected TUI episode:

```sh
cargo run --manifest-path tui/Cargo.toml --locked -- --autostart \
  --start-payload '{"task":"stellar","policy":"checkpoint","seed":300000,"graph":"data/processed/graphs-v2/graph-2000","checkpoint":"experiments/stellar-pilot-001/checkpoint.pt","dataset":"experiments/stellar-data-pilot-001","spreadsheet_config":"configs/spreadsheet.local.json"}'
```

The TUI shows bindings, selected results/destinations, sheet values, and copy
operations. Press `v` to focus observations. Existing pause/step/abort/save/replay
commands remain version 1. Google calls are synchronous and bounded; pause/abort
is handled between operations, not during an in-flight request. Stellar runtime
rejects `environment: browser` and never opens the HabWorlds preview.

## Validation boundary

Automated tests use an explicitly test-only arithmetic transport. They check
contracts, deterministic expert behavior, offline training, and failure handling.
They do **not** establish live Sheets access, live recalculation, or learned
HabWorlds capability. Those gates remain pending until credential setup and
live collection/evaluation succeed.
