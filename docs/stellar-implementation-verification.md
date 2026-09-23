# Stellar integration verification — 2026-09-23

## Implemented

- Restricted Google Sheets adapter and dedicated-copy configuration.
- Six-output stellar task with exact input/result copying, separate task completion,
  shuffled controls, distractors, and main-sequence applicability.
- Live expert collection, offline recorded-data behavioral cloning, live evaluation,
  provenance checks, and credential-free version-1 replay.
- TUI spreadsheet state and an expanded full-observation view (`v` cycles views).
- Setup and sequential run instructions in [stellar-training.md](stellar-training.md).

The user supplied the working-copy URL. Its ID is in the example configuration.
No Google Sheets cells or HabWorlds preview state were modified during implementation.

## Verified locally

- `python -m pytest -q --tb=short`: **106 passed**, including disposable local
  Chromium fixtures. Chromium needed execution outside the filesystem sandbox.
- **24** spreadsheet/stellar test cases, including 100 deterministic expert fixture
  cases, errors, stale targets, reset isolation, restricted writes, bounded/redacted
  transport retries, runtime pause/step/abort/replay, dataset integrity, and an
  offline one-epoch checkpoint save/reload. These use an explicit test-only transport.
- `cargo test --manifest-path tui/Cargo.toml --locked`: **18 passed**, one opt-in
  bridge test ignored by default. Running that bridge separately: **1 passed**.
- Ruff, Clippy with warnings denied, Rust formatting, and `git diff --check` passed.
- Locked Google authentication dependencies installed with `uv sync --frozen`.
- Real biological graph smoke: **100 deterministic finite 2,000-node forwards**,
  responsive 30,000-node inference, and finite small-graph gradients. Report:
  `experiments/stellar-code-verification-20260923/graph-smoke.json`.

## Not yet verified / not claimed

- Service-account access to the user's copy, live headers/formula fingerprint,
  actual Google recalculation/readback, and the live 100-case expert gate.
- A real Sheets-backed training dataset or learned stellar completion rate.
- Classification, planet/habitability tasks, or any real HabWorlds acceptance run.

The service-account key and pinned local config are still needed. Test-double
success is engineering evidence only, not the live spreadsheet or learned-policy gate.
The 90/100 completion gate is implemented but has not been passed by a trained agent.
The original spreadsheet and existing HabWorlds attempt remain protected.
