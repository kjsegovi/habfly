# HabFly terminal interface

A standalone Rust `ratatui` / `crossterm` client for the local Python runtime. It displays student-visible observations, controls and options, proposed actions, reward, episode progress, real neural activity, and a bounded diagnostic timeline. Chromium, when used by the runtime, stays in its own visible window.

From the **HabFly repository root** with the Python package installed in `.venv`:

```sh
cargo run --manifest-path tui/Cargo.toml -- --autostart
```

The default starts a one-star **scripted expert** simulation with seed 0. The UI labels any untrained observer activity as observation-only; it does not claim the expert's actions are learned. Pass start settings directly to the runtime:

```sh
cargo run --manifest-path tui/Cargo.toml -- --autostart --start-payload '{"policy":"expert","stars":3,"seed":12}'
cargo run --manifest-path tui/Cargo.toml -- --replay experiments/traces/session.jsonl
```

Use `--python /absolute/path/to/python` for another Python environment. The child is launched as `python -u -m habfly runtime --jsonl` without a shell. Use `--trace PATH` to set the destination for the save-trace key (default `experiments/traces/tui-session.jsonl`). The runtime owns trace writing and replay; the TUI owns no training or browser logic.

| Key | Operation |
| --- | --- |
| `s` | Start with configured settings |
| Space | Pause/resume based on acknowledged runtime state |
| `p` / `r` | Pause / resume |
| `n` | Single step |
| `a` | Abort the current episode |
| `t` | Save trace |
| ↑ / ↓ | Scroll visible controls |
| `v` | Cycle overview, full-width neural activity, and observation/actions |
| `q` / Ctrl-C | Abort and close the runtime |

Launch with `--replay PATH` for replay. A terminal of 150×45 displays all panels comfortably. At 80×24, use `v` to focus neural activity or observations and see the detail at full width. Terminals below 65×18 show a resize message.

## Protocol and failure handling

The version-1 JSONL contract uses one object per line. Runtime stdout contains only events; stderr appears in the diagnostic timeline.

```json
{"version":1,"command":"start","payload":{"policy":"expert","stars":1,"seed":0}}
{"version":1,"event":"state","sequence":1,"run_id":"example","payload":{"status":"running"}}
```

The supported commands are `start`, `pause`, `resume`, `step`, `abort`, `save_trace`, and `replay`. Supported events are `hello`, `state`, `observation`, `action_proposed`, `action_result`, `neural_activity`, `episode_summary`, and `error`. Trace commands use `{"path":"..."}`. Both flat payloads and wrapped `observation`, `action`, and `result` payloads are accepted. Unknown event kinds, unsupported versions, malformed JSON, invalid UTF-8, and lines above 1 MiB produce diagnostics; a following valid event can recover.

Separate threads read stdout/stderr and write commands. The event queue is bounded at 256 messages, command queue at 32, timeline at 200 lines, and each neural activity series at 80 points. The render loop drains at most 128 messages per tick so input stays responsive and redraws only when state or input changes. Probability fields must be within [0, 1]; without a calibration marker they are labeled **uncalibrated policy probabilities**. Calibrated probabilities display their evaluation scope (for example, held-out command grounding); this does not imply calibration on browser episodes. Missing probabilities or calibration scope are labeled unknown. Use `v` for the full-width action panel when labels do not fit the overview.

Exit closes stdin, waits briefly for runtime cleanup, then kills/reaps an unresponsive child. Terminal restoration covers normal exit, errors, and panics. The TUI remains open with the exit status if the runtime fails, so its last diagnostics are readable.

## Verification

```sh
cargo test --manifest-path tui/Cargo.toml --locked
cargo test --manifest-path tui/Cargo.toml --locked -- --ignored
cargo clippy --manifest-path tui/Cargo.toml --locked --all-targets -- -D warnings
cargo fmt --manifest-path tui/Cargo.toml --check
```

Tests cover a golden protocol stream, malformed/versioned payloads, exact outgoing commands, bounded log/line behavior, missing subprocesses, subprocess IPC, calibration labels, expert/observer labels, and terminal rendering at full and compact sizes.

The `--ignored` test requires the installed repository `.venv`. It starts the actual Python runtime and verifies start, paused single-step, resume/pause, saving a trace into a new directory, replay, and abort through the same bridge used by the TUI. Its disposable output remains under `tui/target/runtime-test-*`.
