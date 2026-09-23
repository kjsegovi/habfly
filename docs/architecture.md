# Architecture and continuation checkpoints

```text
MaleCNS Feather -> streaming filter -> complete-path nested graphs
                                                |
Curriculum / MiniHabWorlds / Playwright -> Observation -> PyTorch policy
                                                |             |
                                             Action <---------+
                                                |
                                          validated step
                                                |
                                      JSONL events and traces
                                                |
                                    Rust terminal UI or replay
```

## Boundaries

- `contracts.py` defines Pydantic observation, control, action, result, command, and event records. Observations contain only available information; expert solutions are outside the policy input.
- `data/` owns immutable graph arrays and provenance. Fixed graph tensors are model buffers and are omitted from learned parameter checkpoints.
- `model/` owns tokenization, sparse recurrent propagation, heads, optional local image encoding, and neural-state inspection.
- `training/` owns curriculum generation, supervised/imitation learning, calibration, baseline comparisons, checkpoint transfer, and gated PPO.
- `environments/` provides Gymnasium interfaces. `MiniHabWorlds` is a synthetic workflow model; real lesson fidelity remains a separate content-validation checkpoint.
- `knowledge.py` and `packs/stellar_knowledge.json` provide versioned stellar references and restricted deterministic arithmetic, separately from synthetic habitability content. `environments/stellar_common.py` shares supplied-class cases and private grading across local-tool and optional Sheets environments. Each backend uses distinct dataset/checkpoint identities.
- `browser.py` translates visible UI observations and validates/executed actions. It never reads hidden JavaScript application state or supplies an answer oracle.
- `runtime.py` owns the run lifecycle, pause/step/replay, artifact writing, and v1 JSONL. `tui/` is a replaceable presentation client, without model or browser logic.

## Contract details

Observation target IDs are valid only for the current observation. The simulator encodes revision in IDs and rejects stale actions. The browser refreshes its visible observation before acting and stops on drift, preserving evidence rather than resolving a stale locator against a new control.

Action arguments are bounded and validated before side effects. Action kinds are CLICK, TYPE, SELECT, HOVER, DRAG, SCROLL, KEYPRESS, WAIT, and STOP. Pointer coordinates are relative to a visible control's box. A requested browser drag remains within that target's bounds.

The browser's allowed path is a navigation boundary, not an asset firewall: ordinary same-application subresources may load, but top-level navigation and popups cannot escape the designated activity. Only loopback HTTP(S) URLs are accepted by this local implementation. The exact completion text and final-submission labels are required application configuration.

The graph manifest describes arrays, source identities/hashes, selection settings, node ordering, provisional polarity, normalization, versions, and validation. Checkpoints contain trainable weights and optimizer state alongside graph/tokenizer/content fingerprints. Numeric graph files are loaded without pickle; PyTorch checkpoint loading uses its weights-only mode.

## Demo modes

`expert`: deterministic solver chooses actions; an untrained graph model processes the same observations solely for visible diagnostics. Confidence is absent for expert actions.

`untrained`: random-initialized model chooses actions. This exercises mechanics and is expected to fail the lesson.

`checkpoint`: the loaded model chooses actions, subject to environment validation. Its calibration marker records whether confidence temperatures have actually been fitted. Changing graph or training policy invalidates calibration.

`replay`: reads validated saved events without operating the environment. Recorded status is distinguished from current replay status; pause and single-step apply to playback.

## Stellar tool contract (protocol version 1)

`Observation.calculation` is an optional section alongside `spreadsheet`; both
default to empty for old observations. Controls declare `surface: calculation`
and use existing SELECT/CLICK actions. The section exposes the selected operation,
reference card, bindings, episode-local result history, exact copy operation and
recoverable errors. It never exposes private grading references. Reference cards
and measurements have independent bounded character-encoding sections.

Runtime settings accept `calculation_backend: local | google_sheets` and optional
`knowledge_pack`. Legacy settings containing only `spreadsheet_config` still select
Sheets. New stellar settings default to local. Both require a compatible collected
dataset; inference checkpoints must match its content and graph identity.
`task_completed`, rather than termination/submission, is the stellar success flag.
Replay remains backend-free. [Local training checkpoints](stellar-local-training.md)
are the recommended next steps; no actual HabWorlds attempt is started.

## Acceptance status

| Checkpoint | Status |
| --- | --- |
| Canonical data and real 2k/5k/10k/30k graph artifacts | Built and validated locally |
| Synthetic expert one/three/30 stars | Completes in tests |
| Model, training, checkpoint, evaluation implementation | Implemented; smoke checks measure mechanics |
| Learning/generalization thresholds | Pending sustained experiments |
| Rust TUI, JSONL and replay | Implemented with protocol/render/subprocess tests |
| Browser adapter | Verified against local HTML fixtures |
| Exact HabWorlds content and UI mapping | Pending supplied lesson and local application |
| Real one-star, three-star, full-project learned runs | Pending content, trained checkpoint, fresh accounts |
| Pixel chart interpretation on real charts | Pending chart examples and vision training |

The next stellar learning checkpoint is the bounded local-tool smoke experiment,
followed by an explicitly launched pilot. Distinguish tool correctness, scripted
expert success, held-out action imitation, and learned closed-loop completion.
