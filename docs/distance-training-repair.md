# Distance training investigation and bounded repair

This work is local, tool-assisted learning research. HabWorlds and Google Sheets
are not involved. Existing pilot and diagnostic artifacts are preserved; replay
does not require rerunning a model.

## What was measured

The failed `distance-diagnostic-002` checkpoint completes 0/4 training cases and
scores 25% on the 40 expert-observation decisions. Its model source and all four
recorded dataset splits matched the code/data used for the run when inspected.

With the same preceding recurrent history held fixed, swapping all ten workflow
observations changes its raw action logits by at most **0.0001183** in the inspected
case. Repeating the unchanged initial observation ten times moves its CLICK share
among CLICK/SELECT from **0.128 to 0.548**. This supports a weak current-state signal
and recurrent drift, not a conclusion that a browser or calculator is unavailable.

Mean loss components per training decision were action 0.598, target 1.015,
typed string 1.328, and duplicate answer string 1.213. Their corresponding readout
gradient norms were 0.540, 0.214, 0.144, and 0.154. Large string loss alone does not
prove that its gradients dominated the action loss.

The SELECT training objective also differed from deployment: training predicted
characters (and duplicated them into a short-answer task), while inference ranked
legal dropdown choices by mean character likelihood. Raw source IDs did not carry
their visible measurement semantics into a candidate-specific scoring function.

## Code changes

These are explicit experimental model modes, saved in each checkpoint. Existing
profiles and old checkpoints retain `pooled_text_v2` / `characters` defaults.

- `structured_tool_v3`: supplement character encoding with 16 fixed visible-state
  features: selected operation/input/source, binding/result presence and counts,
  selected result/destination, answer/unit presence, tool errors and last operation.
  These features contain neither next-action labels nor grading answers, do not
  repair wrong bindings, and do not depend on step number or control ordering.
  Input still travels through sensory neurons and biological edges. No extra
  learned parameters or per-neuron embeddings were introduced.
- `option_pointer_v1`: score visible options using their visible metadata and the
  selected control's label/role. Training and inference use the same scorer.
  Candidates keep their opaque IDs for execution; IDs and option ordering do not
  determine candidate score. SELECT trains option cross-entropy instead of two
  character-generation losses. TYPE/KEYPRESS retain text losses. Numeric copying
  remains the environment's exact copy action, not generated arithmetic.
- Sequence training reports progress after every epoch, including actual update
  and decision counts. No automatic extension, PPO, resume, or learning-gate bypass.

Regression checks cover graph-dependent action input, visible-only state features,
wrong bindings, option permutation/ID invariance, finite gradients, incompatible
mode rejection, checkpoint round trips and legacy model configuration defaults.

## Controlled short experiments

Each is a fresh seed-0 model with hidden size 16, one CPU thread, the same four
ten-action training cases, 20 epochs, and 80 sequence-level optimizer updates.
Calibration (2 cases) and development (2 cases) remain separate. No unseen test
examples are loaded or evaluated by the repair runner. Loss values across the
character and pointer objectives are not directly comparable.

| Variant | Train exact decisions | Development exact decisions | Seen completion | Development completion |
| --- | --- | --- | --- | --- |
| Original v2 | 25% | 30% | 0/4 | Not measured closed-loop |
| Structured state only | 25% | 30% | 0/4 | 0/2 |
| Structured state + option objective | 50% | 45% | 0/4 | 0/2 |
| Parameter-matched topology-free control + option objective | 55% | 50% | 0/4 | 0/2 |

With option training, correct measurement selection rises to 4/4 and mean option
loss over all decisions falls to 0.00161. Action/target sequencing remains unsolved.
The topology-free control responds more strongly to observation swaps and has
lower action loss (0.078 versus 0.539), but **also fails closed-loop completion**.
It is a diagnostic control, not a replacement HabFly agent. One seed and these
small cases do not establish that biological topology is useful or harmful.

Artifacts:

- `experiments/distance-repair-structured-001/`
- `experiments/distance-repair-options-001/`
- `experiments/distance-repair-topology-free-001/`
- `experiments/distance-repair-audit-001/{baseline,options-80,topology-free-80}.json`

Each training directory contains a checkpoint, provenance, calibration and losses;
each seen/development directory contains actual rollouts and offline JSONL replays.

## Separately approved optimizer-budget comparison

The user explicitly authorized one **800-update** biological-graph comparison:
`experiments/distance-repair-options-800-001/`. It uses the same four cases and
sequence length 10, for 200 epochs / 8,000 supervised decision exposures. This
matches v1's optimizer-update count, **not** its number of data passes or compute.
It is not an automatic increase in the default diagnostic or pilot budget.

This run finished in 1,112.65 seconds with peak process RSS about 764 MB.
It achieved **100% training exact decisions and 4/4 training-case completions**,
with zero invalid actions, tool errors or infrastructure failures. It passes the
training-fit gate. However, development exact decisions were **90% and completion
0/2**: both cases selected the wrong source measurement. Their failed rollouts
recorded 34 recoverable tool errors, not infrastructure failures. Unseen/final
tests remain sealed; passing the training gate alone is not generalization.

The frozen audit reports 40/40 decisions unchanged by reordering controls and
renaming their IDs. Current-observation sensitivity increased from 0.000118 to
1.417 in the same probe. But repeating an unchanged initial observation still
changes action preference sharply: temporal dependence remains and recovery from
off-expert states is not established.

The first 20 epoch losses of the 800-update run exactly reproduce the 80-update
biological option experiment, verifying that the longer comparison starts identically.

## Source-identity repair: approved 80-update follow-up

Frozen-weight inspection isolated another representation problem: rotating only
the numeric payloads among candidates changed source selection in all four training
cases (and both development cases), despite holding the requested quantity, units,
source labels, history and pooled neural state fixed. Candidate JSON put `value`
last; the recurrent text encoder could memorize digit strings instead of robustly
representing quantity/unit/source. This is a counterfactual scoring probe, not a
new valid physical episode or a claim of single-factor causality.

The user separately approved one additional 80-update check. New explicit mode
`option_pointer_semantic_v2` excludes numeric payloads from source **identity keys**;
the numbers remain visible in observations and are copied/calculated unchanged.
It uses the existing value-decoder cell as an independent option query rather than
sharing the workflow's target query. No parameters or neurons were added.

`experiments/distance-repair-semantic-001/` inherits the 800-update checkpoint and
trains only its four `value_decoder.*` tensors for 20 epochs / 80 updates. The
recurrent core, text encoder, action/target heads and all other parameters are
frozen; exact equality before/after is asserted and recorded. Parent path/hash and
the trainable parameter names are recorded in the manifest and final report.

**This follow-up did not pass**: 90% expert-observation decision accuracy on both
training and development, but 0/4 and 0/2 complete rollouts. Source selection remains
wrong. The inherited encoder's semantic keys are too weakly separated for this
small query-only update to repair them. On the inspected first case, their maximum
pairwise Euclidean distance was 0.227; the bounded query implies an optimistic
correct-source softmax upper bound of 0.176. This bounds confidence, not argmax
accuracy, and is not proof that no longer query-only optimization could rank them.

Do **not** promote this failed follow-up as a better agent or resume it silently.
At this point the best completed *training-fit* checkpoint remained the 800-update
parent; neither checkpoint generalized successfully. The user subsequently approved
the isolated recognition stage below.

## Isolated measurement-identity stage

`measurement_identity_v1` adds a small, independently trainable character encoder
and factorized quantity/unit/source option head (6,512 added parameters at hidden
size 16; no per-neuron embeddings). The original graph, recurrent
cell, text encoder, action/target heads and existing option scorer remain frozen.
The new head reads the selected input's visible quantity/unit requirement, the
instruction, candidate quantity/unit/source fields, and the biological readout.
It never reads candidate IDs, numeric payloads or supervised labels as features.
There is no hardcoded correct-choice equality test at inference: an untrained or
incorrectly trained head can select any visible option, including a wrong one.

Routing applies only to the visible `Measurement or result` calculation combobox,
when an input has been selected and all its options refer to measurements. All
other selections, including mixed measurement/result lists, use the unchanged
parent scorer. This is a distance-stage experiment, **not** a repair of general
stellar dependencies, recovery, classification or browser behavior.
The new option head has direct visible-reference/candidate inputs as well as the
graph readout. Its recognition scores are not evidence that the biological
topology alone learned field matching.

The fixed run budget is **200 selector-only updates, batch 12 (2,400 labeled
recognition decisions), CPU, one thread, hidden size 16, seed 0**. No PPO or
end-to-end workflow updates occur. It inherits the successful 800-update parent,
not the failed 80-update semantic follow-up. Original tensors are checked for
exact equality before/after; saving/reloading must preserve weights and recognition
predictions. Defaults for the main training profiles remain unchanged.

Recognition practice has 144 training records and 72 development records, with
different templates, random seeds, numeric payloads and opaque IDs. Each record
has 18 shuffled candidates spanning three quantities, three unit labels and two
source labels. Deliberately mismatched quantity/unit combinations test whether
the learner can distinguish units independently of quantity. These are synthetic
field-recognition exercises, **not physically valid stellar cases**, and are never
sent to the calculator. The held-out task is new wording/numbers/order for the
same finite vocabulary, not unseen scientific concepts. Cached neural contexts
come only from the four original training trajectories.

The runner evaluates the same four training and two already-used development
workflows after training. It does not open the eight-case final-test file, tune on
development feedback, or automatically retry. Confidence remains explicitly
uncalibrated for this new selector stage.

```sh
.venv/bin/python scripts/train_measurement_identity.py experiments/YOUR-FRESH-IDENTITY-RUN \
  --parent experiments/distance-repair-options-800-001/training/checkpoint.pt
```

Artifacts include `manifest.json`, `recognition-{train,development}.json`,
`training-progress.jsonl`, `training/checkpoint.pt` and its provenance sidecar,
`report.json`, and replayable `seen/` and `development/` trajectories. The manifest
records the parent hash, trainable tensors, graph/pack hashes, recognition split
hashes and code identity. Reports separate recognition accuracy by field from
teacher-forced workflow accuracy and actual closed-loop completion.

Numeric-invariance assertions hold the biological readout fixed: numbers still
enter the original frozen observation encoder and calculator. They are not a
claim that all recurrent activity is invariant to changed observations.

### Actual result: partial learning, distance workflow improved

`experiments/distance-identity-001/` completed its one capped run. Loss decreased
from 2.86524 to 0.69347; all gradients were finite. Preparation and training took
19.34 seconds; the full run including reload and workflow evaluation took 23.41
seconds, with peak process RSS about 406 MB. All inherited tensors stayed exactly
unchanged, and the new checkpoint reloaded with identical recognition predictions.

| Check | Before selector training | After 200 selector updates |
| --- | --- | --- |
| Training recognition exact | 8/144 | 72/144 |
| Development recognition exact | 4/72 | 36/72 |
| Development quantity / unit accuracy | 33.3% / 33.3% | 100% / 100% |
| Development source accuracy | 50% | 50% |

Source selection **did not learn the requested star**: all 216 recognition records
select `current star`, even when the instruction requests `reference star`.
The recognition gate therefore fails. The paired source confusion is saved in
`identity-audit.json`; this is not a successful general measurement-identity model.

However, all four original training workflows and both previously used development
workflows now complete in exactly ten actions, with zero invalid actions, tool
errors or infrastructure failures. Both teacher-forced workflow scores are 100%.
These workflows all request the current star, so their 6/6 completion is consistent
with the remaining source bias; it does not establish generalization to requests
for other stars or broad stellar acceptance.

Frozen probes confirm that rotating measurement magnitudes changes source logits
by exactly zero on all six workflows, with history/readout fixed. Forty training
decisions survive control reordering and ID renaming unchanged. The old temporal
drift on repeated unchanged observations remains; recovery is still unresolved.

The final-test split remains unopened and confidence is uncalibrated. **Do not
start the larger pilot or final test based on this result.** The next proposed
bounded experiment is source-request discrimination with paired current/reference
instructions, preserving the newly learned quantity/unit distinctions and frozen
workflow. The subsequently authorized stage is documented below.

Inspect `report.json` for scores, `identity-audit.json` for the source bias and
numeric probe, and `experiments/distance-repair-audit-001/identity-200.json` for
the recurrent/control-order audit. Replay the new successful distance workflow:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-identity-001/development/800000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/distance-identity-001/development/800000.events.jsonl
```

## Paired source-request stage

The user authorized one source-only follow-up, capped at **200 optimizer updates**.
Its parent is `experiments/distance-identity-001/training/checkpoint.pt`; the old
workflow and complete identity head remain frozen. New mode `measurement_source_v2`
replaces only the old source-score contribution with an independent character-level
bidirectional GRU, attention pooling and candidate-source pointer. It retains the
learned quantity/unit score contributions exactly. No hardcoded instruction parser,
correct-source filter, local arithmetic replacement or per-neuron embeddings are added.
The auxiliary source head reads visible instruction/source strings and the frozen
graph readout; it does not prove that the biological topology learned language.

The parent preflight confirms that source instructions do reach its scores and
encoder gradients. A simple current/reference substitution changed the inspected
instruction encoding by L2 0.0951, its logits by at most 0.00630, and produced a
nonzero source-query gradient norm of 0.000859. These are frozen diagnostic probes,
not extra optimizer updates or proof that a particular architecture will succeed.

The new dataset contains 72 training pairs (144 instructions) and 36 development
pairs (72 instructions). Pair members have identical measurements, magnitudes,
opaque IDs, ordering, quantity/unit requests and frozen graph context. Only the
instruction and private supervised label change. Templates include both
"use this star; ignore that star" and "ignore that star; use this star" ordering.
Train/development wording, numeric payloads and IDs are separate. Deliberately
mismatched unit labels remain syntactic exercises, never calculator inputs.

Six complete pairs per update yield 2,400 labeled source decisions. Training uses
only source-selection loss; the source strings are grouped into distinct visible
labels for supervision, but inference still scores every candidate without filtering
wrong choices. CPU, one thread, hidden size 16 and seed 0 remain fixed. The runner
blocks socket connections and Sheets initialization, rejects existing output paths,
and exposes no budget-extension or automatic retry flag.

Before training, it records zero-update instruction/gradient probes. After training,
it asserts exact equality of all inherited tensors and quantity/unit scores on all
new and legacy recognition records, then saves/reloads the checkpoint and compares
predictions. It evaluates the six existing workflows, never the final-test split.

Predeclared gate:

- At least 95% requested-source accuracy on both new and legacy development sets.
- At least 95% development pairs correct on **both** sides of the instruction change.
- 100% quantity/unit accuracy on new and legacy recognition records.
- All six existing workflows complete in at most ten actions, with zero invalid
  actions, tool errors, API failures or infrastructure failures.
- Original tensors and quantity/unit scores remain unchanged; reload is exact.

Confidence remains uncalibrated. Even a passed source gate only proposes evaluating
the eight sealed distance cases next; the script never opens them or launches a
larger pilot. A completed script and a passed learning gate are reported separately.

```sh
.venv/bin/python scripts/train_source_request.py experiments/YOUR-FRESH-SOURCE-RUN \
  --parent experiments/distance-identity-001/training/checkpoint.pt
```

Artifacts: `preflight.json`, `manifest.json`, `paired-{train,development}.json`,
`training-progress.jsonl`, `training/checkpoint.pt` plus provenance sidecar,
`report.json`, `status.json`, and replayable `seen/` and `development/` trajectories.
The report includes per-field recognition, paired-switch accuracy, actual workflow
completion, learning gates, finite losses/gradient norms, timing and peak RSS.

### Actual source-stage result: training learned, development gate failed

`experiments/distance-source-001/` completed exactly 200 updates / 2,400 labeled
decisions. The new source component has 7,440 trainable parameters. Source loss
fell from 0.693153 to 0.00000914; all gradient norms were finite. Optimization took
3.09 seconds; the full preparation, reload and evaluation took 15.64 seconds,
with peak process RSS about 466 MB. No additional optimizer run was started.

| Recognition set | Source accuracy before | Source accuracy after |
| --- | --- | --- |
| New paired training | 72/144 (50%) | 144/144 (100%) |
| New paired development | 36/72 (50%) | 18/72 (25%) |
| Legacy recognition training | 72/144 (50%) | 108/144 (75%) |
| Legacy recognition development | 36/72 (50%) | 54/72 (75%) |

All 72 training pairs are correct on both instructions, but **0/36 development
pairs** are correct on both sides. This is not a passed source-recognition model:
the original 95% development gate fails and must not be relaxed after inspecting
the result. The new checkpoint is an experimental artifact, not a promoted policy.

Quantity and unit recognition remain 100% across all four sets. Every original
tensor and every checked quantity/unit score stayed exactly unchanged. The saved
checkpoint reloaded with identical weights and recognition predictions. All four
training and both existing development distance workflows still complete in ten
actions with zero invalid actions, tool errors, API failures or infrastructure
failures. These familiar workflows do not replace the failed wording tests.

The errors depend on language composition: the "Select ... belonging to ...,
not ..." development template always selects the current star (50%); the
"Disregard ...; obtain ..." template selects the excluded star (0%). The new
encoder was trained from scratch on four templates, so it has no pretrained
English semantics. The new development templates introduce directive vocabulary
and combinations absent from its source-stage training; the evidence does not
establish that more epochs on the same templates would solve this.

`wording-audit.json` records zero-update counterfactuals on the already-consumed
development data, with neural contexts fixed. Replacing only `Disregard` with
`Ignore` improves aggregate accuracy from 25% to 50%, but does not solve the task;
other substitutions do not consistently help. These are diagnostic probes, not
independent evaluation scores or deployed instruction-rewriting rules.

The next proposed checkpoint is explicit source-directive vocabulary grounding
followed by **new combinations of taught words**, with unseen synonyms reported as
a separate harder test. It requires a new versioned curriculum and fresh
development holdouts, preserving the current gate failure rather than relabeling
these consumed examples as unseen. The user subsequently authorized the follow-up
below. The eight final distance cases remain unopened, and confidence remains
uncalibrated.

This run used:

```sh
.venv/bin/python scripts/train_source_request.py experiments/distance-source-001 \
  --parent experiments/distance-identity-001/training/checkpoint.pt
```

Do not rerun into that directory. Inspect its saved `report.json` and
`wording-audit.json`; replay requires neither credentials nor inference:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-source-001/development/800000.events.jsonl
```

## Vocabulary grounding and known-word composition

New curriculum `source-vocabulary-v1` continues the **existing** 7,440-parameter
source encoder from `distance-source-001`; it neither adds model parameters nor
reinitializes learned weights. All non-source parameters, including the biological
workflow and quantity/unit encoder, remain frozen. A fresh AdamW optimizer is used
throughout one fixed run, CPU/one thread, hidden size 16, seed 0:

- Updates 1–50: six vocabulary pairs per update.
- Updates 51–200: five composition pairs plus one vocabulary pair per update.
- Total: 200 updates, 900 vocabulary and 1,500 composition labeled decisions.
  No automatic extension, retry or end-to-end workflow training.

Vocabulary exercises teach `use`, `select`, `obtain`, `find` and negative directives
`ignore`, `disregard`, `avoid`, `not`. Single negative clauses mean choose the
nonexcluded source **within this explicitly two-source exercise**; this does not
define how an unrestricted multi-source browser instruction should be interpreted.
The grammar is used only to create supervised examples, never to select or repair
the policy's inference actions.

| Split | Instructions | Purpose |
| --- | --- | --- |
| Vocabulary | 144 | Isolate directive meaning, with paired source changes |
| Composition training | 432 | Twelve positive/negative combinations in both clause orders |
| Fresh composition development | 144 | Four held-out combinations using taught words |
| Unseen-synonym diagnostic | 144 | Separate harder lexical test after checkpoint freeze |

Fresh development reserves `use + avoid`, `select + disregard`, `obtain + not`, and
`find + avoid`. These combinations exclude the source parent's training and recorded
wording probes. The validator rejects word-coverage gaps, overlapping instructions,
numbers or IDs across splits, and train/development combination overlap. It also
checks the new development instructions against both earlier recognition datasets.
Old failed development examples remain **consumed regression data**, never renamed
as a new holdout. Deliberately mismatched units remain recognition-only exercises.

The harder diagnostic uses untaught `retrieve`, `gather`, `exclude`, and `omit`.
It runs only after the single checkpoint is saved/reloaded and cannot steer
optimization. Its source accuracy is reported separately, not used to redefine
the known-vocabulary gate. This narrower composition scope does not erase the
previous run's failed broader wording gate or establish general English competence.

The fixed gate retains 95% fresh-development source accuracy, 95% pairs correct on
both sides, 95% legacy-development source accuracy as a regression check, unchanged
100% quantity/unit accuracy, and all six existing workflows completed in ten actions
with zero errors. It additionally requires 95% vocabulary recognition. Frozen
non-source tensors/scores and checkpoint reload must remain exact. Even a passed
gate does not automatically open the final distance cases or promote a checkpoint.

```sh
.venv/bin/python scripts/train_source_request.py experiments/YOUR-FRESH-VOCABULARY-RUN \
  --parent experiments/distance-source-001/training/checkpoint.pt \
  --curriculum vocabulary-v1
```

The runner preserves old default behavior when `--curriculum` is omitted. This
explicit curriculum requires a source-stage parent and rejects automatic continuation
from its own output. Artifacts include `vocabulary.json`, `paired-{train,development}.json`,
`synonym_diagnostic.json`, manifests/hashes, stage-labelled progress, checkpoint,
reports and offline workflow replays. Calibration remains explicitly unavailable.

### Actual vocabulary/composition result: controlled gate passed

`experiments/distance-vocabulary-001/` completed exactly the declared 50/150 schedule:
200 updates, 900 vocabulary and 1,500 composition decisions. Optimization took 2.38
seconds; preparation, reload and all evaluations took 30.85 seconds, with peak
process RSS about 459 MB. Loss fell from 15.5328 to 0.0000761; gradients remained
finite. The existing source encoder was continued, not reinitialized or replaced.

| Source-recognition evaluation | Before this run | After this run |
| --- | --- | --- |
| Vocabulary training | 72/144 (50%) | 144/144 (100%) |
| Composition training | 237/432 (54.9%) | 432/432 (100%) |
| Fresh known-word composition development | 81/144 (56.25%) | 144/144 (100%) |
| Legacy identity development, already consumed | 54/72 (75%) | 72/72 (100%) |
| Previous source training, retention check | 144/144 (100%) | 126/144 (87.5%) |
| Previous source development, already consumed | 18/72 (25%) | 36/72 (50%) |
| Untaught-synonym diagnostic | Not measured before training | 54/144 (37.5%) |

All 72 fresh development pairs are correct on **both** source requests, including
both clause orders. All predeclared **known-vocabulary composition** gates passed.
Quantity/unit recognition stayed 100% across all sets. Every non-source tensor and
checked quantity/unit score remained exactly unchanged, and the checkpoint reloaded
with identical weights and predictions. All six existing distance workflows still
complete in ten actions, with zero invalid actions, tool errors or infrastructure
failures; teacher-forced workflow decisions remain 100%.

This is **not general instruction understanding**. The previous broad wording set
is still only 50%, previous source-training retention fell to 87.5%, and unseen
synonyms score 37.5%. The earlier failed run remains failed and unmodified. The
fresh 144-case composition score is not directly comparable to that run's different
72-case development score; the table shows same-set before/after comparisons.

The checkpoint hash is
`4e857c8bc739f77d902403a0e005e9c39446b059750326e3d6dfa3ad242dd32f`.
It remains an experimental, uncalibrated, local-tool-assisted policy. The next
suggested bounded check is a **frozen-checkpoint evaluation** on the eight untouched
distance cases, with no training or selection of alternative checkpoints using
those results. At the end of that training stage the final distance cases remained
unopened. The separately authorized evaluation below has now consumed them;
broader stellar/browser acceptance remains deferred.

The completed command was:

```sh
.venv/bin/python scripts/train_source_request.py experiments/distance-vocabulary-001 \
  --parent experiments/distance-source-001/training/checkpoint.pt \
  --curriculum vocabulary-v1
```

Inspect the saved `report.json` rather than rerunning into this directory. Replay:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-vocabulary-001/development/800000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/distance-vocabulary-001/development/800000.events.jsonl
```

## Frozen eight-case distance evaluation (completed; gate failed)

`experiments/distance-final-001/report.json` records the evaluation of the fixed
`distance-vocabulary-001` checkpoint. The checkpoint SHA-256, graph, complete
knowledge pack, source files, dependency versions/lockfile and test manifest were
verified before opening the test payload. The evaluation-only runner records its
gate and identities first, freezes every parameter, blocks socket connections and
Sheets initialization, and has no training or alternative-checkpoint option.

Actual results:

- **0/8 completed**, all eight reaching the original **32-action limit**.
- **72/80 (90%)** exact decisions on expert observations. The sole failing stage
  was source selection: **0/8**; the other nine stages each scored **8/8**. This
  teacher-forced diagnostic is not autonomous task completion.
- In every autonomous rollout, the policy selected **reference-star parallax**
  instead of current-star parallax. The tool correctly calculated the selected
  measurement and copied its result exactly. After rejected answers, the policy
  repeated copying/unit selection without correcting the binding.
- **0 invalid actions, 0 tool errors, 0 API or infrastructure failures**. A valid
  calculation using the wrong star is a policy mistake, not a calculator failure.
- **0 optimizer updates**; all model tensors and the checkpoint file remained
  unchanged. Confidence remains explicitly **uncalibrated**.
- Evaluation/attribution took **5.75 seconds**, peak process RSS **273,498,112
  bytes**. This is a distance-only diagnostic, not the 100-case stellar gate.

The test instruction says “the supplied star” and “its parallax.” The observed
failure is selecting the wrong source under new wording, consistent with the
earlier limited vocabulary results; no counterfactual wording experiment or
retraining was run after inspecting this test. Recovery from incorrect answers is
also absent here.
Future work should target source references/aliases with new development cases
and a newly sealed holdout, then address recovery separately. Increasing arithmetic
training or opening HabWorlds does not address the demonstrated wrong binding.

All eight cases are **consumed evaluation data** now, not an unseen checkpoint
selection set. No other checkpoint was evaluated against them, no calibration
was fitted, and no follow-up training was started. Earlier reports retain their
historical `not_run` fields; this new directory records the later evaluation.

The completed command was:

```sh
.venv/bin/python scripts/evaluate_distance_final.py experiments/distance-final-001
```

Inspect the saved report instead of rerunning. The runner refuses to overwrite
an existing directory; even a deliberate reproduction under a fresh directory
must be labeled a repeat on consumed cases. Every failed rollout has a trajectory
and version-1 event log. All eight event logs replayed exactly offline (**132 events
each**); a previous source-stage log also replayed unchanged (**44 events**).

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-final-001/test/900000.events.jsonl
cargo run --manifest-path tui/Cargo.toml --locked --offline -- \
  --replay experiments/distance-final-001/test/900000.events.jsonl
```

The new runner's tests use synthetic payloads, not these experiment cases. They
cover hash/scope/episode integrity, split overlap, output preservation, refusal
of training/checkpoint-selection flags, and offline tripwires. No model,
environment, default profile, or runtime protocol changes were made for this run.

## Source aliases and manual TUI readiness

The later continuation is documented in [the manual distance guide](distance-manual-test.md).
Two more source-only runs, each capped at 1,000 updates, retained all old artifacts:

| Run | New development workflows | Original workflows | Gate |
| --- | --- | --- | --- |
| `distance-aliases-001` | 36/36 | 0/4 | Failed retention; final set not evaluated |
| `distance-aliases-002` | 36/36 | 4/4 | Passed all declared development gates |

The second run additionally rehearsed the original full instruction, with both
current/reference source choices. It obtained 324/324 fresh source decisions,
162/162 contrastive pairs and 936/936 consumed-rehearsal decisions. Only the 7,440
source-request parameters were trainable; the biological recurrent workflow and
quantity/unit parameters stayed frozen. No inference parsing rule or expert
fallback was added. Each run used CPU/one thread, hidden size 16 and seed 0; no PPO.

The second checkpoint was frozen **before** evaluating the new 100-case split:
**100/100** completed in ten actions, no invalid/tool/API/infrastructure failures,
16.84 seconds and 271,712,256 bytes peak process RSS. Separate templates and numeric
cases were held out from training/development; this establishes known-alias
distance generalization, not arbitrary language or full stellar/browser competence.
The test split is now consumed, and no further training used its results.

All 12 separate demo cases completed through the actual runtime with pause, step,
resume, save and exact offline replay. The runtime now has a hash- and gate-checked
`task: distance` path. The TUI labels measurement options, keeps a unique visible
target highlighted across observation revisions, and scrolls full observations.
Quitting completed runs no longer emits a misleading second aborted summary.
Protocol version remains 1; old stellar, Sheets and synthetic defaults are preserved.

Completed commands (inspect these directories instead of overwriting them):

```sh
.venv/bin/python scripts/train_distance_aliases.py experiments/distance-aliases-001 \
  --parent experiments/distance-vocabulary-001/training/checkpoint.pt --updates 1000
.venv/bin/python scripts/train_distance_aliases.py experiments/distance-aliases-002 \
  --parent experiments/distance-aliases-001/training/checkpoint.pt --updates 1000
.venv/bin/python scripts/train_distance_aliases.py experiments/distance-aliases-002 --evaluate
.venv/bin/python scripts/verify_distance_demo.py experiments/distance-tui-verification-001
```

The original source runner still defaults to its prior 200-update cap. Expanded
training requires an explicit budget limit, with a hard maximum of 1,000 per run;
the new experiment runner never automatically retries, increases it or opens tests.
The first alias run used curriculum v2, the second v3. Their exact datasets, hashes
and code provenance are retained; rerunning today's generator would use v3.

## Implementation verification

- Full Python regression after manual-demo readiness: **225 passed**, including
  alias separation, explicit budget caps, gated distance sessions, completion/quit
  semantics, representation, pointer, checkpoint and existing environment/browser tests.
- Rust: 21 regular tests plus the Python subprocess bridge passed. The bridge
  now verifies that quitting a completed replay preserves completion, while
  aborting an active episode still reports interruption.
- Ruff, touched-file formatting and whitespace checks pass.
- Both original diagnostic checkpoint SHA-256 hashes remain unchanged.
- The 800-update parent checkpoint is unchanged. The identity checkpoint hash is
  `2b27cd06fcc8e325511551d4240c29b07ceaf4a1bf050c10da62666fd86ea95e`.
- All six new workflow traces and one old successful trace replayed exactly
  offline (44 events each), with socket connections and Sheets initialization blocked.
- The source-stage parent identity checkpoint remains unchanged. Its new experimental
  checkpoint hash is `dde2fc29f4b45eccd3a21a04244026673a6e86ecf317df6f2eb9a396ec3390c5`.
- All six source-stage workflow traces plus one identity-stage trace also replayed
  exactly offline (44 events each). The source-stage learning gate failed despite
  passing engineering regression checks.
- All six vocabulary-stage workflow traces and one older source-stage trace
  replayed exactly offline (44 events each). Source and identity parent checkpoint
  hashes are unchanged. New tests cover vocabulary/composition split coverage,
  paired directive labels, continuation without weight resets, the exact staged
  budget and rehearsal counts, metadata nonleakage and reload compatibility.
- Default stellar and diagnostic budgets are unchanged. Repair modes are opt-in;
  an implementation test is not evidence of learned task completion.

## Reproduction and inspection

Run from the repository root. Use a fresh output directory; these commands refuse
to overwrite an experiment. Dependencies and graph must already be local.

```sh
.venv/bin/python scripts/repair_distance.py experiments/YOUR-FRESH-DIRECTORY \
  --variant structured-options
```

`--variant structured` runs the representation-only comparison;
`--variant topology-free-options` runs the explicitly labeled baseline. Each
defaults to 80 updates. The separately approved run was started with:

```sh
.venv/bin/python scripts/repair_distance.py experiments/distance-repair-options-800-001 \
  --variant structured-options --optimizer-updates 800
```

That run is complete; do not rerun it into its existing directory. `status.json` and
`training-progress.jsonl` record progress; `report.json` is written only after
checkpoint reload and evaluation. The runner blocks Python socket connections and
Sheets initialization (a reproducibility guard, not an OS security sandbox).

The separately approved (failed) semantic follow-up used:

```sh
.venv/bin/python scripts/repair_distance.py experiments/distance-repair-semantic-001 \
  --variant semantic-options \
  --parent experiments/distance-repair-options-800-001/training/checkpoint.pt
```

This mode requires a parent that passed its training-fit gate and rejects the
800-update flag. Other variants reject `--parent` and always start fresh.

Inspect frozen weights without training, using a new output file:

```sh
.venv/bin/python scripts/inspect_distance.py \
  experiments/distance-repair-options-001 experiments/YOUR-FRESH-AUDIT.json
```

Audits record checkpoint hashes and any code drift since saving. The legacy modes
retain old computation behavior, but source changes invalidate old calibration;
the audit uses raw logits rather than claiming recalibrated confidence.

Replay never needs weights, credentials, or network access:

```sh
.venv/bin/python scripts/habfly_offline.py replay \
  experiments/distance-repair-options-001/seen/600000.events.jsonl
```
