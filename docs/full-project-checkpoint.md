# Full-project continuation: first discovery checkpoint

## Current launch boundary (2026-09-28 Phoenix)

The user's latest instruction is to finish the current three-star and readiness
checks, then **stop before starting the 30-star test**. Do not launch the 30-star
profile without a new explicit go-ahead. The historical authorization and
chronological checkpoints below do not override this newer boundary.

### Latest live validation: three workflows passed (2026-09-28)

Fresh run `a90d83bc86aa4a84a0908ba6665e7a01` under
`experiments/browser-project-single-event-three-star/` closed successfully at
`awaiting_assessment`. Argargash, Adraoi and Daeleleb each completed the stellar,
observation-window, No-planet visible-readback and inventory workflow. Canonical
revision 18 has three collected, three verified and zero unresolved stars.
Both fresh-star transitions were automatic. No explicit Save, assessment,
Update Score, Submit or training was performed. The browser/runtime is closed.
Each star copied six numeric fields in 60 policy steps, with unchanged weights
and zero optimizer updates. Numeric stages took 63.57, 94.33 and 121.65 seconds
respectively. Approximate scope-to-report artifact-write duration was 24m38s;
the trace has no wall-clock timestamps, so this is not a precise event timing.

All 3,559 recorded payloads replayed unchanged (3,561 replay emissions), with
network, model, environment and subprocess starts forbidden. Credentials were
absent and the source trace remained unchanged, SHA256
`e75e27bc01f2facf2d0d8c99f4dce25d78c5adfcbeb6203069abea8ec100ec12`.

This establishes the short autosave-assumed workflow and repeated navigation,
not a live full-project score, scientific accuracy or persistence across browser
sessions. All three live cases followed a No-planet branch; the separate native
fixtures provide the new planet-present and terrestrial autosave coverage.
Argargash and Daeleleb used the no-visible-dip rule; Adraoi exercised the
baseline-band allowance. None used the single-event shortcut. All three source
chains, inventory captures, task receipts and canonical journal bindings passed
the independent read-only audit.
The first 30-star attempt remains unlaunched. Full-scale timing is unverified:
numeric elapsed time includes synchronous logging and source verification, and
the campaign rechecks a growing set of prior-star evidence. The closed run's
internal bridge/campaign logs total about 1.34 GiB, versus 24 MiB for its compact
outer trace. No neural-state/listener retention leak or 1,800-second per-star
failure was demonstrated. This is a performance limitation to measure during
the first full attempt, not evidence that 30 stars have already passed.

### Overall timer instruction (2026-09-28)

After seeing the measured short-run pace, the user explicitly rejected an
overall campaign cap. Do not substitute the proposed six-hour limit. The held
30-star profile now selects explicit `"uncapped"` overall time, while
retaining the 30-star finish condition, per-star/action guards, source checks,
manual abort and the separate assessment limits. The existing bounded test
profiles keep their original finite limits. This is a configuration decision
before launch, not automatic extension of a running attempt.

The timer-only change passed 719 targeted Python regressions and 58 focused
checks (overlapping groups), plus 127 Rust tests with one optional bridge test
ignored in that last Rust command. Lint/format and diff checks passed. The final
source-only held-30 check validated 504 local files and reported
`campaign_max_seconds: "uncapped"`, `campaign_timer: "no_overall_timer"`, scoring
enabled, submission disabled and both launch authorization flags false. No
browser, model, inference or training was started by that check. The successful
three-star and previously stopped Diativos traces again replayed every payload
unchanged after the timer change; historical failures were not reclassified.

Implementation is ready for the user's first 30-star attempt, not a claim that
the full attempt has passed. From the repository root, the user may explicitly
launch it with:

```sh
.venv/bin/python scripts/browser_project_tui.py \
  --profile configs/browser_project_thirty_star.json \
  --allow-thirty-star --mascot
```

The launcher prompts for the plain preview URL and hidden login credentials.
The TUI starts paused; `r` resumes, `p`/Space pauses and `a` aborts. Codex has not
executed this launch command. Successful completion now means 30 verified
workflows, both assessments and a verified visible Update Score result; formal
Submit remains disabled.

### Requested finish line: Update Score (2026-09-28)

The user's newest completion checkpoint is to finish the stars, run the
assessments, click **Update Score**, and report the score shown by the course.
A perfect score is not required. The held 30-star profile is configured to stop after that
verified score update rather than pressing Submit. This is a score checkpoint,
not a claim of formal project submission or perfect scientific answers. The
previous missing successful-Submit acknowledgement is therefore deferred and
does not block this newly requested finish line. The 30-star launch hold still
applies; the three-star validation profiles do not perform scoring/submission.

The existing Update Score transport passed all 17 intercepted Chromium tests:
one explicit click, actual visible score readback, unchanged submission checkbox,
and rejection of stale, missing or uncertain acknowledgement. These fixtures
contact no HabWorlds service and do not establish a live 30-star score.

Implementation checks passed: 31 joined finalizer/real runtime-owner tests, a
two-case actual runtime terminal-publication and EOF-replay check, 64 profile/
launcher cases, and 127 Rust tests including the Python bridge. Additional
overlapping Python regression groups and formatting/lint checks passed. The
new optional `score_checkpoint_completed` and `reported_score` fields require a
cleanly closed score-only finalizer and its current canonical assessment/score
proof; zero is a valid score. Failed, stale and old absent-field replay states
cannot acquire this new success claim. Formal submitted/task/project completion
fields retain their old meaning and remain false for this score-only checkpoint.

The frozen held-30 profile's source-only check validated 504 local files, with
scoring enabled and submission disabled. It launched no browser, loaded no
model, and performed no inference or training. Launch authorization remains
false. The next live validation is bounded to three stars and cannot score or
submit.

### User-approved autosave strategy (2026-09-28)

The user clarified that the preview autosaves periodically and asked to stop
explicitly saving. The recommended single-event three-star profile and held
30-star profile now opt into `project_save_strategy: "autosave"`. Other profiles
and the default retain the original `"explicit"` strategy.

This mode does not find or click Save, wait for a banner, or treat two minutes
elapsed as proof of persistence. Each branch still verifies fresh visible
answers and its source chain, then verifies the workflow and collected-star
inventory. Receipts explicitly label the authority as visible readback only,
with `persistence_verified: false` and `save_acknowledgement_verified: false`.
The user-supplied autosave behavior is an assumption, not newly measured site
behavior. Legacy replay retains its original Save meaning, and no stopped or
uncertain Save attempt is reinterpreted or retried under the new mode.

This change follows the closed run `929184271e154d119527a2b52709dc99` under
`experiments/browser-project-single-event-three-star/`. Diativos completed its
stellar work and the fixed observation window, selected the existing assumed-No
branch, then stopped with `unverified_no_planet_save_acknowledgement` before a
Save reservation or click. No star was counted, and no assessment or submission
occurred. The new single-event shortcut was enabled but was not used for that
star. Its 1,295 recorded events replayed unchanged (1,297 emissions); the trace
SHA256 remains
`04bf70725db39be00092ec2365aa29942b16889e3623cc8e6700d439870c75b1`.
That historical failure is preserved. The new autosave workflow must pass
implementation checks before a fresh bounded run; the 30-star launch is still
held.

The merged initial gate passed 437 Python tests (two native tests excluded),
and eight intercepted Chromium tests passed. The latter exercise the new
No-planet path with the Save control removed, both capture strategies, strict
workflow/import and offline source reload, changed-answer/choice/prior-Save
rejection, and the old single-event explicit-Save path. Two new test assertion
and setup mistakes were corrected without changing production behavior.
The upper layer separately passed 124 Rust tests with one existing optional
bridge test ignored. Both the old successful Babrignar trace and stopped
Diativos trace replay unchanged without network, subprocess, environment or
model starts. A further 257-test merged gate passed (two native tests skipped,
two native import cases separated for their required macOS launch permissions).
Six additional intercepted browser cases passed: new planet-present and
terrestrial autosave readback/workflow/import using both capture strategies,
changed-answer rejection, and the older supplied-input positive and terrestrial
explicit-Save chains. The held 30-star profile's source-only check passed with
504 local files and no browser, inference, training or launch authorization.
The two separately run native import cases also passed, for 16 intercepted
browser cases across the new and historical paths. These tests are not live
completion or persistence claims.

The fresh autosave run `fb0a8b80bf494312bdc419b1b31d5a1a` is now closed, not
passed. Diativos completed all six stellar fields, the full observation window,
the No-planet autosave readback, and the two-tab visible workflow. There were
zero explicit Save actions and no persistence claim. The next inventory
coordinator stopped with `project_inventory_steps_unsupported_completed_workflow`
because its admission predicate still required an explicit Save acknowledgement.
No collected-star import occurred; canonical collected/verified counts remain
zero. This is an integration bug after successful readback, not a new learning
failure. No stopped owner is resumed or relabelled.

All 1,283 trace payloads replayed unchanged (1,285 emissions), with network,
model, environment and subprocess starts forbidden. Trace SHA256:
`9ec9672ae7b9d85428b1c05ccceedd2a1cf5c5acd64a2740899774196788c4ac`.
Owner elapsed was 407.16 seconds, including 76.25 seconds for numeric inference
and 243.35 seconds for the window; this is not completed-star throughput.
The follow-up fixes the inventory admission gate and both planet-present and
terrestrial parent completion gates together. Parent handoff regression passed
121 tests, including 18 new selected-authority and malformed-child cases; the
full inventory/campaign integration gate precedes any further live validation.

That follow-up integration gate now passes: 349 parent/runtime tests (two native
cases excluded), 194 focused autosave inventory/campaign tests, and the existing
next-star/finalizer regressions. Two native default/pinned coordinator cases
also passed after correcting only the new test document's heading order. These
exercise the actual inventory owner, two native List/Stellar clicks, inventory
verification and strict canonical import after an autosave workflow. Source-tree
guards now remain attached through No import, next-star transitions, and cached
finalization checks. Old failed traces remain stopped and unmodified.

### Approved single-event demo shortcut (2026-09-28)

After the Jarlarth stop below, the user explicitly approved treating a single
possible dip as **No planet** for this for-fun project, accepting imperfect
answers. This is a separate, opt-in assumption—not a newly learned detector,
physical absence claim, or training label. The old stopped attempt and its
recorded policy stay unchanged; it is not resumed or retroactively passed.

The new `project_single_event_reference` setting selects a separately versioned
local policy. It retains the existing baseline-band rule and additionally
permits one supported, localized feature on a complete 0–5,000-day overview to
be deliberately ignored. Missing coverage, unsupported pixels, clipping and
multiple candidate events are not catch-all No answers. Original chart/axis
hashes, the policy identity and the explicit shortcut reason accompany the
decision through fresh readback and replay. No orbital period/depth is invented.
The existing prior-positive-evidence conflict guard is retained.

The implementation gates passed: 227 analyzer/legacy tests, 212 lower-layer
source/replay tests, 243 upper runtime/wire tests (overlapping groups), and 122
Rust tests with one existing bridge ignored. Root's merged run passed 510 tests
but caught one incorrect new fixture expectation; after correcting that test
without changing production behavior, 173 focused tests passed (two branch
skips). Three intercepted native tests passed: default and pinned No/Save,
workflow verification, canonical import and offline replay, plus a second
feature appearing before selection that correctly blocks the shortcut. These
fixtures do not establish a successful live run or scientific correctness.

The new `configs/browser_project_single_event_three_star.json` preserves the
3-star/5,400-second campaign and 1,800-second/512-advance star caps, frozen
models, paused start, and disabled scoring/submission. The held 30-star profile
adds only the explicit shortcut flag, without changing caps or starting it.
Its read-only launcher check validated 503 local source files and reported
`thirty_star_launch_authorized: false`; no model, inference, training or browser
was started by that check. A fresh bounded three-star live validation follows
these gates. Historical gates below retain their original meaning and do not
supersede this newer explicit shortcut permission.

### Earlier bounded check: one unresolved candidate event (2026-09-28)

The fresh three-star check `5e00bb5681c240c4b376000f041cb464` under
`experiments/browser-pinned-capture-three-star/` is closed and **not passed**.
Its first star, Jarlarth, completed the stellar numeric stage (60 actions,
66.96 seconds, zero training updates) and reached the end of the fixed
5,000-day observation window. The original overview yields one candidate
group, image columns 243–244, near day 4,650. A candidate is not a confirmed
transit; neither the positive detector nor the approved baseline-band No rule
accepts this capture. The shallow sensor therefore stopped before probing with
`shallow_probe_exact_two_overview_hints_required`. No period or No answer is
inferred, no Save was attempted, and no star was workflow-verified. The stopped
owner must not be resumed or retried.

Read-only replay preserved all 1,398 recorded payloads (1,400 playback emissions)
with network, model, environment and subprocess starts forbidden. The original
trace remains unchanged, SHA256
`f5927c99f90be26e212dcd6e192168e51bd9355e0e36377a39c4b16c6e28c05d`.
Canonical revision 0 has zero collected/verified tasks; playback completion is
not task completion. Pixel inspection confirms one below-baseline connected
feature, not the faint baseline-only case: no second event is recovered by any
current hint policy. This does not establish a physical transit or its period.

This is insufficient repeat-event evidence, not a training loss or login error.
The user's no-evidence-within-5,000-days assumption does not resolve a visible
single candidate. Leaving that star incomplete and selecting another requires
a separate explicit handling decision; it must not count an attempt as one of
the 30 verified stars. The 30-star launch hold remains active.

Before this live check, the merged native gate passed 37 tests, including both
Save settlement wrappers, default and pinned real-graph terrestrial chains,
and the final-callback No-Save admission case. Rust passed 117 tests with one
optional model bridge not run. These are implementation checks, not evidence
that the failed live campaign or the full project completed.

### Whole-workflow capture and all-branch Save hardening (2026-09-28)

The explicit `pinned_control_capture` configuration option keeps old readers as
the default. The new `browser_probe_pinned.example.json` changes only that option;
origins, paths, frames, observation limits and submission protection are identical.
The bounded baseline-band three-star profile and held 30-star profile now select
this configuration. Every capture still enumerates fresh controls, retains native
AX/value/visibility checks, rejects changed inventories, and disposes all handles.
There is no cross-capture cache, action shortcut, or budget increase. Launcher
`--check` reports `control_capture: fresh_pinned_handles` without launching.

The AX parser retains the original Python syntax/alias/tag/complexity gate and
uses the installed C safe constructor for compatible shallow data only. Unusual
deep input and C-parser errors retain the original safe-parser behavior. Current
bytes are reparsed every time. Exact-output/error parity and fresh source-chain
checks passed; this is an implementation optimization, not changed observations
or a model update.

Planet-present and terrestrial owners now pin their fixed timing/settlement
options across callbacks. Their opted-in Saves use fresh pinned captures and the
same observed-readback admission rule as No-Save. All branches recheck headroom
after the final callback, immediately before allowing the native click. The
20-second reserved deadline is never extended. A late rejection retains any
durable pre-click dispatch marker conservatively; it never enables a retry.
Failure reporting also preserves conservative uncertainty when that final
admission rejection occurs after the durable marker; it does not downgrade that
known possible-write state to missing evidence. Contradictory acknowledgement
or marker/stop records still report unknown and cannot imply success.

The parent now forwards source-bound failed-Save diagnostics for positive and
terrestrial branches as well as No. Missing, changed or malformed evidence is
unknown, not proof of no write. Acknowledgement, complete readback and verified
workflow remain separate. A successful Save followed by a workflow failure is
not relabelled as an uncertain Save. No canonical counts or journals are changed
by these reporting summaries.

The timing gate remains open. The last completed Babrignar run used 424.47 seconds
from class setup through verified owner completion (445.45 seconds from launch),
before the whole-workflow optimizations above. Multiplying that per-star span by
30 already exceeds the held 10,800-second cap, even without later pagination or
positive/habitability work. It is an extrapolation, not a measured 30-star run.
New live throughput and representative branch coverage must be reported honestly.
The separate missing-success-acknowledgement gate at the end of this document
also remains; no success matcher or completion receipt has been fabricated.

### Save admission and capture-local control handles (2026-09-28)

The follow-up implementation keeps the 20-second reserved Save deadline and
at-most-once dispatch. Fresh No-Save guards opt into a capture-local reader that
enumerates role handles once, retains Playwright visibility/enabled checks and
locator accessibility snapshots, and rejects changed control identity, order,
count or exposure before returning. Handles are disposed after each capture;
none are cached between reads. Other callers retain their existing reader.
Visible values, hidden-control exclusion and observation schemas are unchanged.

Before dispatch, the adapter now requires more remaining time than its existing
three-second maximum click allowance plus the slowest full guarded read measured
in that session. This is a minimum admission estimate, not a guarantee about
future acknowledgement or readback latency. An insufficient budget records
`dispatch-budget-rejected.json` and stops before any Save dispatch; the permanent
reservation remains and no retry is enabled. Post-click deadline/cancellation
checks and uncertainty handling still apply, even after a fresh `Data saved`
notice. The closed Aguilthir attempt is not resumed or modified.

In a fully intercepted synthetic 5,000-sample, 40-control comparison, three
measured reads per variant had median durations of 663.312 ms (current locator
reader) and 512.718 ms (capture-local handles), with identical observations except
the timestamp. These six reads and two warmups used no model or UI actions;
the report is `experiments/save-readback-pinned-profile-001.json`. This small
synthetic measurement is not a live performance or completion claim.

Readiness audit: positive/terrestrial workflows and assessment/score transfer
are implemented but need current live acceptance. Successful final-submission
recognition and its canonical receipt remain deliberately unimplemented until
a real student-visible successful acknowledgement is available. Refusal and
unknown outcomes already stop without retry. No success message is invented.

The held `browser_project_thirty_star.json` is now aligned with the already
approved two-event and baseline-band flags. Its paused start, explicit launch
flag requirement, 10,800-second campaign cap and 1,800-second/512-advance star
limits are unchanged; it has **not been launched**. The historical throughput
concern remains unresolved, so profile alignment is not full-project readiness.

Offline gates passed before the live check: 204 focused browser/Save/observation
tests, 446 owner/runtime/uncertainty tests and 191 profile/launcher tests (these
groups overlap), plus 117 Rust tests with the model bridge intentionally not
run. Formatting, Ruff, diff checks and an independent source review passed.
The aligned held profile passed a 502-file source-only check without browser,
model loading, credentials, training, or launch authorization.

Then one fresh **single-star** integration ran, with campaign mode, assessment,
score transfer and submission disabled and the original 512-advance/1,800-second
limit unchanged. Run `927b134a6fdd4ca088ac610fd69360ac` under
`experiments/browser-save-readback-one-star/` is closed and **passed** for
Babrignar: one explicit Save, fresh visible acknowledgement, full readback within
the original 20-second reserved deadline, strict workflow verification and
collection import. It records one collected/verified task and no completed
project. The new baseline-band 5,000-day assumed-No rule was used; this is not a
scientific absence claim. This save used one reserved full revalidation and
zero notice-wait probes, so the repeated-notice branch remains fixture-tested,
not newly live-accepted by this run. No second star or retry was started.

A separate read-only audit revalidated 68 inventory/workflow source files. All
1,342 recorded events replayed unchanged (1,344 emissions including envelopes),
with network/model/environment/subprocess starts forbidden and no credential
fields. Trace SHA256:
`ef3b3966db76e288aea5daa60d9fcc495e2b6744a61220df47daf549fb2ccbe4`.
Replay retained task completion separately from false project completion. The
older Aguilthir failure also replayed unchanged and remains an uncertain Save.

Next is representative planet-present/terrestrial live coverage and throughput
validation, not another attempt to relabel the old failure. Final submission
recognition still needs grounded successful student-visible evidence. The
30-star launch remains held; no additional live run is part of this checkpoint.

### Baseline-band live check and failed-Save diagnostics (2026-09-28)

The authorized bounded three-star run `4f4d017de98b498487f396539487abf9`
is **closed, not passed**. Abristiri completed the new approximate No path,
was saved, verified and collected. Aguilthir reached the same bounded-window
decision but stopped during Save verification; no third star was attempted.
The result is **one verified star**, not a three-star success. No assessment,
score update, submission, training, model change or budget increase occurred.
Artifacts and the raw event stream are under
`experiments/browser-project-baseline-band-three-star/`.

Aguilthir has one recorded Save dispatch and a fresh visible `Data saved`
acknowledgement. Its final full readback did not finish within the existing
20-second post-reservation deadline (`no_planet_save_reserved_phase_timeout`).
File timestamps put acknowledgement at about 15.6 seconds and the stop at
20.6 seconds; these are evidence timings, not an instrumented performance
profile. The profile's separate 30-second pre-reservation settlement setting
does not extend that deadline. There is no confirmed receipt or final `after/`
capture. The Save may have occurred; its reservation stays retained. Do not
retry, resume, promote, or relabel this attempt. The canonical journal's empty
uncertainty list only describes imported workflows and does not clear this
unimported child failure.

The follow-up patch keeps the intended Save artifact path even when Save fails,
and forwards a nullable `save_outcome_uncertain` plus bounded, hashed
`save_outcome` diagnostic through star, project, campaign and compact JSONL
state. Acknowledgement is explicitly distinct from final readback and canonical
completion. Missing or invalid records stay unknown. These are noncanonical
diagnostics; neither counts nor receipts change, and no retry is authorized.
Rust displays that distinction and clears stale warnings on star/run changes
or authoritative legacy snapshots. Old saved events remain unchanged.

Control capture now combines exposed native input metadata in one read, while
retaining the existing visibility, accessibility, enabled-state, protected
control and inventory-budget checks. Plain buttons keep their original read
path. This removes ten browser calls for the recorded screen's 40-control,
five-native-value shape. An initial broader batching candidate was slower in
the synthetic comparison and was narrowed before delivery. Both comparisons
used only intercepted local fixtures: two warmups and six measured reads each,
with 5,000 synthetic chart samples and exact capture equivalence apart from the
timestamp. They are diagnostic samples, **not proof that the live timeout is
fixed**. No second live attempt was started after the failure.

Final follow-up validation: 431 Python owner/runtime/diagnostic checks plus
142 control-capture and intercepted-browser checks passed. Rust passed 117
checks (one model bridge intentionally not run); Ruff, formatting, Clippy and
diff checks passed. Source-only preflight validated 502 files without launching
a browser or model. Offline replay preserved all 2,455 recorded events (2,457
emissions including replay envelopes), with no network/model/environment start
and unchanged trace SHA256
`2131fcdfa109e43554c9af8d62270a7142052aed4ba2e38ecc24a6e0fb1e8d5d`.
The held 30-star profile and frozen reference/model files remain unchanged.

Next is a fresh bounded integration of the readback change, not a retry of the
uncertain Save and not the held 30-star test. Positive/terrestrial live
acceptance, campaign timing and successful submission acknowledgement remain
separate readiness limitations.

### Explicit coarse 5,000-day No assumption (2026-09-28)

The user clarified that a completed 5,000-day window with no clearly visible dip
means **assumed No planet** for this demo. A new, separately versioned
`user_approved_5000_day_baseline_band_no_dip_v1` implements that rule for faint
supported blue shades within the existing 1.5-pixel baseline tolerance. It
requires complete contiguous original-blue coverage and does not repaint the
chart. Deeper features, unsupported colors, incomplete coverage, invalid axes
or overlays cannot be turned into No. Subpixel features may be missed: this is
not scientific absence, learned perception, a training label, or a reason to
invent zero period/depth values.

The new opt-in is `project_baseline_band_reference`; old profiles, frozen
detectors and old receipt/replay interpretations are unchanged. The separate
`configs/browser_project_baseline_band_three_star.json` inherits the previous
three-star profile's 5,400-second campaign, 1,800-second/512-advance per-star
limits, paused start, and disabled scoring/submission. Saved, fresh and
immediately pre-selection evidence must all match the new policy; saving and
offline replay revalidate the same original bytes. Scope and source changes
stop dispatch. The launcher labels the result as assumed No, not proven absence.

At this implementation checkpoint, no new live attempt had started; the later
bounded integration is recorded above. The previous
two-event three-star attempt `1c1e53a561e74d3c9fcc89e369e50987` is closed:
Ellmar completed and was collected; Ianu stopped before any planet answer or
save, and no third star was attempted. Its final offline replay reproduced
all 2,620 recorded events (2,622 emissions including replay envelopes), with
trace SHA256 `246561af1a12ae0321f6078ac49d9b5fc3a6c45490cebe7ed5a784220a6a2449`.
Do not relabel or resume that failure. Read-only reanalysis of its saved Ianu
crop and the earlier Bisperon crop now produces approximate No under the new
policy, while both older policies retain their original unknown-palette result.

The source-only check passed with 502 pinned files and no model loading,
training, evaluation rerun, browser or credentials:

```sh
.venv/bin/python scripts/browser_project_tui.py --profile configs/browser_project_baseline_band_three_star.json --mascot --check
```

Validation passed 454 combined Python checks (one native check excluded
from that pure batch), plus seven fully intercepted Chromium checks including
the new No/save/workflow/replay path and late deeper-feature rejection. The
separate runtime/launcher/source gate passed 242 checks. The intercepted tests
are synthetic transport evidence, not a fresh live acceptance or a learning
score. No frozen model evaluation, training or deadline increase occurred.
An additional 37 owner tests verify exact option forwarding and cancellation
before dispatch if in-memory options, on-disk scope or source files change.

Next is a separately identified bounded live integration when needed, not a
30-star launch. Positive/terrestrial live acceptance, the campaign timing concern,
and ungrounded successful submission acknowledgement remain separate readiness
limitations. The 30-star hold remains in effect.

### Explicit two-event reference checkpoint (2026-09-28)

The saved Bisperon failure is preserved. Its two overview pixel groups are
search hints, not confirmed dips. The new opt-in `project_two_event_reference`
allows an exactly-two-hint branch before any native action, but only if the
unchanged three-hint selector reports too few groups. Three or more valid
groups still use the original three-event path. A failed third probe can never
be relabelled as a successful two-event result.

Both selected groups must independently pass the existing six-zoom,
consecutive-day native-tooltip recipe, with a below-100% decline bracketed by
100% readings. Both must link to the current star's original overview, and
the plot must be restored after each. The two-event path has a 38-action cap
(two 16-action probes plus two three-action restorations); the 900-second
sensor cap, 180-second probe caps and 5,000-day window do not increase.

`approximate_reference_two_visible_tooltips_v1` has its own method hash and
explicit three-key descriptor. It reports two confirmed sampled features,
one spacing and zero repeated-spacing checks. Treating the features as
consecutive is an assumption; recurrence, missed events, aliasing, physical
period and physical minimum depth remain unverified. The TUI says
“two-event estimate / single interval” and “recurrence NOT confirmed.”
The two source bundles, current overview, spectrum and exact copied values
remain pinned through raw transport and downstream workflows. Legacy
descriptors, default behavior and version-1 replay remain supported.

The new bounded profile is `configs/browser_project_two_event_three_star.json`.
It keeps three stars, a 5,400-second campaign, 1,800 seconds and 512 advances
per star, paused start, and scoring/submission disabled. Existing profiles,
including the held 30-star profile, are unchanged. Enable the optional Sporky
overlay with `--mascot`; it supplies no policy input or answer authority.

Read-only source check (no browser, model, training or credentials):

```sh
.venv/bin/python scripts/browser_project_tui.py --profile configs/browser_project_two_event_three_star.json --mascot --check
```

An explicitly launched integration run uses the same command without `--check`.
It creates a fresh run under `experiments/browser-project-two-event-three-star`;
this is **not** the 30-star test. Synthetic native checks do not establish that
the saved Bisperon hints are transits, nor replace live positive/terrestrial
acceptance. No frozen evaluations or additional training are needed for this
reference-tool change. The campaign-time concern and ungrounded successful
submission acknowledgement below remain separate readiness limitations.

### Sporky presentation checkpoint (2026-09-28)

Optional `--mascot` on `scripts/browser_project_tui.py` enables the generated
transparent winged Sporky and short event-derived status bubbles. The original
user image is untouched; the project asset and generation prompt are in
`src/habfly/assets/sporky-fly-v1.png` and the adjacent Markdown provenance file.
This is a pointer-following browser decoration, not an OS cursor replacement,
hidden-thought display, policy input, or extra decision-making system. The
existing Rust TUI is unchanged; no server or spreadsheet access is added.

The live decoration begins only after initial-star setup, never during login.
Captions come from allowlisted raw version-1 events, with proposal/result labels
and explicit pause/stop states. Numeric values, credentials and observation
prose are never rendered. The outer-document inert, aria-hidden, closed-shadow
canvas cannot intercept clicks, change control order or become an extra chart.
All 15 production evidence screenshot calls use a shared temporary stylesheet
that hides only the mascot without painting a mask over the underlying pixels.
The display is opt-in/default-off and cannot increase budgets or authorize a run.

An offline design preview can be generated with
`python scripts/preview_mascot.py --output experiments/mascot-preview/index.html`.
Opening that HTML manually does not contact HabWorlds or start a model/test.
The preview's sample buttons are explicitly labelled as demonstration states;
live captions describe observed actions, not invented internal reasoning.
Validation: the final mascot/status/capture/two-hint gate passed 123 tests;
10 additional intercepted capture tests verified exact owned-host exclusion
across Page/Locator/ElementHandle and cross-origin frames. The separate runtime,
launcher, compact-wire and finalization regression gate passed 256 tests.
Current local source inspection still validates 498 files without loading a
model, running an evaluation, opening a browser or requesting credentials.
The manually openable preview for this checkpoint is
`experiments/mascot-preview-002/index.html`; Codex browser URL policy does not
allow opening a local `file:` page, so its visual preview is a user-open action,
not a reason to bypass that browser boundary. Actual renderer and capture
neutrality tests use intercepted synthetic browser fixtures instead.

The saved Bisperon run `736e75273b6c424d970dc3c17a95dbc2` remains a stopped,
zero-verified-star attempt. It contains two persistent subpixel hint groups, not
three confirmed transits. All four existing hint policies agree; neither the
No policy nor the positive detector supplies a justified planet decision. The
new portable regression verifies the three-event gate stops with zero native
actions instead of inventing a third hint or correcting the measurements.
The browser closed cleanly and offline replay preserved all 1,435 events
(1,437 emitted with replay envelopes), with original trace SHA256
`cc473ffd37e1796a9a60ede2a166fca7a2ae089e080b70194a8054f08ecf81a2` unchanged.
No additional live attempt, training, frozen evaluation, or 30-star run was
started for this visual checkpoint. Positive/terrestrial live acceptance remains
pending, and the 30-star test remains held.

Separate readiness concern: finalized three-star No-only runs took about
20–24 minutes from first class capture through final inventory. At the same
pace, 30 stars would exceed the held profile's current three-hour campaign cap.
No automatic deadline or profile increase has been applied. A future larger
cap requires an explicit decision and coordinated validation changes; successful
submission acknowledgement is also still ungrounded.

### Supplied-class integration checkpoint (2026-09-28)

The autonomous three-star pass remains a **three approximate-No workflow**
acceptance, not a planet-present acceptance. The separately versioned non-main
planet/temperature controllers now require exact visible stellar M/R provenance,
new frozen transfer gates, current-source checks and owned offline proof archives.
Actual class stays in tools/logs; only the explicitly hashed model view omits its
class feature. Main Sequence retains its original controller and final gate.

The two explicitly capped offline transfer evaluations are complete, once each:

- `experiments/planet-supplied-input-transfer-001`: 100/100 completed, 6,200/6,200
  exact actions, 25/25 per supplied class, zero invalid/tool/infrastructure errors.
  Report SHA256: `1bf7d395ecb287961b0c704332aee29d23dc21706e05d8616eabe9abe4e6f270`.
- `experiments/habitability-supplied-input-transfer-001`: 95/100 completed,
  2,795/6,958 exact actions (40.2%), zero invalid/tool/infrastructure errors.
  Five 128-action-cap failures remain preserved; no retry or budget increase.
  Report SHA256: `2ca9871692194776938dd3384001dae229f803219bc597f86d88c4515970c12b`.

Both used the real 2,000-node graph, one CPU thread, original unchanged weights,
zero optimizer updates, and no network/browser. Strict recorded-evidence readers
passed; original final-test cases/reports were not reused as new-scope authority.
The planet checkpoint predates existing temperature support in shared model code:
the new gate explicitly pins that already smoke-tested execution code separately
from original training-source provenance and records the exact delta. Confidence
is uncalibrated; neither scientific validity nor browser acceptance follows from
these synthetic supplied-input transfer scores. An earlier metadata preflight
stop happened before reservation/cases and is separately preserved.

`configs/browser_project_supplied_three_star.json` is the new bounded integration
profile: three stars, 512 advances/1,800 seconds per star, 5,400 seconds overall,
paused start, scoring/submission disabled. Its read-only source check passed with
497 pinned files. This check reads existing private transfer traces to verify
recorded proof but generates no cases, loads no model, starts no browser, and
grants no launch authority. Actual frozen wrapper loading also passed for both
tasks with inference/environment/expert/training/network paths forbidden.

Owned-archive validation passed against both completed transfer experiments:
`experiments/supplied-transfer-archive-check-001/report.json`. Historical readers
consumed all 233 planet and 230 temperature archive files with original external
checkpoint/case reads blocked. Reuse was idempotent, with no changed archive
hashes or original files and no evaluation, training, network or browser calls.

A guarded intercepted-Chromium fixture also completed the White Dwarf path:
three stellar fields, visible M/R receipt, four derived planet copies, exactly
one Save and strict collection import. Its transfer proof is explicitly
synthetic; this verifies native transport, not live scientific or learned-task
acceptance. Rust and installed-Python bridge checks passed (107 total).
The separate supplied-input terrestrial fixture also passed: native gas and
temperature controls, one Habitability Save, three-tab verification, strict
collection import and idempotency. Changed actual-class provenance was rejected
without changing the journal. Gate/model/chamber evidence remains explicitly
synthetic; the fixture cannot establish live planet acceptance.
The broader project/supplied-input/runtime/native-owner regression run also
passed: 1,759 tests in 17 minutes 42 seconds. This was fixture/offline validation,
not a real preview or 30-star campaign.
The subsequent compact-wire/TUI scope-label fix passed 114 affected Python and
108 Rust/installed-Python-bridge tests, retaining legacy absent-flag display.

Remaining gate: native planet-present/terrestrial workflow through calculation,
Save and strict collected-list import. The first bounded live run is preserved
at `experiments/browser-project-supplied-three-star/7d6840f4339d49ba8fe8324c9fde5e74`.
One resume completed Glus (Main Sequence, approximate 5,000-day No), including
Save and independent collection verification. Kanivermyr's stellar fields and
full overview completed, but the shallow hint reader stopped with
`shallow_probe_unsupported_low_intensity_hint_context`; no planet answer or
second Save was substituted. Result: **one verified star, not three**. Login and
the authenticated welcome-banner refresh passed. Scoring/submission stayed off.
Chromium closed cleanly, with no failed resources.

Offline replay preserved all 2,677 recorded payloads and the original failure,
with network, credentials, model/environment starts and subprocesses forbidden.
Trace SHA256: `97d47d0465bf13313c3d45659bc92be0e3d4f855d8297a8f8c77ce87693a0db7`.
The rejection came from quantized faint pixels whose fitted neutral values fell
just outside their observed background range. A separately versioned
`first_three_overview_quantized_blue_hints_v3` recipe now requires exact rational
feasibility of one shared alpha/background across all three 8-bit rounding bins.
It keeps the same raw candidates, first-three ordering, no-skipping rule,
original blue support and all native tooltip/zoom budgets. A hint remains only
a place to inspect, never a planet decision. Historical v2 metadata/results and
the frozen positive, negative and tooltip-measurement hashes are unchanged.

Validation: 252 focused tests, 191 cross-layer tests and 11 intercepted Chromium
tests passed, including rounded-pixel positive and flat-line rejection cases.
An independent linear-program oracle agreed on all 903 deterministic synthetic
RGB/background cases; independent review found no blocker. The v3 recipe SHA256
is `e22aa71a3e41e0813704c869de30c23ed667969fd764bb29d00956e6a75d2582`.
The failed live run and all frozen transfer evaluations remain unchanged. The
fresh capped attempt `5e1bbaded51a438c87e28f9960936996` stopped after one resume:
Diativos and Dand were independently verified approximate-No stars; Ferahir
stopped at `shallow_probe_three_overview_hints_required` before a planet answer
or Save. Result: **two verified stars, not three**. Scoring/submission remained
off and Chromium closed cleanly. Offline replay preserved all 3,845 recorded
payloads (3,847 emissions) with network, credentials, model/environment starts
and subprocesses forbidden. Trace SHA256:
`bd004a78b46a026f7584b96d5c0223ea9e6816960106b73fbf7df5fcc2d02e44`.
Read-only diagnosis found a continuous antialiased baseline edge within the
existing 1.5-pixel tolerance, not evidence of fewer than three physical transits.
The frozen v2 policy still rejects its unknown palette. A separately versioned,
explicitly opted-in compatibility path now recognizes only a continuous,
original-blue-supported baseline edge inside the same 1.5-pixel tolerance. It
requires uniform off-grid and tick-aligned edge colors, visible neutral samples,
one common quantized blend alpha, complete original trace coverage (allowing only
one dark final border column under the existing endpoint tolerance), and no
below-band blue/unknown pixels or bright overlays. PNG bytes are never repainted.
The failed run is not being rewritten or relabeled, and ambiguous curves stop.

The new policy is `user_approved_5000_day_baseline_edge_no_dip_v1`, SHA256
`d3013611379fab5aab8905f956d62cf8f19c75510602f91ed24908f0f158f470`.
Its analyzer passed 182 focused regressions and an independent 603-case linear
program check. Independent review passed 113 detector tests. The display-only
scope label passed 120 Python and 108 default Rust tests (the opt-in Python bridge
was not rerun for this label-only patch). Old stopped-run replay still preserves
all 3,845 payloads and its original hash. Transport validation passed 399 tests;
independent review passed all 41 new integration tests. Three intercepted native
fixtures passed: one No selection, one acknowledged Save, exact policy sidecars,
strict workflow/list import, and late-dip/overlay stops before a selection.
Their numeric/color prerequisites and tab switching remain explicitly modeled,
not live or learned acceptance. Fourteen affected legacy native regressions also
passed. The final read-only check verified 498 files without loading a model,
rerunning evaluation, requesting credentials, or launching a browser.

Fresh capped attempt `a11d7a17ba064f9791e1827a38f9830d` stopped after one resume
with **two verified stars, not three**: Adimrael (main sequence) and Jaha (white
dwarf), both approximate No and independently imported. Both finished plots
passed the original v2 raster conditions under the recorded new policy; neither
needed its baseline-edge exception. Belebron (main sequence) finished stellar
fields and its observation window, then stopped before any planet answer/Save
at `shallow_probe_unsupported_low_intensity_hint_context`. Its saved raster shows
repeated narrow downward features; diagnostic hint failure is not absence.
Scoring/submission remained disabled and Chromium closed cleanly. Offline replay
preserved 3,566 recorded payloads (3,568 emissions), all decisions and the stop,
without credentials, network, model/environment starts or subprocesses. Trace
SHA256: `f7f6998c1dd405a685435b066bc51d1700f401861fc7bcc1f762cfc2dc10afbf`.
Belebron's public crop SHA256 is
`e36d8c12feb0b706a3f5417dc769c266c3307ec7a308f9227c367111ed69eae2`.
Read-only diagnosis found one unsupported pixel among the first three selected
groups: `(66,22)` is locally quantization-compatible but has a v3-valid
antialiased parent at `(66,21)`, with original blue at `(66,20)`. The no-planet
compatibility branch correctly rejects the crop's deep original-blue features.
A separately versioned hint-only two-row support extension is implemented as
`first_three_overview_two_row_blue_hints_v4`, recipe SHA256
`ea6d5a1b9114fff504b7fa537b63db941b0fb6cc61bae250aa4e1a8ebb9c2db8`.
It uses the immutable v3 parent map, never recursively added support, and still
requires the current pixel's independent local RGB/gray fit. It does not change
grouping, skip malformed hints, or replace native tooltip confirmation. Validation
passed 95 pure tests, 272 cross-layer regressions, six intercepted Chromium cases,
and independent review (28 new plus 127 historical tests). The native successful
case confirmed three bracketed probes in 39 actions/54 advances; flat tooltips
and invalid first support chains remained blocked. Old recipes, frozen detectors,
measurements, weights, transfer evaluations and failed artifacts are unchanged.
The read-only launcher check passed with 498 files. Fresh capped attempt
`691ae41b5366465e8c259c524f099de4` completed **three independently verified
approximate-No workflows** after one resume: Dekeegwyn, Dias, and Emirmanden.
It stopped at `awaiting_assessment`, revision 18, with scoring/submission disabled;
task/project completion remains false. Chromium closed cleanly. Offline replay
preserved all 3,623 payloads (3,625 emissions), decisions, and terminal progress
with network, credentials, model/environment starts and subprocesses forbidden.
Trace SHA256: `ebf7fb273460dbd7b2a2ec668dbd208d6db053c829a40deed06c377255f95b69`.
This is not a positive-planet or terrestrial acceptance.

A separate code audit found an early-positive scheduling defect: a partial dip
could end collection and pin an incomplete PNG that the downstream measurement
reader correctly refuses. New scheduling rule
`complete_rendered_window_before_positive_handoff_v1` retains first-dip evidence
and continues read-only polls under the original deadline/poll/day limits until
the current rendered endpoint is ready. It never asserts physical observation
completion. A later contradictory endpoint stops instead of authorizing No or
substituting a positive answer. The final complete capture remains the sole
measurement source; the first-dip hashes are a veto/provenance record, not numeric
authority. Frozen analyzers, measurement rules, model weights and evaluations are
unchanged. Review also tightened typed readiness, immutable options/scope and
post-capture/callback deadlines. Validation passed 123 scheduler/compatibility
tests, 221 cross-layer/frozen regressions, and 223 other owner/runtime checks.
Two intercepted growing-chart fixtures passed: partial dip through full endpoint
and three measured dips, and full-window two-dip measurement refusal. The existing
native spectrum/positive-child handoff regression passed as well (its first
sandboxed launch was denied by macOS; the isolated rerun passed). Test-owned SVG
progression is not a real chart or learned acceptance. The optional compact-state/
TUI waiting message passed 139 Python and 109 default Rust tests (one opt-in Python
bridge ignored). The read-only launcher check again verified 498 files without
browser/model/evaluation execution. Independent review found no remaining scoped
blocker. Scheduler source SHA256:
`66ccb66592b98b4fdf5417e5ce6164d2a829a3ba0c938db6a31c6c386e7c1c66`.

The fresh post-fix run `99fa3bdcefb040138d381f4e437074d2` also completed **3/3
verified approximate-No workflows** after one resume: Elviv, Caidobrios, and
Ammdas (all supplied main sequence in this attempt). It stopped at
`awaiting_assessment`, revision 18, without scoring/submission or failures, and
Chromium closed cleanly. Offline replay preserved all 3,802 payloads (3,804
emissions) with the same forbidden network/model/environment/subprocess gates.
Trace SHA256: `720b908d90ef2a40fb915296ebe3afca39cbd89d5d1dfb35c81058f8c80e4626`.
No new live positive acceptance is claimed. Public names/positions are not fixed
physical identities across fresh sessions: this Ammdas's visible measurements
differ from the earlier red-giant Ammdas. No old measurements or planet answers
are reused, and no name-targeting/search/skip policy has been introduced.
The supplied three-star and separately held 30-star profiles opt in;
historical captures retain frozen v2 unless they explicitly record the exact new
policy. This is still an approximate No assumption, not proven absence, learned
perception, a training label, or completion.
The held 30-star profile now has the same four supplied-input/baseline settings.
Its 30-star/10,800-second campaign, 512-advance/1,800-second per-star limits,
paused start, scoring/submission settings, separate artifact directory and
explicit `--allow-thirty-star` requirement are unchanged. The 73 offline launch/
scope tests passed, including rejection before prompts/build/process without the
flag, and its read-only `--check` verified 498 files with no browser, model or
evaluation execution. Configuration SHA256:
`c3317a4b66949fed61230e16cd7e64e66e79200d2a4645aa4a1c52c735d166f1`.
This prepares configuration, not live acceptance; **do not run it**.

The next capped run `845d2c68cc1b4929ae5f52bc8f598983` verified Goril and
Amandritia on the approximate-No branch. Azryisil completed three native shallow
transit probes (42 actions), then stopped on `spectrum_tooltip_wrong_side` at
the first spectrum hover. The selected left marker was bound, exposed and
hovered, but its visible label was `656.30000132nm`. The right marker was never
sampled, so geometric inversion versus stale event delivery remains unresolved.
No third-star planet answer, Save, score, or submission was authorized. Chromium
closed cleanly. Offline replay preserved all 4,173 payloads (4,175 emissions)
and the same two-verified, stopped outcome with network, credentials, model/
environment starts and subprocesses forbidden. Trace SHA256:
`4653beb5b8ac38713939445c562502574095a7e296dd0a00aefe4f3c6b60f333`.
The original failure remains immutable; it is not a completed positive workflow.

The subsequent spectrum transport adds `visible_geometric_spectrum_pair_v1`.
It records exactly two left/right native hover events on one pinned, exposed
pair. Blue/red interpretation comes only from the two visible numeric labels,
which must be distinct, positive, straddle 656.3 nm and have exactly symmetric
excursions. A duplicated stale second label must settle without another move or
the run stops. The new branch uses 64-digit Decimal precision for its supported
label domain; it never averages, fabricates or repairs an endpoint. Its owned
evidence reader and all fresh/pre-copy checks explicitly retain the recipe.
Legacy absent-mode captures and their strict blue/red behavior remain unchanged.
The star controller opts in; the direct capture function retains its old default.
This avoids assuming screen-left means the blue wavelength, but does not prove
that assumption caused Azryisil's failure: its right label is still unobserved.
Validation: 42 new evidence/precision tests; 20 new intercepted native
capture tests; 43 legacy/native spectrum and positive-child checks; 263
cross-layer checks plus the two isolated native import chains; 262 replay/wire
regressions; 61 scheduling/held-launch checks; and 109 Rust tests (one opt-in
Python bridge ignored). The first sandboxed native imports were denied at
Chromium launch; both isolated escalated reruns passed. An exact-DOM-spacing
fixture encounters different headless hit-testing from Azryisil and correctly
rejects before dispatch; it is not claimed as a reproduced live success.
Two older readiness tests used invalid PNG sentinels and omitted current typed
readiness fields; their fixtures now use a real partial PNG, visible axes and
independently recomputed metadata, with no validator bypass. The held-profile
test now accounts for its four already-authorized opt-ins and still proves no
launch without the explicit flag. No production budget, graph, trained weight,
frozen detector or transfer evaluation was changed.
The final new native test exercises a reversed-label source through actual Yes
selection and all three raw copies without mocking the fresh spectrum reader.
All seven persisted source/fresh/preselect/precopy artifacts retain their raw
left/right labels and explicit recipe. This is synthetic transport validation,
not live Azryisil, learned or scientific acceptance. The read-only three-star
launcher check passed against 498 sources. Spectrum sensor SHA256:
`6bc16b6e77110c375cab6c41140f1de12df2a78b8eddc5b22925c74c9f8e5288`.

The first post-spectrum-change run `fb2f63be2b5043adb13ceb803a6df591`
completed **3/3 verified approximate-No workflows**: Gilthna, Kalis and Auroddess
(the last was supplied supergiant). One resume reached `awaiting_assessment`,
revision 18, with no failures, uncertain actions, scoring or submission.
Chromium closed cleanly. Offline replay preserved all 3,601 recorded payloads
(3,603 emissions), with source bytes unchanged and credentials/network/model/
environment starts/subprocesses forbidden. Trace SHA256:
`3e462329e89495c429ccf795e23134c8dd479e4ca6e594d2e351ac9e9422b575`.
This confirms continued No-branch transport, **not live spectrum or positive
planet acceptance**. No trained weights, source recipes or limits changed
during this run. The 30-star launch remains held.

### Autonomous-reference implementation (2026-09-26)

The new `configs/browser_project_autonomous_three_star.json` and updated held
30-star profile enable `project_autonomous_decisions`. The runtime now prepares,
records, revalidates, and applies the formerly supplied stellar-class/prefix,
planet-class, gas, and habitability choices in separate scheduled stages. Pause,
single-step, abort, current-star checks, immutable evidence, fixed budgets, and
the existing guarded native action owners remain in force. Unsupported or
ambiguous reference evidence stops with preserved diagnostics, never a default
answer. Legacy profiles continue requiring their explicit reference handoffs.

This is an **autonomous hybrid**, not a newly trained end-to-end classifier:
existing frozen connectome policies choose calculation/transport actions;
deterministic tools handle arithmetic; approximate reference rules choose the
new classifications. Stellar choices use the visible, digitized H-R diagram,
with a separately logged nearest-region heuristic for gaps and explicitly
uncalibrated outside-chart estimates (policy `nearest_diagram_region_v2`).
Planet classes use an explicitly approximate NASA Solar-System analogue policy,
not invented course cutoffs. Gas matching compares seven visible spectrum
templates; greenhouse strength then uses the actual selected-combination
absorption readback. The course habitability rule is terrestrial plus liquid
surface-water conditions, not a water-vapor or lifetime requirement.

The dedicated launcher is `scripts/browser_project_tui.py`. Its `--check`
validates local source hashes, the real 2,000-node graph, checkpoint metadata,
and reference packs without loading weights, prompting for credentials,
training, writing artifacts, or opening a browser. Source validation is not a
learned-performance or native-browser acceptance claim. Normal launch starts
paused; `r` resumes all automatic stages, `n` advances one stage, and `p`/space
pauses. A 30-star launch additionally requires `--allow-thirty-star`; no such
launch has been performed or authorized by this implementation request.

An isolated disposable-preview submission-shape diagnostic clicked Submit once
without analyzing 30 stars. The application visibly refused it with
`You need to analyze and submit at least 30 stars.` Evidence is under
`experiments/browser-submission-shape-006`. This grounds a refusal, **not** a
successful submission acknowledgement or project completion. Positive outcome
recognition remains unverified until the separately authorized 30-star test can
produce real visible evidence. No hidden simulation/application state was read.

The first native autonomous three-star check is preserved at
`experiments/browser-project-autonomous-three-star/61f291a6a146479aaef94ebd6010b534`.
After a single resume command, Bisperon completed its stellar calculations,
5,000-day approximate-No branch, Save, and independent collected-list
verification. Aquilix completed its stellar calculations and detected a shallow
transit, but its spectrum sensor recorded the same red-side wavelength for both
hovered markers and stopped with `spectrum_capture_failed`. The original
artifacts remain unchanged: **one verified star, not three**, no assessment,
score transfer, or submission. A wrong-side duplicated tooltip is not a valid
zero excursion or a value that may be repaired numerically.

Offline replay compared every recorded payload for both this failed autonomous
run (3,597 recorded events) and the earlier reference-assisted three-star run
(4,577 events), with network access, subprocess/environment starts, and model
loading forbidden. Both source hashes remained unchanged; the failed run
retained Aquilix, its automatic-decision provenance, and the original failure.
Current validation includes 192 automatic-runtime/options/launcher/source tests,
176 compact-wire/finalization regressions, and 107 Rust/installed-Python-bridge
checks. These software checks do not erase the native failure or establish
learned scientific accuracy.

The spectrum follow-up retains the original two-hover limit and arithmetic
symmetry checks. After each ordinary hover it waits read-only for at most two
seconds (inside the existing session deadline) for the same exposed tooltip to
stabilize on the requested side of the rest wavelength. Wrong-side, equal,
ambiguous, covered, changed, or indistinguishable marker evidence cannot become
a numeric answer. Failures preserve bounded public marker geometry and a safe
reason; no DOM listeners, application hooks, hidden wavelength data, retries, or
guessed endpoint corrections were introduced. All 64 focused numeric/spectrum
checks passed, including intercepted Chromium fractional-offset, delayed-tooltip,
adjacent-marker and positive-child handoff cases. This does not establish which
timing or pointer-delivery effect caused the historical Aquilix duplicate.

The production submission bindings were also exercised in a fresh disposable
zero-analyzed-star preview (`experiments/browser-submission-production-002`):
exactly one readiness check and one Submit returned normally and produced the
course refusal. Initial hidden frames are now pinned but never read as visible
lesson content; revealing/replacing a frame still stops. Its negative feedback
AX shape matches, but the native exposure proof remained unrecognized, so the
report correctly retained `unknown_pending`, not success or retry permission.

Follow-up `browser-submission-production-003` established that the refusal's
paragraph contains its own painted noninteractive span. All native buttons
already passed exposure checks; only that exact descendant text had been
mistaken for an obstruction. The narrow span-aware check passed eight
intercepted browser tests. In **`browser-submission-production-004`**, the
production controls and feedback reader then passed the real disposable-preview
check: readiness and Submit each dispatched once, `course_refusal` recognized,
no failure or retry, browser closed, all submission/task/project completion flags
false. This verifies the refusal path only, not a successful 30-star submission.

The next fresh run, `270d4ed0e0914060a7e91a68c89d58fa`, stopped before any
scientific answers on Egrithori because the v1 automatic reference policy
rejected outside-plot coordinates. Its temperature/luminosity project to
`[545.679926815259, 421.5371338870886]`, just 2.679926815259 pixels past the
right edge. The separate v2 heuristic ranks distances from the **actual**
log-projected point to unchanged diagram polygons; it never clamps the point,
extends a class region, or defaults to Main Sequence. For this case, the nearest
region is Main Sequence and the local lifetime prefix is Ta. Every such decision
records outside-plot distance, extrapolation, and uncalibrated limitations.
The original diagnostic classifier/pack still abstains outside its image;
neither it nor historical artifacts, datasets, or checkpoints were changed.
Invalid/nonfinite geometry, ties, overlapping regions, and source mismatches
still stop. The 325 combined offline stellar/reference/source/handoff tests
passed, including the saved Egrithori capture and different nearest-class
directions. This is a heuristic completion choice, not course-grade evidence.

Fresh v2 run `e3e4a09caf7b479aac074ab1f7bf688b` completed Kanzain's stellar
fields and the 5,000-day render, then stopped with
`shallow_probe_three_overview_hints_required`; **zero workflows were saved or
verified**. Both unchanged positive/negative detectors reported
`unknown_plot_palette`, so this was not an eligible approximate-No source.
There are seven faint, regularly spaced blue-direction pixels below the
baseline; the old hint-only palette selected just two. The pending sensor
follow-up concerns candidate search, not changing planet evidence or supplying
a period. All three failed autonomous attempts remain immutable and replayable.
Exact offline replay checks preserved Egrithori's 54 records (no decision was
produced) and Kanzain's 1,674 records (including the actual v2 decision), with
zero verified stars, original failures, no network/model/environment access,
and unchanged hashes. The final current-source compact-wire/submission/runtime
regression rerun passed all **312 tests** after adapting one old fake-frame
fixture to the existing visible-iframe contract.

Resumed validation (2026-09-28) passed **121** automatic handoff, authorization,
pause/abort, launcher and 30-star launch-boundary tests. Both profile source
checks still validate 36 local files without loading models or launching a
browser. An independent sequencing audit confirmed Save/inventory, both
assessments, score transfer and submission are wired; it also retained the
existing unsupported positive-planet branch for non-main-sequence stars. That
branch must not be described as arbitrary-star completion support. The new
low-intensity hint recipe passed **226** focused pure tests and **three**
intercepted Chromium cases. Bright and faint positive fixtures each used 39
native actions and 54 advances; faint pixels followed by flat visible tooltips
stopped without measurements or answer writes. Canonical recipe validation
rejects boolean/numeric substitutions and mixed versions. Raw candidate groups
are formed before contextual validation, so a broad invalid early feature
cannot be split into edge hints or skipped. Default/unversioned and explicit v1
evidence remain readable; neither frozen planet detector nor the required three
native bracketed declines changed. A separate planet/terrestrial regression gate
passed 266 pure tests plus two intercepted receipt-chain tests (rerun with
browser process permission after the sandbox denied Chromium startup).

Fresh bounded run `a5e97012eee1413c93ba11166307e541` used the existing three-star
profile, unchanged checkpoints, and scoring/submission disabled. It stopped
before any star with
`setup_login_submit_click_playwright_timeout_error_at_login_pointer_interception`.
The browser closed cleanly; no star/decision/project progress was produced.
Offline replay preserved all nine events (11 replay emissions), the null
pre-project state and failure, with credentials/network/model starts forbidden.
Its unchanged trace SHA is
`9065aa7282e996e514519f2d605115522a645c248900039b80d86a2e74ff8a56`.
The setup blocker was grounded by a fresh read-only login-page diagnostic: Sign
in was initially exposed, then the known cookie dialog intercepted its center
at approximately 1.07 seconds. No credentials, authentication-page captures,
form submission or raw driver logs were used by that diagnostic.

The fix adds a cooperative 1.5-second exposed-form settling stage within the
original 90-second setup deadline. Ordinary known-cookie dismissal resets that
stage; form, modal and exposure checks precede each credential write and the
single native Submit. Unknown covers, changed destinations and submit timeouts
still stop; there is no force click, Enter bypass or retry. **82** pure and
**61** fully intercepted native setup/refresh tests passed, including the late
cookie and unknown-overlay cases; independent review found no remaining scoped
issues. The new fixed-string `waiting_for_login_ui` event is carried by the
existing protocol; runtime regression passed 162 tests plus the three explicit
settling/refresh stage cases. Fresh bounded run
`9b91edc2fbd8485da5209cc671eb834f` nevertheless stopped during
`waiting_for_login_ui` with `setup_login_control_not_exposed`, before entering
credentials or producing a star decision. Chromium closed cleanly. A finer
read-only trace of the real login transition is required; fixture success did
not establish native setup readiness. No 30-star launch is claimed or
authorized.

The finer read-only trace found a fixed non-modal backdrop at 0.609 seconds,
followed by the visible known-cookie dialog at 0.771 seconds. It also exposed
the notice's disappearance animation: the former loop could try Close twice.
The follow-up waits at most three seconds read-only for pre-credential
non-modal cover, restarting the exposed-form interval, and pins the original
notice, Close element and enclosing dialogs through a one-shot dismissal.
Replacements, extra modals, persistent cover, post-credential unknown cover
and timeouts stop without retry. The original 90-second limit is unchanged.
**91** pure and **63** intercepted native setup/refresh tests passed. A fresh
actual-page setup-only check then reached `signing_in` and cancelled before
credentials; no authentication snapshots or submit requests were produced.

Fresh three-star attempt `c51d012cca1444ffa3120ee47709ce43` subsequently passed
real login, refresh, setup and Ammdas's automatic red-giant stellar calculations.
Ammdas, Jonyagolow and Aveleb all completed the approximate-No path, Save and
independent collected-list verification after a single resume command. The run
ended at `awaiting_assessment`, with **three collected, three verified, zero
unresolved**, journal revision 18 and no failure. Chromium closed cleanly. This
passes autonomous three-star campaign handling, not a planet-present branch or
full-project completion. All four checkpoint hashes remain unchanged.
Scoring/submission were disabled, and the 30-star hold remains in force.
Exact offline replay preserved all **4,288 recorded events** (4,290 replay
emissions) with network, model loading, subprocesses and environment starts
forbidden. Trace SHA:
`7c14fdc25b50619eeb8f8801e6b4beee49f5c6dd22e9f3141f07b04b9a0b07bb`.
The current Rust TUI/installed-Python bridge rerun also passed all 107 checks.

The separate offline class-bit diagnostic is preserved at
`experiments/planet-class-ablation-001/report.json` (SHA
`0f3912f42f6c891bfdb2a802a9dc4e0d673b738a19c09eea0031fb0f07b77e84`).
Sixteen frozen rollouts used four existing development cases, one CPU thread,
and zero optimizer updates. Only the policy observation's class changed; the
environment, calculator applicability, cases and checkpoint stayed unchanged.
Main Sequence completed 4/4 (248/248 exact actions); White Dwarf completed 0/4
(80/512 exact actions), with 22 tool errors and four step-limit stops. All four
white-dwarf rollouts first diverged at decision 13, selecting orbital radius
instead of executing the pending calculation. `red_giant` and `supergiant`
completed 4/4 each but both map to **unknown/all-zero** class features; that is
not evidence of recognized-class support. No browser, scientific acceptance,
native transport, or non-main applicability is asserted. The non-main scope
gate remains enabled pending an explicitly versioned extension and validation.

The additive supplied-input extension now has a separate pack and an explicit,
hashed inference adapter. All six original equations, constants, units,
assumptions and source conflicts are unchanged. Actual class remains in
authoritative observations and tool applicability checks; only the deep-copied
planet-policy input omits its class field. No checkpoint/model source or old
acceptance report changed, and confidence is explicitly uncalibrated.
`experiments/planet-supplied-inputs-smoke-001/report.json` records the single
capped 16-rollout development diagnostic: **16/16 local tasks**, **992/992 exact
actions**, zero invalid actions/tool errors, one CPU thread and zero optimizer
updates. The four class labels produced identical actions for each of four
reused development numeric cases. This is an input-adapter/contract check, not
new held-out learning, class-population, course, or browser acceptance. Root's
97-test supplied-input/legacy-pack rerun and independent review passed. Native
non-main Yes-panel availability and supplied mass/radius provenance remain
gates before browser integration; the new branch is not enabled.

The disposable UI-capability probes subsequently exposed all four empty derived
Planet fields with their expected units for White Dwarf (`Atir`,
`non-main-planet-capability-white-dwarf-002`), Red Giant (`Kjincaran`,
`non-main-planet-capability-red-giant-001`) and Supergiant (`Gwarn`,
`non-main-planet-capability-supergiant-001`). All three browsers closed cleanly.
Each probe made one deliberate class selection and one diagnostic Yes selection,
not an evidence-based planet decision; numeric writes, Save, assessments, score
transfer and submission were zero. Visible reconstruction inputs were M=1 Msun,
R=0.01 Rsun for White Dwarf and M=8 Msun, R=70 Rsun for both other probes. Their
physical correctness and default/origin behavior are **not** established, and
none is a workflow or training label. White-dwarf attempt 001 preserved its
captures but failed final reporting on a duplicate cleanup call; it remains
unchanged. The idempotent, sanitized cleanup fix passed 28 pure/native-fixture
checks before these complete probes.

The separate supplied-class temperature scope also passed its capped 16-case
development smoke with no updates, invalid actions or tool errors. It completed
16/16 but used **996 decisions, only 448 exact expert matches**, including
redundant valid selections/local copies. These are not duplicate native writes:
the existing temperature owner copies equilibrium temperature only once after
local proposals finish. Efficiency and broader transfer remain gates; neither
new scope reuses the original final-100 acceptance or claims calibration.
Default-preserving controller hooks passed 55 planet and 83 temperature tests,
including native cancellation/copy fixtures, plus 151 independent pure review
checks. Separate gate-bound subclasses and full non-main source-chain integration
remain pending. No new 100-case evaluation or 30-star run has started.

Status: 2026-09-25. The user authorized autonomous implementation/training/testing
through stellar, planet/exoplanet, and habitability work in the **local preview**,
including saving, assessment scoring, score transfer, and test submission. This
does not authorize changes to Torus, hidden-state inspection, external deployment,
or spending real money. A new preview browser resets for this user's test setup;
this is not a claim about learner-account resets.

## Verified in the visible local preview

These are assistant-operated, tool-assisted reference checks, **not learned-model
evaluation**. The existing frozen four-field reliability result remains separate.
No private star catalog, grading answers, application state, or simulation source
was read. Normal visible fields, controls, reference diagram, and feedback only.

- H-R reference is a raster chart with main-sequence, white-dwarf, giant, and
  supergiant regions. It provides no exact machine-readable boundary equations.
  The future classifier must not present guessed thresholds as course truth.
- Clicking the classification text alone did not select the native/custom circle.
  Clicking its visible circle changed the reconstruction and list-row class.
- Selecting Main Sequence reveals mass, radius and lifetime. Lifetime prefixes
  are exactly `ka`, `Ma`, `Ga`, `Ta` (10^3, 10^6, 10^9, 10^12 years).
- Data Quality -> ASSESS costs **100 simulation dollars**; the visible receipt is
  `DATA QUALITY UPDATED`. It reports stars, planets, habitability and overall
  percentages. It does not itself update the outer lesson score.
- Scavenger Hunt -> ASSESS separately costs **100 simulation dollars** and shows
  `SCAVENGER HUNT UPDATED`. Category text alone is not a checkmark; screenshots
  visibly confirmed Main Sequence and Red Giant after the second assessment.
- UPDATE SCORE produced `Score updated.` and changed the outer score from 0.00
  to **56.59**. Do not recompute this from rounded displayed percentages.
- Autosave displayed `Data saved`. Durable restoration after restart was not
  tested. No explicit Save action or final submission was performed.

### Reference cases and feedback

| Star | Parallax arcsec | Wavelength nm | Flux W/m2 | Selected class |
| --- | ---: | ---: | ---: | --- |
| IRPHAM | 0.044 | 969 | 5.40e-8 | Red Giant |
| EZIREK | 0.069 | 1403 | 9.11e-14 | Main Sequence |

Both classes were inferred from the public diagram and subsequently confirmed by
visible Scavenger Hunt feedback. Existing local arithmetic supplied numbers;
the learned models did not choose these actions.

IRPHAM: distance 74.0909090909091 ly, luminosity 871.1705565209776 Lsun,
temperature 2990.4731682146544 K, color IR. No mass/radius/lifetime was entered.

EZIREK: distance 47.246376811594196 ly, luminosity 0.0005976335604307458 Lsun,
temperature 2065.4087669280116 K, mass 0.11994479931725355 Msun,
radius 0.19277976372771763 Rsun, lifetime 2006995712068.1611 yr, color IR.
The lifetime field committed as **2.007 Ta**. Course input widgets visibly round
other numbers too; the local calculation's exact output is separate evidence.

The first Data Quality check (IRPHAM filled, EZIREK blank) reported stars **50.0%**,
overall **16.7%**. After both stellar reconstructions it reported stars **99.8%**,
planets **0.0%**, habitability **0.0%**, overall **33.3%**. Do not call this 100%.
Scavenger count progressed 0 -> 1 -> 2 of 8. Four assessments spent 400 simulation
dollars, leaving 49,600. Two collected stars do not satisfy the 30-star objective.

### Planet observation findings

EZIREK's planet screen displays the entered stellar mass and radius, a spectrum
centered at 656.3 nm (initial ticks 656.2997 and 656.3003), observation duration,
Play, zoom controls, and a normalized-flux chart. Requesting 10,000 days changes
the axis immediately but the plotted line fills over time. Axis extent is NOT
evidence that observation is complete. One zoom-in changes both axes. No transit
depth, orbital period, Doppler amplitude, or no-planet conclusion was established.
Planet and habitability answers were left blank, not guessed from a flat overview.

The worksheet's planet-mass formula references stellar radius; its physical
interpretation remains unresolved. Do not turn it into a training oracle. The
official project help confirms that small transit dips require zoom/pan/hover and
that long periods reach 10,000 days. A lack of obvious dips at overview scale is
not evidence of no planet. See https://kb.inspark.education/habworlds-project .

## Implemented and verified

- `project_assessment.py`: strict visible-text schema/parser and serializable
  spending ledger. Zero is distinct from missing, cost/mode changes fail closed,
  uncertain actions remain reserved across serialization, duplicate assessment
  of the same data revision is rejected. Default spend budget is zero. Verified
  before/after balance plus matching acknowledgement is required for a receipt.
  Assessment alone never claims Save, score transfer, submission or completion.
- `browser_assessment.py` and `habfly browser map-assessment CAPTURE_DIRECTORY`:
  read-only, offline mapping of hash-verified probe captures. No browser action
  is performed and an old acknowledgement is not represented as a new action.
- `lifetime_prefix.py`: explicit, exact decimal conversion to/from the selected
  prefix. No automatic unit choice, arithmetic-model replacement, or rounding.
- New unit/Chromium fixture tests. Fixture traffic is intercepted locally.

These are foundations, not an integrated autonomous full-project runner. The
assessment actuator must persist a reserved ledger **before** the click and
revalidate project revision, origin/path/frame, controls and modal state. It must
not retry an uncertain charge. Automation-unlock spending is intentionally not
supported by this initial guard.

The existing scientific knowledge pack, frozen numeric/color/lifetime checkpoint
bytes, four-field runner, runtime protocol, and TUI defaults are unchanged.

Validation at this checkpoint: 182 Python tests passed (including four new
network-intercepted Chromium checks), plus the five subsequent malformed-capture
cases passed in the final targeted rerun. The existing four-field `--check`
reports ready. Rust regressions and the installed-Python subprocess bridge were
also run. This was a targeted regression set, not a new full-suite or learned
full-project evaluation.

## Next assistant-owned stages

1. Capture stable classification selection/readback and learned H-R decisions;
   distinguish coarse public-chart reference regions from course-verified cases.
2. Bridge the promoted six-calculation model to the conditional main-sequence
   fields with explicit lifetime prefix transport and same-star identity checks.
3. Add the assessment actuator with fixture-tested reservation/receipt persistence,
   then repeat a bounded live stellar check without human stepping.
4. Build chart observation readiness, tooltip/zoom/pan sampling, planet formulas
   and classifications, then terrestrial atmosphere/water-phase/habitability.
5. Validate representative workflows, then a capped 30-star preview run with
   assessments, independent score transfer, and explicitly verified submission.

Do not reuse final numeric cases for tuning or call scripted discovery a learned
success. No new training run or optimizer update was performed in this checkpoint.

## Continuation: live six-field and assessment transport (2026-09-25)

New implementation (not yet an integrated full-project/TUI launcher):

- `browser_classification.py` resolves the observed custom class circle by its
  adjacent visible caption and verifies its painted `::after` dot. It never
  reads the hidden radio, data attributes, CSS classes, or application state.
  The caller supplies the choice; **classification is not learned yet**.
- `browser_stellar.py` has an opt-in six-field mapping path. Existing three-field
  callers still reject conditional fields. Prefixes remain explicit choices.
- `browser_full_stellar.py` transfers the frozen lifetime-003 policy to all six
  native numeric fields. It does not generate private answers. Exact decimal
  lifetime conversion transports the caller-selected prefix without rounding.
- `browser_full_color.py` runs the already-promoted color model on the classified
  six-field screen. Existing color/four-field defaults remain unchanged.
- `browser_assessment_actions.py` reserves and fsyncs a 100-simulation-dollar
  charge before clicking, then verifies the balance, acknowledgement, unchanged
  visible project rows and data revision. Uncertain actions cannot retry. It
  supports only Data Quality and Scavenger Hunt; no automation unlocks.

### Live evidence

Evidence root: `experiments/full-stellar-probe/20260925-002/`.
Fresh preview contained one star, **JYREMIS**, with parallax 0.052 arcsec,
wavelength 329 nm and flux 1.77e-9 W/m2. Main Sequence was an assistant-supplied
public H-R reference decision, not a learned prediction. Prefix Ga was explicit
deterministic transport, not a learned prefix selection.

1. `numeric-8`: preserved failure at 23 policy decisions, two writes. Autosave
   changed Save's enabled state during binding. No retry occurred inside that
   run. The diagnostic diff contains exactly the Save enabled/AX changes.
2. After the scoped fix, `numeric-11`: **60 learned decisions, six verified
   numeric copies**, 148.661 seconds, zero training updates. This reused the
   development star after the failed attempt, so it is **not a fresh acceptance
   or reliability episode**. Lifetime checkpoint bytes were unchanged.
3. `color-14`: the frozen color policy selected and verified UV in three learned
   decisions, one native write. No training updates.
4. `assessments-18`: two durable reservations and confirmed receipts. Data
   Quality reports **stars 99.9%, planets 0%, habitability 0%, overall 33.3%**.
   Scavenger Hunt progressed 0/8 to 1/8. Funding 50,000 -> 49,900 -> 49,800.
   Both custom OK acknowledgement controls were dismissed after verification.

Numeric results: distance 62.69230769230769 ly; luminosity 20.44472913544793
Lsun; temperature 8807.80699088146 K; mass 2.368382358473305 Msun; radius
1.960699731210828 Rsun; lifetime 1.1584317614494119 Ga. List view displayed
62.69, 20.44, 8808, 2.368, 1.961, and 1.158 Ga respectively.

**Artifact annotation:** the initial `numeric-11` lifetime receipt's `unit`
incorrectly says `yr` even though the exact copied number is in Ga; its
provenance and captured dropdown explicitly identify Ga. This reporting bug was
fixed after the run: new receipts record native unit `Ga` plus calculation unit
`yr` and the original tool value. Original evidence files were not rewritten.

Other observed behaviors now covered by narrow guards: choosing a prefix turns
a blank lifetime field into `0.000` (not an answer); the Save busy flag and paired
six-field autosave footer may change without another task edit; visible ASSESS
has accessible name `Assess`; receipt OK is a custom `BUTTON-POPUP`, not an AX
button. All unrelated fields/controls remain guarded.

Three event streams replayed offline and matched manifest hashes: 118 events
for the failed numeric run, 298 for the successful numeric run, 18 for color.
156 targeted Python regression tests passed, followed by the expanded color
fixture and 12 assessment-actuator fixtures. This is not a full Python-suite
claim. Rust regular tests: 38 library + 3 main passed; bridge is tested separately.

No explicit Save, outer score transfer, planet answers, or submission has been
performed in this new preview. The earlier two-star reference window remains
separate. Next assistant-owned work: planet chart observation, scientific
knowledge validation, learned classification, integrated bounded project runner,
then fresh end-to-end reliability and 30-star acceptance.

## Continuation: planet observation foundations (2026-09-25, in progress)

New independent modules: `planet_knowledge.py` and the physical-candidate pack,
`planet_charts.py`, `browser_planet.py`, `browser_planet_chart.py`, and
`browser_planet_spectrum.py`. Commands `knowledge inspect|validate --task planet`
and `browser map-planet CAPTURE` work offline. Stellar remains the default.

The candidate has six operations and six independent golden checks. It preserves
the original worksheet formula in provenance but explicitly uses stellar MASS
instead of the worksheet T2 radius dependency. It is **not course-validated and
cannot generate training references**. Its hash is
`f5acfb5e1b6ce0a4bcdc49176d20e939b0a2939635f259e4b9a82fdad66299da`.
The caller must explicitly supply the circular, edge-on, low-planet-mass and
semi-amplitude assumptions. Planet classes and habitability remain pending.

The reference preview is still JYREMIS in the owned Chromium process. After
10,000 days of observation, spectrum +/- controls change the displayed interval
by factors of ten. Two thin painted excursion markers expose ordinary hover
tooltips. Both full tooltip labels are readable in saved crop evidence:
`spectrum-marker-84-469.png` = `656.299999371nm`;
`spectrum-marker-84-529.png` = `656.300000629nm`.
Relative to `656.3nm`, the semi-amplitude is exactly `0.000000629nm`, not the
peak-to-peak difference. The reference diagnostic copied it into the Doppler
field, which committed as `6.29000e-7` with exact numeric equality. Has Planet was
set to Yes during reference exploration, revealing orbital radius (au), mass
(ME), radius (RE), and density (g/cm3). These and transit fields remain blank at
this checkpoint; no planet class or new assessment has been committed.

Important visibility correction: initial raw `inner_text` tooltip scans suggested
347 consecutive days and transits at 2794/2902/3010, depth 0.008%, period 108 days.
**Do not use those scans or `transit-analysis-79.json` as training evidence.**
HabWorlds retains tooltip text even when zoom/pan moves the plot point outside the
rendered viewport. The production guard detected this and now verifies all glyphs
and complete clipping bounds, not merely DOM presence or center visibility.
`guarded-chart-93.json` also predates the full-glyph rule. A new visibility-checked
scan is required. Partial or off-screen tooltips never establish completion.

`guarded-transit-96` hit its 900-second deadline and remains a failed diagnostic,
with partial events preserved. The revised transport batches only the read-only
safety checks to reduce browser round trips; it retains frame/URL, auth/modal,
answer, chart replacement, occlusion, action-count and deadline checks. The new
scan uses narrower overlapping panned windows so complete tooltips fit on-screen.
Do not infer a final period from sparse samples; the analyzer requires continuous
daily coverage and at least three complete, regularly spaced transits. It never
automatically emits a No Planet answer or a browser answer choice.

Validation: combined planet calculation/chart/field/spectrum, full stellar and
assessment suite: **139 tests passed**. An additional oversized-integer input
case and original stellar calculation regressions passed separately (70 tests).
No new training or optimizer updates have been run. Full-project functionality,
learned classification, learned planet policy, habitability, integrated TUI,
score transfer and final 30-star acceptance remain unfinished.

## Continuation: first course-verified planet reference (2026-09-25)

This supersedes the earlier blank-planet and pending-new-scan status, not the
failed or quarantined evidence. Same development star JYREMIS; no training updates.

- `guarded-transit-100`: 550 bounded hovers and nine pans yielded **344 consecutive
  daily samples (2918–3261)** with three complete transits at 3010, 3118 and 3226.
  Full-glyph clipping/occlusion checks were active. The visible drop is **0.008%**,
  period **108 days**. These are scripted reference measurements, not learned
  chart navigation. Diagnostic `kind/payload` logs are not v1 runtime replay files.
- `guarded-spectrum-101.json` independently reproduced both fully visible line
  excursion labels and a **0.000000629 nm semi-amplitude**.
- `planet-reference-copy-103` used the physical-candidate pack and displayed
  stellar mass 2.368 Msun/radius 1.961 Rsun. Verified native commits: orbital
  radius **0.5919 au**, mass **3.804 MEarth**, radius **1.912 REarth**, density
  **3.002 g/cm3**. Original full-precision values and rounding are retained.
- `planet-class-104`: Terrestrial was an assistant reference selection, not a
  learned classification rule. A comparison stopped because obsolete spectrum
  tooltip text disappeared from the duration label prefix. The mapper now
  normalizes only that prefix and verifies the duration's `days` label. All
  numeric values/units were unchanged. `receipt-reconciled.json` confirms the
  painted choice without repeating the click.
- The next bounded two-assessment stage reports **stars 99.9%, planets 99.9%,
  habitability 11.4%, overall 70.4%**. No habitability answers had been entered;
  that partial score must not be presented as completed habitability. Scavenger
  count rose **1/8 -> 2/8**; `frame-117.png` visibly confirms the Main Sequence
  and Terrestrial checkmarks. Funding **49,800 -> 49,700 -> 49,600**. Both charges
  and acknowledgements are confirmed; no uncertain retry, score transfer or
  submission occurred.

This supports the physical candidate on **one development case**, including its
stellar-mass dependency rather than the worksheet radius reference. It does not
validate all planet types, general classification boundaries, or a learned policy.
The pack's training-oracle gate remains disabled pending representative coverage.

New `browser_planet_numeric.py` accepts seven explicit, unit-checked numeric
copies on an already-selected Yes planet. It does not choose measurements,
calculate, classify, score, or submit. It reserves each write durably, verifies
exact fill and controlled native rounding, protects other fields and outside
frames, and permanently stops on uncertain writes. The comparison policy matches
all six saved live reference transitions offline; live execution of this new
adapter remains to be exercised on fresh blank fields.

## Continuation: terrestrial/habitability observation (2026-09-25)

New read-only `browser_habitability.py` and `browser map-habitability CAPTURE`
map albedo, pressure, entered temperature, selected gas labels, greenhouse choice,
surface-temperature readout, and water phase from hash-verified captures. No gas,
phase, or habitability answer is inferred. A blank gas menu is not proof of no
gas; painted defaults are not learned policy actions. `browser_classification.py`
now reads the same observed painted circles for habitability, including the live
whitespace-padded captions.

`habitability_knowledge.py` and `habitability_physical_candidate.json` add two
explicit-input calculations: radiative equilibrium and the supplied greenhouse
temperature increment. Separate from the synthetic ContentPack and stellar pack.
Hash: `95fec7129d126ed7b5866916538f7df16833fecdad8e4606d74f63015abe23f5`.
The physical candidate is based on full-surface blackbody energy balance, with
luminosity in Lsun, radius in au, and albedo as a fraction. Its assumptions and
constants are explicit; no generic gas or water-phase oracle exists. The course
greenhouse bands preserve even the published decimal gaps; gaps return errors
rather than rounding to a convenient category. Both CLI knowledge commands accept
`--task habitability`; training-reference generation remains disabled.

Visible JYREMIS development sequence (assistant reference, not learned):

- Albedo **0.05**, pressure **9 atm**, displayed luminosity **20.44 Lsun** and
  orbital radius **0.5919 au**. Equilibrium tool result 759.3787951199357 K copied
  exactly and committed as **759.4 K** in `equilibrium-reference-126`.
- Trace Gases is a custom checkbox overlay over a native select. Click attempts
  on the covered native select timed out without making a gas choice. The visible
  overlay reveals CH4, CO2, H2O, H2S, N2O, NH3, O3. Normal log-scale spectrum
  hovers expose wavelength only, not numeric flux. No SVG paths/data arrays read.
- O3 was an assistant candidate from the visible ~0.6 micrometer absorption
  feature and the public Chappuis-band reference. `gas-reference-133` records one
  checked O3 choice, **6.882%** reconstruction absorption. This is not a general
  spectral classifier or proof that all gases have been identified.
- `greenhouse-reference-139`: course Weak (+10) selection, native value `Weak`,
  verified surface readout **769.4 K**. The mapper preserves label/value mapping.
- The publicly linked Pressure-Temperature Chamber requires **Enter** to commit
  pressure. Tab left the attempted 9 atm at **0.380 atm**; `chamber-reference-136`
  and `-137` are invalid condition checks, not phase evidence. After Enter,
  `chamber-reference-138` verifies **9.000 atm, 769.4 K**, liquid/solid icons fully
  transparent, gas icon fully opaque and exposed. No task answers were modified
  by the help tool. Raw screenshots and readbacks preserved.
- `water-reference-140`: explicit Gas selection from that verified chamber check.
  Not Habitable was already the site's default; a later explicit reference click
  confirmed the same choice without altering other reconstruction values. No
  learned inference or successful whole-project completion is claimed.

Verification since the planet checkpoint: 43 intercepted Chromium tests for the
planet copy/chart/spectrum transports, 69 habitability knowledge/mapper tests,
44 existing environment/runtime and stellar-knowledge regressions, six painted
choice fixtures. These are targeted checks, not a full-suite result. No training
updates. Habitability course assessment and representative cases are still pending
at this paragraph's checkpoint.

Subsequent bounded assessment `habitability-assessment-149` (one 100-simulation-
dollar reservation, confirmed and dismissed) reports stars **99.9%**, planets
**99.9%**, habitability **73.7%**, overall **91.1%**. Funding is **49,500**. This is
a **failed habitability-validation gate**, not a successful full-system model.
The O3-only spectral candidate may be incomplete; the precise source of the lost
habitability credit has not yet been established. Do not use this reconstruction
as a correct training label or silently infer missing gas names from the score.
The earlier broad targeted command accidentally ran four Chromium fixtures in
the filesystem sandbox: 184 passed, four launch errors. Re-running with the
required browser permissions passed **all 188**; no test was weakened. Rust has
38 library + three main tests passing, plus the explicit installed-Python bridge
test passing. The owned preview remains open for further visible diagnostics.

### Resolved on the same development system: visible spectral comparison

Further ordinary chart zoom/pan exposed infrared dips absent from the O3-only
model. `gas-candidates-158` temporarily checked and restored each of six other
visible gas checkboxes (12 bounded writes, no grading calls). Candidate chart
crops, per-choice captures and restoration receipts are preserved. **Its initial
`baseline.png` is an incomplete redraw and is explicitly excluded by
`review.json`**; `spectrum-ir-157.png` is the reviewed earlier plot. No hidden
spectral arrays, SVG path coordinates, application state or private gas lists
were read. Exact chart paint readiness still needs a production guard.

Assistant visual comparison identified complementary CO2 and N2O features.
`gas-combined-159` selected **CO2 + N2O + O3** and the combined visible curve
matched the observed infrared features. Absorption became **46.88%**.
`greenhouse-revision-160` explicitly changed Weak to **Moderate (+30)** using the
published reference, producing **789.4 K**. `chamber-reference-161` verifies
**9.000 atm, 789.4 K, gas phase** with pressure committed by Enter. The Gas and
Not Habitable task choices remain appropriate to this verified condition.

The next single-charge assessment (revision 4) reports **stars 99.9%, planets
99.9%, habitability 99.9%, overall 99.9%**. Funding is **49,400** after six total
confirmed 100-simulation-dollar assessments across the entire development run.
This resolves the previous 73.7% habitability checkpoint on the same star. It is
**one course-verified reference system**, not a fresh learned evaluation, broad
physics/classification validation, saved score transfer, or 30-star completion.
The failed candidate remains preserved and is not a correct training label.

A complete 1,048-test Python regression run finished: **1,039 passed, nine
skipped, zero failures**, in 1,002.26 seconds under the 1,200-second cap. JUnit:
`experiments/full-project-regression-20260925-001/junit.xml`. The existing
collected test/code paths were held unchanged during the run. New standalone
water-chamber files added afterward were tested separately, not counted in that
full-suite result.

### Water-phase transport hardening

`browser_water_chamber.py` now provides a one-query, bounded helper adapter. The
caller supplies pressure/temperature and units; the adapter neither chooses
conditions nor writes task answers. It verifies visible Water material, exact
committed numeric readback (pressure Enter, temperature Tab), a fully opaque and
exposed stable phase indicator, unchanged task controls, pinned frame/dialog/input
identity, and at-most-once close. It saves reservations, observations, screenshots,
confirmations or failures without turning a failed read into a phase answer.

The first live adapter check, `guarded-chamber-167`, stopped **before entering
conditions** because a dialog container was incorrectly subjected to click-target
hit testing. The helper remained open because its handle had not been retained.
That failed attempt remains preserved; no task answer or assessment was changed.
The fix treats the unique visible dialog as a container and separately verifies
actual Close/input targets, retaining identity before subsequent checks.
**33 intercepted Chromium tests pass** including this geometry regression,
clamping, stale/replaced controls, wrong material, clipped/faint/multiple phase
icons, task mutation, and uncertain-close no-retry behavior. Live verification of
the corrected adapter is pending; this is not learned water-phase success.

The temporary interactive harness now has narrowly scoped owned-helper inspection
and close commands before the generic no-modal gate. Existing running Python
processes do not reload that main loop. The older JYREMIS browser remains intact
with its helper open; a separate `20260925-003` preview was opened for continued
Playwright testing. Native-window recovery was abandoned after a tool timeout;
**Mission Control and desktop-window automation are not HabFly requirements and
are not part of the supported workflow**.

The separate preview's corrected live check is
`experiments/full-stellar-probe/20260925-003/guarded-chamber-5/confirmed.json`:
9.000 atm, 789.4 K, fully exposed gas icon, all other phase icons transparent,
helper closed, complete task capture unchanged. The earlier `guarded-chamber-0`
and `-4` failures remain recorded. Live inspection showed that Close contains a
visible child span and the inputs appear before loading has completed; the
adapter now accepts descendant captions but rejects sibling occlusion, and waits
for both input controls to become exposed before writing. No Mission Control
interaction was needed for the working recovery or successful query.

The new preview contains **HOMOROM**, visible parallax 0.024 arcsec, peak
wavelength 348 nm, flux 2.51e-10 W/m2. Explicit unclassified-common calculations
give 135.83333333333331 ly, 13.610247413630969 Lsun, 8326.920977011494 K. Those
luminosity/temperature coordinates lie in the visible public H-R diagram's
main-sequence band (`reference-8.png`); this is an **assistant reference choice**,
not a learned classification. `reference-7.png` captured incomplete image loading
and is excluded as diagram evidence. No exact course boundary is inferred.

Final chamber-specific regression run: **35 passed in 53.78 seconds**. The
intermediate diagnostic-frame-detachment failure was corrected without weakening
the task-mutation assertions; screenshot diagnostics no longer replace the
original safety exception or read a detached/authentication frame. Ruff and
`git diff --check` pass. The full-suite count above remains the earlier
1,048-test inventory; the new 35-test chamber module is separate.

HOMOROM continuation: guarded class selection `class-selection-10` confirmed the
explicit reference choice. Frozen `numeric-12` completed **60 learned decisions,
six verified native copies**, 161.70 seconds, zero optimizer updates, unchanged
checkpoint. Native readbacks: 135.8 ly, 13.61 Lsun, 8327 K, 2.108 Msun,
1.790 Rsun, 1.549 Ga. The 298-event version-1 replay parses offline and its hash
matches the manifest (`eeeb584483377b475fae307a48fd74208bc6da90ccc63d199c76c968c8699ca4`).
Frozen `color-13` selected UV with a verified native copy; no course assessment
has yet been performed for HOMOROM. Neither successful transport result is a
full task completion or learned classification claim.

Planet capture parsing now rejects malformed frame/control inventories with
structured mapping failures instead of incidental Python attribute/key errors.
The combined planet/habitability mapping and knowledge set passes **141 tests**;
browser transport regression remains a separate live-process check.

### Planet tool-use simulator and bounded learning experiments

The new `planet_calculations` task is explicitly **independent physics with
supplied measurements**, not course acceptance. It exposes six operations and
four derived answer fields (orbit, mass, radius, density), visible reference
cards, shuffled controls and reference-star distractors. Inputs and intermediate
results are selected through the same interface by expert and learned policies;
incorrect bindings execute as selected. Private inverse-generated references
never enter observations. The course grading oracle remains disabled.

The scripted expert completes **100/100** deterministic cases in **62 actions**
each. Incorrect result lineage is counted as an incorrect binding, not silently
treated as belonging to the current star. There is a new, incompatible planet
observation/control encoding; old stellar encodings and replay remain supported.

`experiments/planet-calculation-smoke-001/report.json` records the actual real-graph
smoke: 2,000 nodes / 132,365 edges, four training cases, two epochs, 64 optimizer
updates, hidden size 16, one CPU thread, no network/Sheets or PPO. Losses were
**4.020611 → 3.178649**, reload verified, 242.54 seconds, process peak RSS
1,368,358,912 bytes. **Learned completion is 0/4 training and 0/2 development**, not
a passed learning gate. Exact held-out action imitation is 9.68%. The policy
repeats a source-selection action without choosing an operation; there are no
invalid actions or infrastructure/tool errors. Missing attempt metrics are null,
not 100% or zero-denominator successes.

`experiments/planet-calculation-sequence-001/report.json` is an explicit additional
**80-update** full-episode comparison (20 epochs, same four cases, no new cases).
The text encoder was frozen and exactly memoized; new optimizer moments and
parent hashes are recorded. Loss **2.989691 → 2.259372**, exact train/development
action imitation **20.97%**, but completion is still **0/4 and 0/2**. The 114.53-second
run's parent and frozen parameters are unchanged; reload passes. This negative
result is preserved. No final-test cases were generated or used.

The next separate experiment, `planet-calculation-transfer-001`, explicitly
transfers the already trained stellar v6 core and shared tool controls, not its
answers or demonstration navigation. Only genuinely shared mass/radius semantics,
units and status flags are remapped; unrelated planet feature columns start at
zero and must be learned. The exact column map, stellar parent hash and an
additional 80-update cap are persisted. **Its result is pending** at this entry.
This is not a silent checkpoint migration or a claim of learned planet behavior.

New commands (fresh output directories; these perform no HabWorlds actions):

```sh
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-calculation-smoke-NEW
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-calculation-sequence-NEW --sequence-from experiments/planet-calculation-smoke-001
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-calculation-transfer-NEW --sequence-from experiments/planet-calculation-smoke-001 --stellar-parent experiments/lifetime-003/training/checkpoint.pt
cargo run --offline --manifest-path tui/Cargo.toml -- --replay experiments/planet-calculation-smoke-001/expert-train/10000000.events.jsonl
```

The last command is **scripted-expert replay**, not a learned run. The local
planet observations display the task's pending course scope, units, bindings,
results, copies and tool errors in the Rust TUI without changing JSONL version 1.
A 190-event expert trace replays offline with exact task payload equality after
removing the runtime's explicit `replay` marker.

Regression verification: **145 Python tests passed in 69.07 seconds** across the
new task, recorded training, legacy stellar/local/Sheets, model gradients,
checkpoint compatibility, sequence training and runtime. Seven transfer/parent
validation tests pass separately (six overlap that earlier set). Rust: **39 library
and three binary tests pass**, and the ignored Python bridge was explicitly run
and passed before the interruption. Ruff and whitespace checks pass.

HOMOROM's last confirmed reference chart action is `20260925-003/chart-action-33`:
ordinary Playwright zoom produced a visible day 4200–4420 crop. No period or
transit depth has been verified for it, and no planet answers or assessment
charges were made. The tool interruption lost the old PTY handles and the exact
preview harness is no longer running. Existing unrelated browser processes were
not touched; no window-manager recovery, browser reset, score update or submission
was performed. Saved browser evidence remains intact. Do not claim the old
preview can still be controlled or that its attempt state was restored.

Transfer result: `planet-calculation-transfer-001` finished its 80 updates.
Held-out exact action imitation rose to **57.26%** (action kind 88.71%, target
69.35%), but **completion remains 0/4 training and 0/2 development**. The six
failed trajectories preserve 12 recoverable calculation errors, zero invalid
actions and zero infrastructure failures. They are failures, not correct expert
examples. The first actual divergence is choosing `orbital_radius` before
creating the prerequisite `period_years` result; the policy then binds an
incompatible measurement and attempts an incomplete calculation.

That failure motivated a separately identified **period-conversion prerequisite**,
not another four-field completion claim. `PlanetPeriodEnv` keeps all six tool
options and the same physical measurement cases, but requires only an explicit
days-to-years calculation, result copy and unit choice. The expert completes
100 deterministic cases in 10 actions each. The new scope and split/template
hashes prevent silent reuse as a full-planet checkpoint. `planet-period-001` is
an explicitly bounded 80-update diagnostic with unchanged underlying numeric
cases, private inverse-generated references, the same CPU/graph/hidden-size
budget, and no final-test cases. Its result was pending at this entry.

```sh
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-period-NEW --period-from experiments/planet-calculation-transfer-001
```

Period result: **4/4 training and 2/2 development tasks completed**, exactly 10
actions each, all selection/binding/copy/unit/numeric metrics correct, zero invalid
actions, tool errors or infrastructure failures. Loss **3.356345 → 0.009304** across
the fixed 80 updates. This is a learned small-development checkpoint, not the
100-unseen-case gate. The frozen stellar and failed four-field checkpoints remain
unchanged. Learned replay is available at
`experiments/planet-period-001/learned-development/12000000.events.jsonl`.

The explicit curriculum now has period (10 actions), orbit (18), orbit+mass (38),
orbit+mass+radius (50), and four derived fields (62). These lengths are verified
by **100 deterministic expert cases per stage**, not imposed on learned action
choices. All six operations and distractor measurements remain available.
Stage-specific required fields, instruction templates and content hashes are
separate; a checkpoint must name the immediately preceding scope. An orbital
stage is running from the learned period checkpoint, again capped at 80 updates
on the same four numeric training cases. No automatic multi-stage run or final
test is started by a command.

```sh
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-orbit-NEW --curriculum-from experiments/planet-period-001 --stage orbit
cargo run --offline --manifest-path tui/Cargo.toml -- --replay experiments/planet-period-001/learned-development/12000000.events.jsonl
```

The latter is now **learned period-only replay**, not a full planet or browser run.
Latest common Python regression before the curriculum addition: **148 passed in
66.21 seconds**. The curriculum's nine tests pass separately, including the five
100-case expert gates and unchanged original full/period observations.

Orbit result: `planet-orbit-001` completes **4/4 training and 2/2 development** in
18 actions, with zero errors and correct inputs, copies, units and numbers.

Mass extension: `planet-mass-001` reached **92.11% exact held-out imitation**,
but completes **0/4 and 0/2**. Its first divergence skips the orbital-radius unit
and moves on to radial velocity. A fresh-AdamW same-stage refinement,
`planet-mass-002`, also fails all six tasks, with a restart loss spike to 5.56 and
64 total recoverable tool errors. Both negative results remain preserved and
were not promoted to the radius stage.

Same-stage optimizer continuation is now explicit via `--resume-optimizer`
(requires `--refine`). It deep-copies the saved optimizer state, checks finite
moments and learning-rate consistency, preserves the original checkpoint, and
records the initial optimizer step separately from this run's update count.
Default training and new-stage initialization retain their previous behavior.
Regression tests verify step counters, parent-state immutability, nonfinite
rejection and mismatched-rate rejection.

`planet-mass-003` compares 80 resumed-moment updates from **mass-001**, not the
failed mass-002 branch. Loss **0.325821 → 0.126256** without the restart spike;
**4/4 training and 2/2 development tasks complete in 38 actions**, all bindings,
calculation choices, copies, units and numeric results correct, zero invalid
actions/tool/infrastructure failures. Optimizer step 80 is restored; this run
adds 80 updates. A radius extension is now running from this successful branch.
The original period/orbit models remain intact.

Advancement is now enforced: except for the deliberate initial period reset,
the previous stage must pass its **learned closed-loop** train/development gate.
Expert success and teacher-forced accuracy cannot bypass it. A same-stage
refinement is explicitly separate from advancement. New planet replay events
also name their confidence calibration scope and real neural activity source;
older stellar event behavior and version-1 parsing remain unchanged.

```sh
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-mass-REFINE --curriculum-from experiments/planet-mass-001 --stage mass --refine --resume-optimizer
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-radius-NEW --curriculum-from experiments/planet-mass-003 --stage radius
```

## Continued local planet learning after the service interruption

`planet-radius-001` and `planet-radius-002` both failed closed-loop completion.
The second first chose the radius unit control after copying the mass answer.
An explicit further 80-update resumed-optimizer refinement, `planet-radius-003`,
passes **4/4 train and 2/2 development cases in 50 actions**, with all selection,
binding, copy, unit and numeric metrics correct and zero errors. Its losses are
0.163200 → 0.030828. The unsuccessful branches remain preserved.

The new frozen diagnostic command tests whole recurrent expert trajectories,
not only option scoring with a fixed hidden state. It changes numeric payloads
and reverses control/option order without updating weights or repairing actions.
For radius-003, all 300 recorded decisions remain unchanged under both variants;
original exact action agreement is 100%. This is selection-invariance evidence,
not a new physical-case or closed-loop completion score.

`planet-full-001` adds density but fails: after 49 correct actions, all six
episodes repeat radius instead of selecting density. `planet-full-002` adds one
explicit 80-update resumed-optimizer refinement. It completes **4/4 train and
2/2 development in 62 actions**, with all metrics correct and zero errors.
Loss is 0.220010 → 0.034433. All 372 original decisions and their numeric/order
counterfactuals pass the frozen diagnostic. These remain small development gates.

```sh
.venv/bin/python scripts/diagnose_planet_calculations.py experiments/planet-full-002 experiments/planet-full-diagnostic-NEW
.venv/bin/python scripts/train_planet_calculations.py experiments/planet-pilot-NEW --pilot-from experiments/planet-full-002
```

The separate pilot is fixed at **64 train, 16 calibration, 16 development,
five epochs / 320 additional updates**, CPU one thread, hidden size 16, seed 0,
real 2,000-node biological graph, no PPO or network/Sheets calls. It preserves
same-scope optimizer moments, verifies parent case/trajectory hashes, collects
the 100-case scripted gate separately, and refuses a failed full-scope parent.
It does not automatically open final-test cases. `planet-pilot-001` was running
at this entry; inspect its final report before claiming a pilot result.

The final command is separate and requires a verified passing pilot:

```sh
.venv/bin/python scripts/evaluate_planet_calculations.py experiments/planet-pilot-001 experiments/planet-final-NEW
```

It reserves the fixed test set once at
`experiments/planet-final-test-v1.reservation.json` before generating cases,
freezes checkpoint/calibration, evaluates 100 cases, preserves every trajectory,
and cannot reuse that set under a new checkpoint as an unseen test. Failure or
interruption retains the reservation. The target is at least 90 completions and
zero invalid actions/tool/infrastructure failures. This is supplied-measurement
physics-tool competence only, never chart sensing, classification, habitability,
course assessment or 30-star acceptance.

`browser_planet_policy.py` adds a separately gated four-derived-field bridge.
Raw period/depth/shift must already be populated and positive in visible fields;
stellar mass/radius are read from visible reconstruction text. The class remains
explicitly supplied. No expected answers or simulator grades exist in this
adapter. It stops on policy/tool/units/transport errors, uses at-most-once native
copies, and reports transport separately from task completion. Browser confidence
is deliberately uncalibrated. It cannot save, assess, update scores or submit.
Only a matching successful frozen final report can enable its learned runner.
No live preview has been started by these local learning commands.

Pilot result: `planet-pilot-001` completes **64/64 training and 16/16 development**
tasks, exactly 62 actions each, all selection/binding/copy/unit/numeric metrics
1.0, and zero invalid actions, tool errors, infrastructure failures or API calls.
Five epoch losses are 0.031072, 0.009252, 0.005925, 0.004553, 0.002971. The saved
AdamW counter resumes at 160 and adds 320 updates; frozen-text/reload/parent
checks pass. Runtime is 529.47 seconds; process peak memory is 2,223,046,656 bytes.
Its parent hash preserves the full diagnostic lineage. The pilot metadata's
`inherited_stage_updates: 0` means no earlier pilot stage; actual restored optimizer
step 160 is recorded in the training report, not a random initialization claim.

The held-out **992 exact actions are all correct**. The calibration split is
another 16 whole episodes (992 decisions), with temperature scaling recorded
as `heldout_recurrent_expert_trajectories`. Its measured zero ECE is only that
limited offline scope, not browser-domain confidence. The old free-text
`answer_exact_accuracy: 0.0` field is an unused head, not the numerical-copy
metric; a clarifying note is now emitted in future planet training reports.

Current checks: 141 focused common/local Python regressions; 10 browser bridge
unit tests; 22 intercepted Chromium planet-copy tests, including dismiss/stop on
unexpected JS confirmations; one scripted native four-field Chromium fixture;
42 Rust tests plus the separately invoked installed-Python bridge. The latter
scripted fixture is transport verification, not learned/browser acceptance.

`planet-real-graph-smoke-001` verifies finite new planet observation outputs and
gradients at **2k, 5k, 10k and 30k nodes**, with 100 deterministic forwards at 2k
and two at each larger size. All models have the same 35,488 learned parameters;
there were zero optimizer updates. This is a random-initialized resource/contract
check, not evidence that a transferred larger graph solves tasks.

The separate frozen `planet-final-001` test was started only after the pilot
passed. It is pending at this entry; the reservation must not be deleted or its
cases reused for training if it fails.

Final result: **100/100 unseen tasks completed in 62 actions each**, all input,
calculation, copy, unit and numeric metrics 1.0, zero invalid actions/tool/API/
infrastructure failures. `planet-final-001/report.json` sets the local learning
gate true, but course acceptance false. It made **zero optimizer updates** and
did not refit calibration. The unchanged pilot checkpoint SHA-256 is
`36967a3261616aa1bbccb66727ae555b620b5473b4abe08f69d0ca873a3ad2a5`.
Final evaluation took 164.96 seconds, process peak 1,695,875,072 bytes. Final
seeds 13000000–13000099 are now consumed; do not tune on them or call a rerun
fresh unseen evidence. All 100 event streams remain available for offline replay.

```sh
cargo run --offline --manifest-path tui/Cargo.toml -- --replay experiments/planet-final-001/learned-test/13000000.events.jsonl
```

This replay shows the learned four-derived-field local policy, not chart sensing,
planet classification, habitability, live course scoring, or a 30-star completion.
The next assistant-owned gate is the exact frozen checkpoint through the local
Chromium fixture, followed by a bounded new owned Playwright preview. No window
manager or unrelated existing browser should be used for recovery.

The exact promoted planet checkpoint then passed
`planet-browser-fixture-001`: **62 learned decisions, four verified native
copies, zero optimizer updates, unchanged checkpoint**. This is intercepted
Chromium fixture traffic, not a live HabWorlds result. Scope metadata is retained
in `fixture-scope.json`; transport remains distinct from task completion.

A fresh owned Playwright preview, `full-stellar-probe/20260925-004`, automatically
signed in, traversed the intro and collected INDANGOLAC. Existing browsers remain
untouched. Its visible measurements are parallax 0.027 arcsec, peak 811 nm, flux
7.19e-13 W/m2, metallicity -0.21. Public H-R reference capture `reference-1.png`
supports an **assistant reference**, not learned, main-sequence choice at about
3573 K and 0.03080 solar luminosity. `class-selection-2` verified the choice;
the explicit lifetime prefix is Ga. The frozen six-field stellar inference is
running at this entry. No live planet measurements, assessments, save, score
transfer or submission have been performed in this new attempt yet.

INDANGOLAC update: `numeric-4` verifies all six stellar copies in **60 frozen
model decisions**, 101.47 seconds, zero updates. Visible readbacks are 120.7 ly,
0.03080 Lsun, 3573 K, 0.3700 Msun, 0.4625 Rsun and 120.1 Ga; exact values and
rounding receipts remain in its manifest. `color-5` independently verifies IR.
One 10,000-day observation was requested after opening Planets; its displayed
axis is not proof that the animation has filled the interval.

Initial spectrum sampling could not identify both rendered excursion markers.
A later visible-layout read confirmed two one-pixel markers; `guarded-spectrum-12`
then verified 656.2999877 nm and 656.3000123 nm around 656.3 nm: semi-amplitude
**0.0000123 nm**, not the 0.0000246 nm peak-to-peak difference.

`browser_transit_sampling.py` implements a capped **scripted evidence sampler**,
not learned chart perception. It uses guarded zoom/pan/hover, records every
sample, rejects daily-coverage gaps, and requires three complete transits before
reporting a period. It never chooses Has Planet or writes an answer. All-flat
coverage remains inconclusive. Budgets: 2,048 actions, 900 seconds, 32 windows,
12 zooms. Partial/failed scans remain separate artifacts.

Live `transit-sampling-13` stopped on a clipped/missing tooltip after zoom; no
period or answer was substituted. `flux-axis-15` independently read fully
exposed y-axis glyphs and placed the 100% baseline at y=19.96 of 195 pixels.
`transit-sampling-16` tried to center it but stopped when the resulting position
was not verified. `flux-axis-17` shows y=30.45, only about 10.5 pixels of the
requested 60.0-pixel movement. The transport now paces its ten native drag
segments at 25 ms with a context guard between segments; this hypothesis is
being tested live. The centering validation has not been relaxed.

New chart tests: four pure sampler/axis checks, one intercepted native
zoom/hover/pan sampler, and all existing chart/spectrum guards. Latest combined
fixture pass before pacing: 30 tests; after pacing, 26 chart/sampler tests pass.
No planet answer, assessment charge, Save, UPDATE SCORE or submission has yet
been performed in the INDANGOLAC attempt.

The pacing hypothesis did not help: `transit-sampling-18` and `flux-axis-19`
still place the baseline at y=30.45. The original native drag implementation was
restored. A separately guarded hover, `chart-action-21`, then read day 239 at
100% with every tooltip glyph exposed. Exact y=80 centering was unnecessarily
restrictive: positioning now accepts a verified y=30–120 band, while retaining
the strict full-glyph/occlusion check on every actual tooltip. An unreadable
tooltip still stops the run. `transit-sampling-22` is a new bounded scan; flat
coverage remains inconclusive, never a No Planet answer.

Added `browser_planet_evidence.py` to transfer three raw measurements only from
matching-star, independently rechecked visible spectrum and continuous transit
evidence. It rejects inconsistent/tampered evidence and already populated raw
fields, persists source hashes, and uses the existing at-most-once native copy
transport. This is explicitly **scripted evidence transfer**, not learned
measurement selection. It does not fill the four derived fields; those belong
to the frozen promoted planet policy. Eight pure evidence tests and three
intercepted Chromium evidence/sampler tests pass. No live raw copy has occurred
at this entry.

`transit-sampling-22` reached its 900-second deadline after **731 consecutive
days, 429–1159**, all at visible 100%. No planet answer was inferred. Its original
failure and journals are immutable. Explicit read-only recovery verifies every
sample against confirmed HOVER result events and the same-star scope before
writing `recovered-report.json`; it cannot recover unknown visibility failures
or silently perform browser actions. New scans save immutable per-window
checkpoints and can explicitly continue a same-star, fully rechecked report.
Gaps, conflicting values and inconsistent status claims are rejected.

Profiles `chart-reader-profile-24.json` and `-25.json` identified repeated
Locator resolution as a significant cost. Cached, still-connected document
handles and one batched full-glyph tooltip read reduce live hover latency from
about **0.807 to 0.198 seconds** (three samples each). This is a diagnostic
timing comparison, not a broad benchmark. Body replacement, frame/navigation,
auth/modals, altered answers, stale charts, clipping and occlusion still stop
the reader. Thirty-one chart/spectrum/native-sampler fixtures passed, followed
by two focused replacement/native-sampling checks; 21 pure sampler/evidence
tests pass. The accidental top-level copy of the new sampler was removed; the
actual implementation remains `src/habfly/browser_transit_sampling.py`.

`transit-sampling-26` continues the saved daily evidence with the faster reader.
It is pending here. No INDANGOLAC planet answers, classification, assessments,
Save, UPDATE SCORE or submission have been performed.

The broad Python regression run completed: **1,161 passed, 10 skipped** in
857.97 seconds. It began before the latest chart/runtime edits; those changes
have separate focused coverage below. Rust remains **42 passed**, plus the
installed-Python bridge checked separately.

Added the offline `planet_calculations` runtime task, a strict promoted-pilot/
matching-final gate, and `scripts/planet_tui.py`. This uses only recorded
development seeds 12000000–12000015; it never opens the sealed final cases or
labels manual reuse as unseen evaluation. It starts paused and preserves v1
step/pause/resume/abort/replay semantics and real neural telemetry. The actual
promoted checkpoint passed `planet-runtime-001`: **62 decisions, completed local
four-field task, unchanged checkpoint, zero updates, and verified offline
replay**, with Python network connections and Sheets initialization denied.
The preceding mock protocol test initially had an incorrect test assertion
(`protocol_version` instead of the event's `version`); corrected focused tests
now pass (36 passed, one opt-in skipped). The real-checkpoint case itself passed.

```sh
.venv/bin/python scripts/planet_tui.py --check
.venv/bin/python scripts/planet_tui.py
```

This is an available local manual demo, not full-project readiness. In the TUI,
`n` steps, space resumes/pauses, `a` aborts and `q` exits. No credentials, browser,
training, class inference or habitability inference are involved.

Live `transit-sampling-26` stopped near day 1903 on the full-glyph visibility
guard. `chart-27.png` shows a dip extending below the highly zoomed vertical
view. Ordinary zoom-out action 28 exposes it; **guarded hover 29** confirms
**98.272% on day 1903**, a candidate drop of 1.728%. This is not yet a period.
Zoom 30 selects an intermediate scale that keeps this deeper point readable
while retaining approximately daily horizontal resolution. New scan 31 resumes
from immutable `transit-sampling-26/window-22.json` (coverage 429–1892), not from
the failed or hidden tooltip. A native fractional-pointer fixture confirms that
subpixel coordinates cannot simply be assumed to supply missing daily samples.

`transit-sampling-31` completed its 32-window budget with **2,997 consecutive
days, 429–3425**, one complete one-day dip at 1903 and depth 1.728%. Period is
still null. Scan 32 explicitly continues from that same-star report. Neither
the observed drop nor a hypothetical recurrence was substituted for a measured
period, and no planet answers have been written.

New `browser_habitability_numeric.py` supports one explicit equilibrium-
temperature copy before greenhouse selection. It validates K units, initial
blank/default state, current target identity, exact fill, controlled rounding,
the dependent surface-temperature readback and unchanged surrounding choices.
No calculation, gas selection, phase, classification, assessment, Save or
submission occurs inside this adapter. Reservations and uncertain outcomes are
immutable, with no retry. Twenty intercepted native fixtures passed initially;
an additional Save-busy-state test covers the narrowly permitted footer change.
Its projection also matches both hash-verified captures of the prior JYREMIS
equilibrium reference transition. This new adapter has not yet been used live.

The latest post-runtime/scan focused Python checks pass **59 tests**, with one
opt-in real checkpoint skipped (that case passed separately and saved
`planet-runtime-001`). Combined chart/spectrum/sampler/evidence-copy fixtures
pass **34 tests**. The installed-Python Rust bridge passes again. Whole-repository
lint reports four pre-existing issues in `scripts/build_neuron_table.py` and
`scripts/inspect_connectome.py`; those unrelated files were not changed. Scoped
lint and `git diff --check` pass.

## INDANGOLAC measured transit gate and guarded transport continuation

`transit-sampling-32` retained 4,486 consecutive days, 429–4914, and two
complete dips (1903, 3930). Scan 33 explicitly continued those same-star
records and finished after 23 windows: **5,543 consecutive days, 429–5971**,
three complete one-day transits at **1903, 3930, 5957**, measured period
**2027 days**, and depth **1.728%**. Spectrum record 12 independently
supplies **0.0000123 nm** semi-amplitude. The raw evidence validator recomputed
all three measurements successfully. This is scripted student-visible
measurement acquisition, not learned chart perception or a completed project.

Added an at-most-once `browser_planet_presence.py` selector requiring those
same-star spectrum/transit records before a caller requests Yes. It never
infers absence from a flat interval. Eleven intercepted fixtures verify its
expected blank conditional panel, existing-answer protection, modal/frame
guards, unchanged axes and uncertain-write retention. The read guard cannot
copy numeric answers before Yes. No class or numeric answer is selected by
this presence transport.

Ten planet-class transport fixtures pass. A newly reproduced false stop came
from a chart tooltip also being appended to the SVG's accessible name. The
comparison now removes only that known transient suffix; axis changes still
fail. This projection is not a chart measurement source. The initial failing
fixtures and subsequent passing reruns were retained in temporary test output.
Habitability's one-temperature transport has 21 passing intercepted fixtures;
neither temperature nor class transport has yet established learned scientific
classification. No final-test training or additional optimizer updates occurred.

Live presence step 35 and evidence transfer 36 passed. Has Planet is Yes and
the three measured inputs are copied with verified native readbacks. Numeric
reader profile 37 measured roughly 6.5-second initialization / 7.6-second full
current-state validation; the inference budget remains capped at 900 seconds.

`planet-learned-38` preserved a genuine transport failure after 35 completed
model decisions: orbital radius **2.251340396909765 au** displayed as **2.251**
was verified, then the pending mass copy **57.354917176709556 MEarth** displayed
as **57.35** changed the dependent reconstruction `ORBIT (years)` from **5.552**
to **5.551**. The original capture comparison rejected that dependency. No
other changed field/axis/widget was found; the checkpoint is unchanged and no
optimizer updates occurred. The original failed report is retained, not fixed
up into a passing run.

The comparison now permits the specific mass-dependent orbit readout as well
as the existing orbital-radius dependency. It requires a finite nonnegative
readout, a positive changed result, and no increased positive period after
adding positive mass at fixed orbit. Other fields remain protected. Twenty-six
numeric fixtures pass. This is UI-dependency handling, not a course formula or
grading oracle.

Added explicit frozen-prefix recovery for this one failure family. It verifies
source capture/event hashes, current native readbacks and unchanged surrounding
state, then reconstructs the identical original learned action prefix offline.
Both existing numeric copies become recorded replay receipts with **no repeated
native writes**. Live state is rechecked before continuing into untouched
fields. Two native/pure recovery checks pass, including tampered events, changed
measurements, checkpoint mismatch and attempted repeated continuation. No live
recovery result is claimed here yet.

Also added a one-click planet Save adapter with 10 passing intercepted fixtures.
Its receipt explicitly distinguishes a visible Data saved banner from verified
cross-session persistence, scoring and submission. The broader affected-module
regression passed **119 tests, 1 opt-in skipped** before the later recovery
change. Frozen numeric and color policies passed eight intercepted fixtures
covering all four **supplied** stellar classes; class selection is still not
learned. Neither fixture result establishes real-course classification accuracy.

Live recovery **`planet-learned-40` passed**: 62 identical frozen policy
decisions in 282.73 seconds, original orbit/mass readbacks reconciled without
rewrites, and only radius/density copied natively. Exact values and displayed
readbacks are:

| Quantity | Exact learned-tool result | Browser display |
| --- | --- | --- |
| Orbital radius (au) | 2.251340396909765 | 2.251 |
| Mass (MEarth) | 57.354917176709556 | 57.35 |
| Radius (REarth) | 6.626895223255005 | 6.627 |
| Density (g/cm3) | 1.086695251938891 | 1.087 |

No optimizer updates; checkpoint unchanged. Trace SHA-256:
`966a39fba460df08bd66b1dd9d62eb1297042ae56426b7c27d816011c7f7cde2`.
The failed 185-event trace and successful 319-event trace both passed offline
replay with network and Sheets initialization denied; neither claims task or
project completion. TUI timelines now label reconciled receipts as **no browser
write**. Latest checks: **69 affected Python tests**, **43 Rust tests**, and the
installed-Python subprocess bridge pass. `scripts/planet_tui.py --check` reports
ready, still local supplied-measurement development reuse only.

Probe command 39 failed during a temporary harness module refresh before any
browser operation or run artifact was created. Correcting module reload order
allowed command 40 to perform the above explicit recovery. This import failure
did not trigger a repeated native answer copy.

Class step 41 verified the painted **Gas Giant** selection, explicitly marked
assistant/reference rather than learned. Its rationale is recorded separately
in `planet-class-reference-source-41.json`: NASA GSFC's published Jovian size
category includes the measured 6.627 Earth radii, but it is not represented as
HabWorlds' own verified classification boundary. Course assessment is pending;
no knowledge-pack hashes or training labels were changed by this reference
decision. Save step 42 is being attempted separately with its tested adapter.

Save attempts 42 and 45 stopped **before clicking**. Visible layout captures
44/46 showed a transparent loading span inside the otherwise visible native
Save button intercepting center hit tests. The narrowed native-button check
accepts its own descendants, not sibling overlays or an invisible parent.
Fourteen fixtures passed after that change. Save 47 then clicked **once** but
timed out waiting for an acknowledgement: its reservation remains uncertain,
and it must not be repeated. List 48 independently showed all entered values
and Gas Giant intact; this is not cross-session persistence evidence.

The Save adapter now observes a brief, fully exposed Data saved notice in the
same footer before doing slower full-screen validation, and keeps the last
verified capture on failure. Fifteen fixtures pass, including a notice that
disappears during full validation. This later fix was **not** used to retry
uncertain live Save 47.

**INDANGOLAC assessment 50 is mixed, not a passing full workflow:** confirmed
100 simulation dollars spent (funding 49900), stars **99.9%**, planets **88.8%**,
habitability **0.0%**, overall **62.9%**. Planet readback success therefore did
not prove all scientific/classification choices correct. Habitability fields
have not been completed. No score transfer or submission occurred. The original
reference gas-giant inference is not promoted into a training label or rule.
Scavenger feedback and habitability inspection remain the next diagnostic steps.

## Continued after the service interruption: isolated classification error

The owned browser remained alive. Assessment 53 confirmed only one scavenger
category; screenshot 55 visibly showed Main Sequence checked, not Gas Giant.
Habitability navigation 57 showed only the terrestrial prerequisite, with no
editable habitability controls. No hidden fields were accessed.

Added explicit classification revisions to `browser_planet_classification.py`:
the caller must name the expected existing class and a diagnostic reason;
ordinary initial selection still refuses overwrites. One native click swaps only
the two expected painted circles. Numeric values, other controls and the third
circle remain protected. Eighteen intercepted classification tests pass.

Reference diagnostic 62 changed only **Gas Giant -> Ice Giant**. Data Quality
assessment 65 then reported stars **99.9%**, planets **99.9%**, habitability
**0.0%**, overall **66.6%**. This isolates the prior planet deficit to the
reference classification; the learned four derived answers were not changed.
Scavenger assessment 70 confirmed **2/8**; screenshot 72 visibly checks Main
Sequence and Ice Giant. Funding is **49600**, four confirmed 100-simulation-dollar
charges in this preview. Attempt 68 stopped before reserving/clicking because
its observation changed; inspection 69 established unchanged rows and funding
before the separate bounded attempt 70. Uncertain Save 47 was never repeated.

This is one course-verified Ice Giant development example, not an inferred
universal size/density threshold or a new training label. Habitability remains
unavailable for the selected non-terrestrial planet and its assessment score is
still zero; no full-project completion is claimed.

The planet bridge now uses its pinned visible measurements for purely local
calculator actions. Every native answer copy still performs full pre/post live
validation, and final transport completion requires another full validation.
Changed measurement, modal, escaped page, replaced target and final-state
fixtures all stop before an unsafe write/completion. Eighteen affected tests
pass, with the opt-in trained fixture run separately.

`experiments/planet-browser-fixture-002` passed the unchanged frozen 2,000-node
checkpoint: **62 decisions, four native copies, zero optimizer updates**, 4.89 s
versus the earlier fixture's 17.51 s (separate runs, not a controlled benchmark).
This is intercepted fixture transport, not a new unseen evaluation or live
course-completion result. No graph, pack, dataset or checkpoint was changed.

Added `browser_score_transfer.py` with an explicit opt-in, hash-checked receipts
for both assessment kinds at one revision, matching visible rows, one durable
source-revision reservation, unchanged unchecked submission box, and a required
fresh visible acknowledgement. Its source-revision claim prevents uncertain
transfers from being retried through a different output directory/process.
Seventeen intercepted fixtures pass, including changed data, missing/hidden
acknowledgement, checked submission, duplicate controls, tampered evidence and
read-only reconciliation.

Live transfer 73 clicked UPDATE SCORE once and displayed **Score updated.**;
its initial conservative comparison stopped because Torus adds a known feedback
block (OK, Open/Close Feedback, and close icon). Saved before/after captures
showed only those added controls and consequent observation-local renumbering.
The adapter now permits exactly that observed block, preserving comparison of
every other control and all simulation rows/widgets. Reconciliation 77 verified
the original acknowledgement, source hashes, live unchanged project and the
outer score **0.00 -> 108.26**, with **zero further browser actions**. The
original stopped transfer and its reservation are preserved. Score transfer is
confirmed; explicit Save persistence and submission remain unverified.

Combined affected regression before the final feedback-block patch: **134
Python tests passed, 1 opt-in skipped**. The final score-transfer suite adds the
three feedback/reconciliation tests above. Rust: **43 passed, 1 ignored**.
`scripts/planet_tui.py --check` remains ready for its offline, supplied-measurement
development example; it is not the complete browser workflow.

## Second collected star and stale native class paint

The next-star picker selected AMAS once (80), excluding INDANGOLAC's prior
visible point. Its retained Planet tab used an unnamed image with visible `1`,
not an accessible image name. The initial transition stopped; the explicit
continuation 85 clicked only the Stellar tab, never the star or View Star Data
again. AMAS has parallax 0.114 arcseconds, wavelength 1709 nm, and flux
6.03E-14 W/m2. Its numeric answers and color are blank.

Both native classification widgets retained previous-star paint: Ice Giant on
Planet and Main Sequence on Stellar. This paint is recorded as **unverified**.
One explicit Main Sequence click (87) produced no normalized screen change and
failed the conditional-field guard; it is not a successful answer receipt.
The original capture and its one-attempt reservation remain intact. List 89
independently shows **two collected stars**, AMAS's classification and eight
stellar answer columns blank, and INDANGOLAC unchanged. Thus prior assessment
65/70 and transferred score 73/77 no longer describe the current collection.

Added `browser_collected.py`: an exact name-scoped, blank-row, visible eye-icon
navigation adapter. Eight intercepted fixtures cover multiple rows, hidden/
obscured controls, changed identity/measurements, outer-state changes, and stale
paint. The live SVG eye receives clicks through its own painted child, handled
without reading SVG path data. Reopening AMAS (96) verified unchanged blank data
but did not clear the stale Main Sequence paint. No answers were changed.

Added an explicit, at-most-two-click blank-radio recovery with a durable claim
in the original failed selection. It requires hash-checked failed-click and fresh
blank evidence, names the temporary White Dwarf UI state separately from the
intended Main Sequence reference choice, and protects all numeric values and
other controls. The temporary state is **not a scientific hypothesis or training
label**. Seven intercepted fixtures pass, including interrupted second clicks,
changed evidence, attempted retries, and rejection of nonblank-answer revisions.
The reference Main Sequence choice remains course-unvalidated for AMAS.

The broad Python snapshot finished **1,292 passed, 11 skipped, 928.59 seconds**.
It was collected before these final navigation/recovery changes; latest affected
tests are being run separately. No training updates or new final-test cases were
used for the browser transport changes.

Live recovery 99 succeeded with exactly two class clicks and **zero numeric
writes**; all six fields became available. Prefix 101 selected Ga. Frozen
lifetime checkpoint run 102 then made six verified copies in 59 completed
decisions (124.16 seconds), with its 60th/final-check proposal stopping on
`screen_changed_during_binding`. The preserved diff contains exactly one change:
the accessibility footer's `Data saved` notice disappeared while the separately
read text already lacked it. Values and controls did not change.

The six-field footer projection now matches the exact text and accessibility
suffixes independently, leaving notices elsewhere and all other state protected.
Read-only reconciliation 104 verifies the hashed stream/diff, original six
readbacks, unchanged checkpoint, current class/values and final learned check
proposal, using **zero browser actions**. The stopped run is never rewritten or
retried. Confirmed native displays: distance 28.60 ly, luminosity 0.0001449 Lsun,
temperature 1696 K, mass 0.08002 Msun, radius 0.1409 Rsun, lifetime 5521 Ga.
These remain transport results, not course correctness or project completion.

The stellar bridge now avoids full browser reads for local menu/binding choices,
while retaining full checks at calculation, every copy (including after approval),
and completion. The in-flight 102 run used the earlier implementation; no model
was changed mid-run. Nine focused tests pass, including four real-graph frozen
class variants and stale measurement/modal/navigation/replaced-target/final-check
rejection. The earlier affected snapshot completed **118 passed in 167.64 s**;
ten subsequent footer/reconciliation fixtures pass. Latest combined affected
tests are running separately.

The final combined affected stellar/navigation run passed **68 tests in 140.68
seconds**, including frozen numeric/color policies. AMAS color 105 selected IR
with one verified write and zero optimizer updates. Numeric 102 (298 events) and
color 105 (18 events) both replay offline with socket access denied and unchanged
trace hashes. Numeric's original stopped outcome remains visible in replay;
reconciliation is a separate read-only receipt, not an edited success trace.

AMAS observation 108 requested 10,000 days once. No repeat Play action was taken.
Initial spectrum 109 found only one exposed excursion marker, and the uncentered
chart hover 113 rejected a missing/occluded tooltip. Neither is an absence-of-
planet result. Screenshot 112 and visible geometry are retained; the owned page
and simulation both report visible, so no native window management is needed.
Bounded, centered visible-tooltip sampling is the next evidence gate.

Scan 115 stopped at `daily_transit_coverage_gap`. Its first window reported days
5919-5971 (the prior INDANGOLAC view) despite the new visible 0-10000-day axis;
the first drag then produced days 43-2391. All of 115, including its otherwise
continuous first window, is explicitly invalidated for measurement/continuation.
Original samples remain untouched. The sampler now refuses a parent directory
with `invalidated.json`, and new recorded scans check every hovered day against
fully exposed numeric x-axis tick glyphs. No SVG paths, bound scales, event
handlers or hidden model values are read. Dragging uses ten separately paced
native segments with guards between them. Forty-nine affected chart/spectrum/
sampler tests pass; the invalidation continuation test was added afterward.

By spectrum 118 both excursion markers were exposed. Strict hover results are
656.29995212 and 656.30004788 nm around 656.3 nm: semi-amplitude **0.00004788 nm**.
This is visible reference measurement evidence, not learned perception. A new
independent axis-checked scan is running; it does not inherit any of 115.

Scan 119 stopped before recording samples because the unzoomed chart clamped
vertical panning and kept its baseline near y=20.45. Slower drag segments did
not fix this; reviewing the earlier 18/19 evidence confirms the same pacing
hypothesis had previously been rejected. The original native drag implementation
is restored again, not presented as an improvement. Ordinary reference zoom 122
changed the visible time axis to roughly 500-5000 days. The sampler's bootstrap
now spends from its existing 12-zoom budget if a verified baseline cannot be
positioned while unzoomed, then rechecks readability before any sample. No budget
increase, answer change or hidden chart state is involved. Fourteen pure sampler
tests pass, including full-scale pan clamping and zero-budget rejection.

Read-only axis diagnostic 125 showed the first visible time tick at x=43.5
(500 days), whereas the first sampled native pixel is x=31 (tooltip 239 days).
The new agreement check initially rejected this valid half-pixel/first-tick
case. It now permits at most one tick interval of linear extrapolation inside
the known plot bounds, with a 1.1-native-pixel rounding tolerance. A fixture
accepts 239 and still rejects the stale 5919 value at the same pixel. Scan 124
remains stopped. Scan 126 is recording independent continuous axis-checked daily
evidence from day 429; no part of invalidated 115 is inherited.

Subsequent reader changes combine exposed tooltip and x-axis text/geometry into
one atomic browser evaluation, retaining the pre-read context guards and the
per-glyph visibility check. Zoom calibration targets 42-54 days across 54 native
pixels with smaller wheel steps; excessive duplicate-day sampling is avoided
without raising the 12-zoom, 2,048-action or 900-second budgets. The in-flight
126 batch keeps its original implementation. Fifty-seven latest focused chart/
sampler/evidence tests pass; 16 pure sampler tests cover the later provenance
and replay additions. New scan streams use version-1 JSONL with an explicit
`scripted_reference_sensor` label and offline replay compatibility. Legacy
raw journals remain readable by the evidence recovery code, but new live
continuations require recorded per-hover time-axis validation; old unverified
evidence is not silently promoted.

The second broad Python regression completed **1,330 passed, 11 skipped in
1,091.38 seconds**. This is a code-check result, not project acceptance. The
subsequent chart/sampler/evidence suite passed **57 tests in 18.84 seconds**.

Scan 126 reached its 900-second cap. Read-only recovery retains **857 consecutive
verified days, 429-1285**, with no complete transit and no planet answer. Original
journals remain unchanged. Strict reader profile 127 measured three guarded
hovers at approximately 0.264-0.285 seconds each. Continuation 128 uses the new
reader/calibration and inherits only the recovered, axis-checked 126 evidence.
It does not inherit invalidated scan 115.

Recovery now preserves a timed-out continuation's inherited evidence, not just
its newest rows. Future batches retain an exact `prior-report.json` snapshot;
recovery checks its recorded hash, sample count, star, coverage and axis-validation
provenance. Older in-flight batches can supply their original prior report
explicitly, subject to the same hash checks. Missing, altered, invalidated and
gapped prior evidence fails closed. Twenty-three pure sampler tests pass.

`record_transit_campaign` runs an explicitly chosen 1-10 sequential batch limit,
retaining the existing per-batch budgets and 17,600-sample cumulative ceiling.
Only a successfully completed batch can start the next; a safety failure stops
without automatic recovery or retry. Three verified periodic transits end the
campaign early. Exhausting batches is inconclusive, never a No Planet choice.
Eight orchestration tests pass. This remains scripted sensing with zero answer
writes and zero optimizer updates.

The TUI now displays chart samples, batch-window progress, safety stops and the
scripted-sensor boundary instead of empty reward placeholders. Its new shared
golden stream retains failed/incomplete status and no fabricated neural activity.
Rust checks passed **45 tests with one normally ignored integration test**; that
installed-Python subprocess integration was also run separately and passed.
These are replay/transport improvements, not learned chart-perception acceptance.

Continuation 128 completed 32 windows: **2,145 consecutive days, 429-2573**, all
flat so far. Campaign 129 was explicitly capped at six batches, but stopped in
its first batch at day 2625 because a full tooltip was unavailable. Successful
window checkpoints retain inherited evidence; no automatic retry or planet
answer followed. The crop in axis diagnostic 130 shows a deep dip extending below
the current 98.4-100% flux viewport. Ordinary pans/zooms 131-137 confirm coupled
x/y zoom and restore the narrow view; wide-zoom hover 135 also remained unreadable.
No clipped tooltip value was used. Period evidence remains incomplete.

The new explicit `probe_hover` returns no sample for a zero-exposed-tooltip
result while retaining every context/identity/time/action guard. Strict `hover`
still fails on the same condition. `exposed_flux_sample` uses at most 64 ordinary
vertical search pans per reading, then reverses them and verifies restoration
against rendered axis glyphs; it neither changes the time scale nor infers an
unseen value. Failed restoration or exhausted budgets stop. Sixty-two affected
Python tests and a separate native Chromium deep-dip fixture pass; 37 pure
sampler/campaign/exposure tests pass after integration. Live diagnostic 138 is
testing one reading before any campaign continuation.

Live deep-flux diagnostic 138 succeeded: **day 2625, 81.162% brightness**, 37
ordinary chart actions, verified restoration, zero answer writes. Campaign 139
continues from 129/batch-001/window-02 under the same six-batch cap; its first
window verified one complete transit. No period or planet answer is claimed yet.

The supplied-temperature curriculum is now separate from course habitability
classification. Its inverse energy-balance fixtures expose luminosity, orbit,
albedo and one of four supplied greenhouse increments plus reference-star
distractors. Two explicit operations produce equilibrium and surface temperature
results; exact copying and K-unit selection remain decisions. The forward pack's
course-reference oracle is still disabled, and no gas/phase/habitability rule was
invented. The scripted expert passes **100/100 cases in 28 actions each**.

Versioned `structured_habitability_tool_v1` / `semantic_habitability_tool_v1`
encodings preserve the biological recurrent architecture and reject accidental
stellar/planet checkpoint reuse. Model/environment/checkpoint/replay checks passed
**26 tests with one optional test skipped**, including finite gradients, edge
dependence, private-answer exclusion, exact copying and old planet runtime checks.
The temperature smoke command is
`.venv/bin/python scripts/train_habitability_calculations.py experiments/habitability-smoke-001`.
It is explicitly limited to the biological 2,000-node graph, hidden size 16,
one CPU thread, seed 0, four training cases, two epochs, **32 optimizer updates**,
two calibration and two development cases. Network and spreadsheet calls are
denied, PPO is absent, and final-test cases are neither generated nor evaluated.
The run finished in **68.22 seconds**, peak process RSS **1,006,043,136 bytes**.
Loss fell from **3.5693 to 2.9529**, and checkpoint reload was exact. Learned
closed-loop completion was **0/4 training and 0/2 development**: every episode
reached the 128-action limit without invalid actions, tool or infrastructure
errors. The first preserved failure repeats a measurement selection instead of
advancing the workflow. Finite loss and a reloadable checkpoint pass the code
smoke, not the learning gate.

An explicit whole-episode diagnostic now reuses exactly that hashed dataset and
checkpoint: 20 epochs, 28-step sequences, **80 additional optimizer updates**,
frozen text encoder, fresh optimizer moments, no new numeric cases, no PPO or
final-test use. Parent split/trajectory/checkpoint hashes are checked before
training. The last fixed-budget epoch is evaluated; development is not used to
select a best epoch. Command:
`.venv/bin/python scripts/train_habitability_calculations.py experiments/habitability-sequence-001 --sequence-from experiments/habitability-smoke-001`.
Twelve affected model/lineage tests pass; the run is being evaluated separately.

`habitability-sequence-001` finished: **0/4 train, 0/2 development**, finite loss
2.7303 to 2.0263, 60.03 seconds, peak RSS 543,883,264 bytes. Teacher-forced exact
actions are 25.89% train / 25% development. Frozen diagnostic
`habitability-diagnostics-001` records the first mistake at action zero in every
case (measurement control instead of calculation). Reversing controls/options
changes zero decisions; numeric counterfactuals change 1/112 train and 0/56
development decisions. This is mostly a workflow-learning failure, not a claim
of complete numerical invariance.

`habitability-transfer-001` explicitly copies the planet pilot's recurrent core,
shared controls, and **only identical orbital-radius quantity/unit semantics**.
Unrelated temperature/albedo/luminosity columns begin at zero; no formula/action
rules, neuron embeddings, or expert action repairs are added. Its 80-update run
improves teacher-forced exact decisions to **75%** on both recorded splits but
still completes **0/4 and 0/2**. The first rollout selects and binds the first two
inputs correctly, then reselects luminosity instead of albedo. It incurs **42
training / 18 development recoverable tool errors** (missing inputs), not API or
infrastructure failures; invalid-action count is zero. Loss 10.5037 to 0.5146,
62.48 seconds, peak RSS 577,077,248 bytes. No extra numeric cases or final tests.

A separate explicit same-task refinement reuses the transfer checkpoint's AdamW
moments for **80 additional updates**, same four cases and 28-step sequence. It
validates dataset/checkpoint hashes and the saved optimizer step count; it does
not reset moments, relabel failed cases or increase the number of examples.
Eleven transfer/refinement/optimizer tests pass. The run is
`habitability-refinement-001`; its scores are pending.

The refinement completed **4/4 training and 2/2 development**, all in 28 actions,
all selection/copy/unit/numeric metrics 1.0 and all error counts zero. Loss
0.6499 to 0.006475, 60.95 seconds, peak RSS 516,440,064 bytes. The explicit
64/16/16 pilot then completed **64/64 training and 16/16 development**, with
the same zero-error 28-action behavior. `habitability-pilot-001` used five epochs,
**320 additional updates**, inherited AdamW state, unchanged source checkpoint,
323.83 seconds and peak RSS 908,017,664 bytes. It has 60 new training cases and
14 new cases each for calibration/development; final cases remained sealed.

`habitability-final-001` is the one reserved **100-case frozen evaluation**:
**100/100 completed**, 28 actions each, all recorded selection/copy/unit/numeric
metrics 1.0, zero invalid actions/tool/API/infrastructure failures. It took
**93.98 seconds**, peak RSS **594,771,968 bytes**, with zero optimizer updates,
no recalibration, and unchanged weights/checkpoint. SHA-256:
`44cc4cc06c2e2d8cfa3e1af9a7f255058be95d6e5139d655b725a068a75c4984`.
Seeds **18000000-18000099 are consumed**; do not tune on them or reuse them as
unseen data. The exclusive reservation is
`experiments/habitability-supplied-temperature-final-v1.reservation.json`.

This passes **only supplied-temperature local tool use**. Gas identification,
greenhouse-strength selection, water phase and habitability classification are
not learned or course-accepted. The version-1 runtime/TUI profile reuses
development seeds 17000000-17000015 and requires the matching pilot/final gate.
Use `.venv/bin/python scripts/habitability_tui.py --check` for a read-only local
readiness check, then `.venv/bin/python scripts/habitability_tui.py` from an
interactive terminal. `n` single-steps, Space runs/pauses, `a` aborts, `q` quits.
These commands never open HabWorlds or train a model. The opt-in real biological
graph planet/temperature runtime and final-gate suite passed **28 tests in 10.35
seconds**, with sockets and spreadsheet construction denied. Both paused
checkpoint demos complete, emit real neural diagnostics, preserve the checkpoint
and replay offline. The temperature launcher's `--check` reports **ready**. Rust
now passes **46 tests, one normally ignored**, including the explicit
supplied-temperature/not-habitability scope display. Ruff and `git diff --check`
pass for this checkpoint.

The initial broad regression invocation hit a macOS sandbox limitation creating
Chromium's MachPortRendezvousServer (permission denied): **944 passed, 10 skipped,
448 browser-fixture setup errors**. These were not interpreted as model/browser
behavior failures. The isolated assessment fixture passed all four tests with
the required process-launch permission; a permission-enabled full suite is in
progress against synthetic fixtures without touching the owned HabWorlds page.

Adding the versioned temperature encoding correctly invalidated the earlier
planet checkpoint's model-source calibration fingerprint. The optional planet
TUI regression caught this before use. `planet-calibration-refresh-001` refits
only temperatures against the already-recorded 16 calibration trajectories;
its **16/16 recorded development cases reproduce the original semantic action
streams exactly** with zero invalid actions or tool/infrastructure errors. The
frozen checkpoint file and all model weights are unchanged. No sealed final
test was reused, and this is not a new unseen evaluation.

`configs/planet_tui.json` explicitly names this calibration receipt. The loader
checks model-source, checkpoint, content, calibration-data and development
artifact hashes and rejects changed actions or inconsistent temperatures before
applying it. The opt-in real-graph runtime/replay plus calibration suite passes
**28 tests**; `scripts/planet_tui.py --check` reports ready. Rust passes **45 tests,
one normally ignored**. Historical evaluations remain untouched.

The permission-enabled broad Python suite completed **1,393 passed, 11 skipped
in 1,177.50 seconds**. New runtime/evaluation tests added after its collection
were run separately as described above. Rust's normally ignored installed-Python
bridge also passes (2.03 seconds).

AMAS campaign 139 completed in five of its six allowed batches: **8,083
consecutive verified daily readings, days 429-8511**, with complete dips at
**2625, 5563 and 8501**. The measured period is **2,938 days**, brightness drop
**18.838%**, using only exposed hover tooltips and per-hover visible-axis checks.
No answer was written by the sensor. Spectrum 118 remains the same-star source
for the **0.00004788 nm semiamplitude**. This is scripted/reference chart sensing,
not learned perception or full-project completion.

Read-only readiness 140 confirms AMAS's Has Planet and raw fields are still
blank, with the previous star's Ice Giant circle still painted. A narrow
presence-selection option can explicitly preserve that exact paint while
choosing evidence-backed Yes; it performs zero class writes and records
`class_selection_verified: false`. Default behavior still refuses preselected
paint, and any unexpected class/readback/control change still stops. Twelve
controlled Chromium presence tests pass. This is not permission to interpret
inherited paint as the current star's classification.

Presence action 141 selected Yes once, then stopped because the native page
cleared the inherited Ice Giant paint. Its failed receipt is preserved. The
before/after field projection is otherwise identical apart from Yes and the
expected blank derived-field panel; read-only check 142 confirms Yes, seven blank
numeric answers and no class selection. Future presence runs permit only this
exact transition to the same fully unselected radio rendering, or unchanged
paint—not a different selected class. Explicit read-only reconciliation validates
the source hashes, captures, unchanged current fields and stable native targets;
it never repeats the selection. Fourteen Chromium presence tests pass, including
legacy stopped-readback reconciliation with all action methods configured to fail.

Read-only reconciliation **143** passed without another Yes click. Raw transfer
**144** copied semiamplitude 0.00004788 nm, drop 18.838% (displayed 18.84%) and
period 2938 days. Frozen planet run **145** completed all **62 decisions** with
four verified native copies, unchanged checkpoint and **zero optimizer updates**:
orbit 1.731 au, mass 91.04 Earth masses, radius 6.666 Earth radii and density
1.695 g/cm3. Runtime was **151.57 seconds**. The event stream SHA-256 is
`d81afb8adc1defeeee19faaa11690155d94376e06cae144c8cdf78dfd7f3d178`.

Reference class **146** selected Ice Giant by explicit comparison with the
same-session assessed INDANGOLAC case, not a learned or universal class rule.
The two-star Data Quality assessment **149**, revision 3, returned **99.9% stars,
99.9% planets, 0% habitability, 66.6% overall**. Its fresh receipt consumed 100
simulation dollars and is separate from the previous one-star assessments. This
is course-visible evidence, not proof of an independent learned classifier.
Save, score transfer, and final submission are not implied.

`browser_habitability_policy.py` now provides the frozen supplied-temperature
bridge. It exposes visible luminosity/orbit/albedo plus an explicitly supplied
warming increment, runs the same learned calculation controls, then performs
**one guarded equilibrium-temperature copy**. The surface-temperature proposal
remains local: the live surface readout is derived by HabWorlds, not a field to
overwrite. The bridge does not select gases, greenhouse strength, water phase or
habitability and does not contain a browser grading oracle. Browser confidence
is deliberately uncalibrated. It journals version-1 events and exact provenance.

The offline bridge suite passes **18 tests**, including a real frozen biological
model completing 28 decisions with sockets/Sheets denied and unchanged weights.
The intercepted Chromium bridge/numeric suite passes **24 tests**, covering
single-write transport, stale-page rejection, and uncertain-write no-retry. The
bridge has not yet run on a fresh terrestrial planet in the current attempt.

Scavenger assessment **152** stopped at `assessment_changed_after_reservation`,
the guard immediately **before** `button.click`. Read-only capture **153** shows
the same visible snapshot and funding still **49,500**; no assessment click was
retried, and the original pending reservation is retained. The exact transient
trigger was not captured, so no unsupported root cause is asserted. The older
failure artifact conservatively says writes may have occurred even for this
pre-dispatch exception. Future actuator failures now record the dispatch flag,
the three comparison outcomes (snapshot, project rows, native control), and the
pre-dispatch capture. This changes diagnostics, not retry authorization.
**62** assessment/ledger/score-transfer regression tests pass. This stage still
lacks a current confirmed scavenger assessment and cannot authorize score
transfer from its partial ledger.

Next-star navigation **156** opened **ISSA** at rendered point (385.5, 351) in
the unchanged 950x600 viewport, excluding both visited points. Fresh stellar
answers are blank. Visible observations: parallax 0.029 arcsec, peak wavelength
1258 nm, flux 3.52e-14 W/m2. Explicit unclassified reference calculations give
2303.47 K and 0.00130726 solar luminosity; the assistant's supplied Main Sequence
choice follows the cool/faint H-R region and remains unassessed, not learned.

`synchronize_fresh_main_sequence` consumes a hashed fresh-star receipt and
handles only verified blank new data. When Main Sequence is inherited paint
without its fields, it uses exactly two class clicks (temporary White Dwarf,
then supplied Main Sequence), with no numeric writes or training labels. Other
unselected/different paint uses one click. There is no automatic retry or
same-option failure requirement. **68** class-recovery/full-stellar/next-star
fixtures pass. Live **157** succeeded with two clicks and zero numeric writes;
**158** selected Ga. The frozen stellar numeric run is **159**.

Both `.venv/bin/python scripts/planet_tui.py --check` and
`.venv/bin/python scripts/habitability_tui.py --check` still report **ready**,
paused, with unchanged checkpoints; neither opens a browser or starts training.

ISSA numeric **159** passed: **60 learned decisions**, six verified copies,
unchanged checkpoint, zero optimizer updates, **43.81 seconds**. Displays:
112.4 ly, 0.001307 Lsun, 2303 K, 0.1500 Msun, 0.2292 Rsun, 1147 Ga.
Frozen color **160** selected IR in one native write. Planet tab **161** remains
blank; observation **162** requested 10,000 simulated days once. The previous
star's day-8511 tooltip is stale and must not be used as ISSA evidence; per-hover
time-axis checks remain mandatory.

The offline `record_undispatched_assessment` function records only the exact
known pre-click guard with a matching hashed before/current capture. It leaves
the original pending ledger untouched and authorizes no retry; delivered or
ambiguous failures are rejected. New failures require an explicit false dispatch
flag. Legacy guard review is separately opted in. **51** assessment/ledger tests
pass, including both versions and refusal of uncertain/delivered failures.
The explicit legacy review of AMAS attempt 001 produced
`amas-assessments-149/attempt-001-undispatched.json`, with zero browser actions,
unchanged funding, and the unused $100 reservation retained as audit history.
No second scavenger assessment or score transfer was performed.

ISSA spectrum **163** initially found an ambiguous marker inventory and stopped
before a mouse move. Read-only geometry **164** and crop **165** subsequently
showed two exposed excursion markers. Separate sensor read **166** verified
656.29995548/656.30004452 nm, semiamplitude **0.00004452 nm**; the earlier failure
is preserved. No answer was written.

Transit diagnostic **167** stopped at `flux_exposure_restore_not_verified`
during initial calibration. Its chart changed from the inherited 98.4-100%
axis to 0-100% after an ordinary pan; the inverse drag did not restore the old
axis. The sole exposed calibration reading (day 43, 100%) is **not accepted as
period evidence**, and no sampled-day set was completed. Read-only axes **168**
verified the new visible scale. A separate bounded scan **169** starts from
that inspected viewport, with no inherited samples and no tolerance relaxation.

Offline version-1 parsing passes for planet **145** (321 events), stellar **159**
(298 events) and color **160** (18 events), preserving their original bytes/hashes.

The combined latest temperature-bridge, assessment, score-transfer, fresh-class,
flux-exposure and transit regression gate passes **159 tests in 51.94 seconds**,
including the real frozen 2,000-node temperature policy. No additional training
or sealed-final evaluation was run. Ruff and diff-whitespace checks pass.

The next-star picker accepts an optional bounded fractional sky-region anchor;
the default remains (0.4, 0.55). It still selects only a rendered compact bright
dot, excludes every visited pixel location, and records the anchor as geometry-
only navigation, not a scientific class filter. Recovery reuses the recorded
anchor; older receipts retain the previous default. **49** setup/navigation
tests pass. No anchored selection has yet been made in the live attempt.

ISSA scan **169** completed all 32 bounded windows: **1,542 continuous daily
readings, days 426-1967**, with complete dips at **755 and 1642**, depth **6.438%**.
It used 1,829 chart actions and six calibration zooms. Two dips are not promoted
to verified period evidence. Explicit continuation **170** allows just **one**
additional 32-window/900-second batch, carries the exact hashed prior report,
and stops early if a third consistent transit is established. No new observation
request, answer write, assessment, or training is part of the scan.

Continuation **170** stopped successfully at window 13: **2,119 continuous
days, 426-2544**, complete dips at **755, 1642, 2529**, verified period **887 days**
and brightness drop **6.438%**. Its batch report hash is
`00fe2b6e659a57070d1627e9c7b729695c70050d5cffe3442cf05c8a89054ab3`.
Combined with spectrum **166**, this is same-star scripted visible measurement
evidence, not learned chart perception. Presence transition **171** is next.

`browser_project_assessment.py` provides explicit bounded **new revision**
stages (one or two $100 simulation assessments). It validates all earlier
reservation/confirmation history and source hashes, blocks uncertain dispatched
actions, refuses duplicate/older revisions and unchanged project rows, and
checks the caller's expected collected count. A separately reviewed non-dispatch
disposition remains audit history and never authorizes retry of that revision.
No Save, score update or submission occurs during stage setup. **46** project-
assessment/actuator/score-transfer tests pass. Read-only validation of the actual
attempt finds five confirmed assessments and one preserved unissued revision-3
scavenger reservation; no new live assessment has yet used the stage helper.

## 2026-09-26 continuation after interruption

ISSA presence **171** and raw transfer **172** passed. Frozen planet run **173**
completed 62 decisions and all four native copies in 168.39 seconds, with zero
updates and an unchanged checkpoint: 0.9604 au, 86.33 Earth masses, 6.339 Earth
radii, 1.869 g/cm3. Its event hash is
`d4526e8657538396b92370035c71cd5e2f57a828fcd500a4a2833fff331bed56`.
Reference-assisted Ice Giant selection **174** passed readback; it is not a
learned classification or a confirmed course answer.

Revision-4 assessment **177** dispatched once, then stopped with
`project_rows_changed_during_assessment`. No post-change capture was saved by
the old code. On continuation the owned harness process was no longer running.
The original reservation remains unresolved, never retried or marked complete;
`interrupted-session-disposition.json` records that limit. The earlier two-star
99.9% stellar/planet result is not a verified three-star score.

The actuator now saves a hash-verified changed-row capture and the before/after
row hashes before stopping. It still rejects the mismatch without retrying.
`browser_habitability_actions.py` adds one-shot reference greenhouse lookup and
same-star/same-condition confirmed chamber-to-water-phase transport. It cannot
infer gases, alter temperatures, save, assess, or submit. Published greenhouse
band gaps and the unavailable explicit None option stop without repair. Context,
native identity, source hashes, unchanged answers and dependent surface readback
are checked. **137** combined menu/numeric/mapping/chamber/assessment tests pass.

A fresh, separate owned run `experiments/full-stellar-probe/20260926-005` opened
Galbarrya using sky anchor (0.75, 0.4), point (698.5, 266) in the 950x600 frame.
This is a user-authorized fresh preview test, not recovery or validation of the
lost attempt. Class reference **1** is explicitly supplied Main Sequence from
the cool/faint H-R region. Ga prefix **2**, frozen stellar run **3** (60 decisions,
six copies, 36.07 seconds), and frozen color **4** (IR) passed. No new training
or sealed-final evaluation was run.

Chart guards now combine chart identity/parent checks with the same native-value
read instead of separate browser round trips, and read each body's visible text
once. Every pre/post-action guard and full-glyph/day-axis check remains enabled.
**75** chart/spectrum/transit/deep-exposure regression tests pass. No speed gain
or new scientific perception capability is claimed before live measurement.

### Continued integration: gas comparisons, safe revisits, and chart guards

Galbarrya revision-1 data-quality assessment **7** was confirmed once: stellar
**99.8%**, planet/habitability **0%**, overall **33.3%**; simulated funding fell
from 50,000 to 49,900. Its receipt was dismissed once (**8**). No scavenger
assessment, score transfer or submission was performed. The two unlearned
decisions so far are the supplied Main Sequence classification and observation
duration; all six numeric choices/copies and IR selection used frozen policies.

Observation **11** requested 10,000 days once. Spectrum diagnostic **12** stopped
before a hover because two excursion markers were not visible. Read-only image
**16** and marker layout **17** still showed only the central line; that is not
promoted to a measured zero-amplitude answer. Scan **15** recorded 1,542
continuous days (426–1967) with no transit. Campaign **20** is explicitly capped
at two further batches, with the same per-batch limits and no answer writes.
Flat finite intervals remain inconclusive, not automatic No labels.

`browser_gas_controls.py` now provides explicit candidate checkbox comparisons,
chart-only crops with hashes, exact native readbacks, and an optional explicit
visual-reference combination. Candidate selection is **not learned**, no best
fit is automatically inferred, and correctness remains unverified. Comparison
images, source capture, current star and conditions must agree before applying a
combination. Uncertain writes are recorded and stopped, never restored/retried;
driver failures are sanitized. The expanded gas/revisit subset passes **48
tests**. Earlier combined gas/habitability checks passed **67 tests** before the
combination helper was added. Live gas validation on a new terrestrial planet
is still pending.

`browser_collected_revisit.py` reopens a **currently visible** named stellar row
by its exposed eye icon, verifies displayed measurements, numeric answers,
color, class paint and lifetime prefix against the detail screen, and performs
zero answer writes. It does not paginate or infer classification correctness.
The original blank-row adapter remains unchanged. **32** old/new navigation
tests passed; live populated-row validation remains pending.

For ordinary sibling frames, chart guards now validate the retained iframe
elements together with the main document's native controls. They reject a
newly hidden frame **before** reading its child controls; nested frames retain
the original per-element guard. Every action still has pre/post checks, fully
exposed tooltip glyphs and day-axis validation. **82** chart/spectrum/transit/
exposure/campaign tests pass, including added before/after hidden-frame cases.
This reduces browser round trips; no measured speed-up is claimed yet. Frozen
models, optimizer state and sealed evaluation cases have not been changed.

Campaign **20** completed its two batches: **4,520 continuous days, 426–4945**,
with no transit; outcome `batch_limit_inconclusive`. Reopen **23** successfully
verified the populated Galbarrya stellar row and all displayed answers with
one eye click and zero answer writes.

`browser_planet_absence.py` adds a separate **explicit reference hypothesis**
transport, not an extension of the verified positive-transit rule. It requires
an unchanged, continuous flat scan of at least 1,000 exposed days (an engineering
evidence minimum, not a scientific confidence threshold), one exposed central
spectral line without excursion markers, an explicit limited-evidence rationale,
and blank planet answers. It records the interval, spectrum crop and hashes;
`absence_proven`, `correctness_verified`, `training_label` and `task_completed`
remain false. It does not enter invented zero-valued measurements. **12**
intercepted tests pass. This allows a falsifiable reference choice without
misrepresenting finite observations as an absence proof or learned prediction.

Live hypothesis **25** selected **No** for Galbarrya once, with successful native
readback and no numeric/class writes. Its course validation is pending; the
original positive-transit sensor still never automatically chooses No.
The broader integration gate passes **195** tests; Rust passes **46** regular
tests plus its separately invoked Python protocol bridge test.

Revision-2 assessment **27** confirmed one $100 simulation charge after the
explicit No hypothesis: stellar **99.8%**, planet **25.0%**, habitability **0%**,
overall **41.6%**. This is **not** planet completion or proof of absence. The
three raw measurement fields remain blank; no invented zero-valued period or
Doppler measurement was entered. Funding is 49,800. The changed result is kept
as private course feedback, not policy input or a ground-truth training label.
The positive/negative-reference/chart evidence regression subset passes **56**
tests. Both planet and habitability TUI preflight commands still report ready,
paused, frozen development-case reuse, and zero optimizer updates.

Dismissal **28** stopped at the old `receipt_changed_before_dismiss` guard,
before any acknowledgement click. Navigation **29** also stopped before its
header click because the receipt covered it. Read-only capture **30** matched
the stored confirmed assessment exactly, including 49,800 funding; the transient
discrepancy was not saved by the old code and its cause is unknown. The separate
`legacy-ack-preclick-review.json` records that limit; no assessment was retried.

Receipt dismissal now records `ack-NNN` reservations/confirmations separately
from assessment spending records, captures stale pre-click observations, binds
native identity, and blocks repeats after a dispatched/uncertain dismissal.
**50** assessment/project-history/score-transfer fixtures pass. New dismissal
**31** passed once with an immutable acknowledgement receipt. The original
assessment/history receipts remain unchanged; read-only history validation still
finds exactly two confirmed data-quality assessments.

### Fresh-star continuation and generalized habitability choice

Aramintor opened at **33**. Its explicit cool/faint Main Sequence reference
selection **34**, Ga prefix **35**, frozen six-field policy **36** (60 decisions,
six verified copies, 36.34 seconds) and frozen IR selection **37** passed. Native
stellar values were 50.94 ly, 0.0001441 Lsun, 1696 K, 0.07989 Msun, 0.1405 Rsun,
5543 Ga. Observation **39** requested 10,000 days once. Read-only spectrum **40**
and screenshot **41** showed one central line and a flat overview; this is not
measured planet absence. No planet answer was written.

Kjin opened at **43**, point (396.5,332) from rendered sky pixels, not the former
session's catalog identity. The explicit Main Sequence reference **45**, Ga
prefix **46**, frozen stellar **47** (60 decisions, six copies, 39.13 seconds),
and frozen IR selection **48** passed. Native values: 75.81 ly, 0.002027 Lsun,
2447 K, 0.1700 Msun, 0.2528 Rsun, 838.8 Ga. Observation **50** requested 10,000
days once; spectrum **51** again showed one central line. Navigation-layout
inspection **52** verified that the named Play control belongs to the observation
panel, not the separate reconstruction animation. No planet answer or absence
claim was made. These new stellar runs used zero optimizer updates and unchanged
checkpoint hashes; they are not new sealed-final evaluations.

`browser_habitability_choice.py` replaces the old JYREMIS-specific reference
click with a bounded, same-condition phase-receipt transport. It verifies the
confirmed water-phase menu receipt, hashed underlying chamber evidence, current
star/pressure/temperature/answers and native identity. One explicit choice,
including confirmation of default paint, is reserved before dispatch; uncertain
writes stop and retain evidence. A Habitable reference choice requires confirmed
Liquid water but is not equated with scientific correctness. It does not choose
gases, train a classifier, save, assess or submit. **49** targeted choice/menu
Chromium fixtures pass, with stale-source, phase/condition changes, replaced
controls, modal/auth changes and interrupted side effects. Live validation of
this generalized choice adapter is still pending.

### User-approved bounded planet window

New observation requests are capped at **5,000 days**. Once the plotted trace
reaches that window, no visible dip permits an explicit **reference assumption
of no planet**. This is the user's accepted completion/accuracy tradeoff, not a
scientific absence proof, learned classifier, or training label. Long-period,
shallow and subpixel transits can be missed. Existing 10,000-day artifacts are
preserved; only their first 5,000 days may support this rule. No graph panning or
longer observation campaign is required to reject a flat window.

`browser_planet_observation.py` commits a blank duration with Tab, rebinds the
visible Play control and dispatches it once. Durable per-star reservations also
recognize legacy starts. **81** start/numeric transport tests pass. Dispatch and
axis extent are not collection completion: the real chart visibly fills over
time. `browser_observation_progress.py` checks the rendered trace endpoint,
exposed plot and unchanged visible axes; **17** pixel/exposure tests pass.

Begollo's legacy start **56** and additional diagnostic Play **62** remain
recorded. The second click may have restarted collection; the earlier fill was
not proven lost. Captures **65** and **66** show the trace extending over time.
Read-only **69/72** reached the first 5,000-day endpoint. Geometry-only exposure
check **70** incorrectly rejected a covered background screen; actual plot
hit-testing plus pixel validation fixed this at **72**, without another Play.

Begollo's separately supplied Main Sequence reference **74** and Ga prefix
**75** allowed frozen stellar run **76** to finish all six native copies in
60 decisions, **33.31 seconds**, zero optimizer updates. Values: 28.85 ly,
13.82 Lsun, 8351 K, 2.118 Msun, 1.793 Rsun, 1.532 Ga. Its numeric event hash is
`a7dbcbecc98037ef75517727947e823c0dd502155e99275a387d0f086c8b5244`.
This is calculation/transport evidence, not full-star or project completion.

The new offline `ProjectJournal` distinguishes predictions, readbacks, explicit
No/N/A or nonhabitable branches, uncertain writes, assessment, score transfer
and submission. Its **80** ledger/assessment tests pass. The Rust project view
accepts optional version-1 progress events; **58** regular tests plus the
Python bridge pass. Runtime coordination and branch-aware live completion are
still being integrated; these isolated test passes do not constitute a working
30-star autonomous runner.

The bounded-window selector passed **54** intercepted browser tests including
legacy positive/strict-absence regressions. Live **79** then independently
recaptured Begollo's chart twice and selected **No** once with native readback.
The three visible raw numeric answers stayed blank; no planet class, zeros,
habitability answer, Save, assessment or submission was dispatched. The policy
is `user_approved_5000_day_raster_no_dip_v2`, hash
`1e376eb83c896c6828aa9335ca98820e95ffbf0328ae4a6f33ce53118e16e826`.
The crop at selection had continuous blue trace across all 116 pixels in the
requested first-half window, without the previous hover cursor gap.

Offline replay is available without credentials or a browser:

```sh
.venv/bin/python -m habfly browser analyze-planet-window experiments/full-stellar-probe/20260926-005/observation-progress-72
```

It reports the policy/hash, coverage, source hash and reference assumption,
with `write_authorized: false`. Replay/CLI tests pass **16** cases; the combined
offline ledger/policy/readiness/replay gate passes **135**. The TUI now explicitly
shows the policy label and observation cap: **61** regular Rust tests plus the
Python bridge pass. Live No-save and per-star workflow verification are pending.

Save **81** stopped **before its click** when an already-visible `Data saved`
notice appeared during the reservation checks. Its immutable failure explicitly
records `save_may_have_occurred: false`; no second Save was attempted. Read-only
capture **82** retained the No branch and blank raw fields. Reconciliation **83**
verified that exact pre-dispatch history, but found no currently visible save
notice, so it records a blocked disposition, not a save receipt or persistence
claim. The original reservation remains in place.

A separate timing test reproduced the risk of missing an 80–280ms acknowledgement
while full-screen validation takes 350ms. Save now polls only visible notice,
control identity and context until acknowledgement, records it, then performs
full validation. **46** new Save/reconciliation and **69** existing browser
regressions pass. The ledger now treats conditional N/A applicability separately
from native habitability-field writes; **87** ledger/assessment tests pass.

`PlanetWindowSession` is the cooperative scheduler around the tested observation,
readiness and No-selection components. Each `advance()` performs one bounded
component or read-only poll. Abort before the separate No-selection step prevents
that write; time/poll limits never restart Play, pan, extend the observation, or
convert unresolved evidence into No. **17** scheduler tests pass. Live use on a
fresh 5,000-day star and the integrated project runtime are still pending.

The same-intent Save continuation **84** also stopped before its sole dispatch
when a fresh footer check found a notice after full-screen reads. Neither the
original nor continuation has a dispatch marker; both reservations remain.
Read-only footer sampling **85** saw no notice in 40 samples over 11.78 seconds.
This is a timing edge, not evidence that any native Save ran or proof of an
autosave cadence. The new continuation/reconciliation fixtures pass **69**
tests; a current acknowledgement is still required for the workflow receipt.

The frozen stellar components now expose one learned decision per `advance()`.
Their **36** checks include real 2,000-node inference against an intercepted
browser fixture (six numeric writes plus UV, zero updates). A separate
`BrowserStarSession` composes numeric, color, navigation and bounded observation,
with a distinct positive-dip handoff and guarded No-save/workflow verification.
It is not yet the full project runtime. Callback-abort/reentry regressions were
found in review and fixed before live use; the two scheduler suites pass **40**
tests. Pre-write class-provenance checks and live fresh-window validation remain
in progress. No new training or sealed evaluation was run for these changes.

Dulat was selected from a rendered dot at **88** (sky hash
`cdae5cb511694361f29000b1870b4c8f11b0d5c2f105243b1ac2a1bbc915e2b2`).
The new scheduler at **90** committed **5000** and dispatched one Play with
unchanged answers; this proves the new cap/readback, not yet trace completion.
Main Sequence reference **92** and Ga prefix **93** precede the new cooperative
frozen numeric component **94**: six verified copies, 60 decisions, 47.28s,
zero updates. Display values are 12.78 ly, 0.001306 Lsun, 2303 K, 0.1500 Msun,
0.2292 Rsun and 1148 Ga. Event hash:
`3886c70d944b379fe181b79b776dab782a661c3df218e5087d536e5c2991672d`.
Cooperative color **95** then selected IR with native readback, event hash
`1cb61bc4af9a7187697b1e2da67df973624abb7961e56f99f874f4a798694ca0`.
The separate class-source validator passed the actual 16-file Dulat evidence
chain offline. Class remains a reference prediction, not learned/class-scored.

Dulat's first window poll **97** captured complete 0–5000-day coverage and
visible repeated blue dips, but v2 correctly failed closed before any No choice
on `unknown_plot_palette`. Offline inspection identifies antialiased blue dip
edges over black/gray grid pixels. A supplemental positive-only detector is
being validated without changing the pinned v2 No policy. The scheduler's
stopped report is preserved; neither Play nor observation duration was retried.
Guarded spectrum **99** read the exposed excursion labels 656.29995212nm and
656.30004788nm, giving semi-amplitude 0.00004788nm. A static-view hover at **100**
read day 870, brightness 100%. Eight fractional-day probes at **101** produced
seven repeated day196 readings and day217: mouse hover quantization prevents
claiming continuous daily coverage in this unzoomed view. A separate approximate
raster measurement mode, explicitly uncertainty-bearing/reference-only, is
being validated instead of calling those samples exact period evidence.

The supplemental positive detector now recognizes Dulat's **25** supported
blue dips without weakening the pinned v2 negative policy. Offline replay of
the original progress capture reports `dip_observed`; its original stopped
scheduler report remains unchanged. The independent fixed-window estimator
reports an approximate **195.34-day** period (pixel-resolution interval
193.68–197.01 days) and **6.96%** brightness drop (6.19–7.73%). These are
conditional raster-resolution intervals, not statistical confidence intervals,
scientific verification, exact daily sampling, or training labels. Missed or
aliased events remain possible. The estimator/positive/negative/chart suites
pass **138** tests; browser input transport remains a separate gate.

Read-only inventory **104** verified five fully exposed stellar rows against
the displayed `viewing 1-5 of 5` and `Total collected 5`: Galbarrya, Aramintor,
Kjin, Begollo, and Dulat. This is collection evidence, not five completed tasks.
List revisit **105** stopped before clicking because a transient `Data saved`
footer changed during its guard. A narrow projection now ignores only that
known footer transition, preserving row values, scores, other notices and
context checks; **70** collected/navigation regressions pass. Revisit **107**
then reopened Begollo once and verified retained stellar fields.

Begollo's read-only save reconciliation **109** still stopped: cross-visit
context differed from its original Save reservation, despite identical mapped
scientific fields. Its native duration remained 10000, while the shared chart
axis now displayed 0–5000; collection navigation availability and a covered
starfield tooltip also changed. A separately tested historical-equivalence
check is being developed; current-session guards remain strict. Neither Save
nor its consumed continuation was repeated, and no workflow-completion or
persistence receipt is claimed yet.

Offline chart replay now includes `approximate_measurements` with the measured
period/depth intervals when supported, or a structured measurement error for
insufficient dip geometry. Negative/partial captures do not invent estimates.
Both original Begollo and Dulat captures replay with unchanged outcomes and
`write_authorized: false`. The read-only crop helper accepts an explicitly fixed
1–120-second capture budget (default 30), independently of the 5,000 simulated
days; it never extends a running capture or the observation window. **95**
capture/replay/estimator/scheduler tests pass. The broader offline runtime,
ledger, source-validation and replay selection passes **214**, with **2** opt-in
checks skipped. Rust passes **62** regular tests; its optional Python bridge
was not rerun in this particular regression invocation.

Cross-visit No-save verification passed **137** browser/workflow tests and
**115** downstream evidence/progress tests. Live read-only reconciliation
**110** then accepted exactly the three documented view differences, verified
unchanged Begollo answers, but observed no current save acknowledgement. It
records `no_current_save_acknowledgement`, not task completion. No Save was
dispatched or retried.

Return-to-list **111** passed. Planet-list to stellar-list navigation **112**
stopped before its click on a changing target/screen; capture **113** matched
its original before-state, so the transient difference was not captured and
its exact cause is unproven. Navigation now preserves a separate pre-click
capture and applies the existing exact-footer-only status projection on list
screens. Rows, scores, other notices, context and target geometry remain
guarded. **74** navigation/revisit tests pass, including footer acceptance and
rejection of simultaneous row/score changes. Detail comparisons are unchanged.

Dulat continuation **118–120** passed. The public native star label `Dulat`
and chart heading `DULAT` are compared case-insensitively without changing the
original captures or hashes; the earlier pre-write stop **117** remains intact.
One confirmed Yes selection was followed by three exact raw copies: spectral
semi-amplitude 0.00004788 nm, approximate raster depth 6.961534486328223%, and
approximate period 195.34337944664028 days. Controlled display rounding is
recorded separately. These are reference measurements, not learned perception
or scientifically verified labels.

The frozen planet policy then made **62 decisions and four verified native
copies in 158.19 seconds**, with zero optimizer updates. Browser displays are
0.3502 au, 56.07 Earth masses, 6.592 Earth radii and 1.079 g/cm3. Event SHA-256:
`d32956800dd35aebf051359a779a01fbe8b65a7d67ecaa9117ee0471e2630cf4`.
This is numeric transport completion, not a completed star or assessed answer.

Cooperative components now cover reference raw inputs (**40 checks**), learned
planet calculations (**66 checks**) and habitability temperature proposals plus
a separate native-copy step (**124 checks, two opt-in checks skipped**, with a
separately enabled frozen-model fixture). They preserve existing receipts and
do not own the browser, perform training or silently continue after abort.
The new outer-project event relay passes **18 offline checks**: child lifecycle
events remain nested, original envelopes are retained, and a child's successful
termination cannot end the whole project. Runtime integration is still pending.

Dulat **121** selected Ice Giant as a case-based reference prediction; its
mass/radius/density resemble the previously course-assessed INDANGOLAC example.
The rationale is saved separately and is not a universal classification rule
or training label. **122** delivered one Save click and observed a fresh exposed
`Data saved` notice (`notice_was_already_present: false`), with answers unchanged.
Read-only two-tab verification **123** then passed the complete non-terrestrial
workflow chain, including stellar/color/raw/derived/class sources and current
readbacks. Habitability is explicitly not applicable to this selected class;
no hidden habitability fields or course answer key were inspected.

The canonical journal for `habworlds-local-preview` / `20260926-005` now contains
**five collected, one workflow-verified, four unresolved** stars. The positive
importer checked the complete saved source chain against inventory **104**;
it did not fabricate historical write/Save receipts. Current-revision course
assessments, score transfer and submission remain unverified. This is a passed
development workflow, not a passing scientific score or a 30-star acceptance.

The production spectrum capture wrapper records exactly two guarded visible
marker hovers in the existing raster-input source format, keeping partial
failure events without issuing a completed measurement. **12 spectrum tests**
pass, including source-loader compatibility, wrong-star/callback rejection and
budget checks. No model or frozen chart-policy source was changed.

Offline replay of Dulat's stellar, color and planet traces passed with socket
connections and spreadsheet initialization explicitly denied (298, 18 and 321
events). Re-importing its workflow was idempotent and left the journal bytes
unchanged. The new read-only initial-setup handoff captures the first star after
the existing login/intro picker without selecting it again; **16** navigation
tests pass, including the class-session handoff and changed-source rejection.

A separate **experimental, unwired** H-R reference matcher uses coarse manually
digitized interiors from the visible help image, with the complete pack and
source-image hashes recorded. It is not a learned or course-validated classifier.
**17** geometry/abstention checks pass; Begollo falls in its Main Sequence
interior, while Dulat and Kjin abstain outside its conservative interiors. Those
abstentions are not silently converted to Main Sequence, and none of its
candidates authorizes browser writes or becomes a training label. Independent
review is pending before deciding whether this is suitable for any runtime use.

Kjin continuation **128–131** passed. The first 5,000 visible days contained no
supported dip; **129** selected No under the explicitly approved reference
assumption (not proven absence). The hardened fresh-Save path **130** used a
fixed 30-second pre-reservation settling budget, delivered exactly one Save
click, and observed a new exposed acknowledgement with the answers unchanged.
Two-tab workflow verification **131** passed, including the original stellar,
color and class source chains; habitability is explicitly not applicable.

The same canonical attempt journal now has **five collected, two verified,
three unresolved**, revision 15. Re-importing Kjin was idempotent and did not
change journal bytes. Current-revision assessments, score transfer and final
submission remain unverified. Begollo's earlier undispatched Save reservations
and consumed continuation were not retried or modified. The Save settling and
workflow regression gate passed **163 tests** before Kjin's live Save.

The combined star scheduler now supports four explicitly supplied stellar
classes and a verified No/N/A branch. Main-sequence positive detections can
continue through visible spectrum capture, three reference raw measurements
and four frozen-policy derived copies, then stop for explicit classification.
Positive non-main-sequence calculations remain unsupported rather than filling
in missing stellar mass/radius. **128 scheduler/relay/child tests** pass,
including a fully intercepted browser pipeline; the live combined-path check
is the next bounded development experiment, not a 30-star acceptance attempt.

Fresh-star **134** opened Hisfael: parallax 0.055 arcsec, peak wavelength 636 nm,
flux 1.72e-11 W/m2. An explicit assistant interpretation of the visible H-R
reference selected Main Sequence; no learned classification is claimed. The
first class preflight stopped before any reservation because the exact
three-field autosave footer was not normalized in the six-field comparison.
The narrow layout match now covers both existing footer representations;
measurement/answer/other-notice changes remain guarded. **58 full-stellar/footer
tests** passed before the confirmed two-click inherited-paint setup **138**.

Hisfael's `cooperative-star-142` then completed **60 frozen stellar decisions,
six verified native copies**, and one learned Red color selection. Numeric
readbacks: 59.27 ly, 0.1776 Lsun, 4556 K, 0.6103 Msun, 0.6829 Rsun, 34.37 Ga.
The numeric checkpoint is unchanged with zero optimizer updates; event hash
`ecdf9b80191d7756287fe2457b040a48c3b898e492fe40bda76f857149db55f9`.
Color event hash:
`57b4fac38d767edd9693fd63a70cde1eae13bcdc302afe4248a3c91f617eb81f`.

The combined scheduler stopped after its Planet-tab click. Read-only diagnostic
**209** identified stale exception-class aliases in the long-lived temporary
worker after module reloads; fresh-process scheduler regression passes **81**
tests. Reconciliation **210** confirmed the saved before/after destination and
the identical current Hisfael Planet screen without another click. The original
coordinator remains stopped, its failure evidence unchanged. A separately
recorded window continuation is not retroactively a passed combined run.

The hardened H-R diagnostic remains unwired: **97** schema/geometry/source tests
pass, actual source PNG bytes/dimensions verify, and malformed metadata fails
closed. Its conservative polygons and uncalibrated two-pixel margin were not
expanded to force classifications. Rust library regression passes **67** tests
with its separate Python bridge also passing; three main-binary tests were not
included in that library invocation. No new training or sealed-test evaluation
was performed during these integration checks.

Hisfael observation **211** reached the fixed 60-second start deadline after
committing duration 5000 but **before Play dispatch**. The original claim,
committed capture and Play reservation remain intact; no observation restart,
duration extension or completed-window claim is made. The emitted stop records
`play_click_may_have_occurred: false`. Future-start read-count optimization and
an explicitly bounded, proven-predispatch continuation are being validated
separately; neither silently retries the consumed request.

Visible collection inventory **214** verifies **1–6 of 6**, with six fully exposed
named rows and both pager buttons disabled. This still does not establish the
actual page capacity or enabled pagination behavior. A separate collection-only
importer passed **11 focused tests** (the preceding combined 10-case version plus
legacy evidence/progress tests passed **104**). It added only Hisfael's collected
record: canonical revision **16**, **six collected, two verified, four unresolved**.
Existing scientific and workflow records are unchanged; a second import is
byte-identical. No Save/assessment/submission receipt was manufactured.

Hisfael's explicit pre-Play continuation **218** passed after read-only timing
check **217**. The fixed 60-second continuation delivered exactly one Play click,
performed no duration or answer writes, and preserved fresh current class paint.
The original **211** stop and reservation are unchanged. Historical class paint
was not captured in that source and is explicitly not claimed verified. This
confirms action/readback only; chart progress and any planet decision remain
separate. The continuation's current implementation passed **40 fixtures**,
including exactly three full reads and pre-/post-dispatch deadline checks; the
preceding start/recovery/window combination passed **118**.

The fresh-browser project runtime bridge passed **39 injected lifecycle tests**
and **187 combined controller/component regressions**, with no test browser
launched. Runtime/CLI and Rust wiring are in progress. It still requires explicit
classification evidence and stops at unsupported handoffs; this is not yet an
unattended 30-star runner. The terrestrial final-Save adapter and three-tab
workflow verifier are separately fixture-validated (**56** and **25** cases);
terrestrial journal import and a fresh live proof remain pending.

Hisfael **222–225** passed the agreed negative branch: a guarded endpoint crop
supported the approved 5,000-day reference No assumption; **223** made one
confirmed No selection, **224** delivered one Save and a fresh acknowledgement,
and **225** verified both tabs against the original class/numeric/color and new
choice/Save source chains. Workflow SHA-256:
`44e1af523d80faceec8abb2f8ce803778e7e25948476dc34359fc9e077f67e3d`.
The stopped combined scheduler remains stopped, so this explicit continuation
is not reported as an uninterrupted autonomous run.

Strict ingestion against inventory **214** added six records and passed a
byte-identical repeat import. Canonical revision **21** now has **six collected,
three verified, three unresolved**, with no current-revision assessment, score
transfer or submission. All four frozen checkpoint hashes still match. Root
also reran **138** offline source/ledger/relay/footer checks and **93** injected
project-runtime/controller checks successfully.

`configs/browser_project_checkpoint.json` is a paused one-star integration
profile for the new runtime route, not a 30-star launch profile. Classification
and inventory receipts remain explicit handoffs, and scoring/submission remain
disabled in this route. No credentials or session URL are included in it.

Runtime/CLI integration passes **217 checks** (one opt-in checkpoint skipped and
one deselected); Rust project-mode integration passes **81 tests**, with the
separate legacy Python bridge smoke also passed. Root independently reran all
81 standard Rust tests. An actual CLI start with closed stdin produced a paused
`launch_pending` event followed by abort-on-close, with no browser launch or
credential consumption; run ID `044b8d7b3e654ef294727d0b9045de7b`.

The new profile explicitly sets `browser_no_planet_save_settle_seconds: 30`,
matching live Saves **130/224**. The old default remains 20 seconds, while the
observation-start budget stays 60 seconds. This is a declared preflight budget,
not an automatic extension/retry. **222 offline integration checks** pass with
the setting and Save-abort preservation; the single intercepted Chromium
component fixture passed separately in 47 seconds after its sandboxed launch
was denied. No real preview was accessed by that fixture.

Fresh-star **227** opened Grim with blank data: parallax 0.037 arcsec, peak
wavelength 342 nm, flux 6.69e-10 W/m2. The frozen local common formulas give
approximately 88.1081 ly, 15.2629 Lsun and 8473.01 K. An explicit assistant
reading of the same visible H-R diagram selected Main Sequence; the separately
recorded rationale is not a learned or course-verified classification.

Grim class/prefix setup **228–229** was followed by combined owner **231**.
The first launcher **230** stopped before any action because the temporary caller
supplied a doubly prefixed journal path; correcting the caller to absolute owned
paths did not relax the controller's history boundary. Owner **231** completed
60 frozen learned stellar decisions and six verified native copies by **237**:
88.11 ly, 15.26 Lsun, 8473 K, 2.179 Msun, 1.831 Rsun, and 1.427 Ga as displayed.
The recorded elapsed time includes operator scheduling pauses and is not a
continuous-throughput benchmark. Color/window/Save/inventory remain pending for
this owner; the canonical journal is still revision 21, six collected and three
verified. Grim has not yet been imported as a collected or completed record.

Root reran **154** offline relay/progress/inventory/positive/terrestrial importer
checks successfully, with two intercepted browser cases explicitly deselected.
An earlier unfiltered invocation reached those two fixtures but their Chromium
launches were denied by the sandbox; that was not a passed native-browser gate.
New cooperative class setup and inventory producers separately pass **47** and
**56** offline checks. Their runtime integration and live proof are still pending.

Grim owner **231** subsequently completed color **238** (UV), Planet navigation
**239**, and duration/Play **240** without restarting or extending the observation.
Progress **241–243** reached the 5,000-day endpoint (230 trace columns, seven at
the endpoint), supporting only the approved reference No assumption. **244**
selected No; **245** delivered one Save with a fresh acknowledgement; **246**
verified both tabs. Immutable workflow SHA-256:
`dbc7bc5fb9cac03fac6a86df9e685c615add329064aee7db5a1e13109c1f524f`.

The separate inventory producer **247–248** stopped before its List click:
`Data saved` appeared in the exact final Planet footer between captures. The
original `click_may_have_occurred: false` evidence is preserved. Navigation and
inventory guards now share narrow known-footer normalization; all other text,
inputs, chart content, star identity, controls and target geometry remain
checked. **89 offline checks** and one separately intercepted native footer
fixture passed. The first fixture version used a different AX footer structure
and correctly failed; matching the observed structure fixed only the test.

Explicit one-use, proven-predispatch continuation **249** navigated to the list,
then **250** opened its Stellar tab and read-only **251** verified all seven
visible rows. Both pager controls were disabled; page capacity is still unknown.
Owner import **252** added Grim's real workflow evidence. Canonical revision
**27** now reports **seven collected, four verified, three unresolved**, with no
pending uncertain journal writes and no current assessment/transfer/submission
receipts. A repeat strict import was byte-identical. The failed inventory
producer is not relabeled successful, so this is not a clean unattended full
runtime acceptance run.

Explicit class setup is now wired into the outer runtime through the version-1
`step.reference_class` payload (class, rationale, applicable lifetime prefix),
with separate scheduled class/prefix writes and validated evidence handoff.
The original class-source handoff still works. Fresh runtimes opt into automatic
inventory navigation/import, but neither inference nor scoring/submission is
enabled by this change. Combined offline integration passed **382 tests**; two
real intercepted class fixtures then passed. Rust passed **86 standard tests**,
its separate Python protocol bridge, and Clippy. No model updates or new sealed
evaluation were performed.

Fresh star **254** opened Adraoi with blank stellar fields (0.064 arcsec,
1258 nm, 1.71e-13 W/m2). Explicit reference class owner **255–258** completed
three scheduled actions: clear inherited Main Sequence paint via White Dwarf,
select Main Sequence from the recorded visible H-R reference rationale, then
select Ga. All three passed under the original fixed 180-second deadline.
The owner reports setup verification only, not learned classification or task
completion. Class receipt hash:
`587d8cef687cc7dff287bb5498a9ab9f24c8a423cb7cd7fc8b48b6a31e704516`.
Prefix receipt hash:
`5f52d0cfbb7016178e63c9713d382784006faf79cdb0e6add987fb89bb2d1ab1`.
Known exact stellar-footer status normalization now also applies to this
cooperative setup's comparisons; answer/class/target changes remain guarded.

The positive-planet continuation is now wired through outer runtime options
`project_reference_planet_continuation` (strict boolean, default false) and
`browser_positive_save_settle_seconds` (fixed 0.1–30 seconds, default 20).
The paused checkpoint profile explicitly enables continuation and selects 30
seconds. At `awaiting_planet_class`, `step.planet_class` records a supplied
name/rationale offline; separate subsequent advances select the native class,
initialize the gas/ice finalizer, verify readback, Save once, verify both tabs,
then produce/import inventory. No class is automatically selected. Terrestrial
is still an explicit incomplete handoff until its runner is integrated. The
TUI schedules these phases but never manufactures a classification choice.

Validation: **163** offline outer-runtime/reference-class/options checks and
**167** owner/positive/inventory integration checks passed (overlapping suites,
not additive coverage claims). **88** standard Rust tests, the separate Python
protocol bridge, and Clippy passed. The genuine terrestrial native fixture is
still being validated; no live terrestrial acceptance is claimed. Scoring has
a separately tested offline controller, but no new real assessment or
submission has occurred. The canonical journal remains revision 27 with seven
collected, four verified, and three unresolved; Adraoi is not yet imported.

The isolated genuine terrestrial end-to-end fixture subsequently passed in
70.79 seconds. It generated receipts through real intercepted native controls:
fresh setup/class/Ga, frozen stellar numeric/color inference, raster/spectrum
and raw measurements, frozen planet calculations, explicit terrestrial/gas
decisions, frozen temperature inference, chamber/phase readback, explicit
not-habitable decision, exactly one fresh Save, strict three-tab verification,
native inventory and strict canonical import. No model/source-validator patches,
fabricated receipts, prior Save, training, or real preview access occurred.
Earlier fixture attempts exposed a wrong graph-manifest filename, unsupported
fixture layout and missing displayed Ga suffix; those corrections were confined
to the new test fixture. This is not real terrestrial browser acceptance.

Live Adraoi owner **260** started with automatic inventory and explicit planet
continuation enabled. The temporary launcher **259** failed on an outdated
in-process module import before owner creation or any native action; reloading
the actual dependency resolved the caller issue. No stopped browser action was
retried. Root's combined outer/owner/scoring/progress offline gate passed
**189 tests** (overlaps the earlier suites).

The opt-in terrestrial coordinator subsequently passed **195 injected tests**.
It preserves the legacy terminal terrestrial handoff unless explicit frozen
habitability model paths, graph and gas candidates are supplied. New outer
runtime options pass those sources into separate initialization/active stages;
`step.gases` and `step.habitability` provide reference decisions offline and
require a subsequent bounded step/resume. The paused checkpoint profile now
supplies the existing habitability model and all seven observed comparison
candidates, with a fixed 1,800-second child ceiling and 30-second Save readback
budget. It does not preselect gases, greenhouse increment or habitability.
Root's focused runtime/options gate passed **97 tests**; Rust remained at **88
standard passing tests**, with Clippy clean. These counts overlap earlier gates.

Two additional offline-only components are available: a separate paginated
capture-chain consistency validator (**130-test gate**) and read-only submission
preflight (**156-test gate**). Neither is a browser pager or submission actuator;
neither changes existing inventory/scoring contracts or creates a submission
receipt. Live enabled-pager behavior and actual submission acknowledgement are
still ungrounded. No larger training or new sealed evaluation was run.

Live Adraoi owner **260** completed at **277** with no in-owner failure or
external recovery. Commands **261–265** completed 60 frozen stellar decisions
and six exact copies; displayed results were 50.94 ly, 0.001304 Lsun, 2303 K,
0.1499 Msun, 0.2289 Rsun and 1150 Ga. **266** verified IR; **267–271** navigated
and completed the single fixed 5,000-day observation. **272** selected the
approved reference No assumption, **273** Saved once, **274** verified both
tabs, and **275–276** automatically produced a full eight-row inventory.
Strict import **277** completed the task: canonical revision **33**, **eight
collected, five verified, three earlier unresolved**, no pending canonical
write and no current assessments/transfer/submission. Source hashes:

- Workflow: `a9b586906edf603109752c2af753954d3fee96ba8af3c85e5f7fbf2fbb9fd2b1`.
- Inventory: `c957eea3d154c2cccddd3a09718b4d7b356af19a06bf2513620615287afac468`.
- Numeric manifest: `e695e1e715f9eae4a13c47b9fbf39810c98a30c670ad884e81f06081002543b6`.
- Color manifest: `4ca74b2ca05e96de21b12d56a4ff527135ae1ae54bccd38332e950778da0220b`.

This was assistant-scheduled cooperative execution with an explicit preceding
reference class, not fully learned classification or a fresh 30-star attempt.
Four model checkpoint hashes were reconfirmed unchanged. Combined outer/owner
integration passed **321 tests**; legacy simulator/planet/habitability runtime
regressions passed **37**, with two opt-in checks skipped.

The same run exposed a scheduling inefficiency: advances **71–85** consumed
15 owner-count slots while the child was simply waiting for its next poll.
There was no failure in this run, but the 0.2-second automatic cadence could
exhaust the 512-advance cap before the original 600-second window deadline.
A narrow readiness fix is being tested; budgets and deadlines are not raised.

The readiness fix is now implemented and passed **33 focused tests** plus the
broader offline gates. A fake-clock real-owner/real-star/real-window test makes
1,001 ticks over 200 seconds but counts only **42 work advances** (constructor,
Play, 40 due polls). Idle ticks still check pinned sources, cancellation and
original wall deadlines. The 512-work cap and 600-second window budget are
unchanged. Root's combined readiness/terrestrial/outer-runtime gate passed
**127 tests**. Existing run 260 was not reloaded or reinterpreted by this fix.

Navigation **278** returned to the starfield; **279** opened fresh Avan with
0.025 arcsec parallax, 128 nm peak wavelength and 1.69e-12 W/m2 flux. Common
unclassified tools give 130.4 ly, 0.0844542 Lsun and 22638.8 K. The explicit
visible H-R reference interpretation selected White Dwarf: hot and faint,
not the much brighter main-sequence band at that temperature. **280–281**
completed one class selection and no lifetime-prefix action. This remains
reference classification, not a model prediction or grading answer. Class SHA:
`315187731ceafe73ff77e3f1341b6a0f88684dfa609f3a88b8d3e625627be8ae`.

Fresh owner **282** loads the readiness fix and explicit optional terrestrial
configuration, without changing the browser attempt or frozen checkpoints.
It begins with the three common white-dwarf calculations; Avan is not yet a
canonical collected/task-completed record. The journal remains revision 33,
eight collected/five verified/three unresolved until a verified inventory import.

Avan **283–286** verified all three common numeric fields and UV, with no
main-sequence-only numeric or lifetime-prefix writes. **287–290** navigated,
started one 5,000-day observation, and captured its partial trace. **291**
stopped owner 282 on `unknown_plot_palette`: its saved crop contains one
chromatic antialiased frontier pixel at (162,20), RGB (32,39,42). This is not
a learned failure or evidence of a planet. The owner remains permanently stopped.

The scheduler now allows fresh read-only polls for this particular unresolved
palette state only when the existing public progress reader establishes a blue
frontier short of the endpoint. The final raster decision policy/hash, 600-second
deadline, 60-poll cap, single Play, and 5,000-day range are unchanged. Complete
ambiguous crops still stop. The focused scheduler/policy/readiness gate passed
**106 tests**; broader star/owner/positive/terrestrial/replay regressions passed
**268**, with one native fixture explicitly deselected. An earlier unfiltered
invocation had those 268 pass but the native fixture failed to launch under the
macOS sandbox; that was not a browser acceptance pass or an application assertion.

Independent read-only captures **292–293** did not resume owner 282. The latter
contains the complete trace and passes the unchanged no-dip policy. A separate
explicit recovery intent `avan-window-continuation-294.json` pins the stopped
reports and the new evidence, authorizing no observation restart/extension.
**294** selected the approved reference No assumption from three freshly checked
captures; **295** Saved once and observed a newly appearing acknowledgement.
Strict workflow readback and inventory import are still pending at this note.

Parallel implementation delivered an offline-tested next-star transition
(84 focused /252 combined tests), explicit versioned first-selection provenance
(236 initial/next-star/runtime/protocol tests), and assessment panel navigation
(39 focused /126 combined). These counts overlap. The original single-star
runtime remains the default, and the current legacy attempt is not retrofitted
with the fresh-campaign selection contract. No current same-revision assessment,
score transfer, final submission, new training, or new sealed evaluation is claimed.

Avan strict readback **296** passed. Navigation **297–298** and read-only inventory
**299** verified all nine collected stars on one exposed page. Importing the
separate recovery appended seven canonical records: **revision 39, nine
collected, six verified, three earlier unresolved**, no uncertain canonical
write. Owner 282 remains stopped, not relabeled a successful unattended run.
Workflow SHA `f4e6b8305d055e983d06be85bacc1b981b41de6717039f44aa47f747e994eebf`;
inventory SHA `735b2fbd8cecc5de99971b9393b0b17701c3941d5c4e07bf28092bc6d89d32ba`.

Assessment-panel navigator **300–301** passed its first real read-only check:
the already-open Data Quality panel was verified with **zero navigation clicks**,
zero assessments and zero spending. The displayed 41.6% is the old diagnostic
assessment, not a score for the current revision. Both collection-page buttons
are still disabled at nine rows; enabled pagination remains ungrounded.
Latest focused chart gate is **107 passing tests** after strict type checks;
initial-source plus outer/class/planet/terrestrial runtime regressions passed
**143** (overlapping earlier gates).

Assessment-panel navigators **302–304** and **305–307** then verified a native
switch to Scavenger Hunt and back to Data Quality: one tab click each, no Assess
clicks, no charge, and unchanged $49,800 displayed funding. This validates panel
navigation only, not the charged scoring coordinator or current-revision scores.

Fresh Dand **309** was explicitly classified from the public H-R reference as
main sequence (**310–312**, with Ga lifetime prefix). Its immutable owner **313**
verified all six learned numeric fields and UV, then exercised the partial-chart
wait fix live. Automatic observation **322** took 14 read-only polls and waited
through the unresolved partial palette without restarting Play or widening the
5,000-day window. The unchanged final no-dip policy supported the approved
reference No assumption. **323–327** Saved once, verified the workflow, read the
complete ten-row collection, and imported the task: **revision 45, ten collected,
seven verified, three earlier unresolved**, no pending canonical write.
Owner 313 finished without external recovery; reference classification and
planet absence remain explicitly not learned or scientifically established.
Workflow SHA `6f01489aa624cfcb7d8bbc3538326b8f36b9222f83bf6f290fbd5e15a3902832`;
inventory SHA `7e80f8e64bb8b2c945070c5857d1f567b3591473982f95b8872980d67443e9a2`.

Read-only pager inspection **328** captured all ten rows and both disabled
30-by-30 footer buttons. The rendered arrows are left/right at frame-relative
x 323.34/357.79, y 555, but no enabled transition had yet been observed. A new
star is being opened solely to ground that boundary; it is not a verified task.

The new opt-in campaign has **48 focused /277 combined offline tests**. It keeps
single-star defaults, explicit source-bound reference decisions, frozen model
dependencies, separate bounded next-star transitions, and a terminal
`awaiting_assessment` handoff that never claims project completion. Runtime
integration is under additional test; real 30-star launch remains disabled
pending live pagination acceptance. The two-star profile is
`configs/browser_project_two_star.json`, with an explicit 3,600-second whole-run
budget and unchanged per-star limits; no live campaign has run from it yet.

TUI campaign adoption now clears old star observations, actions, neural traces,
reference choices and child summaries while preserving initial-star provenance.
All **92 standard Rust tests**, the real Python subprocess/replay bridge, and
Clippy pass. These are protocol/UI gates, not native campaign or course acceptance.

The next native diagnostic opened **Gamor (330)** with blank answers, then
returned to the stellar list (**331**). Inspection **332** shows 1–10 of 11 and
the exposed right footer button enabled. Native one-click probes **333/335**
went forward to 11–11 (only Gamor) and back to 1–10, preserving every captured
row hash. The footer's centered group shifts horizontally with range-text width;
direction is grounded by the rendered arrows and ordered native pair, not fixed
absolute coordinates. Separate reinspection **334** captures the second-page
geometry and its left-enabled/right-disabled state.

After forward probe **336**, exact native panel clicks **337/338** switched
Scavenger Hunt then Data Quality. Both preserved page 11–11 of 11, the visible
row, funding and existing scores; neither clicked Assess. These are public
navigation probes, not a complete paginated inventory receipt or a Gamor task.
Canonical counts remain ten collected/seven verified pending a validated
full-collection import. The new live paginated producer and its consumer path
are being implemented separately from the prior offline consistency validator.

The controlled submission Chromium fixture passed **13 native tests** after
correcting fixture-only duplicate-widget routing (initial request frame names
were unavailable). It exercises the real native preflight, exposed checkbox,
stable handles, durable canonical reservation, one Submit, callback abort,
overlay/rebinding refusal, and guarded post-click diagnostics. Every request is
locally intercepted; actual eligibility sources are explicitly synthetic seams.
The result remains `unknown_pending` even for generic “Project submitted” text.
No real project submission, acknowledgement, course score or learned success is
claimed. The actual preview was idle throughout this fixture run.

The opt-in campaign runtime passed **273 combined offline tests** and a final
**99-test protocol gate** (overlapping counts). Integration review fixed an
unpaused-start child scheduler stall and checks evidence again after the final
summary callback. TUI terminal handoffs now retain their specific assessment or
pagination guidance while still disabling further steps. No new training or
sealed evaluation ran.

### Paginated inventory integration gate (September 26, continuation)

The separate live producer now has explicit first-page, forward traversal,
reverse revisit and final first-page anchor proofs. The strict shared inventory
accessor and all four importers passed **304 offline tests** (two native fixtures
excluded). The inventory wrapper, next-star and campaign consumers passed
**464 offline tests** across two overlapping suites. Legacy single-page source
and binding bytes remain unchanged; an offline page-chain consistency report
still cannot impersonate a live receipt. These tests are not live acceptance.

Native attempt **340** stopped before any pager dispatch
(`navigation_dispatch_attempts: 0`). Its first capture contained the exact
paired list-footer **Data saved** notice, which disappeared before advance two.
Read-only inspection **341** confirmed the same 1–10 of 11 page, rows, funding
and scores; the notice was the only comparison difference. The failed directory
is retained at its accidentally nested temporary-harness path:
`experiments/full-stellar-probe/20260926-005/experiments/full-stellar-probe/20260926-005/paged-inventory-340`.
The harness now resolves output paths before passing them to the source owner;
the failed evidence was neither moved nor rewritten.

The fix waits read-only for that exact paired footer notice to clear, for at
most **three seconds per read inside the unchanged component deadline**. It
does not strip text, normalize hashes, retry a click or extend a budget. Raw
page/row/source comparisons and assessment hashes remain strict. **165 pager
and offline-chain tests passed**, including notice timeout, cancellation,
popup/deadline stops, preservation of changed rows and unchanged raw evidence.
The replacement live inventory attempt and native fixture gate remain pending.

Replacement native inventory **342** passed: five bounded advances, two pager
clicks, all eleven rows forward/revisited, and the actual first-page anchor.
The strict accessor independently reloaded the entire immutable source tree.
Receipt SHA `102117829cc28249f65a28e4a597f1726080f10f20d00a0f0db18e3e680697bc`;
whole-collection SHA
`d00cfd5dae9a0947ccc2fc177aadc55190c768076f8057563c66e3b99be1fb6a`.
Collection-only canonical import added Gamor without any task/science/write
receipt: **revision 46, eleven collected, seven verified, four unresolved**.
The fresh campaign, full 30-star charge/submission and terrestrial live gates
are still pending. The separate native fixture gate exposed fixture scaffolding
issues and is being repaired; it does not invalidate the recorded preview pass.

Panel attempt **343** stopped before dispatch on a transient row-fingerprint
change; independent inspection **346** again exactly matched inventory342's
text, accessibility and row hash. The shared bounded notice wait is now used
before panel reads, with raw changed-row diagnostics retained on any later
mismatch. **347** was an undispatched temporary-harness module-reload failure;
the producer export was refreshed before the replacement component.
Production panel runs **348/349** then verified Scavenger Hunt and Data Quality
respectively using inventory342: one tab click each, preserved first-page rows,
unchanged $49,800 funding and prior scores, and **zero Assess clicks**.
The pager/panel focused regression gate passed **149 tests**. Low-level scoring
settling is being wired as an explicit new-mode opt-in; legacy hashes and
captured content remain unmodified.

The isolated native pagination fixture gate now passes **8/8**. It exercises
real 11/30-row traversal, native enabled/exposure/identity guards, exact-notice
settling and strict source reloading, with every request intercepted locally.
Fixture-only fixes were needed for Playwright's callback wrapper, nonzero text
rectangles in the footer and AX separation between a row and the notice; no
production validator was relaxed to make those cases pass.

### Assessment regression and shallow-chart stop (September 26)

The paginated assessment/transfer notice settling is explicitly opt-in and
retains raw capture hashes, unchanged deadlines and at-most-once writes. The
combined offline scoring gate passed **143 tests**, with a final **39-test**
settling gate (overlapping counts). Root then ran all **50 native assessment,
assessment-stage and score-transfer fixtures**, passing in 18.37 seconds with
all network requests intercepted. The actual preview was idle; these fixtures
did not assess or transfer its score.

Gamor's new reference class decision **352** used the exposed course H-R card
and current common calculations. Main sequence is an explicitly tentative
cool/faint reference candidate, not learned classification or a validated
course boundary. Class setup **352–355** and frozen owner **356** completed
all six numeric fields and IR color transport with zero optimizer updates.
The owner then opened Planet and started one 5,000-day observation.

Observation batch **365** stopped after fourteen polls with
`planet_window_visual_evidence_unresolved`; the owner stopped at advance 83.
The final raster reached the requested endpoint but was not classified as
No or as a supported positive dip. No planet answers, Save, assessment or
canonical task completion followed. Revision 46 remains eleven collected,
seven verified and four unresolved.

Offline replay of `project-owner-356/star/window/progress-013` exactly reproduces
both stored analyses. The negative policy rejects three antialiased pixels;
the positive detector explains their colors but finds only one original-blue
pixel below baseline, short of its independent multi-row component requirement.
Two shallow candidate locations are visible; even positive detail would not
satisfy the existing three-component raster measurement requirement. The chart
SHA is `3382236867c4d6fc893b16504c7d6e8bbcd1e7a5faa06aaa45a41c34feeb67a4`.
The **73 policy/detector tests** pass unchanged. No palette widening, fabricated
absence, measurement-rule relaxation, new training or sealed evaluation ran.

The next diagnostic is an explicitly separate, single ordinary wheel zoom of
that same saved/current chart, with visible axes and crop evidence retained.
It does not restart the observation, extend beyond 5,000 days, revise the
stopped owner or grant authority to write a planet answer.

Read-only capture **366** confirmed the same Gamor Planet screen and blank
planet answer fields. The separate one-action detail diagnostic **367** made
one ordinary wheel zoom and saved before/after crops, native event evidence and
visible axis labels. Its narrower view did not resolve a usable measurement:
the visible tooltip and densely packed axis glyphs prevented a complete exposed
time-axis read. No daily scan, restart, answer selection or measurement claim
followed. The original failed owner and 5,000-day evidence remain immutable.

The native pagination suite now passes **11 tests**, including explicit
viewport geometry. Both 1600×1100 and 1280×720 pass with the simulation embedded
at x=325, y=100. A separately clipped y=200 embedding at 1280×720 stops before
dispatch. The first test expectation incorrectly assumed the smaller viewport
alone caused clipping; the test was corrected to distinguish measured fitting
geometry from the actual clipped case. No production visibility rule changed.
An explicit project viewport option is being added for reproducible parity with
the diagnostic, not as a proven explanation of any prior live failure.

The optional `project_viewport` implementation now passes **269 combined offline
runtime/CLI/campaign/reference/planet/terrestrial tests**, including its 34 focused
cases. Omission preserves the legacy context call; the two-star profile requests
1600×1100 before page creation and records that fixed geometry. Changed requested
or actual geometry stops without resizing. Lint and format checks passed.

Fresh production JSONL campaign `ebc9eb3a887741fb88cc7c7380fa5138` has been
started under `experiments/browser-project-two-star`. This is a new local preview
context, separate from diagnostic attempt `20260926-005`; its frozen models,
3,600-second campaign budget and 1,800-second/512-advance per-star budgets remain
unchanged. Setup and current-star reference handoffs are still required; launch
is not acceptance. Credentials are supplied through private process input and
consumed before Chromium launch, not stored in the source or artifacts.

The production campaign verified **Crabiltia** through six frozen numeric copies,
color, one 5,000-day approved no-visible-dip assumption, Save, detailed readback
and strict collection import. It then automatically selected **Gisesius** and
paused on its distinct fresh class source. Canonical revision 6 has one collected,
one verified, no unresolved task and no pending action. Gisesius's hot/faint
visible measurements (141 nm, 0.026 arcsec, 1.03e-12 W/m2) support an explicit
reference white-dwarf candidate from the course H-R card, not learned class
prediction. Its source is `campaign/transitions/001-to-002/picker`; its separate
reference decision was supplied and the same campaign resumed. Star two is
not yet a verified task at this checkpoint.

The actual outer production trace revealed display-only replay defects. EOF
now retains the latest recorded star/progress with `replay_finished` and
`recorded_status`, rather than restoring initial launch context or claiming task
success. Neural source reads current `source` and legacy `activity_source`.
Meaningful stage changes clear prior observation/neural history across transient
null envelopes. The header explicitly says **configured stellar checkpoint**;
it does not guess an active component checkpoint absent from the event.
Validation passed **189 Python tests** (two optional skips), **96 Rust tests**
(one installed-environment integration ignored), Clippy, lint and format checks.
A streaming prefix of the actual outer trace independently verified these fixes
without loading the growing file. Replay the outer sibling `.jsonl`, not the
bridge's internal `events.jsonl`, once its writer is stable; replay does not tail
or attach to a running browser campaign.

One scaling problem remains under investigation: after the first workflow, the
outer and bridge traces are approximately **128 MB each**. Their duplicated
payloads must be understood before a 30-star launch. This is logging overhead,
not additional model training or accepted task evidence.

The fresh two-star campaign has now **passed** its scoped live gate. Its terminal
report is `awaiting_assessment` / `handoff` with `target_workflows_verified: true`.
Canonical revision **12** records **two collected, two verified, zero unresolved
and zero pending actions**. Crabiltia exercised six-field Main Sequence work;
Gisesius exercised applicable-field White Dwarf work. Both used the approved
5,000-day no-visible-dip reference assumption, not learned planet absence.
The automatic fresh-star transition, separate source-bound class decisions,
Save/readback/import and final campaign handoff passed. No scoring, score
transfer or submission was performed. A fresh live terrestrial workflow remains
pending; this two-star pass is not a claim that all interaction families passed.

The completed two-star terminal report SHA is
`59a193b0369e07a8767b705ea16a9ecdc32312e3e3d1fc6e59735cddcd9919`.
Its browser is retained at handoff; it is not resumed or relabeled as a 30-star
attempt. The separate 30-star profile now passes **255 offline tests**, including
26 new launch-contract cases. Only explicit campaign targets 2, 3 and 30 are
accepted. The profile fixes 10,800 seconds overall and retains 1,800 seconds /
512 advances per star; it starts paused with a 1600×1100 viewport. This is launch
support, not a live 30-star or scoring/submission pass.

The compact display projection's independent Rust review streamed the fixed
first-star prefix of 1,700 events. All **544 rendered view comparisons** matched
the legacy stream, and all **301 native data envelopes** reconstructed exactly.
Replay EOF retained the current star, canonical progress and false task/project
completion. No TUI changes were needed for this projection. Helper regression
tests remain pending; raw component and bridge evidence journals stay unchanged.

In diagnostic attempt `20260926-005`, bounded navigation 368–371 returned the
stellar list from page 11–11 to page 1–10. Fresh selection **373** then selected
Ayistash from an ordinary visible-star anchor at (0.20, 0.65), without catalog or
planet filtering. The fresh stellar capture SHA is
`071a914329bf2df646ce1cc1b50736f816e794b454de935337e120a5b3718770`.
The visible measurements (0.278 arcsec, 383 nm, 1.69e-08 W/m²), common tools and
the course H-R card support an explicit main-sequence reference candidate.
Class setup **374** completed in three advances. Frozen owner **378** is now
testing the whole workflow under its unchanged 1,800-second/512-advance limits.
Scheduling slices do not extend those limits or supply reference decisions.

Ayistash owner **378** verified all six numeric copies and color, then stopped
truthfully at the 5,000-day window endpoint after nine polls (**382**):
`planet_window_visual_evidence_unresolved`. Both raster checks reject unexplained
plot colors in `star/window/progress-008`. No planet answer, Save or completion
receipt followed. Its original stopped owner remains immutable. This is a second
shallow/ambiguous rendered-chart case to diagnose, not a no-planet success.
Canonical revision 46 remains historical at eleven collected/seven verified;
the browser's newly selected Ayistash has not been imported as a completed task.

Root's focused offline continuation/detector regression passed **281 tests**.
One additional native fixture initially could not launch in the macOS sandbox;
it was explicitly excluded from that offline count and is being rerun separately
with both real browsers idle. No application assertion failure was observed.

The completed compact-wire gate passed **229 Python tests**. The exact runtime
projection of the fixed 1,700-event prefix is **127,497,438 → 10,084,361 bytes**
(92.09% smaller), with a maximum line of 13,196 bytes. These implemented numbers
replace the initial estimate. Default legacy behavior and raw evidence journals
are unchanged; future two-/30-star profiles explicitly opt into compact output.

The previously sandbox-blocked, fully intercepted native positive-child fixture
passed separately (**1 test, 51.44 seconds**) with both real previews idle.

Offline Ayistash crop diagnosis identifies a rendering/evidence limitation:
unchanged baseline pixels appear in multiple progress frames, while the number
of exposed gray grid pixels drops below the positive detector's required support
as the trace fills. No bright-neutral overlay signature is present. There are
also three separate one-row-deep candidate groups, so this is not a justified
No decision. The frozen negative and positive rules remain unchanged.

Separate chart-only diagnostics **383/384** made three zoom-in wheel actions and
one small zoom-out within the same completed 5,000-day observation. They expose
a shallow feature and restore readable horizontal tick spacing. No observation
restart, form write, planet answer, policy recovery or measurement claim occurred.
All original owner378 failure evidence is preserved.

Chart-only vertical pan **385** made the tooltip readable. Read-only axis capture
**386** exposes eight linear ticks from 2,200 through 2,900 days. Three bounded,
axis-checked hovers **387–389** then reported **day 2335: 100%, day 2338: 99.238%,
day 2342: 100%**. This is genuine visible evidence of a shallow dip, stronger than
the original overview crop; it is not a period, minimum-depth guarantee, completed
planet measurement set or planet answer. No form, Save, score or submission was
changed. The original owner remains stopped. The result motivates a separately
bounded zoom/tooltip follow-up for shallow candidate features, not relaxing the
frozen no-dip policy or retroactively turning the failure into a success.

The separate `browser_shallow_transit_probe` diagnostic now reproduces that
bounded interaction recipe without writing answers: a caller-selected visible
candidate, four ordinary wheel actions, at most one vertical pan and at most
eight axis-checked neighboring hovers, under a fixed 16-action/180-second guard.
It requires explicit report/PNG hashes and exact current overview/axes before
the first action. Independent review caught an optional-hash loophole; explicit
64-hex pins are now mandatory before creating output or constructing a session.
**34 offline tests** and **two fully intercepted native Chromium tests** pass.
The first native fixture used a fractional horizontal embedding and produced an
unsupported 281-pixel crop; only the fixture placement was corrected. Production
crop restrictions were not widened. The diagnostic is not connected to answer
entry, period estimation, owner recovery or task completion.

The source-linkage regression now distinguishes an actual visible decline from
confirmation of the original overview hint. Ayistash's day-2338 reading is
outside the selected hint's approximately 2543–2609-day interval. Affine axis
checks preserve the zoom anchor within 0.00005 day, so an adapter x-coordinate
offset is not supported by the evidence. The probe records
`visible_decline_not_source_linked` instead of claiming correspondence. All
**35 offline tests and four intercepted Chromium cases** pass, covering both
linked/unlinked features and optional baseline panning. No historical owner or
frozen detector was changed.

The separate post-campaign finalization runtime is now implemented: explicit
30-star scoring/submission flags, preserved historical campaign evidence, fixed
600-second scoring and 180-second submission budgets, and the existing
pause/step/abort controls. The integration agent reports **317 Python tests**
passed. Root reran the TUI suite: **99 passed**, with one installed-environment
integration intentionally ignored. No live assessment or submission followed
from these fixtures. Without an authoritative visible acknowledgement, the
submission state remains `unknown_pending`, not successful completion.

### Daily-tooltip reference path and fresh acceptance (September 26, continued)

The shallow diagnostic now uses the separately versioned `daily_focus_v2`
recipe: six bounded zoom steps, optional vertical centering, explicit pointer
clearing, and consecutive integer-day tooltips. Four-day sampling missed a
one-day decline and is not used by this new recipe. Frozen positive/negative
raster detectors and trained checkpoints remain unchanged.

Live diagnostic sources **410, 412, 408** independently bracket declines at
days **1160, 2338, 3516**, respectively. The pinned validation receipt is
`experiments/full-stellar-probe/20260926-005/tooltip-reference-validated-001/report.json`
(SHA256 `f6e969360a9ecf188d99559527ca28c5b81914a02bfecb3150d3e58d924fe964`).
It reports an observed recurrence of **1178 days** with an open bracket-compatible
interval (1177, 1179), and a maximum sampled decline of **0.762%**. These are
approximate reference measurements, not verified physical period/minimum depth,
learned decisions, training labels, or completion. No answer was written by
these diagnostic probes. Owner378 remains failed and immutable.

The restored-overview identity retains complete PNG/report hashes and compares
exact plot RGB pixels, axis geometry and rational affine time mapping. Its only
grounded exception is the zero-axis glyph/gutter outside the plotted data:
actual sources401/409 differ there but have byte-identical plot pixels. No pixel
tolerance or frozen detector relaxation was added.

`ShallowTransitSteps` integrates three probes and four restorations under fixed
900-second / 64-native-action / 80-advance caps, inside the existing parent
1,800-second limit. The explicitly opted-in parent then passes the independent
measurement receipt to the existing native answer-copy and frozen calculation
paths. Pause/abort applies between bounded calls; no failed owner is resumed.
The TUI labels sampled decline and bracket compatibility without fabricated
physical bounds. Legacy evidence/replay defaults remain supported.

Root's current-source checks passed **182 focused offline tests**, **189
parent/window/runtime tests**, **84 launch/runtime tests**, the two existing
native positive-child regressions, and **12 fully intercepted Chromium tests**
for the new tooltip transport (including frozen derived-policy inference).
These suites overlap and are not additive counts of unique coverage. Native
fixture tooltip source bundles are synthetic; actual measurement evidence is
the separately pinned live diagnostic set above.

The new `configs/browser_project_shallow_three_star.json` profile explicitly
enables this reference path, targets three stars, retains per-star caps, uses a
5,400-second campaign cap, and disables assessment/submission. The original
two-star profile is unchanged. Its first fresh launch
`e37f811e5bbc41a48545aa110a53bed8` stopped during login with
`setup_browser_operation_failed`, before star selection or any answer writes.
It is preserved as a setup failure, not a policy failure or accepted live gate.
A credential-safe login diagnostic and fresh reproduction are next. Full
terrestrial acceptance, 30-star scoring, score transfer, and a grounded visible
submission acknowledgement remain pending.

The second fresh launch `868b364a3b7f4788826ba43af3fd8a4b` isolated the
setup failure to `setup_login_submit_click_playwright_timeout_error`. The new
fixed-label diagnostic records the failed operation/error family, and future
submit timeouts distinguish only allowlisted destination/click-stage categories;
it never emits credentials, raw driver messages, or authentication-page captures.
No login guard, timeout, or retry behavior changed. Root verified **65 direct
diagnostic tests** and **seven intercepted native login/form/redaction cases**.

The third fresh launch `a841534eb81a49ba82b97d799bd8238a` authenticated and
selected **Jhako** without reproducing the timeout. The historical intermittent
login failure therefore remains unresolved, not claimed fixed. This active
three-star run uses the new opt-in profile; fresh stellar measurements and the
public H-R card support an explicit assistant main-sequence/Ga reference choice.
The choice is not a learned classification, grading answer, or scientific proof.
Root also reran the TUI gate: **103 tests passed**, one pre-existing installed-
environment integration ignored, and strict Clippy passed.

### Three-star stop and first cooperative sensor diagnostic

The `a841534eb81a49ba82b97d799bd8238a` campaign is now **stopped**, not active
or accepted. Jhako and Ayissa completed their approved bounded-window No
workflows, Save/readback and inventory verification. Canonical revision 12
contains two verified stars. Ayissa's captured H-R helper abstained; its
assistant-selected main-sequence/Ga continuation was explicitly recorded as
uncertain reference assistance, not learned classification or a verified answer.

Kjinson passed numeric/color transport and the bounded No choice, but its Save
owner stopped with `no_planet_save_stale_acknowledgement`. The exposed footer
changed from absent to present during the roughly 7.5-second final guard gap.
The captured task controls and values did not change. The durable stop records
`reservation_created=true`, `save_may_have_occurred=false`; no Save dispatch
exists. This third star is not imported or counted as verified. No scoring,
submission, training update or retry followed.

Coordinator error propagation incorrectly relabeled that ordinary child stop
as `event_forwarding_failed`. A full read-only audit reconstructed all 908
native envelopes in 4,373 raw events without compact-transport failure. The
narrow coordinator patch preserves the first sanitized causal stop, rejects
subsequent action proposals, and still distinguishes actual callback failures.
Its 321-test offline gate passed; root independently reran 126 focused
coordinator/budget tests. The original stopped artifacts are unchanged.

The standalone Ayistash chart diagnostic `shallow-owner-414` also stopped:
after five advances and three confirmed chart-only actions, its first redundant
overview zoom produced `raster_planet_unverified_settled_overview`. Original
source 413 had 230 trace columns and a visible endpoint; the restored capture
had only six detected trace columns and no verified endpoint despite matching
axis labels. No probe measurements, form answers, Save or owner recovery were
performed. This is a failed cooperative-reader gate, not a successful extension
of the earlier independent daily-tooltip measurements. The frozen detectors
and historical source/receipt files remain unchanged.

For **future fresh owners**, the cooperative reader now verifies two exact
initial captures without any preliminary zoom. Three subsequent restorations
retain the full strict plot/axis comparison and use the explicitly recorded
`post_probe_center_0_5_0_4_v2` pointer recipe. The 900-second, 64-action and
80-advance caps did not increase. The correction passed 353 offline/injected
regressions; it does not retrospectively complete diagnostic 414 or establish a
fresh live shallow-planet workflow.

New star-session Save calls explicitly select
`bounded_read_only_pre_dispatch_settle_v1`. This retains the one canonical
reservation and, before any dispatch, may wait for the existing known notice
to clear and repeat the full guard. A single original 20-second final-phase
deadline covers waiting, click, acknowledgement and final readback. The legacy
Save API keeps its old default behavior. Late acknowledgement/readback,
cancellation and source/control changes still stop; no second click or failed
owner recovery is introduced. The intercepted Chromium gate passed 140 legacy
and 27 focused cases. The new diagnostic files must also be adopted into the
workflow evidence chain before a fresh acceptance launch.

Root additionally verified 153 sensor/parent/profile/replay cases and 206
runtime/replay cases (overlapping suites). One mixed-suite browser fixture
could not launch inside the macOS sandbox; it is pending its separate native
run, not reported as passing. Hashes of all four frozen checkpoints still match
the recorded stellar/color/planet/habitability identities. Completed/stopped
owned browser processes were closed normally; their saved evidence remains.

The corrected cooperative sensor then passed **two fully intercepted Chromium
fixtures in 21.82 seconds**. The successful synthetic three-dip case reached
`measurements_ready` in 54 advances with 39/39 confirmed chart actions and zero
answer writes. Its independent validator reproduced the fixture's observed
1,000-day recurrence, open (999, 1001) bracket compatibility and 0.762% sampled
decline with no physical-depth bounds. The mutation case stopped at 15 advances
and 10 actions before another action or restart. These are real native fixture
executions, not a live HabWorlds shallow-planet success or new learned scores.

Root's new No-Save/workflow compatibility fixture and existing positive-child
handoff fixture also passed **2/2 in 49.59 seconds**, outside the macOS sandbox
with all network requests intercepted. The opt-in No workflow now independently
validates and pins all settlement diagnostics; its 65 focused pure evidence
tests passed, and preserved legacy Jhako/Ayissa source sets stayed exactly
40/44 hashes respectively. Review prompted a same-raw-bytes reservation check
and hash pin, preventing a change between those two reads from being adopted.

The same footer race was found in production gas/ice and terrestrial Save
owners. Their equivalent opt-in fixes and independent evidence checks are in
progress; production parent forwarding tests passed 103 cases. No fresh live
attempt has yet exercised these changes. All historical failed reservations,
checkpoints and training budgets remain unchanged.

The gas/ice and terrestrial fixes subsequently passed **96 offline tests** and
**129 fully intercepted native regression tests in 650.32 seconds**, with all
fixture browsers closed afterward. Root also passed **29 native No-Save
settlement cases**, including reservation mutation and JSON boolean/number alias
rejection, plus **373 overlapping coordinator/runtime/replay/parent tests**.
These checks do not establish a successful live campaign.

A fresh three-star attempt, `723a0f23f2454d8792187c1865cc939d`, now exercises
the released changes. It retains the original limits and frozen models, with
scoring/submission disabled. A temporary console filters terminal display only;
the production runtime still persists the complete canonical version-1 event
stream. Historical failed owners are not resumed or relabeled.

Attempt `723a0f23f2454d8792187c1865cc939d` subsequently stopped on **Athe**,
before any shallow chart action, planet answer, Save, or inventory import. Its
stellar policy completed 60 decisions and six verified copies in 59.22 seconds,
with an unchanged checkpoint. After 14 chart polls the full 5,000-day overview
exposed 48 narrow rendering-feature groups; the shallow hint helper rejected
the count with `shallow_probe_too_many_candidate_groups`. The causal stop now
propagates correctly through the campaign rather than being masked as a relay
failure. The owned browser was closed normally; the failed owner is immutable.

The saved PNG contains 60 hint pixels, all on row 21 and 1.05 pixels below the
visible baseline, in groups separated by 4.5–5 pixels rather than the 23-pixel
grid spacing. These are rendering hints, **not confirmed transit measurements**.
The frozen detectors remain insufficient. A future opt-in hint-selection patch
will inspect only the first three raw groups, without skipping or repairing
selected groups, and require the same guarded native daily-tooltip brackets.
Neither the action/time budgets nor the detector/measurement equations change.

The explicit `first_three_overview_groups_v1` implementation now selects
Athe's saved columns `[34]`, `[38,39]`, `[43]` as search hints only. The legacy
helper still rejects the same 48-group image, and both frozen detectors remain
`insufficient_visual_evidence`. Selected groups are never skipped, repaired,
or replaced; broad/edge selected groups still stop. Later groups do not acquire
an inferred planet/absence status. New diagnostics pin the selection policy,
and the independent reader re-derives selected columns and visible-axis hint
intervals, verifies actual tooltip linkage and requires indices 0, 1, 2.

The focused probe gate passed **67 tests**; root's independent reference,
project-evidence, replay and shallow-owner gate passed **167 tests**. The saved
Ayistash daily-tooltip reference still reloads to the exact same complete JSON,
all 63 source hashes, and unchanged measurement-method hash. This does not
resume either failed owner. The new native fixture/live acceptance gates remain
pending at this entry.

The selected-hint cooperative gate then passed **38 offline tests** and **three
fully intercepted Chromium cases in 39.30 seconds**. Both sparse three-hint and
dense 48-hint cases used the real native probe and independent validator,
finishing in 54 advances with 39 chart actions and zero answer writes. Their
respective 1,000-day and 100-day sampled recurrence fits stayed reference-only.
The visible-answer mutation case stopped before another action. All fixtures
closed their browsers; limits remain 900 seconds, 64 actions and 80 advances.

Fresh attempt `4718f4b74128488287f7926eeb5eaf7e` started after those gates.
Its first star is **Brel**. The visible measurements yield approximately
2,515 K and 0.002482 Lsun; the H-R helper abstained outside its digitized
interiors. The assistant explicitly recorded an uncertain main-sequence/Ga
reference choice rather than labeling it a learned or course-verified answer.
The frozen numeric stage passed 60 decisions and six native copies. Planet,
Save, inventory and whole-attempt outcomes remain pending at this entry.

Brel subsequently passed the full 5,000-day approximate-No path, one Save
dispatch with a fresh notice, independent workflow verification and collection
readback. Canonical revision 6 contains one verified star. Its new reserved
Save policy needed one full revalidation and no footer-wait loop; this confirms
the live opt-in path, not a live reproduction of the notice-race branch.

The attempt then stopped on **Aradi** during fresh class setup, before numeric
or planet work on that star. The inherited Main paint requires the existing
explicit intermediate White Dwarf selection before setting Main again. Its
child recorded `unsupported_class_circle_rendering`, one proposed class action
and one possibly dispatched click, but no successful class receipt. The outer
runtime masked this as `project_runtime_class_setup_unverified`. The saved
counter establishes that pre-click binding returned; exact unsupported paint
properties were not recorded. Neither the class action nor the campaign is
retried, and the owned browser has been closed normally.

Failure-only allowlisted paint diagnostics and faithful outer causal reporting
are being added for a short fresh two-star class-transport reproduction. This
does not relax accepted paint, repeat an uncertain action, or change budgets.

The class-runtime causal fix passed **392 offline tests**: valid same-star
stopped reports preserve their sanitized cause, and cleanup failures cannot
replace it. Failure-only paint diagnostics passed **77 tests**, including
allowlisted-value redaction, exact read/dispatch phase, unchanged successful
payloads, and terminal behavior when diagnostic persistence fails. The fresh
`experiments/class-rendering-diagnostic-001` probe now isolates two class setups
without calculations, Save, scoring, submission or resuming a failed owner.

Diagnostic 001 stopped before its first class click because the temporary
console callback incorrectly expected an event object instead of the serialized
dictionary. That helper-only error was corrected and tested; the failed owner
was not resumed. Fresh diagnostic **002** then completed the first class setup
on Junnaenda and reproduced the second-star failure on Ellaen. The returned
White Dwarf click was followed by a read of the old Main dot at opacity
`0.00683295`, with its cyan border and transparent background already in the
unselected state. A later read-only snapshot shows exact unselected Main and
selected White Dwarf paint. This establishes a post-click fade-out race, not a
missed click. The diagnostic browser closed without Save, scoring or submission.

A narrow read-only settlement fix is in progress. It must retain exact final
paint validation and one click, fail on unrelated changes, and remain bounded
inside the existing class-setup deadline. Neither historical failure is retried
or relabeled as a successful setup.

The paint fix passed **93 offline cases**, **five intercepted native class
cases**, and **18 native planet-classification regressions**. Only known
fractional opacity on the old/new class circle is eligible for a fixed
two-second/40-probe read-only wait. Exact final paint, native handle identity,
full public-state checks and the original parent deadline remain required;
successful event payloads are unchanged. Permanent animation and unrelated
answer mutations stop after the single original click. Root also passed
157 overlapping pure class/reference/planet-guard cases; an initial sandboxed
native launch could not create a browser, and the 18 native cases were then
run successfully with authorized process permissions. Fresh live diagnostic
003 is now running; its result is not yet established here.

Live class-only diagnostic **003 passed both fresh setups**: Garodhy used one
class click plus Ga; Astasanor exercised the inherited Main -> White Dwarf ->
Main transition using exactly two class clicks plus Ga. Both final reports
verify setup; neither claims scientific correctness or a completed stellar
task. Its browser closed normally, with zero Save/assessment/submission.
Fresh three-star campaign `80e1d3dff5e4474984783a5273ed7594` now starts with the
released fix, unchanged frozen models, original caps and scoring/submission
disabled. Earlier failed campaigns remain immutable.

Campaign `80e1d3dff5e4474984783a5273ed7594` verified **Demtrios** through all
six numeric copies (60 decisions, 57.97 seconds, checkpoint unchanged), color,
the approved full-5,000-day approximate-No branch, Save acknowledgement and
independent collection readback. Canonical revision 6 contains one verified
star. Its next star, **Gorasaz**, stopped on the returned intermediate White
Dwarf click: the old Main circle has the same cyan/transparent geometry but
`dotOpacity` was rejected by the diagnostic's string grammar and recorded as
null. The settlement loop therefore did not run. Scientific notation for a
very small opacity is a hypothesis, not yet recovered live evidence. No Gorasaz
numeric/planet/Save operation was run. The failed owner remains immutable and
its browser closed normally. A native serialization fixture is the next check;
exact final paint and single-click limits will remain unchanged.

The isolated Chromium CSSOM fixture confirmed computed opacity strings
`1e-07`, `1e-12`, `1e-38` and `1.4013e-45`. Diagnostic parsing now accepts
bounded unsigned numeric notation with exponent magnitude at most 45 and an
exact Decimal domain of 0..1. NaN, infinity, negatives, malformed/overlong
values, extreme exponents and values just above one remain rejected. This does
not recover or assert Gorasaz's original spelling. Exact final painted-state
acceptance, the single original click and the two-second/40-read limit are
unchanged. The final gate passed **117 offline tests** and **seven intercepted
native tests**, including an actual `1e-07` old-Main dot settling to exact zero.

Fresh three-star campaign `cab28dfc50aa4e5396831446b10e6185` starts on
**Bestiniri**, after those gates, with unchanged models and limits. The public
measurement capture `0112e0a715af7bb1bde327fcc98a13eb7bccb95e546bd204dd31c70959d06b83`
yields about 6,205 K and 1.632 Lsun; the assistant supplied the diagram-based
Main/Ga reference choice, not a learned classification or grading answer.

Failure-only submission diagnostics now preserve permitted public outcome
text/accessibility and the current viewport under the original deadline. They
allow outcome shape changes relative to the pre-submit screen but pin current
visible frames and URLs throughout the capture. Authentication, unknown visible
frames, popups and stale/hidden frames refuse content. Native dialogs retain
only their type; no prompt text, acceptance or dismissal occurs. Interruptions
cannot replace the original failure or lose its pending reservation. This adds
no acknowledgement recognizer, success claim or retry. The existing normal
post-submit capture path and its payloads remain unchanged. Validation passed
**152 offline tests** and **five intercepted Chromium tests**.

Root replayed the complete failed `80e1d3dff5e4474984783a5273ed7594` outer
stream with socket connections and live runtime/browser start explicitly
forbidden. All 1,759 replay emissions completed, source bytes stayed unchanged,
and final playback preserved Gorasaz, one verified star, the exact class-rendering
failure and false task/project completion. Replay termination did not promote
the failed browser attempt to success.

Campaign `cab28dfc50aa4e5396831446b10e6185` verified **Bestiniri** through
six numeric copies (60 decisions, 60.57 seconds, checkpoint unchanged), color,
the approved full-5,000-day approximate-No branch, Save and independent
collection readback. Canonical revision 6 contains one verified star. On
**Arthinna**, both inherited-class transitions were confirmed, but the single
Ga selection stopped at `screen_changed_during_class_read`. No prefix readback,
numeric work, planet work or Save was confirmed for Arthinna. Its pending
prefix claim remains unresolved; the browser closed and the owner is not
retried.

The saved class-after and prefix-before public captures are exactly identical:
six blank answers, blank prefix, enabled Save and no Data-saved notice. They do
not contain the two internal snapshots that disagreed. Delayed lifetime
initialization and a known autosave-footer transition are hypotheses, not a
proven cause. The next narrow change aligns inner reads with the parent's
existing exact footer projection and preserves both already-read snapshots on
future mismatches. Actual answer, unit, measurement or unknown-feedback changes
must still stop; neither a mismatch nor a pending action is promoted to success.

The readiness audit also passed 104 Rust tests (including the actual Python
subprocess bridge), offline Clippy and formatting, and 283 Python runtime/replay/
compact-wire cases. A fresh root pass of launch contracts, finalization, replay
and scheduler readiness passed **141 tests**. One stale test assertion was
updated to expect the already-implemented original child timeout cause instead
of the generic wrapper; the timeout, action count and no-retry assertions remain
unchanged. These are offline/fixture checks, not a 30-star attempt or proof of
real project completion.

The inner class/prefix consistency change passed **140 offline tests** and
**11 intercepted native Chromium tests**. Only the established exact autosave
notice and Save-busy projection is shared; delayed numeric initialization
inside a read pair still stops. Failure-only evidence saves both original
public captures and their raw/projected differences, pins their hashes, and
distinguishes class-click from prefix-selection dispatch. Capture, difference
and pin I/O failures retain the original terminal cause. Independent read-only
review found no actionable issues. Fresh live class-only diagnostic 004 starts
after this gate; no claim about its outcome or Cab28's lost mismatch is made yet.

Live class-only diagnostic **004 passed both fresh setups**: Dulkyargor used
one class click plus Ga; Kail used the inherited Main -> White Dwarf -> Main
transition plus Ga. Both report `setup_verified: true`; the process exited
normally and closed its browser. This diagnostic made no calculations, Save,
assessment or submission and makes no scientific classification claim. A fresh
three-star check can now exercise the integrated flow under its unchanged caps;
the separate 30-star test remains prohibited until the user explicitly releases
the launch boundary.

The unopened `browser_project_thirty_star.json` profile now explicitly enables
the same reference-assisted shallow-transit continuation as the three-star
profile. Its 30-star target, 10,800-second campaign cap, per-star limits,
paused start, frozen models and viewport remain unchanged. All **27 launch
contract tests passed without opening a browser**. This configuration
preparation does not authorize or launch the deferred 30-star test.

Fresh campaign `9dda11b5679847d3a41f1626998789d5` verified **Ezirek** on the
white-dwarf reference path: three numeric fields, color, 14 lightcurve polls,
the full-5,000-day approximate-No branch, Save and independent collection
readback. Canonical revision 6 contains one verified star. The assistant's
first reference command omitted the explicitly required null lifetime-prefix
argument; protocol validation rejected it before constructing a class owner or
performing a browser action. The corrected complete command was then accepted.

Star two, **Dathanhir**, stopped on its first class-setup advance with
`fresh_class_steps_operation_failed`, zero class/prefix attempts and no native
proposal or class-selection reservation. The fresh capture, manifest and all
six source pins remain valid. Both independent mapping checks and required
frame/label checks pass on the saved capture. Inherited White Dwarf targeting
White Dwarf correctly plans Red Giant -> White Dwarf, without a lifetime prefix.
The failing live snapshot/exception was not retained, so neither a driver error
nor a frame/mapping failure is established. The failed browser closed normally;
no owner is retried. Failure-only, sanitized initialization diagnostics are the
next bounded change. The three-star gate has **not** passed; no 30-star run has
started.

Failure-only initialization diagnostics passed **149 offline tests** and
**three intercepted native Chromium cases**, including inherited White Dwarf
-> Red Giant -> White Dwarf with exactly two clicks and no prefix. The new
`class-operation-failure/diagnostic.json` records allowlisted exception type,
code, phase and bounded repository-code locations, never raw messages, local
paths, arguments or credentials. A public capture is saved only when it already
returned from the approved reader. No extra query, retry or success-path change
was added. Diagnostic annotation and I/O failures preserve the original stop.
Fresh live diagnostic 005 exercises this white-dwarf path without calculations,
Save, assessment or submission; Dathanhir's historical cause remains unknown.

Live diagnostic **005 passed both white-dwarf setups**: Ara'in used one click;
Iane exercised inherited White Dwarf -> Red Giant -> White Dwarf with two
clicks. Neither required a lifetime prefix. Both reports verify setup, the
browser closed normally, and no calculation, Save, assessment or submission
was performed. This rules out a deterministic failure in that class transport
on these two fresh examples, not the unknown failure in the earlier integrated
run. A fresh bounded three-star run will retain the new diagnostic evidence if
the integrated failure recurs.

## Clean three-star regression and launch hold (2026-09-26 Phoenix)

Campaign `569aaefaf861447bb567603cc2a8c6dc` reached its intended three-star
handoff with **Aguilthir, Efin and Daphion independently verified**. The final
outer report is `handoff / awaiting_assessment`, `target_stars: 3`, and
`target_workflows_verified: true`. Canonical revision 18 records three collected,
three verified, zero unresolved and no uncertain actions. All three were
explicit reference-assisted Main Sequence/Ga choices and used the approved
full-5,000-day approximate-No branch. Each numeric stage made 60 decisions and
six verified copies: 18 field copies total, with unchanged checkpoints and zero
optimizer updates. Numeric-stage times were 57.63, 82.92 and 101.55 seconds.

No assessment, score transfer or submission occurred. `task_completed`,
`project_completed` and `submitted` remain false, as required for this bounded
checkpoint. The runtime exited normally and its browser closed. This is a
clean three-star **regression**, not a scored result or proof of every planet,
shallow-transit, terrestrial or habitability branch in one live campaign.
Those integrated branches and actual post-submission acknowledgement remain
unverified. The historical stopped attempts remain unchanged; their unknown
causes are not retroactively declared fixed.

The final offline replay check exposed a display-context defect: the outer
stream ends with a genuine terminal `episode_summary`, but replay retained the
preceding `running / verifying_star` state at EOF. The live final report and
canonical receipts are valid and unaffected. A narrowly identified parent
campaign-summary replay fix now binds the genuine outer run, source context,
project/attempt and target before accepting its terminal display context. Child
summaries cannot become parent completion. All **215 focused offline regression
tests passed**, followed by an independent **116-test replay/launch pass**.
The actual closed three-star trace replayed with sockets and browser/runtime
startup forbidden: 4,579 emissions, correct `handoff / awaiting_assessment`,
three verified stars, and false task/project/submission claims. Source SHA-256
remained `5244e2181478b5e4bdc5ccb023b78d2e24260a5de1cbfa0c6c02e02c6ca1111c`.
No historical event was rewritten. The Rust default suite also passed all 103
tests, followed by the separately enabled actual Python subprocess bridge test
(104 Rust tests total).

The same terminal-summary-only ordering could leave the live TUI's main state
on its preceding running snapshot. Future runs now publish a terminal `state`
after the existing post-summary proof checks and revalidate pinned proof after
that new callback. Changed sources or failed forwarding persist a corrective
stopped state instead of accepting the handoff. All **160 offline runtime,
campaign and integration tests passed**; an independent **22-case terminal
regression pass** also passed. These cover clean two-/three-star handoffs,
source changes at both terminal publication boundaries and failed callbacks.
The actual historical trace replay was checked again successfully and remained
unchanged. No additional browser attempt was needed for these reporting fixes.

The **30-star test has not started**. Its profile is prepared, but the user's
launch hold remains in effect. No automatic browser-budget increase or new
training budget is authorized by this checkpoint.

Work is stopped at this launch boundary, with all live browsers closed. The
next live action belongs to the assistant only after a new explicit user
go-ahead for the first 30-star attempt. This preparation does not certify all
planet/habitability branches, assessment scoring or final submission.

## Post-login welcome-message refresh (2026-09-26 Phoenix)

Automatic setup now refreshes the authenticated preview document exactly once
after its own login submission and before any intro or star action. The page,
browser context, authentication and original 90-second setup deadline are
preserved; no browser process/context restart or credential storage is added.
Already-authenticated previews and later star navigation are not refreshed.
The version-1 setup events expose `refreshing_authenticated_preview` and
`post_login_refresh_verified` through the existing runtime/TUI path.

Authentication loss, an unexpected redirect/modal/popup, refresh failure, or a
persisting visibly painted Welcome-back message stops setup without a second
refresh or login submission. Callback cancellation and changed page identity
are checked before refresh dispatch. The text visibility check handles nested
and pointer-disabled labels without mistaking hidden/covered text for a toast.

Validation: **172 offline tests and 73 intercepted browser/setup/next-star
regressions passed**; three separately opt-in class fixtures were skipped.
The live login-only diagnostic `experiments/browser-login-refresh-001/report.json`
confirmed Welcome-back text visible before refresh and absent afterward, with
the same page/context and authenticated session. It made zero intro actions,
selected no star, started no policy, assessment or submission, and closed its
browser. No credential, session query, authentication screenshot or storage
state was saved. The **30-star launch hold remains in effect**.

## Successful submission evidence still required (2026-09-28)

The submission actuator, public outcome diagnostics, and canonical receipt schema
exist. A successful student-visible acknowledgement has **not** been observed.
The immutable `browser-submission-production-004` captures ground only the exact
course refusal, `You need to analyze and submit at least 30 stars.` Its guarded
feedback can be rebuilt offline; it is not a 30-star attempt, a canonical receipt,
or permission to retry. Generic success words, a returned click, a checked
readiness box, disappearing controls, and `Score updated.` are not submission
proof. No new success recognizer or inactive success-commit code is added on the
strength of those observations.

Before any eventual submission action, the existing preflight must still prove
exactly 30 unique completed task receipts, a current complete paginated inventory,
both assessments and score transfer bound to the same current revision/rows,
unchanged current score and authorized page context, explicit submission opt-in,
and no uncertain/pending prior write. Assessment scoring and project submission
remain distinct. The canonical reservation and native dispatch markers remain
durable and at most once; another output directory cannot bypass them.

The minimum missing public evidence is a fresh response to that one authorized
Submit: the exact exposed response wording/roles and native control identities,
current permitted page/frame identity, and a post-auth viewport plus text/AX
captures linked by hashes to the before capture, dispatch marker, current journal
revision, assessment receipts and score transfer. Capture must distinguish a new
course response from preexisting instructions, quoted text, transient control
state, or score feedback. The existing diagnostic path preserves unrecognized
public responses where its privacy/deadline guards permit; its
`consistency_unverified` evidence cannot itself become an acknowledgement.
Unknown/authentication/popup contexts stop without relaxed guards or another click.

Only after the actual response is grounded can a narrow recognizer and strict
recorded-evidence verifier map it to the existing canonical submission receipt.
That change must also update the finalizer's currently unknown-only verifier and
the runtime's false-completion-only finalization/replay contract. These are real
remaining code gates, not a missing configuration flag. It must test
source/identity mutations, stale or contradictory outcomes, callback
abort, duplicate reservations, canonical current-revision matching and replay.
It must not relabel historical failed or unrecognized outcomes as successes.
Until then, a future authorized first 30-star run is a **readiness/evidence
attempt**, not a promise of verified project completion. The 30-star launch hold
remains active; this audit performs no browser action, training or evaluation.

Validation: **49 offline feedback tests and 98 canonical-progress/submission-
preflight tests passed**. Seven of the feedback cases use optional hash-pinned
public production-004 captures and skip when those local artifacts are absent;
the remaining fixtures are self-contained. The recorded refusal rebuilt exactly,
its original files remained unchanged, and edited positive wording, score
feedback, stale responses and absent exposure proof could not become success.
