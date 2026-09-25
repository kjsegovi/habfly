# Stellar browser preflight — 2026-09-24

Status: Firefox requirements inspected; read-only Chromium probe implemented and
fixture-tested. **No learned browser attempt has run.** The promoted lifetime
checkpoint still performs only local, supplied-class, tool-assisted tasks.

Completed follow-up: the user captured **ALTHINAGON in a separate Chromium
session**, and its [offline stellar field mapping](browser-stellar-mapping.md)
is now available. This does not require CRABILTIA or Firefox's attempt state.

## Current preview inspected

Inspected the user's existing Guest / Attempt 1 preview through visible Firefox
UI. Navigated introductory guidance, the collected-star list, the eye-shaped
view control beside CRABILTIA, and the H-R/color references. Opened and dismissed
the color dropdown without choosing an option. No answers, classification,
collection, deletion, assessment, Save, UPDATE SCORE or SUBMIT PROJECT action was
performed. The page's own autosave remains active.

The existing collected star survived reopening: CRABILTIA, total collected 1,
funding $50,000, data quality 0.0%, scavenger hunt 0/8. Its three numeric answer
inputs still displayed `0` and the list displayed unanswered `?` values; these
zeros are not successful answers. Color/class were unselected. The submission
checkbox was unchecked. The preview was left on CRABILTIA's stellar detail tab.

The exact outer activity path is in `configs/browser_probe.example.json`.
The current `preview_sequence_id` is **session-specific**, accepted only as an
explicit pinned value (including its `q:...:...` format), not written into saved
observations or committed configuration. Firefox's **View Frame Info**, not
application source/state, verified these HTTPS frame URLs:

- `https://sim.argos.education/habworlds-star-project/1.5.2/index.html`
- `https://sim.argos.education/spr-widget-buttonwidget/4.4.1/index.html`

Two score-widget frames share that URL; they must not be identified by URL alone.
The probe gives them separate snapshot-local IDs. Those IDs are evidence labels,
not executable browser targets or selectors.

<a id="observed-controls-and-remaining-gaps"></a>

## Visible field map and missing capabilities

| Student-visible field | Observed value/options | Bridge requirement |
| --- | --- | --- |
| Selected star | CRABILTIA | Retain identity across list/detail views |
| PARALLAX (`"`) | 0.032 arcseconds | Convert the displayed unit label to `arcsec`, preserve value |
| PEAK WAVELENGTH (nm) | 212 | Preserve `nm`; do not use metallicity as a measurement |
| FLUX | 5.15E-13; list header supplies W/m² | Parse scientific notation and preserve measurement provenance |
| DISTANCE (ly) | Input displays 0 | Associate the unnamed input with its visible adjacent label |
| LUMINOSITY (solar) | Input displays 0 | Same; map solar luminosity to `Lsun` |
| TEMPERATURE (K) | Input displays 0 | Same; preserve kelvin |
| PEAK λ COLOR | IR, Red, Orange, Yellow, Green, Cyan, Blue, Violet, UV | New categorical selection; not learned by the current policy |
| Stellar class | MAIN SEQUENCE, RED GIANT, SUPERGIANT, WHITE DWARF | New classification decision; current training supplies class and collapses giants |
| Mass/radius/lifetime | Hidden until applicable class | Do not select a class merely to reveal controls during inspection |
| Save / UPDATE SCORE / SUBMIT PROJECT | Separate controls | Distinguish persistence, scoring and submission; none proves one-star completion by itself |

Native Firefox AX labels and visual adjacency are verified, **not Playwright DOM
selectors**. Some native accessibility clicks failed to activate their intended
control; screenshot-grounded coordinates were used only to navigate/read. Do not
hard-code those screen coordinates into the agent.

The H-R reference is a raster plot with temperature, luminosity, main-sequence,
giant, supergiant and white-dwarf regions. It supplies no exact textual grading
thresholds. The local checkpoint has never learned this classification. It needs
a separately validated reference interpretation and training/evaluation gate;
do not infer class from a hidden catalog or inject a solved class into a claimed
autonomous run.

The color reference was reread: UV <380 nm; Violet 380–450; Blue 450–475;
Cyan 475–494; Green 495–570; Yellow 570–590; Orange 590–620; Red 620–744;
IR >744. Shared endpoints and the 494–495 gap remain unresolved. At 212 nm the
reference is unambiguous, but this is not evidence that a policy learned color.

Conditional main-sequence fields, lifetime magnitude-prefix options, answer
format/precision, tolerances, one-star completion and save acknowledgements still
need direct verification. Nothing in this checkpoint expands into planets or
habitability.

## Next bounded command: managed Chromium observation

The probe launches an **isolated headed Chromium**, not the existing Firefox
session. You own any normal sign-in and read-only navigation. It neither copies
Firefox cookies/profile nor saves authentication state. It never launches a
model, trains, clicks, types, spends funds, submits, or writes to Sheets.
Internet is needed for the remotely hosted simulation assets.

First validate the profile offline:

```sh
.venv/bin/python -m habfly browser inspect \
  configs/browser_probe.example.json experiments/browser-preflight-001 --check
```

Then, in a terminal at the repository root, replace the placeholder below with
the current preview URL copied from the address bar. Keep the single quotes;
paste the plain URL, not Markdown `[link](URL)` syntax:

```sh
HABFLY_PREVIEW_URL='PASTE_FULL_PREVIEW_URL_HERE'
.venv/bin/python -m habfly browser inspect \
  configs/browser_probe.example.json experiments/browser-preflight-002 \
  --url "$HABFLY_PREVIEW_URL" --new-test-session
```

For a **separate Chromium test session**, use `--new-test-session` as above. Leave
Firefox alone. Navigate through the introductory pages, manually select/collect
one star in Chromium, and open View Star Data → stellar tab. Any star is suitable;
it need not be CRABILTIA. This manual setup is not learned behavior. Do not reset,
enter answers, choose a classification, assess, Save, update score or submit.
Close help panels, then confirm capture in the terminal. The site's own autosave
may run even while our probe only reads. The completed capture in `-001` remains
valid; this command is only for a later, explicitly separate capture.

Without `--new-test-session`, the probe retains existing-state instructions: if
the intended star is absent, cancel rather than collecting a replacement. A new
Chromium session does not automatically restore Firefox's attempt.

If sign-in lands on the author dashboard, paste the **original URL supplied to
this command** into the same Chromium tab after signing in. Opening another
preview from the dashboard may create a different `preview_sequence_id`, which
is intentionally rejected. Do not reset the attempt or change the configured URL
implicitly. The explicit new-test-session mode permits only the one-star manual
setup described above, not automated collection or answer entry.

Expected artifacts in a **new** output directory:

- `observation.json`: visible text/AX and control inventory for the exact allowed
  frames, outer controls, protected-control annotations, pending gates.
- `manifest.json`: observation SHA-256, zero actions, and
  `browser_acceptance_passed: false`.

No screenshots, raw HTML, request/response traces, cookies, storage state,
passwords, or URL query values are saved by this probe. Treat the resulting
student-visible evidence as local/private and inspect it before sharing.

It rejects a changed outer URL/query, missing/duplicate/hidden required frames,
wrong screen/readiness text, visible password controls, accessible dialogs and
oversized observations. Unknown frame bodies are excluded. For address, login,
modal or readiness failures, Chromium now stays open for up to three manual
capture attempts. Address failures print the expected/current paths and which
URL components differ; query values and credentials are redacted. The allowed
URL is never automatically changed. Correct the screen/address and then confirm
again, or answer `n` to cancel. Other failures stop immediately.

The original probe also had a confirmed stale-address bug: while its synchronous
terminal prompt waited for input, Playwright did not process queued navigation
events. Its cached `page.url` could still be the login page after the human had
already returned to the correct preview. The probe now performs a zero-duration
Playwright protocol round trip after confirmation, **before** checking the current
URL and open tabs. A synthetic timed-navigation regression reproduced the failure
before this fix. The strict URL check itself was not weakened.

On failure, no observation directory is created, so the same output directory
can be reused on the next run. Once a capture succeeds, use a new directory for
later evidence. The probe does not bypass login/security warnings.

This command does **not** attach the lifetime policy to the page. The existing
runtime prohibition on stellar browser execution is retained and regression-tested.
The general `evaluate browser` command is not an alternate route around this gate.

## After capture

1. Validate actual Chromium field associations/custom controls against this map.
2. Implement the semantic browser/calculation bridge and fixture-test exact
   copying, scientific notation, conditional fields, stale targets and units.
3. Add color and four-way classification learning with honest held-out checks.
   Obtain a defensible course reference for boundaries before making grader claims.
4. Run one controlled stellar browser trial with spending, deletion, score updates
   and final submission disabled; report numeric-only or human-assisted work as
   such, not a completed autonomous HabWorlds star.

Keep the successful local lifetime checkpoint, its consumed final cases, and all
previous experiment artifacts unchanged.

## Verification

- Full Python suite after preflight implementation: 361 passed, 9 existing skips.
- Follow-up CLI credential-error redaction check: all 6 CLI tests passed.
- Blocking-prompt/navigation follow-up: all 55 browser, probe and CLI tests passed.
  Returning to the original URL during the prompt now succeeds; leaving the
  allowed page during the prompt still stops before capture.
- Targeted preflight/browser/CLI tests cover query pinning/redaction, same-URL
  siblings, hidden/nested-unlisted frames, auth/modals, missing frames, readiness,
  observation budgets, submission punctuation and the stellar runtime boundary.
- Changed Python files pass Ruff and formatting checks; `git diff --check` passes.
  Repository-wide Ruff also reports four pre-existing findings in
  `scripts/build_neuron_table.py` and `scripts/inspect_connectome.py`; those unrelated
  learning scripts were not changed.
- Promoted lifetime checkpoint SHA-256 remains
  `e383ce87cc94e04cf5147f6e3a09ad33caf0578c79b2e85ce27b196527d52c07`.
- No training, new final-test consumption, Rust changes, or live Playwright
  acceptance run was performed for this preflight checkpoint.
