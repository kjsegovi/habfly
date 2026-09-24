# HabWorlds preview: adapter and learning requirements

Inspection date: 2026-09-23. Status: requirements discovery, not a browser acceptance run.

Follow-up on 2026-09-24: the existing CRABILTIA attempt was re-inspected without
answer edits. Exact HTTPS frame URLs are now verified through Firefox Frame Info;
the local read-only Chromium probe and the remaining stellar bridge/classification
gates are documented in [stellar browser preflight](browser-stellar-preflight.md).
Earlier unverified implementation items below remain pending unless that follow-up
explicitly supersedes them.

## Evidence and boundary

Inspected the user's Firefox Project preview through its visible UI and accessibility tree. Navigated introductory screens, the starfield/list views, three analysis tables, instructions, and the assessment menu. No answers entered, stars deleted, funds spent, score updated, or project submitted by the inspection.

The preview is a Guest / Attempt 1 authoring preview, not a fresh learner account. Initially the activity showed funding $50,000, data quality 0.0%, scavenger hunt 0/8, and total collected 0. The user subsequently selected/collected CRABILTIA and authorized continuing through VIEW STAR DATA. Follow-up inspection read its detail tabs and sampled 1,000 simulated days without entering reconstruction answers.

Observed outer origin: `http://localhost` (no explicit port). Activity path:

`/authoring/project/habworlds_accessible_version_w/preview_fullscreen/ct9gg_project_diq2o`

The URL also contains `preview_sequence_id`. Treat its value as session-specific configuration; do not hard-code the inspected instance as a stable learner URL.

Observed embedded documents:

- `sim.argos.education/habworlds-star-project/1.5.2/index.html`: main simulation.
- `sim.argos.education/spr-widget-buttonwidget/4.4.1/index.html`: two separate frames, UPDATE SCORE and SUBMIT PROJECT.
- A Vimeo welcome video on the introduction, not a necessary agent action surface.
- Opening Pressure-Temperature Chamber adds a help frame at `s3.us-west-2.amazonaws.com/etx-habworlds/phases-of-matter/index.html`. This needs its own narrowly scoped optional frame rule.

The native accessibility representation omits URL schemes on those child documents; verify actual frame URLs in a managed browser before creating exact origin rules. Frames initially appeared as `about:blank`, then loaded asynchronously.

No real Playwright selectors were validated during this inspection. Accessible labels below are verified UI evidence; DOM locators, frame identification, and any canvas semantics remain implementation checks. The computer-control tool could operate accessible elements but returned `windowNotFoundAtPosition` for starfield coordinate clicks. This is an inspection-tool limitation, not evidence that Playwright cannot click stars.

## Required browser changes

| Area | Current implementation | Required behavior |
| --- | --- | --- |
| Session | Launches a new Chromium context | Keep headed managed Chromium as the first target; a manually opened Firefox tab is inspection evidence, not an attached runtime session. Add managed Firefox only if wanted. Obtain learner login separately; never copy a personal browser profile. |
| Start URL | Rejects all query strings | Allow an explicit preview query schema for inspection mode. Separate preview and delivery profiles. Preserve narrow origin/path checks and reject unknown parameters/credentials. |
| Frame scope | Reads only the top-level page | Discover and allowlist the main simulation and the two navigation-widget frames. Scope all locators, text, safety checks, and target IDs to their frame. Do not indiscriminately expose every iframe. |
| Load readiness | Outer DOM loaded | Wait for simulation identity, funding display, and expected controls; distinguish loading from a missing activity. |
| Targets | Role-based controls | Add visible image/icon controls after DOM verification. Starfield/list switches and numbered analysis tabs appear as images, not buttons in the native tree. Resolve unnamed next/previous controls by container and visible purpose. |
| Observation | Whole body text, generic tooltips | Extract current screen/table, selected star, row identity, column labels/units, values, funding, quality, collected/analyzed counts, scavenger categories, and save status. Exclude browser extensions and unrelated outer chrome. |
| Submission | Exact case-insensitive label match | Actual checkbox label is `I am ready to submit project.` with a trailing period. Default configured label lacks it. Protect explicit frame-scoped controls plus tested label normalization. SUBMIT PROJECT is in a child frame. |
| Enter handling | Any final control present blocks Enter | Final submission controls coexist with the activity. Scope Enter safety to the actual target/form so entering observation days can work while final submission remains forbidden. |
| Spending/deletion | No dedicated budget gate | Independently gate ASSESS and deletion. They are not ordinary free navigation. Default to zero assessment spend and no deletion during observation-only runs. |
| Charts | First chart crop; HOVER/DRAG/SCROLL | Scope chart versus starfield versus spectrum. Test visible tooltip extraction, point clicks, zoom/pan, frame-relative bounds, and separately labelled crops. Do not read internal star catalog or answer state. |
| State freshness | Whole-observation fingerprint | Preserve target validation but separate asynchronous load/autosave/animation changes from action-relevant changes. Reobserve on benign drift; never silently reuse a changed target. |
| Completion | Text contains `Project submitted` | Verify actual post-submit signal in a later explicitly authorized run. Track analyzed stars separately from collected stars, score, Save, UPDATE SCORE, and final submission. |
| Limits | 300 steps / 600 seconds | Measure budgets at one-star gate before sizing three/30-star runs; do not assume current defaults cover the full project. |

The loopback restriction on the outer lesson does not imply every asset is local: this preview uses remotely hosted simulation assets. Local inference/training and fully offline browser execution are different requirements. Do not mirror or modify Torus to resolve this without a separate decision.

Playwright page locators operate in the main frame unless a frame is selected; frame-aware routing is therefore essential, not just a selector refinement. See [Playwright frames](https://playwright.dev/python/docs/frames). Use a Playwright-managed browser binary/session; see [supported browsers](https://playwright.dev/python/docs/browsers).

## Observed lesson content

The UI says at least 30 stars must be analyzed. Total possible score is 260: scavenger hunt 120 plus data quality percentage times 140. Scavenger categories are main sequence, supergiant, red giant, white dwarf, gas giant, ice giant, terrestrial, and habitable world. Perfect score remains outside the acceptance requirement.

| Table | Observations shown | Analyzed fields shown |
| --- | --- | --- |
| 1: Stars | Parallax (arcseconds), peak wavelength (nm), flux (W/m²) | Distance (ly), luminosity (solar), peak-wavelength color, temperature (K), classification, mass (solar), radius (solar), lifetime (years) |
| 2: Planets | Doppler shift (nm), brightness drop (%), brightness-drop period (days) | Has planet?, orbital radius (au), mass (Earth), radius (Earth), density (g/cm³), classification |
| 3: Habitability | Modeled albedo, modeled surface pressure (atm), equilibrium temperature (K), trace gases present | Greenhouse effect, estimated surface temperature (K), water phase, habitable? |

These are the UI's grouping labels, not a claim that every item under Observations is supplied without calculation. In particular the assessment menu offers Equilibrium Temperature ($5000), so its acquisition/editing behavior must be inspected with a selected star.

Assessment mode contains AUTOMATION, SCAVENGER HUNT, DATA QUALITY; an assessment-field menu; stars selected; and ASSESS. Its explanatory text says accurate column assessments can unlock automatic calculations for all stars and planets, and every attempt spends non-renewable funding even on failure. No assessment was executed.

Supporting lesson resources are Instructions, Color Wavelength, Pressure-Temperature Chamber, Tutorial Videos, H-R Diagram, and Greenhouse Strength. The follow-up below records the inspected references; the chamber's controls, tutorial content, and exact classification boundaries still need inspection.

### Follow-up: selected star and reference panels

VIEW STAR DATA appears as text inside an image container in the native accessibility tree. It opened successfully through that text element, but retained analysis tab 3 from prior list navigation. Do not assume it opens tab 1.

CRABILTIA's stellar observations: parallax 0.032 arcseconds, peak wavelength 212 nm, flux 5.15E-13 (table labels establish W/m²), metallicity -0.27, and a visual spectrum. Reconstruction contains distance, luminosity, temperature, peak-wavelength color, and four stellar-class controls. Several text fields expose only `0` rather than a field name in the native tree. Link them to visible adjacent labels in verified DOM rather than guessing from numeric values.

- Color options: IR, Red, Orange, Yellow, Green, Cyan, Blue, Violet, UV.
- Stellar choices: MAIN SEQUENCE, RED GIANT, SUPERGIANT, WHITE DWARF. They appear as text/containers rather than radio roles in the native tree.
- The UI explicitly says mass, radius, and lifetime apply only to main-sequence stars. Classification is a prerequisite, not merely another independent answer.
- Planet tab: a spectrum viewer with separate unnamed zoom buttons and a 656.3 nm reference; observation-days field and Play; a normalized-flux chart; editable Doppler shift, brightness drop (%), and brightness-drop period (days); HAS PLANET? choices Yes/No; GAS GIANT, ICE GIANT, TERRESTRIAL controls; a separate orbital reconstruction visualization with its own controls.
- Entering 1000 in OBSERVE FOR and clicking Play changed the horizontal axis from 0–100 to 0–1000. A later screenshot showed a line near 100% flux. This establishes sampling/axes, not a detected transit or proof of no planet. Hover/zoom semantics and Doppler movement remain unverified.
- Habitability tab displays a terrestrial-planet prerequisite instead of its fields when that prerequisite is not met. Do not expose or fill hidden conditional controls.

The H-R help is a raster diagram, exposed with a filename as its accessible description. It plots luminosity versus surface temperature and shows main-sequence, giant, supergiant, and white-dwarf regions. A reference-image interpretation or verified content-pack mapping is needed; the label alone is insufficient.

Color Wavelength help states: UV <380 nm; Violet 380–450; Blue 450–475; Cyan 475–494; Green 495–570; Yellow 570–590; Orange 590–620; Red 620–744; IR >744. Preserve those published bands; exact shared endpoints and the 494–495 gap need explicit handling rather than invented grading boundaries.

Greenhouse Strength help supplies course-specific approximations: None (0–0.49% absorbed) adds 0 K; Weak (0.5–39.99%) adds 10 K; Moderate (40–59.99%) adds 30 K; Strong (60–100%) adds 100 K. The source of absorbed-energy percentage and boundary precision still need verification in the terrestrial workflow.

The page advertises autosave every two minutes and separately exposes Save, UPDATE SCORE, and SUBMIT PROJECT. Introductory guidance warns that some users encounter score-saving failures and recommends screenshots of assessment views. Preserve evidence and verify persistence; do not equate a button click with saved completion.

## What the agent decides, versus what Playwright does

Playwright is an actuator and observer. It must not quietly solve the task, choose favorable stars using hidden data, or inject values into application internals.

The policy must learn to:

1. Choose an unexplored visible star, inspect it, and decide whether to collect it.
2. Carry a star's identity and measurements across views; avoid duplicate or mismatched rows.
3. Choose an observation duration, distinguish insufficient observations from no detected planet, locate transit dips, and decide when to zoom, pan, or sample longer.
4. Read values and units; compute quantities in dependency order; choose categorical answers from allowed visible options.
5. Combine stellar, orbital, physical, atmospheric, and water-phase evidence into a habitability decision.
6. Decide whether to revise a result, continue searching, or request an assessment within the configured budget.
7. Know when one/three/30-star objectives are actually complete and, only with explicit authorization, submit.

The deterministic expert may perform validated calculations to generate demonstrations. The user explicitly authorized both the assistant and HabFly to use the provided spreadsheet unchanged. Its use is an allowed calculation aid, not evidence of learned arithmetic: label assisted evaluations accordingly and record inputs, formula/source cell, units, and returned result. See [spreadsheet formula inventory](habworlds-spreadsheet-reference.md). Do not alter source formulas or silently substitute a different reference.

Candidate calculation dependencies to verify against course material (not yet approved formulas):

- Parallax -> distance; distance plus flux -> luminosity.
- Peak wavelength -> temperature/color; luminosity plus temperature -> H-R class and radius; applicable stellar relation -> mass and lifetime.
- Transit timing/depth plus stellar properties -> period, orbital radius, planet radius; Doppler measurement with its reference wavelength/conventions -> planet mass; mass and radius -> density/class.
- Luminosity/orbit/albedo -> equilibrium temperature; gas mixture -> greenhouse strength -> surface temperature; pressure and temperature -> water phase; terrestrial class plus liquid-water conditions -> habitability.

Do not invent constants, rounding rules, stellar-class relations, spectrum baselines, or gas coefficients. Exact formulas, unit conversion factors, tolerances, categorical options, missing-value behavior, and dependency applicability belong in a versioned verified content pack.

The [official project help](https://kb.inspark.education/habworlds-project) additionally confirms that lifetime uses a numeric quantity plus magnitude prefix, transit observation uses Enter/play and zoom/pan/hover, and habitability requires terrestrial composition with liquid-water-compatible pressure/temperature rather than detection of water vapor. It also warns that reopening preserves data rather than resetting it. These documented behaviors still need end-to-end confirmation in this preview/version.

## Simulator and protocol gaps

The current `synthetic-stars-v1` pack has only lifetime, period, radius, and a simplified habitability rule. It is not an adequate real-project expert. Its radius field uses stellar units, whereas the real planetary table asks for Earth radii; observed transit depth is percent, requiring explicit conversion before any square-root relation. Keep the synthetic pack intact for engineering fixtures.

A verified real-project pack and simulator need:

- All three tables, categorical and numeric field types, unit/prefix controls, conditional fields, and row identity.
- Observation uncertainty, no-transit cases, long periods, tiny dips, multiple visible chart views, and view-local values rather than privileged ground truth in observations.
- Starfield and list navigation, analysis tabs, pagination, collection lifecycle, funding, assessment costs/unlocks, and distinct save/score/submission state.
- Water-phase classification from a validated pressure-temperature relation, not the current rectangular temperature interval plus minimum-pressure shortcut.
- Progress counters that distinguish collected, filled, analyzed, correct, saved, and submitted.
- Frame-qualified observation-local targets and typed visual surfaces. If stars or tooltips have no usable visible DOM semantics, add scoped visual perception; the current chart-only crop/encoder may not cover star selection.
- TUI displays for selected star, analysis stage, measurement provenance/uncertainty, funding and proposed spend, persistence, and preview-versus-learner mode.

## Bounded implementation gates

1. **Read-only managed-browser probe.** Verify exact URL/query policy, frame origins/paths, readiness, controls, and student-visible text. No collection, spending, answer edits, or submission. Produce a sanitized observation fixture.
2. **One-star UI inventory.** With a disposable preview/account, inspect detail tabs, units/dropdowns, charts/tooltips, collect/edit feedback, and help resources. Record exact formulas and one independently checked worked example. Keep spending and submission locked.
3. **Adapter contract tests.** Include same-URL sibling frames, frame reload/stale targets, unnamed icons, missing semantics, custom dropdowns, async chart state, punctuation-safe submission protection, Enter for observation days, and blocked assessment spending.
4. **Content/simulator fidelity.** Build deterministic one-star demonstrations only after the field/formula contract is verified. Then test three representative stars and held-out scenarios; no claim of learned success from the expert.
5. **Learned one-star gate.** Train/evaluate in simulation before browser inference. Only afterward expand to three stars and a fresh-account full project with explicit final-submission authority.

## Still unverified

- Exact collection transition, conditionally revealed main-sequence/planet/terrestrial fields, editable list-row widgets, actual graph tooltip text/DOM, and native versus custom dropdown implementation.
- Spreadsheet formula mappings are now inventoried, but course grading tolerances, classification boundaries, assessment success thresholds, full habitability calculations, and save acknowledgements remain unverified.
- Authenticated learner delivery URL/account, durable reset procedure, actual successful-submission signal, and runtime budgets.
- Whether preview data persists identically to delivery; Guest is not evidence of a valid acceptance account.

Do not enable browser acceptance by merely replacing the placeholder URL in `configs/browser.example.json`.
