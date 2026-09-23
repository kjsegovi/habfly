# HabWorlds: permitted spreadsheet calculation reference

Inspected 2026-09-23 through Google Sheets' visible formula bar and column headers. The user authorized the assistant and HabFly to use the sheet and explicitly requested that it remain unchanged. No source cells or formulas were intentionally changed; this is a reference inventory, not a spreadsheet repair or a full workbook audit.

Source: [habworlds spreadsheet, Sheet1](https://docs.google.com/spreadsheets/d/1xr6vbzvlCZRRLpHrgXhzTOL1_tt1rH0Qq4mL-Q-XCPA/edit?gid=0#gid=0).

## Inspected formulas

Exact row-2 formulas below are preserved as observed. A formula's presence does not establish its domain, input units, grading tolerance, or applicability to every star. Do not use the sheet's example rows as answers for a different live star.

| Cell | Header / purpose | Formula as observed |
| --- | --- | --- |
| C2 | distance (LY) | `=(3.26)/B2` |
| D2 | distance (m) | `=C2*9460500000000000` |
| E2 | lum (W) | `=A2*4*pi()*(D2^2)` |
| F2 | lum (Ls) | `=E2/3.827E+26` |
| G2 | Mass(Ms) | `=F2^(1/3.5)` |
| H2 | Radius(Rs) | `=(F2^(1/2))/((M2/5800)^2)` |
| I2 | Lifetime | `=10^10*G2^(-2.5)` |
| K2 | emitted light | `=2897768.5/J2` |
| M2 | temp | `=2897768.5/L2` |
| Q2 | radius of planet (xre) | `=(O2/100)^(0.5)*(P2)*109` |
| R2 | radius of planet (x rj) | `=Q2*0.0892` |
| S2 | orbital radius | `=(N2^2*G2)^(1/3)` |
| T2 | mass | `=11.177*(SQRT((X2^2)*S2*H2))` |
| U2 | mass(g) | `=T2* 5.97E+27` |
| V2 | Radius (cm) | `=Q2*637000000` |
| W2 | density | `=U2/((4/3)*PI()*V2^3)` |
| X2 | radial velocity | `=(Y2/656.3)*300000000` |

Relevant input headers: A flux, B para, J temp, L peak wavelength, N period, O % bright drop, P radius of star, Y max line shift. M is the wavelength-derived temperature used by H; J is a separate input used by K. P is a separate stellar-radius input used by Q, not a link to H in the inspected row. Do not conflate these columns or assume every row describes one consistent system.

## How HabFly may use it

- Treat the sheet as a permitted calculation aid. The policy still selects the target measurement, supplies visible inputs, chooses the appropriate calculation, and transfers the result to the correct live field.
- The current approved default is a versioned local knowledge pack transcribed from this reference. Preserve the source workbook unchanged. See [local-tool training](stellar-local-training.md); numerical transcription is golden-tested independently before generating private grading references.
- The dedicated Google Sheets working copy remains an optional backend, with writes restricted to A2/B2/L2. See [the optional setup guide](stellar-training.md). That backend collects/evaluates through Sheets, has its own formula fingerprints/cell mappings, and never silently substitutes local calculations. Do not mix its demonstrations/checkpoints with local-tool runs.
- Record `calculation_mode`, source URL/tab/cell, formula version, inputs and units, output, and applicability checks in trajectories and TUI. Separate spreadsheet-assisted results from model-only arithmetic accuracy.
- Keep undefined inputs, zero, unavailable measurements, and not-applicable fields distinct. Report source calculation errors rather than substituting plausible zeros.

## Unit and applicability decisions

- B is parallax in arcseconds; C returns light-years. Flux is W/m² and D converts the distance to metres before luminosity calculation.
- L is wavelength in nm; M returns K. The live color choices are IR/Red/Orange/Yellow/Green/Cyan/Blue/Violet/UV.
- The live app limits stellar mass/radius/lifetime reconstruction to main-sequence stars. Do not apply G/H/I indiscriminately just because the sheet returns numbers.
- Lifetime I is in years; the browser input needs a numeric quantity plus its selectable magnitude prefix. Confirm exact unit options when the field is revealed.
- O is percent, not a fractional depth. Q explicitly divides by 100 and converts solar to Earth radii using 109. R is a Jupiter-radius conversion and is not the Earth-radius answer requested in the real planet table.
- The observed period conversion at N9 is `=1841/365`. N13 contains the literal `1841`, without that conversion. N's header does not declare units. S requires an explicit period-unit contract before it is used on browser measurements in days. These are different recorded inputs, not grounds to rewrite the source.
- T2 references H2, whose header is Radius(Rs), while G2 is Mass(Ms). Preserve the supplied formula and record this dependency. Its physical interpretation needs confirmation before calling it a validated planet-mass expert: the [NASA Exoplanet Archive radial-velocity reference](https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html) describes stellar mass, not radius, among the mass-inference dependencies. No correction was applied.
- Missing inputs already cause visible `#DIV/0!` and `#NUM!` results in parts of the sheet. This reference is useful without treating all cached example outputs as training labels.

## Worked mapping from the current star

For CRABILTIA, the live UI supplied parallax 0.032 arcseconds, flux 5.15E-13 W/m², and peak wavelength 212 nm. Applying the inventoried distance/luminosity/temperature expressions locally, without writing either UI or sheet:

| Output | Derived result |
| --- | --- |
| Distance | 101.875 ly |
| Luminosity | 0.015708042 solar |
| Temperature | 13,668.71934 K |
| Peak-wavelength color | UV, using the lesson's <380 nm band |

These are spreadsheet-derived calculations, not feedback-validated answers. The hot, low-luminosity location is consistent with the white-dwarf region of the visible lesson H-R diagram; that classification is an inference, not a submitted/graded result. Do not manufacture a main-sequence mass/lifetime for this star before resolving its class.

## Remaining content

The inspected numeric columns do not supply a complete habitability workflow. Continue with lesson references and the terrestrial detail screen to capture equilibrium temperature, albedo, gas/absorption interpretation, pressure-temperature water phase, classification thresholds, and exact grading/rounding rules.

The Greenhouse Strength popup does provide course-specific temperature additions: None +0 K, Weak +10 K, Moderate +30 K, Strong +100 K. See the [preview requirements](habworlds-preview-inspection.md) for the published absorption bands and current browser gaps.
