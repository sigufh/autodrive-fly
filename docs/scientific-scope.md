# Scientific scope and fidelity contract

## v7 staged causal contract

The isolated `v7-experimental` path must pass controlled R1–R6 visual-response
tests before central-complex navigation, descending readout, mushroom-body
learning or city integration may begin. Linear luminance and signed frame
difference with type-specific time constants recover aggregate ON/OFF and some
looming timing, but the tested full-graph and visual-subgraph candidates still
fail T4/T5 direction and mirror gates. The legacy backend also fails all response
gates and does not consistently beat preliminary shuffled controls. v5/v6
remain frozen and v7 deployment is disabled. See `docs/driving-v7.zh-CN.md`.
An anatomy-only optic-hex calibration fitted on 3,428 stratified T4 cells
generalizes to zero-shot T5 (83.5% cardinal accuracy) but fails disjoint T4
validation (46.6%, 50.8-degree median error) and the preregistered cross-eye
mirror gate. This is evidence for a partial T5 spatial-offset signal, not a
validated retinal coordinate system or functional motion circuit.
A nested 5x4-fold T4 source audit subsequently identifies Mi1 versus Mi4+C3
offsets with 96.38% outer-fold cardinal accuracy. These cells were already used
in exploration: nested folds do not undo prior analyst exposure or establish
fresh independent validation. This is anatomy-only evidence. The frozen
three-branch candidate still has 0.0024 median T4 direction contrast and 0.5053
legacy mirror error (0.4346 under the revised weighted metric). It
therefore does not establish a functioning T4 circuit or authorize stage 2.
The published single-compartment conductance equation and parameter values were
also evaluated without tuning. All five input populations had dynamic activity,
but no retinal backend passed T4 direction or mirror gates. This tests transfer
to the current dimensionless proxy; it is not a refutation or reproduction of
the original electrophysiology.
A strictly split phenotype-constrained fit (published T4 direction and ON labels,
not measured voltage traces) reaches only 51.7% train and 51.6% validation
direction accuracy while retaining 79.9% ON/OFF validation accuracy. Fitted
parameters were not evaluated on that fit's test stimuli or the original
battery after validation failure. The cells' anatomy and the original battery
were used previously; neither is a project-wide untouched final test. A single
failed optimization does not establish a model-capacity limit.
The retained R1--R6 coverage is asymmetric (1,107 left versus 2,237 right). An
exactly paired 957-per-eye ablation produces zero receptor-level mirror error,
but discards 1,430 receptors and is not a biological eye reconstruction. A
legacy relative-error metric was inflated by near-silent OFF responses. The
experimental mirror gate now uses summed absolute mismatch divided by summed
absolute response amplitude of population means, retaining the old diagnostic.
This is a post-result protocol revision, not a pass of the original preregistered
gate. Strong populations can mask weak mismatches; silence also scores zero,
so this metric alone cannot establish circuit function or cell-level equivariance.
The layerwise audit covers only four stimulus pairs, not the entire battery.
Balancing total eye drive reduces but does not pass this revised gate. The
957-per-eye subset passes it while all T5 polarity contrasts become negative.
Selection changes both count and spatial coverage, so their causal contributions
are not isolated. The default 3,344-receptor mapping remains unchanged.

The supplementary temporal-input audit leaves scoring and model parameters
unchanged. Independent pixel filters generate position-dependent whole-window
edge contrasts without any spatial motion detector: median absolute contrast is
0.4373 for sustained luminance and 0.00207 for frame differences. Local alignment
removes the control's contrast, but does not explain all neural failures. Uniform
ON steps produce the expected negative Mi9 peak in only about half the cells;
these are qualitative proxy measurements, not matched local electrophysiology.
See `artifacts/v7-temporal-input-audit.json`; it cannot authorize stage advancement.

An anatomy-selected local-input audit samples three bilaterally covered columns
(24 cells). All six sampled Mi9 cells show ON-negative centre and ON-positive
annulus responses under both encodings, but those same cells are also ON-negative
under full-screen steps. This does not overturn the all-cell result or establish
population-wide recovery: coverage selection and unequal receptor drive remain
confounds. Local Mi4 polarity is not uniformly correct, and linear-luminance Mi9
peaks hit the observation boundary. Left L3 and bilateral Tm3 lack direct column
annotations and are excluded, not assigned guessed positions. The diagnostic is
not a T4 direction, navigation or learning validation.

The same audit now stratifies all 1,775 Mi9 cells using anatomy-only same-eye
receptor-column coverage. Under ON steps, covered cells have negative peaks in
84–94% of cases, versus 5–26% without same-column receptors across the two tested
encodings and eyes. All-cell statistics exactly reproduce the earlier audit.
Missing coordinates remain separate. This is an association, not a causal lesion:
no same-column receptor does not imply no neighbouring or recurrent input, and
neither the model nor its release gates were changed.

A follow-up within-model intervention holds only the external code of same-column
R1–R6 receptors at baseline. For all six previously inspected Mi9 cells and both
encodings, ON dominant peaks switch from negative to positive; matched-count
nearest-column and far-column controls retain negative peaks. Sham trajectories
are unchanged. This establishes a contribution of those inputs in the tested
model, not a biological lesion result or a rescue of missing retinal coverage.
Controls match receptor count, not synaptic weight, and T4 direction remains unvalidated.

Raw T4 input-weight auditing retains all 6,861 T4a–d targets and all unknown
coverage denominators. Covered Mi9 sources contribute 31.7% of left and 57.5% of
right Mi9-to-T4 synapse weight; Tm3/CT1 remain unlocated rather than uncovered.
Declared branches plus explicitly omitted inputs reconstruct total raw weights.
These are anatomy-only constraints on interpretation, not evidence that inputs
are inactive or that a covered T4 subset has passed functional gates.

A body-ID-joined coverage/response audit reproduces the frozen T4 population
scores while retaining all 6,861 targets, including nine not using the branched
equation. Even the >=50% covered-Mi9-input strata fail the 0.10 direction-contrast
criterion under frame-difference encoding; T4a remains negative. This is reused-data
association, not proof that coverage is irrelevant, since other inputs may remain
uncovered or unlocated and whole-window scoring retains temporal confounds.
No subset score authorizes stage advancement.

A sign-only comparator identifies an inverted monotonicity in the legacy
three-branch Mi9 term: -max(-state,0) increases inhibition when Mi9 becomes more
negative. Replacing it with -max(state,0) corrects this monotonicity but does not
supply tonic inhibition below the arbitrary proxy zero. Both tested encodings
still fail direction and mirror gates. The independent comparator changes no
published candidate, default policy or release threshold and is not a validated
conductance-based disinhibition mechanism.

A threshold-order comparator holds source matrices, normalization and published
parameters fixed while moving rectification before weighted source aggregation.
Both equations pass a synthetic tonic-inhibition/disinhibition interaction table,
but neither passes direction or mirror gates in the connectome model. Constant
background probes after 128 substeps are finite-adaptation measurements, not
verified equilibria; frame-difference encoding yields identical states for all
three constant brightness levels. Nonzero normalized Mi9 conductance therefore
does not establish a physiological, light-dependent resting baseline.

A fixed-horizon background audit shows persistent changes after 2,048 microsteps:
with constant gray luminance, the largest tail-window step change is about 0.174
in the original conductance model versus 0.000042 in the typed visual model.
Large conductance-model residuals are concentrated in left T4 populations, but
ranking does not identify a causal loop. A zero-state, zero-drive frame-difference
control remains zero by construction. Finite windows, one initialization, and
lag tests do not establish asymptotic stability or biological oscillations.

Fixed-parameter feedback ablations zero 20,623 direct T4-to-input edges or all
476,323 T4 output edges without renormalizing surviving weights. Under constant
gray luminance, the final tail-window maximum step change falls from 0.1744 to
0.1126 or 0.00000739, respectively; the latter still has 1,128 varying cells by
the window-range criterion. Removing all outputs also destroys feedforward
signalling. This demonstrates output dependence in the tested model, not a
localized feedback mechanism, biological stability or usable vision.

A timing-only conductance comparator makes T4 read the previous state instead of
same-step upstream updates, preserving all weights and input normalization. It
passes one-step causality tests but does not improve functional gates; with
frame-difference encoding the final background step change rises from 0.000148 to
0.16056 and mirror error reaches 0.9835. Synchronous scheduling is not validated
biological timing, and its altered feedback phase does not identify a causal
microcircuit or justify deployment.

A fixed-state interface Jacobian audit finds large inverse calibration spans
(e.g. median Mi1 normalization slope about 156 left versus 6.8 right for linear
luminance). Normalization, threshold and output clipping are included, and a
fixed-direction finite-difference check agrees within about 6e-5 at 1e-3-span
perturbations; smaller perturbations expose float32 rounding. These are local
input-to-T4 row-sum bounds, not the recurrent Jacobian or a proof of instability.
No normalizer, model parameter or release gate was changed.

Full mixed-update tangent propagation and paired nonlinear replays were compared
for two fixed source-perturbation directions, two amplitudes, and 32 microsteps
from two frozen gray-background states. All tested end-to-initial L2 ratios are
below one (about 0.020–0.069), despite large local interface bounds. This neither
proves global stability nor rules out growth in untested directions. The unit-slope
tangent comparison changes only a derivative factor along the original trajectory,
not a deployable model; finite differences retain threshold and float32 errors.

A phase-balanced periodic-grating diagnostic rejects independent-pixel filters
(direction contrast about 1e-16) and detects a synthetic two-pixel correlator.
Neither frozen neural model shows the expected aggregate T4/T5 preference under
the two tested encodings; all T4 population medians are nonpositive. Both scored
cycles and every T4a–d/T5a–d cell are retained. This is a post-inspection diagnostic
at one frequency, not independent validation or an ON/OFF-specialization test.
Gray-sham subtraction does not remove stimulus-dependent recurrent dynamics.

An isolated anatomical-sign comparison identifies a convention mismatch: the
historical axis fit maps centre-minus-proximal onto preferred motion, whereas
Groschner et al. (2022, https://pmc.ncbi.nlm.nih.gov/articles/PMC8891015/) describe
preferred motion encountering distal Mi9, central Mi1/Tm3, then proximal Mi4/C3.
Negating both axes within each eye changes the frozen conductance T4 aggregate
contrast from -0.10380 to +0.09358 for linear luminance and from -0.07228 to
+0.06616 for frame differences. Physiological labels, graph, dynamics and source
normalization remain fixed; historical reports are preserved and hash-verified.
This is a literature-motivated, post-inspection coordinate comparator, not a
score-selected label flip. Reversing input space can itself exchange direction
responses, so the result is not independent functional validation or learning.
The subtype-fitted axis and reused anatomy still need independent validation.

The comparison retains all 13,580 T4/T5 targets and missing anatomical-vector
counts. Continuous per-eye reversal differs from a discrete half-image mirror
at 1,655 of 3,344 receptors with the unchanged historical rounding rule; one left
eye receptor rounds onto the right image half. Those ambiguities are disclosed,
not silently corrected in the same intervention. Gray-background dynamics are
unchanged, mixed-polarity single-frequency gratings do not establish ON/OFF or
looming specialization, and neither v7 defaults nor advancement gates change.
See `artifacts/v7-geometry-sign.json` for paired all-cell results and coordinates.

A subsequent per-eye pixel-centre bilinear sampler has zero tested cross-eye
support and zero random-image reflection error at all 3,344 retained receptor
positions. However, the original rectified phase-cycle score gives independent
receptor filters a spurious direction contrast up to 0.0289 after fractional
sampling. Integer and half-pixel test locations miss this failure. An explicitly
post-failure 33-position diagnostic initializes filters at their analytic periodic
fixed point and still finds rectified-score bias (up to 0.02786/0.02833), ruling
out finite warmup as its necessary cause in that control. Mean-square response
has contrast near 1e-16 here, but is only a diagnostic, not an adopted neural
metric. Sampler algebra, stimulus discretization and scoring must be validated
jointly; prior nearest-pixel controls do not establish interpolation invariance.

`artifacts/v7-pixel-sampling.json` retains all receptor denominators, cycle means,
input hashes and synthetic periodic traces. Its input control explicitly fails,
and no new neural model evaluation was run. The diagnostic revision after failure
is recorded, existing scores and thresholds remain unchanged, and interpolation
must not be described as a measured receptor receptive field or a functional fix.

A follow-up spectral diagnostic (`artifacts/v7-spectral-controls.json`) retains
signed DC, F1 power, total power and input-referenced complex transfer separately.
Independent filters at all 3,344 receptor positions and 33 fractional offsets per
eye pass the fixed 1e-6 negative tolerance in both encodings and both finite and
analytic-periodic windows; maximum F1 contrast is about 4.4e-16. A synthetic
opponent correlator has signed DC contrast near +1, and near -1 after voltage
inversion, but almost equal power for opposite motion. Power alone therefore
misses that direction code and cannot establish depolarization or ON/OFF identity.
Missing input phase references are flagged explicitly; no unique reference
receptor is assumed for downstream neurons. These are post-failure measurement
controls, not a replacement visual gate or neural validation. No new neural
response evaluation or deployment was performed in this diagnostic stage.

A frozen neural replay then compares nearest-pixel and within-eye bilinear
sampling while holding coordinates, graph, labels, dynamics and conductance
normalization fixed (`artifacts/v7-neural-spectra.json`). All 13,580 T4/T5 IDs,
two cycles, per-cell DC/F1/total-power values and activity denominators are
retained. Bilinear sampling leaves uniform-gray trajectories unchanged and makes
only modest aggregate F1 changes. Conductance T4 DC is positive across all eight
T4 populations, but conductance T5 DC remains negative in aggregate and only two
of eight T5 groups are positive under frame-difference input. Typed-model DC
under interpolated frame differences has no cells above the 1e-6 response
denominator: its nonzero normalized contrast is a tiny-denominator artifact, not
a direction signal. Power metrics are mostly positive but lose voltage sign.
Thus the replay supports neither complete T4/T5 direction validation nor a new
gate; background conductance dynamics, ON/OFF identity, looming and topology
specificity remain unresolved. No downstream phase reference is invented and no
default model or policy changes.

The official Edmond dataset for Groschner et al. (DOI 10.17617/3.8G) is now
audited without committing source arrays. Thirteen Fig. 1/Fig. 3 files are
verified against repository sizes, MD5 values and frozen SHA-256 hashes. Numeric
arrays are loaded with pickle disabled; the Fig. 1d object array is not
deserialized. The Fig. 3 notebook uses 1-kHz processed traces, class-averaged
inputs, one global ON/OFF min-max normalization and synthetic ±160-ms shifts for
Mi9 versus Mi4/C3. Its fixed parameters replay the same fitted T4 averages with
pooled RMSE 1.765 mV and correlation 0.855 in the 2.5–5.5-s display crop. This is
training-data reproduction, not independent validation: the paper reports
least-squares fitting plus manual tuning, and the notebook contains no optimizer
objective/history or split.

Although current v7 parameter values match the paper, its computation does not.
It applies them to image-driven per-node recurrent MaleCNS states normalized by
synthetic calibration, integrates row-normalized source matrices per T4 cell,
then maps predicted voltage into a leaky normalized drive. It neither consumes
the paper's five class-average voltage traces nor applies their fixed ±160-ms
shift in the conductance equation. Accordingly it is a paper-parameter-derived
connectome candidate, not a reproduced or fitted electrophysiology model. Fig. 3
cannot validate T5, per-cell MaleCNS activity or driving behavior. No fitting or
stage advancement is authorized by this audit.

A separate timebase audit finds no physical time unit in the current v7 visual
runtime: 16 stimulus frames and four brain substeps per frame are abstract indices.
The branched proxy uses fixed 0/2/3-substep lags. No single positive solver step
can make both 2 and 3 substeps equal the paper's 160-ms spatial offset, and fixed
past-state lags cannot reproduce the paper's PD/ND-dependent sign reversal between
Mi9 and Mi4/C3. Borrowing the paper's angular scales also implies incompatible
frame durations for current horizontal edge, vertical edge and periodic grating
generators. Most importantly, the conductance candidate reads same-substep updated
source state and bypasses branch delays entirely. Existing leaks and delays are
therefore dimensionless ordering parameters, not calibrated milliseconds or
biological time constants. No `dt`, delay or dynamics parameter changes in this
audit (`artifacts/v7-timebase-audit.json`).

Fig. 5 provides a distinct 36-direction ON-edge target condition: the unchanged
Fig. 3 model correlates 0.979 with the 25-cell GFP population mean (RMSE
0.733 mV), with matching peak direction. This is external-condition support, not
an independent cell holdout, because target arrays contain no cell IDs and the
model reuses Fig. 3 input-class averages. Model Mi9 removal and measured
GluCl-alpha RNAi both reduce direction selectivity, but they are explicitly not
treated as equivalent interventions. The source notebook's flattened 18-element
roll preserves group mean/SEM while scrambling individual rows, and its plotted
0–360-degree coordinate differs from the internal 0–350-degree indices; the audit
uses the unrolled canonical indices. Behavioral CSVs are excluded. This evidence
is retrospective, T4-only, and cannot advance MaleCNS or driving gates
(`artifacts/v7-fig5-validation.json`).

The offline electrophysiology interface keeps processed Fig. 3 arrays on an
explicit native 1-ms grid and marks Fig. 5 direction curves as untimed derived
readouts. Roles are immutable: Fig. 3 is training reproduction, Fig. 5 GFP is
inspected external-condition evidence, GluCl-alpha RNAi is a mechanism challenge,
and Nmdar1 RNAi is a measured negative control. Stable cell identities, independent
Fig. 5 inputs, T5 recordings and an untouched final test are absent. The interface
cannot resample, map millivolts to v7 state, inject target activity, or be imported
by the runtime (`artifacts/v7-ephys-interface.json`). Fitting remains disabled.

A T5 availability audit identifies a genuine whole-cell candidate rather than
collapsing all optical signals into electrophysiology. Gruntman et al. (2021,
doi:10.1016/j.cub.2021.09.072) publicly describe CC-BY Figshare datasets with
T4/T5 single-bar responses (6.32 GB), minimal-motion responses (4.31 GB), a
3.63-MB unified-model package and 17-KB support functions. In this environment,
the article APIs returned HTTP 403 on 2026-09-15, so the large raw datasets were
not downloaded. This is a dated access observation, not a claim that the public
data are unavailable. A later bounded retrieval through Figshare's widget public
endpoint verified the 3.63-MB model archive and its 17-KB support archive. They
contain a MATLAB target-level `t5_simple_wrap` model and fitted T4/T5 tables, but
not the separate Python `axolotl.tmodel` imported by the Kohn--Portes Figure 6
notebooks and not a Tm1/Tm2/Tm4/Tm9/CT1-to-MaleCNS source mapping.
The related public GitLab project metadata for `rbehnialab/axolotl` is visible,
but its repository content is not anonymously readable: GraphQL returns a null
repository, repository REST endpoints return 403/404, no public forks are listed,
and no Wayback repository snapshot was found. All 28 accessible commits of
`flexible-filtering` contain Figure 6 imports but no `TModel`, `create_sine_wave`
or `apply_stim` definitions. PyPI and Conda packages with the same name are the
unrelated LLM trainer first published in 2024. These are bounded availability
observations, not a global source-absence claim.

A source-only temporal-identifiability precheck then applied the four measured
saline Tm kernel shapes to R1--R6-driven Tm activity over the fixed 5-by-3 local
edge grid. Both preregistered readouts failed before direction scoring: the
summed-centroid readout had shuffle/ordered-residual and static/ordered energy
ratios of 31.507 and 0.9996, the fast-pool-versus-Tm9 centroid difference had
ratios of 2.557 and 1.099, and the fixed one-frame Reichardt temporal difference
had ratios of 1.921 and 0.996 (both limits 0.50). No source gain was fitted and
no target activity was injected. This rejects only those three readouts and does not
establish that every possible T5 model fails.
The convolution is now hash-bound to the audited 10 ms offline stimulus-frame
contract, with exactly one kernel sample consumed per stimulus frame. This is a
discrete sampling alignment only: the diagnostic probe performs one brain update
per frame, but that update has no independently calibrated biological solver
interval. External-recording-to-probe-solver alignment and physical source-dynamics
transfer therefore remain unauthorized.
The same three candidates and 120-stimulus control battery were repeated with
the standard offline four brain updates per frame. Their shuffle/static ratios
were 5.136/0.9999, 2.421/0.9278, and 2.153/0.9763, respectively. All three
therefore remained negative, and no candidate passed at both one and four updates
per frame. Direction scoring remained stopped; this numerical-resolution check
does not convert update counts into a biologically calibrated solver interval.
An additional support-coverage audit verified that the author prediction path
passes stored kernels to `lfilter` without reversal, supporting causal index zero.
The exported module does not explicitly import `lfilter`, so self-contained
execution is not claimed. Each local-edge trace has only 27 post-baseline samples
(270 ms; maximum lag 260 ms), versus 499 coefficients (4.99 s nominal support).
All population-kernel absolute peaks fall inside the window, but the prefix covers
only 43.61%, 40.82%, 46.84%, and 47.17% of Tm1/Tm2/Tm4/Tm9 L1 mass. The result
therefore rejects those readouts over early support only; a full-support negative
conclusion and direction scoring remain unauthorized.
The same fixed candidates were then evaluated over complete 525-sample FIR outputs
by appending 498 zero source-drive samples after each observed 27-sample trace. No
visual input was added and the neural network was not advanced during the tail. The
shuffle/static ratios at one update per frame were 2.235/0.9999, 1.443/0.7917, and
0.992/0.5963; at four updates they were 2.187/0.9995, 1.511/0.7626, and
1.197/0.6615. No candidate passed both frozen 0.50 limits at either resolution.
This excludes the zero-source-tail full-FIR variant only; it is not evidence for a
continued network response or an independently validated biological source model.
Population-kernel robustness was then tested without outlier removal using the
existing Fig. 3 source-kernel thresholds. Tm1, Tm4, and Tm9 passed all four shape
gates. Tm2 failed both the 0.80 held-out-recording-versus-rest median gate
(0.7857) and the exhaustive near-equal-partition p05 gate (0.7695). Although each
leave-one-recording-ID-out population mean remained correlated above 0.97 with the
full mean, that jackknife stability does not replace agreement among held-out
records. Recording IDs are not verified fly IDs, so neither this audit nor the
population mean supplies independent biological validation or kernel transfer.
The author aggregation path was audited separately. `return_mean_temporal` defaults
to `baseline=None`, which uses zero offset; all nine bounded calls in Figure 2 and
flash analysis omit the baseline argument, while the inspected Figure 5 and
available Figure 6 notebooks do not call that function. The current no-baseline
kernel therefore matches the author default, and an end-tail-baselined alternative
is not authorized. The author averages payload rows, whereas v7 deliberately
averages within recording ID and then weights IDs equally. This differs only for
Tm1 (8 rows, 7 IDs): the unit-L1 means correlate at 0.99956 with maximum absolute
difference 0.001331, which is close but not exact reproduction.
To test whether that deliberate de-duplication caused the negative result, the full
zero-tail FIR audit was repeated with the author's payload-row weighting. Only Tm1
was allowed to change; Tm2/Tm4/Tm9 remained bit-identical. The three candidates'
shuffle/static ratios were 2.252/0.9999, 1.427/0.7920, and 0.986/0.5872 at one
update per frame, and 2.215/0.9995, 1.499/0.7573, and 1.204/0.6581 at four updates.
None passed both 0.50 limits, so row weighting did not rescue temporal
identifiability and direction scoring remained unauthorized.
The only source failing the inherited kernel-robustness gate, Tm2, was then varied
across all five leave-one-recording-ID-out kernels while the other sources, full
zero-tail FIR, three candidates, thresholds, and one/four-update conditions stayed
fixed. None of the 30 candidate-by-fold-by-update evaluations passed. Across all
folds and update counts, the three candidates' shuffle ratios ranged from
2.102–2.353, 1.365–1.506, and 0.986–1.229; static ratios ranged from
0.9994–0.9999, 0.7467–0.7941, and 0.5921–0.6742. Tm2 recording-ID omission did
not rescue temporal identifiability. Because recording IDs are not verified fly
identities, this remains a sensitivity analysis rather than independent validation.
The measured-kernel path is also explicitly a cascade, not a replacement of the
existing source dynamics. Signed frame-difference input first propagates from R1–R6
through the typed visual subgraph's signed recurrent adjacency, tanh, and fixed
Tm1/Tm2/Tm4/Tm9 leaks of 0.28/0.55/0.34/0.16. Baseline subtraction, positive
half-wave rectification, and synaptic spatial moments are then computed before the
measured FIR is applied. The delay and target-correlator backends are inactive in
this diagnostic, but the typed recurrent/leak dynamics are not. Because no external
mV or filter-output mapping to the v7 signed state exists, these negative results
apply to that composite cascade and cannot be attributed to the measured kernel
alone or interpreted as a single-stage biological source model.
Direct inputs to those Tm states were partitioned exhaustively on the actual
target-normalized visual-subgraph adjacency. Tm/T4/T5 recurrent-or-feedback mass
fractions were 6.24%, 4.12%, 19.61%, and 23.06% for Tm1/Tm2/Tm4/Tm9; Tm9 also
received 9.07% from CT1. The base recurrent update reads the prior state before the
same-step L1/L2/L3 retinal half-wave overwrite, so the observed Tm trace is neither
instantaneous synaptic drive nor a pure same-step feed-forward signal. A true
measured-kernel replacement therefore requires a separately specified intervention
that removes or bypasses these recurrent dynamics; it has not been evaluated or
authorized here.
A corresponding lamina-only intervention was then evaluated. External input still
entered only R1–R6; only L1/L2/L3 half-wave states were advanced, and Tm source
values were replaced by signed target-normalized lamina preactivation before the
measured FIR. Tm leak/tanh, non-lamina inputs, Tm/T4/T5 recurrence or feedback, and
CT1 were removed. Only 3/1/0/1 Tm1/Tm2/Tm4/Tm9 source cells lacked such input and
remained zero. Static controls improved, but every candidate still failed temporal
shuffle at both one and four updates per frame (shuffle ratios 1.570–2.361). Thus
the original recurrent/leak cascade is not sufficient to explain the failure; the
isolated lamina-only replacement also fails to distinguish ordered motion from
shuffled frame transitions. No direction scoring or physical transfer is authorized.
The temporal-shuffle control preserves the post-baseline frame multiset exactly,
but it is not energy matched in the signed-frame-difference input space. Relative
to ordered motion, mean absolute pixel transition energy is 6.934 times larger and
R1–R6 signed-frame-difference energy is 5.838 times larger at both tested update
counts. Positive-half-wave lamina-only Tm preactivation energy is 4.710–4.908 times larger at one
update per frame and 5.127–5.158 times larger at four. The observed output ratios
remain valid under this stronger perturbation, but they do not by themselves
establish failure of equal-energy temporal selectivity. No post-hoc
energy-normalized acceptance gate is introduced, and all downstream gates remain
closed (`artifacts/v7-t5-temporal-shuffle-input-energy-audit.json`).
As a post-hoc control discovery, the initialization transition was held fixed and
the remaining 26 whole-image temporal increments were permuted before reconstructing
the image sequence from baseline. Across all 120 S1-T01 stimuli, reconstructed
images remained in [0, 1], terminal frames were preserved within 1.49e-8, and the
pixel-increment and R1–R6 signed-drive multisets were preserved for every stimulus.
R1–R6 mean absolute energy was 1.0 within 1e-12; after ten zero-drive settle frames,
lamina-only source-energy ratios were 0.99904–0.99997 at one update and
0.99997–1.00000 at four. Nevertheless, the three full-FIR candidates'
control-to-ordered residual ratios were 1.009–1.385 rather than attenuated. Because
the control was discovered on S1-T01, it is not an independent validation and does
not replace the existing temporal gate, define a new threshold, or authorize
direction scoring (`artifacts/v7-t5-increment-order-control-discovery-audit.json`).

Wienecke et al. (2018) measure relative T5 voltage with two-photon ASAP2f at
about 15 Hz; this can constrain polarity, F1 and relative response amplitude but
not absolute millivolts. Ramos-Traslosheros and Silies (2021) provide calcium
activity and receptive-field constraints, while the Shinomiya T5 dataset is
structural connectomics only. These modalities cannot substitute for one another.
The separate 2021 raw datasets remain unverified here; a subsequently identified
2019 repository is not treated as a substitute for those raw files
(`artifacts/v7-t5-data-audit.json`).

The fixed `reiserlab/T5ConductanceModel` commit
`fe52053dda84d49a124e6c1f141dd461eba9630c` corresponds to Gruntman, Romani and
Reiser (2019, doi:10.7554/eLife.50706). All 38 repository files are size-,
Git-blob- and SHA-256-bound under GPL-3.0; none is committed here. Its MAT files
expose processed baseline-subtracted whole-cell T5 voltage (`vm - vl`) from 17
cells. They are not absolute resting-voltage traces or raw acquisition files.
Native ragged traces use 2.5-ms or 5-ms intervals, so the read-only interface
preserves per-trace time vectors instead of resampling them into the T4 1-kHz
bundle.

The repository code fits only width-2 single-bar flashes. Across cells this is
635 conditions and 124,381 samples; all 973 `_spfr` traces are exact copies of
condition-matched traces in `_all`. Other single bars, moving bars, minimal-motion
stimuli and available gratings therefore test within-cell stimulus-condition
generalization, not independent cells. The paper's 1000 initializations and
best-error 1% require external multi-seed orchestration; the provided function
runs at most 10 attempts per seed and saves one model per cell. That ensemble was
not reconstructed, no fitting was run, and no untouched final test exists. The
interface is read-only with `fit_allowed=false` and remains runtime-isolated
(`artifacts/v7-t5-conductance-audit.json`,
`artifacts/v7-ephys-interface.json`).

A measured-only phenotype extraction finds 134 matched moving-bar direction-code
pairs across all 17 cells. Numeric code 1 has the larger peak in 130 pairs, with
median normalized code-1-minus-code-0 contrast 0.4922; all 17 within-cell medians
are positive. Swapping the codes exactly negates the metric, while a fixed random
within-pair label control has median -0.0253. The paper confirms that T5 responses
were aligned to each cell's PD/ND axis and defines DSI, but neither the repository
nor the accessible model code maps numeric code 0/1 to PD/ND. The audit therefore
does not infer the label from response magnitude, assigns no biological PD code,
does not score a v7 model, and cannot advance a gate
(`artifacts/v7-t5-phenotype.json`).

The missing mapping is now localized to the Figure 4 data-and-code package
(doi:10.25378/janelia.11328086.v1). DataCite verifies its title, 1,391,804,749-byte
size, CC-BY-NC-4.0 license and declared `readme.txt`; however, the file API, page
and archive endpoints returned HTTP 403 in this environment and no file manifest
was obtained. The full archive exceeds the frozen 10-MB audit budget. A future
bounded retrieval must hash and inspect only the readme/plotting source and find
an explicit response-independent code map before any T5 model scoring
(`artifacts/v7-t5-label-audit.json`).

A new stage-1 stimulus protocol freezes disjoint development, validation, OOD and
reserved-final sets before another model is fitted. Each split contains 172 stimuli
covering uniform luminance, ON/OFF four-direction edges, looming/receding/static discs,
translation and rotation. Parameters, noise levels and seeds are disjoint; all
cross-split identity and frame-hash overlaps are zero. Mirror pairs share exactly
mirrored noise. Final per-stimulus rows are omitted and remain unevaluated, but
their generator settings are committed, so this is not a blinded one-time test.
A release final still requires an externally custodied manifest.
The declared 10-ms frame and nominal 2.5-ms substep are engineering scheduling
units only: they are not applied to the runtime and do not constitute biological
time calibration (`artifacts/v7-stage1-split.json`).

The official Europe PMC supplementary bundle is retrievable within a 10-MB cap.
Supplementary file 1 is parsed only as nested ZIP/XML and its 121,076-byte DOCX
member is fixed by SHA-256. It contains ten free-parameter definitions, units and
bounds plus fixed VE=0, VI=-74 and VL=-65 mV, but no fitted vector for any of the
17 cells. Those bounds cannot support zero-fit model replay or measured/model
comparison; Figure 4 result files are still required
(`artifacts/v7-t5-supplement-audit.json`).

The frozen strict stage-1 scoring contract requires every T4/T5 subtype-by-eye
group to pass signed direction and polarity criteria separately; no pooled median
can authorize a pass. LPLC1, LPLC2 and LC4 must each prefer expansion over both
receding and a size-matched static disc. Per-cell IDs, denominators and invalid
coverage are retained. Silence, label reversal, insufficient valid coverage,
static-equals-looming and silent mirror controls all fail as intended. Mirror
equivariance also requires active-pair coverage, preventing silent populations
from scoring as perfect. No old artifact is retrospectively rescored and final
authorization additionally requires external custody
(`artifacts/v7-stage1-scoring.json`).

The only authorized development neural screen evaluates 172 stimuli with the
frozen published-parameter T4 conductance proxy on `nested_t4_axis_v1`. The proxy
replaces 6,749 of 6,852 T4 targets that have all five declared source types; all
remaining T4 cells and every T5/LPLC/LC cell retain legacy recurrent dynamics.
Neither retinal encoding passes: direction is 0/16 groups, looming is 0/6 groups,
polarity is 9/16 for linear luminance and 11/16 for frame difference, and mirror
passing is 24/132 versus 4/132. Per-cell IDs, signed contrasts, denominators and
invalid cells are retained. This blocks validation; OOD, reserved final and
topology controls were not run (`artifacts/v7-stage1-development.json`).

An anatomy-supported development A/B negates both continuous within-eye axes while
holding stimuli, labels, normalization, parameters, dynamics and scoring fixed.
All eight T4 direction medians move positively, but the full gate still fails.
Linear luminance changes direction/polarity/mirror pass counts from 0/9/24 to
3/10/48 (out of 16/16/132); frame difference changes them from 0/11/4 to
0/10/0. Looming remains 0/6 for both. Only linear-luminance T4a_R passes the
complete per-group direction rule. This is retrospective development evidence,
not independent validation, and does not authorize label reversal or validation
(`artifacts/v7-stage1-geometry-ab.json`).

An anatomy-only source audit shows that 6,717/6,719 T5 targets have all three
candidate fast inputs (Tm1/Tm2/Tm4) and at least one delayed candidate (Tm9/CT1),
while 6,712 have all five. Missing edges are therefore not a population-wide
explanation. The proposed delayed sources cannot be collapsed by sign: Tm9 cells
are cholinergic, whereas the two large-field CT1 cells are GABAergic. Tm4 has only
49.9% optic-hex coordinate coverage and CT1 has none.

LPLC1 and LPLC2 nearly always receive both T4 and T5 inputs, but LC4 is structurally
different: only 1/126 LC4 cells has any direct T4 input while all 126 have T5
input. A single T4+T5 looming integration rule is therefore not justified across
all three targets. Every target ID, source weight, edge count, same-side fraction
and located-weight fraction is retained; connectivity is not interpreted as
functional sign (`artifacts/v7-visual-target-input-audit.json`).

The development-only R1-R6 preflight evaluates 172 stimuli without running a
neural model. Both the default 3,344-receptor map and the 1,914-receptor exact
bilateral-pair control pass family-specific dynamic range and spatial-support
coverage, exact ON/OFF complementation, and zero cross-receptor leakage under
linear-luminance and signed-frame-difference encodings. The paired map has zero
input mirror error. Finite-view per-receptor motion energy is retained as an
observation rather than incorrectly required to be direction invariant. This
authorizes development neural evaluation only; validation, OOD, reserved final
and central-complex stages remain closed
(`artifacts/v7-stage1-input-audit.json`).

The machine-readable objective coverage audit maps all numbered requirements 0–8
to current evidence. Only version/checkpoint isolation is complete; the city/LLM
pause is a boundary-only pass. Controlled vision is failed, topology/baseline work
is incomplete, and central-complex, descending, separated-signal, mushroom-body
and release stages remain unauthorized (`artifacts/v7-goal-audit.json`).

## Supported claims

- The recurrent model retains every edge in the documented MaleCNS v1.0 canonical
  subgraph.
- A dopamine-like, eligibility-gated rule changes only existing structural inputs
  to four documented descending neurons.
- In v4, an engineered original/mirrored full-graph readout produces exactly
  mirrored trajectories on 16 held-out road pairs. This is imposed symmetry,
  not a biological discovery or evidence of useful obstacle avoidance.
- The v4 frozen candidate has 0% completion and travels less far than a
  zero-steering baseline. It fails publication gates. The historical v2
  checkpoint is not loaded by v4; no usable driving policy is claimed.
- Historical v2 constraint and plasticity results remain archived and are not
  transferable to the changed v4 road, retinal proxy, reward or readout.

## Unsupported claims

- Model state is not measured membrane voltage or a calibrated electrophysiology
  simulation.
- The PPL101 variables are not measured dopamine concentration.
- DNp20/DNpe017 steering and speed plus MDN reverse are engineering interpretations, not known
  natural autonomous-driving functions.
- The MDN reverse readout, per-eye coordinate orientation and odd/even visual
  readout are also engineering assumptions, not calibrated biological mechanisms.
- Training-only clearance shaping is not a claim that a fly has geometric ray
  sensors or that PPL101 naturally represents this signal.
- Constraint benefit on the current road generator is not dopamine-learning
  benefit, general autonomous-driving competence, biological learning validation,
  emotion or consciousness.

## Required next controls

- shuffled retinal coordinates;
- source-preserving, target-shuffled graph;
- a parameter-matched MLP and linear policy;
- KC→MBON plasticity versus current DN-input plasticity;
- unseen obstacle density, curved roads, sensor noise and speed shifts;
- continuous-state dynamics versus a calibrated LIF backend.

The connectome-specific hypothesis is stronger only if the real topology remains
better than those controls across repeated seeds and out-of-distribution tasks.
