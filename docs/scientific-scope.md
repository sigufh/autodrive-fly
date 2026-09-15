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
