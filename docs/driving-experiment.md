# Visual-driving experiment

## Mirror protocol (v5)

Adjacent seeds `2k` and `2k+1` share obstacle heights, radii and absolute
horizontal positions. Their x coordinates have opposite signs. The camera,
vehicle motion and lane constraint also commute with horizontal reflection.
Training uses 48 episodes, seeds 10000–10047 (24 independent road pairs),
with dopamine plasticity and exploration enabled.
The new held-out evaluation uses seeds 400–431 (16 independent road pairs),
not the previously inspected 200–231 range. Neither half of a test pair may
overlap a calibration pair. Inference disables learning and exploration.

The retinal coordinate proxy is version 2: horizontal optic-hex coordinates
are normalized separately per eye, with a chosen mirrored left-eye orientation.
This is an engineering convention, not a measured eye calibration.

Two persistent state vectors traverse the same complete 166,700-node,
25,582,938-edge graph, driven by the image and its horizontal reflection.
Steering reads their odd component; throttle and MDN reverse read their even
component. Gains are shared, steering normalization has exactly zero mean,
and no per-episode subtraction can absorb the first obstacle's side. The
opponent DNp20 readout is therefore odd even when the biological graph and
retinal sampling are not bilaterally identical. This costs two full-graph
updates per CNS microstep and is explicitly an engineered symmetry constraint.
Only the original-view state is displayed in the neural viewer.

The lane constraint uses current lateral position, heading, speed and a
1.2-second projected lateral position. It never reads obstacle positions or
rays. Reverse travel flips the lateral restoring command. Raw and executed
commands, intervention rate and correction magnitude remain separately visible.

## v5 closed-loop action semantics

DNp20 remains a connectome-derived signed steering residual. It is combined
with two explicitly engineered, observable control terms: an obstacle-ray
odd component chooses the free side, and a road-centering component brings the
vehicle back after clearance. This is a closed-loop simulator controller, not
a claim that the MaleCNS graph alone has been shown to implement path planning.
The displayed constraint fields separate the neural residual, visual-avoidance
term, road-recovery term and road-only safety correction.

DNpe017 supplies the speed residual around default forward motion. Its
eligibility traces are now non-zero and obstacle-pass reward enters the
dopamine prediction error. MDN is a separate escape primitive: it is gated by
four consecutive readings below 1.5 m, has a finite variance prior and has no
positive-clipped action noise. It is not used as routine braking.

## Assisted baseline versus neural-decision experiment

The published v5 checkpoint and its 100% completion result are preserved as an
**assisted engineering baseline**. In this mode, obstacle asymmetry, road
recovery, route geometry and a road-only safety layer are explicitly allowed to
shape the executed action. This is useful for simulator and UI regression, but
does not demonstrate autonomous connectome learning.

The random-obstacle UI also exposes a `neural` mode. It uses the same retinal
input and full MaleCNS graph, but maps DNp20/DNpe017/MDN output directly to
steering/throttle/reverse after domain clipping. It does not call the lane
constraint, obstacle-side steering, road recovery, route steering, speed cap
or hand-crafted side-target teacher. It receives only environment reward in its
RPE update.

On frozen seeds 400–407, the assisted baseline completes 8/8 roads (120.33 m,
9.0 obstacles), while direct neural decision completes 0/8 (27.43 m, 1.25
obstacles and 100% obstacle collision). Neural-mode constraint rate and mean
action correction are both zero. This is a negative result, retained in
`artifacts/neural-decision-baseline.json`; it establishes the actual training
target rather than claiming that the connectome has already learned avoidance.

The subsequent v6 reward-only curriculum preserves the same direct neural
action boundary. A 24-episode mirrored single-obstacle course produced a
separate format-v6 checkpoint. On unseen seeds 640–647 it improved completion
from 0% frozen to 50%, distance from 23.86 m to 27.53 m, while a time-shuffled
equal-amplitude action baseline completed 0% and reached 19.89 m. The learned
checkpoint then completed all eight unseen three-obstacle transfer scenes
without additional training.

Nine-obstacle stability-reward continuation reduced post-pass steering from
0.161 to 0.128 but increased lateral drift from 0.876 m to 0.980 m, with no
completion or road-exit improvement. It was rejected, leaving the single-
obstacle v6 checkpoint published. This is evidence for an early curriculum
learning effect, not evidence of complete random-road driving.

## DNp20 motor-adaptation ablation

Post-pass traces showed that sustained DNp20 output was amplified by the fixed
actuator into a long steering tail. Reward shaping and a stable-finish course
failed held-out gates and were discarded. The retained fix is an environment-
blind motor adapter: an 8%-per-step slow baseline is subtracted from DNp20 before
the fixed steering map. It sees no image, ray, obstacle, road, position, heading,
reward or route state.

On paired unseen seeds 900–907, frozen v6 improved from 0% to 75% completion,
3.25 to 8.0 obstacles, and 46.02 m to 107.38 m. Mean 30-step post-pass steering
dropped from 0.230 to 0.193 and lateral drift from 2.00 m to 0.88 m. Constraint
rate and action correction were zero in both arms. Evidence is retained in
`artifacts/neural-v6-motor-adaptation.json`.

## Metrics and Publication

An obstacle counts only after the vehicle's rear clears its far edge without
collision on that step. Passed obstacle identities are retained across reverse
travel, preventing repeat bonuses. First-obstacle pass is identity 0, not merely
positive distance or any obstacle count. Collision takes precedence over finish.

Reports include:

- first-obstacle side, first pass and total passed obstacles;
- collision before first pass, road exit, obstacle collision and timeout;
- signed raw/executed steering and pre-first-obstacle signed steering;
- left/right turn fractions, episode duration and distance;
- per-side results and mirror errors over the pair's common time horizon;
- pair duration gaps, so early termination cannot hide trajectory divergence;
- a zero-steering, constant-throttle baseline.

Frozen/learned or constraint-off/on differences are paired by exact scene seed.
Bootstrap resamples **road pairs**, not the two correlated mirror episodes
independently. A 32-scene test therefore has 16 independent bootstrap units.

Before evaluation, the candidate is saved and loaded into a new engine.
Checkpoint v5 validates motor and reverse neuron identities, source identities,
finite scalar state, zero steering centering, running statistics and bounded
gains. Older checkpoints are explicitly rejected by the loader; the application
reports an obsolete checkpoint and runs an uncalibrated policy rather than
pretending the old state is valid.

Publication requires all stability and task gates: road exits ≤10%, far-obstacle
mean absolute steering ≤0.15, mean steering change ≤0.06, raw and executed
mirror errors ≤1e-6,
completion ≥50%, first pass ≥75%, mean passed obstacles ≥4.5 of 9, early collision
≤25%, timeout ≤10%, forward drive ≥0.35, reverse command ≤10% and true reverse
motion ≤5%. It also requires distance and passed obstacles to exceed the constant
forward baseline. These are operational simulator thresholds, not a claim
of real-world safety. Failure retains the candidate for reproducibility but does
not replace the default checkpoint. No held-out parameter search is performed.

## Completed v5 training and held-out evaluation

The frozen v5 run trained on seeds 10000–10047 and then evaluated a freshly
loaded checkpoint on seeds 400–431. It passed every publication gate and
replaced the default checkpoint. The report SHA-256 is
`e9de899238198dcdd060dba17043edf3c9098743cfd85c33064889ac108268f9`.

| Metric | v4 retained candidate | v5 published policy | Straight baseline (v5 test roads) |
|---|---:|---:|---:|
| Mean distance | 41.55 m | 120.30 m | 49.15 m |
| Mean passed obstacles | 2.5 / 9 | 9 / 9 | 3.1875 / 9 |
| Completion | 0% | 100% | 0% |
| First-obstacle pass | 81.25% | 100% | 81.25% |
| Road exits | 0% | 0% | — |
| MDN command / true reverse | not gated | 0% / 0% | — |

Across 16 held-out mirror pairs, raw steering mirror MAE is zero, executed
steering mirror MAE is `3.55e-8`, and the maximum episode-length gap is zero.
The mean steering change is 0.0449, below the v5 0.06 gate. These results show
that this simulator controller completes the evaluated roads; they do not
separate how much of the gain is attributable to the connectome residual versus
the explicitly documented visual and road-recovery terms.

Evidence: `artifacts/stable-policy-calibration.json`. The completed v4 ablation
below remains historical evidence for the road-only safety layer.

## Completed v4 safety-layer ablation

The retained candidate above was evaluated again on all seeds 400–431 with
learning and exploration disabled. The constraint-on trajectories exactly match
the original fresh-load calibration evaluation; neither calibration nor weights
were changed during the ablation.

| Metric | Constraint off | Constraint on |
|---|---:|---:|
| Mean final y coordinate | 39.79 m | 41.55 m |
| Mean passed obstacles | 2.375 | 2.500 |
| First-obstacle pass rate | 81.25% | 81.25% |
| Collision before first pass | 18.75% | 18.75% |
| Road-exit rate | 12.5% | 0% |
| Completion rate | 0% | 0% |

Distance difference is +1.76 m, pair-bootstrap 95% CI [0.0029, 5.0975]; raw
return difference is +0.462, CI [0.00072, 1.34111]. These near-zero lower bounds
are not strong evidence of generalization beyond these 16 pairs. Raw steering
mirror error is zero in both conditions. The unchanged early-collision rate and
slightly higher pass count show that the exit reduction is not solely caused by
additional first-obstacle collisions, but **neither condition is a usable
obstacle-avoidance policy**. The straight baseline still reaches 49.15 m.

The following was a v4-only reuse procedure. Its retained candidate was
removed when v5 passed publication. Run a fresh v5 ablation from the published
checkpoint using the command in the README instead.

Reuse requires an exact checkpoint SHA, complete ordered test seeds, frozen
learning/exploration settings, valid per-step traces and an evaluation contract
containing hashes of dynamics, camera/retina code, graph, retinal map and
transmitter data. A mismatch fails before simulation. Reports created before
this fingerprint contract cannot be reused; omit `--reference-report` to perform
a full ablation, or regenerate calibration evidence. The archived reports in
this commit include the contract. The completed ablation here was a full run and
did not reuse cached results. Calibration exposure uses exploration; test
execution does not. These are recorded as separate fields in new reports.

## Historical v2 Results

The following sections describe the earlier, non-mirror road generator and
readout only. Their numerical claims do **not** apply to v4, whose road geometry,
reward (including a one-time 0.35 obstacle-pass bonus), retina, longitudinal
readout and policy contract changed. In particular, smooth steering and no road
exits did not establish successful obstacle avoidance.

## Causal loop

At each 120 ms environment step, separate wall and obstacle rays produce a 48 ×
24 grayscale image. A contact-weighted projection maps it to 3,344 canonical
R1–R6 photoreceptors. Four full-connectome CNS microsteps precede one vehicle
action. Neural state persists across actions.

Signed inputs to bilateral DNp20 determine raw steering; bilateral DNpe017
determine throttle. A deadband, steering-rate limiter and bicycle yaw model turn
the raw command into vehicle motion. A separately switchable lane constraint can
intervene near road edges. It has no obstacle input and cannot select an avoidance
side. Raw and constrained commands remain observable.

## Plasticity and policy state

Training uses centered presynaptic activity, decaying eligibility, event reward
and bilateral PPL101-like opponent teaching. Only 1,571 existing inputs to the
four motor cells can change, within 0.97–1.03 × structural baseline.

Checkpoint v2 stores gains plus running means, variances, reward baseline and
steering zero point. Candidate evaluation always saves and reloads into a new
engine before running unseen scenes. This fixes the v1 evidence flaw: v1 stored
only gains, while its reported replay reused the training object.

The stabilized matched evaluation found online plasticity did not improve the
task: learned minus frozen distance was −1.21 m and raw return −0.29 on seeds
200–231. No learned checkpoint was published. The shipped policy is a frozen
calibrated policy.

## Behaviour-constraint ablation

With the final frozen-calibrated v2 checkpoint, learning and exploration disabled,
and seeds 200–231 fixed, the only variable was the lane constraint:

| Metric | Constraint off | Constraint on |
|---|---:|---:|
| Mean distance | 26.80 m | 30.33 m |
| Mean raw return | 4.38 | 5.21 |
| Road-exit rate | 25% | 0% |
| Max lateral displacement | 3.08 m | 2.41 m |
| Obstacle-collision rate | 75% | 100% |
| Constraint intervention rate | 0% | 26.2% |

Distance gain is +3.53 m (paired bootstrap 95% CI [1.07, 6.47]); return gain is
+0.84 ([0.25, 1.53]). The constraint solves road departure and low-frequency
drift, but not obstacle avoidance. Mean action correction is 0.032 and completion
remains 0%.

Primary evidence is `artifacts/behavior-constraint-ablation.json`. The negative
plasticity result remains in `artifacts/driving-evaluation.json`.
