# Visual-driving experiment

## Mirror protocol (v4)

Adjacent seeds `2k` and `2k+1` share obstacle heights, radii and absolute
horizontal positions. Their x coordinates have opposite signs. The camera,
vehicle motion and lane constraint also commute with horizontal reflection.
Calibration uses 48 episodes, seeds 10000–10047 (24 independent road pairs).
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
Checkpoint v4 validates motor and reverse neuron identities, source identities,
finite scalar state, zero steering centering, running statistics and bounded
gains. Older checkpoints are explicitly rejected by the loader; the application
reports an obsolete checkpoint and runs an uncalibrated policy rather than
pretending the old state is valid.

Publication requires all stability and task gates: road exits ≤10%, far-obstacle
mean absolute steering ≤0.15, mean steering change ≤0.03, raw mirror error ≤1e-6,
completion ≥50%, first pass ≥75%, mean passed obstacles ≥4.5 of 9, early collision
≤25%, and timeout ≤10%. These are operational simulator thresholds, not a claim
of real-world safety. Failure retains the candidate for reproducibility but does
not replace the default checkpoint. No held-out parameter search is performed.

Evidence: `artifacts/stable-policy-calibration.json` and
`artifacts/mirror-constraint-ablation.json`. The latter loads the retained
candidate explicitly; it must not be mistaken for a published policy.

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

Optional reuse is explicit, not an automatic fallback:

```bash
.venv/bin/autodrive-fly evaluate-constraints --evaluation-start 400 --evaluation-seeds 32 \
  --checkpoint artifacts/checkpoints/driving-policy.calibrated-candidate.npz \
  --reference-report artifacts/stable-policy-calibration.json \
  --output artifacts/mirror-constraint-ablation.json
```

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
