# Visual-driving experiment

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
