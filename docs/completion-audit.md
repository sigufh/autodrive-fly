# Behaviour-stability completion audit

## Current v5 Audit

| Requirement | Evidence | Status |
|---|---|---|
| Mirror road and camera generation | Adjacent seeds have reflected obstacles, images and dynamics; tests cover odd/even image widths | Complete |
| Separate first-obstacle side and early collision | Per-side groups, full-clearance pass identities, early collision and episode duration in calibration report | Complete |
| Correct obstacle-side direction | Explicit free-side visual term; unit probe verifies left obstacle→right steering and vice versa | Complete |
| Remove fixed steering preference | Shared full-graph odd/even readout; raw MAE 0 and executed MAE 3.55e-8 on 16 unseen pairs | Complete |
| Symmetric road-only safety layer | Position/heading/speed projection; forward/reverse mirror tests | Complete |
| Train and evaluate a fresh policy | 48 `learning=true` episodes (10000–10047), fresh checkpoint load, 32 frozen test episodes (400–431) | Complete |
| Avoid early-stop stability claims | Completion/pass/early-collision/timeout publication gates; 32/32 complete without early collision | Complete |
| Close longitudinal learning | DNpe017/MDN eligibility tests; pass reward enters RPE; forward is default and MDN is close-hazard gated | Complete |
| Demonstrate useful obstacle avoidance | 100% completion, 9/9 obstacles, 120.30 m versus 49.15 m straight baseline on 32 unseen scenes | Complete |
| Sparse MDN behaviour | reverse command, true reverse motion and close-hazard gate all 0% in held-out evaluation | Complete |
| Preserve reproducibility | Published v5 SHA-256 `e9de899238198dcdd060dba17043edf3c9098743cfd85c33064889ac108268f9` in calibration report | Complete |
| Regression coverage | 31 driving Python tests, Ruff, 6 frontend tests, production build and 3 desktop/mobile browser tests passed against the running v5 service | Complete |

The old language-experiment `test_sklearn_declared` still fails in the full
workspace suite because the public driving package intentionally does not depend
on scikit-learn. It is unrelated to this change and was not modified.

Current evidence: `stable-policy-calibration.json`. Mirrored episode pairs are
correlated, so bootstrap operates on the 16 independent road pairs. The v5
checkpoint is published and default-loadable. Its path-planning performance is
an operational simulator result with explicit engineered visual/recovery terms,
not a claim that the biological graph alone produced the behaviour.

## Neural-decision audit

| Requirement | Evidence | Status |
|---|---|---|
| Preserve original random-obstacle front end | Default API/UI reset is `highway` with random mirrored obstacles | Complete |
| Preserve useful v5 demonstration | `assisted` mode retains the published v5 checkpoint and regression route | Complete |
| Map neural output directly to vehicle | `neural` mode maps DNp20/DNpe017/MDN only; actuator adapter only clips ranges | Complete |
| Prevent hidden action assistance | Neural mode bypasses lane constraint, visual-avoidance, road-recovery and route steering; regression test enforces zero override | Complete |
| Avoid hidden supervised teaching | Neural training uses environment reward only; side-target teacher is disabled | Complete |
| Quantify present neural capability | Seeds 400–407: 0% completion, 27.43 m, 1.25 obstacles, 100% collision | Complete negative result |

The two modes must not be compared as equivalent claims. `assisted` is an
engineering baseline; `neural` is the scientific experiment that still needs
dedicated training and ablation.

## Historical v2 Audit

The statements below refer only to the earlier generator and checkpoint.

Objective: confirm the behavioural-control evidence, implement stability fixes,
and retain only claims reproduced by a freshly loaded deployment checkpoint.

| Requirement | Direct evidence | Status |
|---|---|---|
| Diagnose road exits and oscillation | Frozen/no-exploration and empty-road probes separated persistent bias from exploration sign flips | Complete |
| Separate execution, plasticity and exploration | API and UI have independent flags; execution defaults to learning=false, explore=false | Complete |
| Preserve full policy state | Checkpoint v2 stores gains, running means/variances, reward baseline, steering bias and source IDs | Complete |
| Evaluate a fresh process-equivalent policy | Candidate checkpoint is saved and loaded into a new `DrivingEngine` before evaluation | Complete |
| Stabilize vehicle execution | Deadband, steering-rate limit and bicycle yaw dynamics replace direct heading accumulation | Complete |
| Separate obstacle and road sensing | Environment exports independent obstacle and wall ray arrays | Complete |
| Prevent low-frequency bilateral drift | Global and per-episode DN steering baselines remove common-mode motor bias | Complete |
| Make road constraint explicit | Optional lane constraint exposes raw action, executed action, blend and correction; it never reads obstacles | Complete |
| Quantify constraint benefit | `behavior-constraint-ablation.json`: road exits 25%→0%, distance +3.53 m, return +0.84 on seeds 200–231 | Complete |
| Do not misattribute benefit to learning | Final plasticity test is negative: distance −1.21 m with 95% CI below/at zero; no learned checkpoint published | Complete |
| Disclose remaining failure | Stable policy has 0% road exits but 100% obstacle-collision failure and 0% completion on the evaluated seeds | Complete |

The previous v1 checkpoint omitted running normalization state. Its claimed zero
reload error was invalid because evaluation reused the training object. The v1
positive-learning claim is superseded. The shipped v2 checkpoint is explicitly
`frozen_calibrated`; it does not claim dopamine-learning benefit.
