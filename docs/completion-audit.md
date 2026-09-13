# Behaviour-stability completion audit

## Current v4 Audit

| Requirement | Evidence | Status |
|---|---|---|
| Mirror road and camera generation | Adjacent seeds have reflected obstacles, images and dynamics; tests cover odd/even image widths | Complete |
| Separate first-obstacle side and early collision | Per-side groups, full-clearance pass identities, early collision and episode duration in calibration report | Complete |
| Remove fixed steering preference | Shared full-graph odd/even readout; raw/executed mirror MAE 0 on 16 unseen pairs | Complete |
| Symmetric road-only safety layer | Position/heading/speed projection; forward/reverse mirror tests | Complete |
| Fresh calibration and unseen evaluation | 48 calibration episodes (10000–10047), fresh checkpoint load, 32 test episodes (400–431) | Complete |
| Avoid early-stop stability claims | Completion/pass/early-collision/timeout publication gates; candidate rejected | Complete |
| Demonstrate useful obstacle avoidance | 0% completion, 2.5 obstacles passed, 41.55 m versus straight baseline 49.15 m | **Not achieved** |
| Preserve reproducibility | Retained v4 candidate SHA-256 in calibration report; historical v2 is explicitly disabled | Complete |
| Regression coverage | 49 driving-mainline Python tests, 6 frontend tests, 3 desktop/mobile browser tests | Complete |

The old language-experiment `test_sklearn_declared` still fails in the full
workspace suite because the public driving package intentionally does not depend
on scikit-learn. It is unrelated to this change and was not modified.

Current evidence: `stable-policy-calibration.json` and
`mirror-constraint-ablation.json`. Mirrored episode pairs are correlated, so
bootstrap operates on the 16 independent road pairs. Symmetry and smoothness
are not a release-quality controller: the candidate remains unpublished.

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
