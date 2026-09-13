# Behaviour-stability completion audit

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
