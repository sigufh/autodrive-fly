# Doomfly-inspired positive-gain completion audit

Objective: optimize the MaleCNS driving structure using relevant Doomfly methods
until learning produces reproducible positive task gain.

| Requirement | Artifact and direct evidence | Status |
|---|---|---|
| Doomfly-style persistent neural/environment time separation | `DrivingEngine.brain_substeps=4`; every action follows four full 166,700-node, 25,582,938-edge recurrent updates | Complete |
| Slow centered activity and eligibility-based modulation | `DopaminePolicy.running_mean`, `running_variance`, and decaying eligibility traces in `engine.py` | Complete |
| Bilateral dopamine-like opponent compartments | MaleCNS PPL101 11327/11900 states and lateral modulation of existing left/right DNp20 inputs | Complete |
| No runtime heuristic planner | `DrivingEngine.step` obtains steering/throttle from `DopaminePolicy.action`; clearance is training-only teaching and cannot override evaluation actions | Complete |
| Plasticity only on existing structural edges | 1,571 CSR entries entering DNp20/DNpe017; checkpoint verifies every presynaptic body ID and target contract | Complete |
| Fair baseline | Frozen and learned policies receive identical seeds 10000--10047 and exploration; only the update flag differs | Complete |
| Unseen evaluation | Seeds 200--231 are disjoint from training and run without exploration or learning | Complete |
| Positive distance and raw-return gain | 16.53 to 58.32 m (+41.79); 1.90 to 12.77 return (+10.87) | Complete |
| Gain is not only an outlier mean | Median 14.58 to 51.66 m; 20/32 paired wins; completion 0% to 21.9% | Complete |
| Uncertainty gate | Paired bootstrap 95% CI: distance [25.68, 58.40], return [6.59, 15.31], both above zero | Complete |
| Time-scale ablation | One-substep matched experiment gains only +2.87 m versus +41.79 m with four substeps | Complete |
| Deployable learned state | `artifacts/checkpoints/driving-policy.npz`, SHA-256 `06b61da6ae25c9a804a29d3711dd3cc505a8610f2a9c3fb360cf0bda69efb117`; exact 32-scene replay error 0.0 m | Complete |
| Automated and browser verification | 27 public-mainline Python tests, 5 web unit tests, production build, and 2 real-browser tests pass | Complete |

Primary evidence is `artifacts/driving-evaluation.json`; the one-step control is
`artifacts/driving-ablation-one-substep.json`. The result establishes positive
gain in this simulator. It does not establish biological learning fidelity,
general autonomous-driving competence, or a natural steering role for the chosen
descending neurons.
