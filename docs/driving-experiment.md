# Visual-driving experiment

## Causal loop

At each 120 ms environment step, 48 horizontal rays render walls and circular
obstacles into a 48 x 24 grayscale stimulus. A contact-weighted projection maps
this image to 3,344 canonical R1--R6 photoreceptors. The mapping is inferred from
position-annotated postsynaptic optic-lobe partners and is cached as
`data/processed/malecns-v1.0/retina_map.npz`.

Four sparse recurrent microsteps then use every canonical MaleCNS node and edge
before one vehicle action. The state persists between vehicle steps. Acetylcholine
and unknown predictions are positive;
GABA, glutamate and histamine are negative; dopamine, octopamine and serotonin
are excluded from the fast recurrent term. This is a modelling prior, not a
receptor-complete biophysical simulation.

The signed input currents to bilateral DNp20 (body 10059/10162) determine steering;
bilateral DNpe017 (10527/555871) determine throttle. These four engineering choices
are exposed by the API. No geometric avoidance rule overrides their outputs.

## Plasticity and evaluation

Exploration perturbs the two neural actions. The reward is forward progress minus
a small time cost, with -1.5 on collision and +3 on completing the road. Reward
prediction error is the dopamine-like third factor and is also represented as a
signed state on the annotated bilateral PPL101 cells (11327/11900). In training,
a privileged left/right clearance signal supplies compartment-like opponent
teaching to the two DNp20 pathways. It never controls the action at evaluation or
runtime; those actions still come only from neural activity and learned gains. The
signals multiply centered presynaptic/action eligibility traces and update only
1,571 structural synapses already entering the four motor cells. Gains are bounded
to 0.2--3.0. This is reward shaping, not a claim that PPL101 naturally encodes the
simulator's ray geometry.

The final frozen protocol gives both conditions the same 48 exploratory scenes;
only the learned condition enables weight updates. Evaluation uses unseen seeds
200--231 without exploration. Mean distance improves 16.53 m to 58.32 m
(+41.79 m), median distance 14.58 m to 51.66 m, raw return 1.90 to 12.77, and
completion 0% to 21.9%. The learned policy wins 20 of 32 paired scenes. Paired
bootstrap 95% intervals are [25.68, 58.40] m for distance gain and [6.59, 15.31]
for return gain.

The one-CNS-step ablation under the same protocol improves only +2.87 m and +0.66
return, versus +41.79 m and +10.87 with four microsteps. This supports, without
proving biological fidelity, the Doomfly-inspired separation of fast neural and
slower environment time scales. Reproduce with `make evaluate-driving`; full
per-seed results remain in `artifacts/driving-evaluation.json`.

Required future controls include shuffled retinal coordinates, frozen plasticity,
randomized connectome targets and a compact non-connectome policy with comparable
trainable parameters. Success means improvement on held-out road seeds, not merely
nonzero dopamine or changed weights.
