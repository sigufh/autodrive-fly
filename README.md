# MaleCNS Visual Driving

[中文说明](README.zh-CN.md) · [优化复盘（中文）](docs/optimization-review.zh-CN.md)

This is a closed-loop research simulator for asking whether the structural prior
of the adult male *Drosophila* CNS can support obstacle-responsive control and
reward-modulated learning. The primary connectome is **Janelia MaleCNS v1.0**.

```text
ray-cast road camera -> 3,344 mapped R1--R6 photoreceptors
                     -> four persistent full-connectome CNS microsteps per action
                     -> bilateral DNp20 and DNpe017 readout
                     -> steering/throttle -> road reward -> bilateral PPL101 RPE state
                     -> eligibility-gated gain on 1,571 existing motor-input synapses
```

The former dialogue/emotion experiment was retired from the public application.
The API, web UI, package and data manifest do not load a text encoder, dialogue
dataset or language model.

## Run

```bash
./scripts/run-api.sh
npm_config_cache=.npm-cache npm --prefix apps/web run dev -- --host 127.0.0.1 --port 5174
```

Open <http://127.0.0.1:5174>. Use single-step mode to inspect the causal loop, or
continuous mode to run until collision, success, or 500 steps. Execution defaults
to frozen/no-exploration with the lane constraint enabled. “Restore published
policy” discards in-process experimental updates and reloads the calibrated state.

Run paired frozen/learning evaluation with:

```bash
make evaluate-driving
```

Reproduce the deployed stable policy and lane-constraint ablation with:

```bash
make calibrate-policy
.venv/bin/autodrive-fly evaluate-constraints
```

The report is written to `artifacts/driving-evaluation.json`. Always compare the
same held-out seeds and report failures; synapse changes alone are not evidence
of learned obstacle avoidance.

The currently shipped policy is a **frozen calibrated policy**, not a learned
policy. On 32 unseen scenes, enabling the explicit lane constraint reduces road
exits from 25% to 0%, increases mean distance by 3.53 m and raw return by 0.84;
paired bootstrap intervals exclude zero. Online dopamine plasticity did not add
task benefit after stabilization (distance delta -1.21 m), so no learned
checkpoint is published. Obstacle completion remains unsolved.

## Scientific scope

- Retina coordinates are a proxy inferred from R1--R6 postsynaptic partners with
  MaleCNS optic-hex annotations, not calibrated phototransduction.
- The full canonical topology participates in every recurrent step. Activity is
  bounded, persistent and signed by presynaptic transmitter class, but remains a
  dimensionless model state rather than measured membrane voltage. Glutamate is
  treated as inhibitory as an explicit fly-CNS prior; receptor-level exceptions
  are not modeled. Monoamines are excluded from fast transmission.
- DNp20 and DNpe017 are real bilateral descending neurons and all plastic edges
  are real connectome edges. Their steering/throttle interpretation is an
  engineered readout inspired by Doomfly, not an established natural function.
- “Dopamine” in the UI is a scalar reward-prediction error used by a three-factor
  eligibility rule. It is not a measured dopamine concentration and is not a
  claim about happiness or consciousness.
- The simulator has no hidden obstacle detector or path planner. Obstacles affect
  the vehicle only through visual rays, neural dynamics, the documented motor
  readout and collision/reward feedback.

MaleCNS data attribution: Janelia FlyEM, CC BY 4.0. See
<https://male-cns.janelia.org/download/>.
