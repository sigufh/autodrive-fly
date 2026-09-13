# Architecture

## Boundaries

- `data`: pinned MaleCNS sources, checksum verification and audit code.
- `connectome`: canonical sparse graph, positions, complete pathway aggregation
  and on-demand source skeletons.
- `driving`: retinal proxy, persistent CNS dynamics, dopamine-modulated policy,
  road environment, evaluation and checkpoint contract.
- `apps/api`: stateful typed HTTP/NDJSON transport.
- `apps/web`: WebGL anatomy/activity, complete schematic topology, road and
  compound-eye stimulus.

## Canonical graph

MaleCNS v1.0's flat table contains segments beyond curated neurons. The canonical
graph is the induced subgraph whose endpoints both have a non-null `superclass`.
It has 166,700 nodes and 25,582,938 directed edges. Matrix rows are postsynaptic
targets and columns are presynaptic sources, so `A @ x` follows pre → post.

Raw contact counts are retained for provenance. Recurrent dynamics use each
target's incoming-count-normalized CSR matrix. Every vehicle action follows four
full sparse recurrent microsteps, and neural state persists between actions.

## Closed loop

The ray-cast environment produces a 48 × 24 grayscale image. R1–R6 positions are
inferred from contact-weighted, position-annotated optic-lobe partners. This is
an explicit visual proxy, not calibrated phototransduction.

Presynaptic consensus transmitter labels supply a sign prior: acetylcholine and
unknown predictions are positive; GABA, glutamate and histamine are negative;
monoamines are excluded from the fast recurrent term. Receptor-level exceptions
are not modeled.

Signed inputs to bilateral DNp20 produce steering. Bilateral DNpe017 produce
throttle. These are real cells but engineering readouts. No obstacle rule or
geometric planner overrides their actions.

## Learning and checkpoint

Slow running mean/variance removes common-mode DN input. Exploration creates a
policy-gradient eligibility trace. Collision/completion events and training-only
left/right clearance shaping provide global and opponent dopamine-like factors.
Only the 1,571 existing connections entering the four motor cells can change.

The compact checkpoint records target motor IDs, every presynaptic body ID and
its gain. Loading rejects graph/order mismatch, non-finite gains and values beyond
the configured 0.2–3.0 bounds. The evaluator publishes a checkpoint only after
both distance and raw-return gains are positive and both paired bootstrap 95%
intervals exclude zero.

## Visualization levels

- 140,638 neurons use real soma or alternate-soma coordinates. Missing positions
  remain counted but are never invented.
- The complete schematic view aggregates every edge into one of 330 directed
  superclass pairs, including self-loops and groups without coordinates.
- The anatomical detail layer displays the 40,000 strongest positioned edges.
- Selected neurons load official Neuroglancer centerline skeletons.
- Activity colors show dimensionless simulated state and signed contribution,
  never measured membrane voltage.
