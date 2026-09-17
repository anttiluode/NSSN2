# NSSN2 v0 — Developing Event Machine Design

Date: 2026-09-17

## Purpose

NSSN2 explores the macro abstraction that fell out of SimpleNeuron and NotSoSimpleNeuron: large state stays resident, tiny events travel, sparse routes access structured receiver state, and the effective local operator depends on current state/history.

The first version must not simulate a biological brain. It should isolate one developmental question:

> Can a sparse recurrent event-driven substrate, using only local unsupervised adaptation, turn repeated latent causes in a small world into more stable and separable resident trajectories than the same frozen substrate?

The repository should keep negative boundaries explicit. A successful v0 does not establish biological realism, intelligence, object concepts, or superiority over backpropagated neural networks.

## Chosen architecture

Use a compact recurrent dynamical network rather than either extreme:

1. **Not a detailed compartmental neuron simulator.** That would obscure the macro experiment under channel and morphology parameters.
2. **Not a generic RNN renamed NSSN2.** That would erase the distinctions that motivated the project.
3. **Chosen: event-addressed stateful nodes.** Each node has a resident vector state, sparse incoming routes, state-conditioned local dynamics, scalar travelling events, and slow local route adaptation.

## World

The synthetic world contains a small number of recurring latent causes. Each cause produces a noisy sensory event pattern through the same sensor population. Causes recur over time with jitter/noise so exact observations are not identical.

The learner never receives the latent cause label. Labels exist only in the experiment harness for measuring whether resident network trajectories become cause-specific.

A frozen control sees exactly the same world tape and starts from the same network parameters, but route adaptation is disabled.

## Node model

Each node owns a resident state vector `x_i in R^d`.

A discrete update is conceptually:

```
x_i[t+1] = tanh(A_i x_i[t] + G_i(x_i[t]) + sum_r b_r q_r[t])
```

where:

- `A_i` is a stable base operator;
- `G_i(x)` is a small local state-conditioned term, so the local Jacobian depends on resident state;
- `q_r[t]` is a scalar event on incoming route `r`;
- `b_r` is the sparse receiver-side access vector for route `r`.

The scalar event does not carry the high-dimensional vector. Route identity plus receiver-side access expands it into local state.

## Event publication and routing

Each node has a scalar readout. When its activity crosses a threshold/cooldown rule, it emits a scalar event to its fixed outgoing routes.

Routes are sparse and fixed in source/target identity during v0. Structural growth is deferred. Only their receiver-side access pattern/strength adapts.

This separates three timescales:

- fast: resident state and event propagation;
- slow: local route plasticity;
- deferred: structural growth/pruning.

## Local adaptation

No global loss and no backpropagation are used.

When route `r` delivers an event to target node `i`, update only that route's receiver-side access vector toward the current local state, with a resource constraint:

```
b_r <- normalize_nonnegative((1 - eta_decay) b_r + eta * q_r * positive(x_i))
```

A small homeostatic term prevents one route from growing without bound. The exact implementation may use signed access if needed, but v0 should prefer the simplest stable nonnegative resource-constrained rule.

The rule is intentionally described as a synthetic route-conditioned local Hebbian rule, not Oja/PCA and not a biological synapse model.

## Experiment gate

Use a deterministic common world tape for learner and frozen control.

Primary metrics:

1. **within-cause state distance** — average distance between network resident-state snapshots caused by repeated presentations of the same latent cause;
2. **between-cause state distance** — average distance across different causes;
3. **separation ratio** — between / within;
4. **nearest-centroid cause decoding** on held-out episodes, using labels only for evaluation;
5. **route specialization** — cosine separation or entropy of learned access patterns.

Predeclared success boundary for v0:

- learned separation ratio must exceed frozen control on the same tape;
- held-out nearest-centroid decoding must exceed frozen control;
- route specialization must change measurably from initialization;
- a shuffled-label evaluation must return near chance, guarding against a broken metric.

If these fail, record the failure rather than adding extra mechanisms to rescue v0.

## Controls and claim boundaries

Required controls:

- frozen-route common-tape control;
- shuffled-label decoder check;
- event-count accounting so learner does not simply receive more sensory drive;
- deterministic seed receipt.

Explicit non-claims:

- this does not show a child learning objects;
- this does not show a biological dendrite implements the rule;
- this does not prove morphology is required;
- this does not establish an advantage over transformers/RNNs;
- this does not constitute a world model unless later gates establish predictive or counterfactual reuse.

## Browser microscope

`index.html` is a self-contained GitHub Pages demo using plain HTML/CSS/JavaScript so no build system is required.

The demo runs a lightweight JavaScript version of the same conceptual machine and exposes:

- a tiny animated world with recurring latent causes;
- sensor pulses entering the network;
- sparse node/route graph with pulse animation;
- node state/activity heat display;
- learned route-access strengths;
- rolling within/between separation and held-out-style proxy score;
- controls for pause/reset, learning on/off, speed, noise, and seed;
- a short scientific claim/boundary panel.

The page is a microscope, not evidence. Frozen Python experiments remain the scientific receipt.

## Repository structure

```
NSSN2/
  README.md
  pyproject.toml
  index.html
  src/nssn2/
    __init__.py
    world.py
    network.py
    experiment.py
  experiments/
    run_v0.py
  results/
    v0.json
  tests/
    test_world.py
    test_network.py
    test_experiment.py
  docs/superpowers/specs/
    2026-09-17-nssn2-v0-design.md
  .github/workflows/
    static.yml
    ci.yml
```

## Testing

Development follows red-green-refactor.

Tests cover:

- deterministic world tapes;
- stable bounded state updates;
- scalar event routing to sparse receiver access vectors;
- local adaptation changes only active routes and respects resource bounds;
- frozen control remains unchanged;
- experiment receipt schema and shuffled-label sanity check.

CI runs Python 3.11 and 3.12, pytest, and a small experiment smoke. The existing Pages workflow remains responsible for deploying `index.html`.

## Next gates after v0

Only after v0 is frozen:

1. predictive reuse: does resident structure help predict the next sensory state?;
2. intervention: does acting on the world improve identification of latent causes?;
3. route growth/pruning: can slow structural adaptation beat a matched fixed graph?;
4. heterogeneous node/operator types;
5. only then larger embodied worlds and richer oscillatory dynamics.

The first version is intentionally about development of reusable resident structure, not intelligence.