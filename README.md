# NSSN2

> **Large state stays resident. Tiny events travel. Learning slowly changes how routes access the resident machine.**

`NSSN2` is the macro-scale continuation of [`NotSoSimpleNeuron`](https://github.com/anttiluode/NotSoSimpleNeuron).

The earlier project asked what replaces a conventional weight when a nearly scalar routed event lands in a stateful nonlinear receiver. Its strongest boundary was that a linear dendritic machine can be rewritten exactly in another basis, while local active state makes the effective Jacobian depend on resident state and event history.

This repository asks the next question:

> **Can many small stateful event machines develop useful internal structure from repeated causes in a world, using only local route adaptation?**

The first answer is deliberately mixed: **not yet.**

---

## The machine

Each node owns a resident state vector

```text
x_i in R^d
```

and updates it through a stable base operator, a small state-conditioned local term, and sparse routed events:

```text
x_i[t+1] = tanh(A_i x_i[t] + G_i(x_i[t]) + sum_r b_r q_r[t])
```

where:

- `q_r[t]` is a scalar travelling event;
- `b_r` is a sparse receiver-side access vector;
- `A_i` is a stable local base operator;
- `G_i(x)` makes the local effective operator depend on resident state.

Nodes publish scalar events when their resident activity crosses a threshold. Those events travel on fixed sparse recurrent routes and are expanded again by receiver-side access vectors at the next node.

So the v0 abstraction is:

```text
world
  |
  v
scalar sensory events
  |
  v
sparse route -> receiver-side access vector
  |
  v
resident stateful node
  |
  +-- stable base dynamics
  +-- local state-conditioned dynamics
  |
  v
scalar publication
  |
  v
other resident nodes
```

The point is not biological detail. It is to preserve the distinctions that survived `NotSoSimpleNeuron` while making a network small enough to study as a developing system.

---

## Slow local adaptation

There is no global training loss and no backpropagation in v0.

When a route actually delivers an event, only that route can change. Its sparse receiver-side access vector moves toward the currently active coordinates of the target state, then renormalizes to a fixed resource budget:

```text
b_r <- normalize((1 - eta) b_r + eta * q_r * positive(local_state))
```

The support stays fixed in v0. Structural growth/pruning is a later gate.

This is a **synthetic route-conditioned local Hebbian/resource rule**. It is not claimed to be Oja's rule or a biological synaptic rule.

---

## v0 gate — does repeated experience improve resident structure?

The world contains three recurring latent causes. Each cause produces overlapping noisy sensory patterns. The network never receives the cause labels.

Two identical networks see the exact same deterministic world tape:

- **learner** — local route adaptation enabled during the development phase;
- **frozen control** — identical initial network, adaptation disabled.

For v0, fast electrical state is reset between episodes while learned route access persists. Each episode still contains three internal ticks so routed/recurrent activity can unfold. This deliberately isolates slow representational development from arbitrary carry-over between neighboring episodes.

After development, both machines are frozen and evaluated externally. Labels are used only to measure:

- within-cause resident-state distance;
- between-cause resident-state distance;
- between/within separation ratio;
- held-out nearest-centroid decoding;
- route specialization;
- shuffled-label sanity.

The success criteria were fixed before the receipt was generated.

---

## Frozen result: v0 fails

`results/v0.json`, seed 17, 360 episodes:

| Metric | Learner | Frozen |
|---|---:|---:|
| within-cause distance | **0.079843** | 0.088452 |
| between-cause distance | 0.116491 | **0.123016** |
| separation ratio | **1.459004** | 1.390766 |
| held-out accuracy | 0.858333 | **0.891667** |
| route change norm | **2.421928** | 0 |

Additional controls:

- external sensory-event budget: **1071 vs 1071**;
- shuffled-label accuracy: **0.338802**;
- chance: **0.333333**;
- verdict: **`FAIL_DEVELOPING_RESIDENT_STRUCTURE`**.

The learner did not simply do nothing. Its routes changed substantially and its same-cause responses became tighter, increasing the geometric separation ratio from **1.3908 to 1.4590**.

But the representation became **less useful to the predeclared held-out decoder**: accuracy fell from **89.17% to 85.83%**.

That is enough to fail the gate.

The useful statement is therefore narrower:

> **this local rule changes the resident representation in a repeatable cause-related way, but v0 does not show that the change is an improvement.**

That negative result stays in the repository. No threshold or learning-rule tuning was performed after seeing it.

---

## Why this is not a transformer

There is a loose architectural comparison, not an identity claim.

A transformer largely keeps learned parameters fixed during a forward pass while current token states construct dynamic mixing through attention.

NSSN2 v0 does almost the reverse:

```text
Transformer-ish emphasis:
    dynamic routing/mixing over vectors
    fixed learned machinery during inference

NSSN2 emphasis:
    mostly fixed sparse physical routes
    changing resident local dynamics
    slow receiver-side route adaptation
```

Both can be described as repeatedly applying context-dependent effective operators to resident high-dimensional state, but the factorization and timescales are very different.

---

## Three timescales

The project is intentionally separating:

```text
fast       resident state + scalar event propagation
slow       receiver-side route adaptation
later      structural route growth / pruning
```

v0 implements only the first two.

The next useful question is therefore not "make it bigger." It is why the simple local plasticity tightened clusters while degrading held-out discrimination, and whether a local rule with an explicit novelty/prediction signal can improve reuse without labels or global backprop.

---

## Browser microscope

The GitHub Pages `index.html` is a live illustrative version of the same conceptual machine. It shows:

- recurring world causes and noisy sensors;
- scalar pulses entering a sparse recurrent network;
- resident node activity;
- route-access changes when learning is enabled;
- a parallel frozen control on the same world stream;
- a rolling geometric separation estimate.

**The browser demo is not the scientific receipt.** The Python experiment and frozen JSON above are the evidence for v0.

GitHub Pages: <https://anttiluode.github.io/NSSN2/>

---

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --episodes 360 --out results/v0.json
```

CI runs Python 3.11 and 3.12.

---

## Claim boundary

NSSN2 v0 does **not** establish:

- that this is how biological dendrites learn;
- that a child learns objects this way;
- that the network contains a predictive world model;
- that morphology is computationally necessary;
- that this architecture outperforms RNNs or transformers.

It establishes a runnable macro abstraction and a first falsified developmental gate.

That is useful because the next mechanism has something concrete to beat.