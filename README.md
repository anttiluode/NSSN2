# NSSN2

> **Large state stays resident. Tiny events travel. Learning slowly changes how routes access the resident machine.**

`NSSN2` is the macro-scale continuation of [`NotSoSimpleNeuron`](https://github.com/anttiluode/NotSoSimpleNeuron). The earlier repo asked what replaces a conventional scalar weight when routed events land in stateful nonlinear receivers. This repo asks what happens when many such receivers form a sparse recurrent machine that develops from repeated experience.

The answer so far is deliberately negative in two different ways:

- **v0 Hebbian/resource plasticity:** changes the representation and tightens same-cause trajectories, but hurts held-out decoding.
- **v1 route-local prediction residual:** repairs part of that damage, but still does not beat the untouched frozen machine.

No rescue tuning was done after either frozen gate.

---

## The machine

Each node owns resident state

```text
x_i in R^d
```

and updates it through

```text
x_i[t+1] = tanh(A_i x_i[t] + G_i(x_i[t]) + sum_r b_r q_r[t])
```

where `q_r[t]` is a travelling scalar event, `b_r` is a sparse receiver-side access vector, `A_i` is a stable local base operator, and `G_i(x)` is a small state-conditioned local term.

Nodes publish scalar events when resident activity crosses threshold. Those events travel on fixed sparse recurrent routes and are expanded again by receiver-side access vectors at the next node.

```text
world
  ↓
scalar sensory events
  ↓
sparse route → receiver-side access vector
  ↓
resident stateful node
  ├─ stable base dynamics
  └─ local state-conditioned dynamics
  ↓
scalar publication
  ↓
other resident nodes
```

The point is not ion-channel realism. It is to preserve the NSSN distinction between **travelling events** and **resident high-dimensional computation** while making a network small enough to falsify developmental rules.

---

## v0 — naive local co-activity

The first slow rule lets only routes that actually carried an event change. Within the route's fixed sparse support, access shifts toward currently active target-state coordinates and is renormalized to a fixed resource budget.

There is no global loss and no backpropagation.

The world contains three recurring latent causes that produce overlapping noisy sensory patterns. Cause labels are never available to the learner. Fast electrical state is reset between episodes while learned route access persists; each episode still receives three internal ticks of recurrent dynamics.

### Frozen v0 result

`results/v0.json`, seed 17, 360 episodes:

| Metric | Hebbian learner | Frozen |
|---|---:|---:|
| within-cause distance | **0.079843** | 0.088452 |
| between-cause distance | 0.116491 | **0.123016** |
| separation ratio | **1.459004** | 1.390766 |
| held-out accuracy | 0.858333 | **0.891667** |
| route change norm | **2.421928** | 0 |

The representation visibly changed and same-cause states became tighter, but held-out decoding fell from **89.17% to 85.83%**.

Verdict: **`FAIL_DEVELOPING_RESIDENT_STRUCTURE`**.

That failure motivated a sharper question: perhaps co-activity alone erases distinctions because frequent structure is reinforced whether or not it is informative.

---

## v1 — local prediction residual

v1 gives each route one additional slow local state: an exponentially updated prediction of the receiver state normally seen when that route fires.

For an active route,

```text
residual_r = observed_target_state - predicted_target_state_r
surprise_r = ||residual_r||
```

The residual chooses the direction of the route-access update and **surprise gates the amount of plasticity**. If the route receives exactly what it predicted, its plasticity closes. The predictor and access vector are both local to the receiving route; labels, global loss, and backpropagation remain absent.

The v1 comparison is deliberately harder:

```text
same world tape + same initialization
                │
        ┌───────┼────────┐
        ↓       ↓        ↓
   predictive  Hebbian  frozen
     residual     v0      control
```

The gate was frozen before the default run:

- 12 seeds, 360 episodes per seed;
- predictive median held-out accuracy must beat **both** controls by at least `+0.01`;
- predictive must win at least `8/12` paired seeds against each control;
- median late/early surprise ratio must be `<= 0.90`;
- external sensory-event budgets must match.

### Frozen v1 result

`results/v1.json`:

| Metric | Predictive | Hebbian v0 | Frozen |
|---|---:|---:|---:|
| median held-out accuracy | **0.883333** | 0.870833 | **0.900000** |
| predictive paired wins | — | 6 / 12 | 3 / 12 |

Additional receipt values:

- predictive − Hebbian median margin: **+0.0125**;
- predictive − frozen median margin: **−0.016667**;
- median late/early surprise ratio: **0.931919**;
- external sensory budgets: **matched for every seed**;
- verdict: **`FAIL_PREDICTION_RESIDUAL_GATE`**.

So prediction residual **did repair part of v0**: its median decoder score is 1.25 percentage points above the naive Hebbian learner, and its route changes are substantially smaller. But it did not establish useful development, because the unmodified birth machine still performed better overall.

The surprise result is equally important. The local predictor learned something—late surprise was generally lower—but not enough to cross the frozen `0.90` criterion. One seed even had late surprise slightly above early surprise.

No parameters, seeds, or thresholds were changed after seeing this result.

---

## What v1 actually teaches us

The v1 predictor is **unconditional per route**. It asks:

> "When this route fires, what target state do I usually see?"

But the same route can participate in several overlapping causes and several different resident contexts. Its average prediction therefore mixes contexts. Residual plasticity can reduce repeated expected structure without knowing **which local context made that expectation appropriate**.

That suggests the next mechanism should not simply increase learning rate or add more nodes. A cleaner next question is:

> **Does locally context-conditioned prediction preserve the distinctions that unconditional route prediction averages away?**

A minimal v2 would let a route predict the receiver's next state from a small projection of the receiver's **pre-event resident state**, then use only the local prediction error to gate access plasticity. The attacker remains the same three-arm/frozen-world setup.

That is still a synthetic computational question, not a claim that cortex implements this exact rule.

---

## Why this is not a transformer

There is a useful comparison, not an identity claim.

A transformer typically keeps learned parameters fixed during a forward pass and dynamically mixes token vectors through attention. NSSN2 instead keeps a sparse route graph mostly fixed on the fast timescale while resident state changes continuously and receiver-side access adapts slowly.

```text
Transformer-ish emphasis
    dynamic mixing over travelling/resident vectors
    fixed learned machinery during inference

NSSN2 emphasis
    tiny travelling events
    high-dimensional state resident in receivers
    state-conditioned local operators
    slow local changes in route access
```

Both can be written as context-dependent effective operators, but the factorization and timescales are very different.

---

## Three timescales

```text
fast       resident state + scalar event propagation
slow       receiver-side access + local predictive traces
later      structural route growth / pruning
```

The first two exist now. Structural growth remains intentionally out of scope until a local developmental rule actually beats the frozen substrate.

---

## Browser microscope

The GitHub Pages demo is an illustrative live microscope. It shows the same sensory stream driving three matched machines and lets you inspect **predictive**, **Hebbian**, and **frozen** modes without confusing the animation with evidence.

GitHub Pages: <https://anttiluode.github.io/NSSN2/>

The Python receipts, not the browser animation, determine the scientific verdicts.

---

## Run

```bash
python -m pip install -e '.[test]'
pytest -q
python experiments/run_v0.py --episodes 360 --out results/v0.json
python experiments/run_v1.py --episodes 360 --seeds 12 --out results/v1.json
```

CI runs Python 3.11 and 3.12 and regenerates both frozen experiments.

---

## Claim boundary

NSSN2 does **not** establish that:

- biological dendrites learn by these synthetic rules;
- a child develops object concepts this way;
- the network contains a predictive world model;
- morphology is computationally necessary;
- the architecture outperforms RNNs or transformers.

What it does establish is narrower and useful: a runnable resident-state/event-routing abstraction now has **two falsified developmental rules with matched controls**, giving the next mechanism something concrete to beat rather than a story to confirm.
