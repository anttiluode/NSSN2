# v1 note — prediction residual is not enough

The frozen v1 gate compared three matched arms across 12 seeds:

- route-local prediction-residual plasticity;
- the original v0 Hebbian/resource rule;
- a frozen birth machine.

The predictive arm improved median held-out decoding over the Hebbian learner (`0.883333` vs `0.870833`) but did not beat frozen (`0.900000`). It won 6/12 seeds against Hebbian and 3/12 against frozen. Median late/early surprise was `0.931919`, above the frozen `0.90` requirement.

Verdict: `FAIL_PREDICTION_RESIDUAL_GATE`.

The result suggests that an unconditional per-route expectation is too coarse when the same route participates in multiple resident contexts. The next clean question is context-conditioned local prediction, not more scale or post-hoc tuning.
