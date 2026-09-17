import math

from nssn2.experiment_v1 import (
    V1_MIN_MEDIAN_MARGIN,
    V1_MIN_PAIRED_WINS,
    V1_SURPRISE_DECLINE_RATIO,
    run_v1_suite,
)


def test_v1_gate_constants_are_frozen():
    assert V1_MIN_MEDIAN_MARGIN == 0.01
    assert V1_MIN_PAIRED_WINS == 8
    assert V1_SURPRISE_DECLINE_RATIO == 0.90


def test_v1_suite_is_deterministic_matched_and_label_free():
    kwargs = dict(seeds=(3, 4, 5), episodes=120, noise=0.20)
    first = run_v1_suite(**kwargs)
    second = run_v1_suite(**kwargs)

    assert first == second
    assert first["configuration"]["seed_count"] == 3
    assert set(first["aggregate"]) >= {
        "predictive_median_accuracy",
        "hebbian_median_accuracy",
        "frozen_median_accuracy",
        "predictive_vs_hebbian_wins",
        "predictive_vs_frozen_wins",
    }
    assert first["verdict"] in {
        "PASS_PREDICTION_RESIDUAL_GATE",
        "FAIL_PREDICTION_RESIDUAL_GATE",
    }
    assert first["claim_boundary"]["labels_used_for_learning"] is False

    for seed_result in first["seeds"]:
        assert set(seed_result) >= {"seed", "predictive", "hebbian_v0", "frozen", "event_accounting"}
        accounting = seed_result["event_accounting"]
        assert accounting["same_external_sensory_budget"] is True
        assert accounting["predictive_sensor_events"] == accounting["hebbian_sensor_events"]
        assert accounting["predictive_sensor_events"] == accounting["frozen_sensor_events"]

        for arm in ("predictive", "hebbian_v0", "frozen"):
            assert 0.0 <= seed_result[arm]["heldout_accuracy"] <= 1.0
            assert math.isfinite(seed_result[arm]["separation_ratio"])

        assert seed_result["predictive"]["route_change_norm"] > 0.0
        assert seed_result["frozen"]["route_change_norm"] == 0.0
        assert seed_result["predictive"]["early_mean_surprise"] >= 0.0
        assert seed_result["predictive"]["late_mean_surprise"] >= 0.0
