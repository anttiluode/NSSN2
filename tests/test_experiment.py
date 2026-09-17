import json
import math
from pathlib import Path

from nssn2.experiment import run_v0


def _assert_receipts_close(actual, expected, *, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_receipts_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)), path
        assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (
            path,
            actual,
            expected,
        )
        return
    assert actual == expected, (path, actual, expected)


def test_v0_receipt_is_deterministic_and_has_honest_controls():
    kwargs = dict(
        seed=5,
        episodes=120,
        sensors=8,
        causes=3,
        nodes=8,
        state_dim=5,
        routes_per_node=2,
        noise=0.20,
    )
    first = run_v0(**kwargs)
    second = run_v0(**kwargs)

    assert first == second
    assert first["configuration"]["episodes"] == 120
    assert first["event_accounting"]["learner_sensor_events"] == first["event_accounting"]["frozen_sensor_events"]
    assert first["frozen"]["route_change_norm"] == 0.0
    assert first["learner"]["route_change_norm"] > 0.0

    for arm in ("learner", "frozen"):
        assert math.isfinite(first[arm]["within_cause_distance"])
        assert math.isfinite(first[arm]["between_cause_distance"])
        assert math.isfinite(first[arm]["separation_ratio"])
        assert 0.0 <= first[arm]["heldout_accuracy"] <= 1.0

    chance = 1.0 / kwargs["causes"]
    assert abs(first["shuffled"]["mean_accuracy"] - chance) <= 0.18
    assert first["verdict"] in {"PASS_DEVELOPING_RESIDENT_STRUCTURE", "FAIL_DEVELOPING_RESIDENT_STRUCTURE"}
    assert first["claim_boundary"]["labels_used_for_learning"] is False


def test_frozen_v0_receipt_matches_default_gate():
    stored = json.loads(Path("results/v0.json").read_text(encoding="utf-8"))
    _assert_receipts_close(run_v0(), stored)
