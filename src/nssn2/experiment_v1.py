from __future__ import annotations

import numpy as np

from .experiment import _evaluate, _route_specialization
from .network import NSSNNetwork
from .world import make_world_tape


V1_MIN_MEDIAN_MARGIN = 0.01
V1_MIN_PAIRED_WINS = 8
V1_SURPRISE_DECLINE_RATIO = 0.90


def _encode_training_episode(
    net: NSSNNetwork,
    event: np.ndarray,
    learn: bool,
) -> tuple[np.ndarray, int, int, float]:
    """Run one development episode and expose only route-local surprise.

    As in v0, fast electrical state is reset between episodes while slow
    route access and predictive traces persist. Labels never enter here.
    """
    net.reset_state()
    sensor_count = 0
    recurrent_count = 0
    surprises: list[float] = []

    first = net.step(event, learn=learn)
    sensor_count += first.sensor_events
    recurrent_count += first.recurrent_events
    if first.total_events:
        surprises.append(first.mean_surprise)

    zero = np.zeros_like(event)
    for _ in range(2):
        step = net.step(zero, learn=learn)
        sensor_count += step.sensor_events
        recurrent_count += step.recurrent_events
        if step.total_events:
            surprises.append(step.mean_surprise)

    mean_surprise = float(np.mean(surprises)) if surprises else 0.0
    return net.snapshot(), sensor_count, recurrent_count, mean_surprise


def _route_change(net: NSSNNetwork, sensor_initial: np.ndarray, recurrent_initial: np.ndarray) -> float:
    return float(
        np.linalg.norm(net.sensor_access - sensor_initial)
        + np.linalg.norm(net.recurrent_access - recurrent_initial)
    )


def _single_seed(
    seed: int,
    episodes: int,
    sensors: int,
    causes: int,
    nodes: int,
    state_dim: int,
    routes_per_node: int,
    noise: float,
    train_fraction: float,
    sensor_threshold: float,
) -> dict[str, object]:
    tape = make_world_tape(
        seed=seed,
        episodes=episodes,
        sensors=sensors,
        causes=causes,
        noise=noise,
    )
    sensory = np.where(tape.events >= sensor_threshold, tape.events, 0.0)

    common_seed = seed + 1000
    predictive = NSSNNetwork.create(
        seed=common_seed,
        sensors=sensors,
        nodes=nodes,
        state_dim=state_dim,
        routes_per_node=routes_per_node,
        learning_mode="predictive",
    )
    hebbian = NSSNNetwork.create(
        seed=common_seed,
        sensors=sensors,
        nodes=nodes,
        state_dim=state_dim,
        routes_per_node=routes_per_node,
        learning_mode="hebbian",
    )
    frozen = NSSNNetwork.create(
        seed=common_seed,
        sensors=sensors,
        nodes=nodes,
        state_dim=state_dim,
        routes_per_node=routes_per_node,
        learning_mode="hebbian",
    )

    predictive_sensor_initial = predictive.sensor_access.copy()
    predictive_recurrent_initial = predictive.recurrent_access.copy()
    hebbian_sensor_initial = hebbian.sensor_access.copy()
    hebbian_recurrent_initial = hebbian.recurrent_access.copy()
    frozen_sensor_initial = frozen.sensor_access.copy()
    frozen_recurrent_initial = frozen.recurrent_access.copy()

    predictive_initial_specialization = _route_specialization(predictive.sensor_access)
    hebbian_initial_specialization = _route_specialization(hebbian.sensor_access)

    train_count = int(round(episodes * train_fraction))
    train_count = min(max(train_count, causes * 4), episodes - causes * 4)

    counts = {
        "predictive_sensor_events": 0,
        "hebbian_sensor_events": 0,
        "frozen_sensor_events": 0,
        "predictive_recurrent_events": 0,
        "hebbian_recurrent_events": 0,
        "frozen_recurrent_events": 0,
    }
    predictive_episode_surprise: list[float] = []

    for index in range(train_count):
        _, ps, pr, surprise = _encode_training_episode(predictive, sensory[index], True)
        _, hs, hr, _ = _encode_training_episode(hebbian, sensory[index], True)
        _, fs, fr, _ = _encode_training_episode(frozen, sensory[index], False)

        counts["predictive_sensor_events"] += ps
        counts["hebbian_sensor_events"] += hs
        counts["frozen_sensor_events"] += fs
        counts["predictive_recurrent_events"] += pr
        counts["hebbian_recurrent_events"] += hr
        counts["frozen_recurrent_events"] += fr
        predictive_episode_surprise.append(surprise)

    reference_events = sensory[:train_count]
    reference_labels = tape.labels[:train_count]
    eval_events = sensory[train_count:]
    eval_labels = tape.labels[train_count:]

    predictive_metrics, _, _ = _evaluate(
        predictive,
        reference_events,
        reference_labels,
        eval_events,
        eval_labels,
        causes,
    )
    hebbian_metrics, _, _ = _evaluate(
        hebbian,
        reference_events,
        reference_labels,
        eval_events,
        eval_labels,
        causes,
    )
    frozen_metrics, _, _ = _evaluate(
        frozen,
        reference_events,
        reference_labels,
        eval_events,
        eval_labels,
        causes,
    )

    surprise_window = max(1, train_count // 4)
    early_surprise = float(np.mean(predictive_episode_surprise[:surprise_window]))
    late_surprise = float(np.mean(predictive_episode_surprise[-surprise_window:]))
    surprise_ratio = float(late_surprise / max(early_surprise, 1e-12))

    predictive_out = {
        **predictive_metrics,
        "route_change_norm": _route_change(
            predictive, predictive_sensor_initial, predictive_recurrent_initial
        ),
        "initial_route_specialization": predictive_initial_specialization,
        "final_route_specialization": _route_specialization(predictive.sensor_access),
        "early_mean_surprise": early_surprise,
        "late_mean_surprise": late_surprise,
        "surprise_late_over_early": surprise_ratio,
    }
    predictive_out["route_specialization_delta"] = float(
        predictive_out["final_route_specialization"] - predictive_initial_specialization
    )

    hebbian_out = {
        **hebbian_metrics,
        "route_change_norm": _route_change(hebbian, hebbian_sensor_initial, hebbian_recurrent_initial),
        "initial_route_specialization": hebbian_initial_specialization,
        "final_route_specialization": _route_specialization(hebbian.sensor_access),
    }
    hebbian_out["route_specialization_delta"] = float(
        hebbian_out["final_route_specialization"] - hebbian_initial_specialization
    )

    frozen_out = {
        **frozen_metrics,
        "route_change_norm": _route_change(frozen, frozen_sensor_initial, frozen_recurrent_initial),
        "route_specialization": _route_specialization(frozen.sensor_access),
    }

    same_external = (
        counts["predictive_sensor_events"]
        == counts["hebbian_sensor_events"]
        == counts["frozen_sensor_events"]
    )
    counts["same_external_sensory_budget"] = bool(same_external)

    return {
        "seed": int(seed),
        "predictive": predictive_out,
        "hebbian_v0": hebbian_out,
        "frozen": frozen_out,
        "event_accounting": counts,
    }


def run_v1_suite(
    seeds: tuple[int, ...] = tuple(range(12)),
    episodes: int = 360,
    sensors: int = 8,
    causes: int = 3,
    nodes: int = 8,
    state_dim: int = 5,
    routes_per_node: int = 2,
    noise: float = 0.20,
    train_fraction: float = 2.0 / 3.0,
    sensor_threshold: float = 0.42,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if not (0.5 <= train_fraction < 0.9):
        raise ValueError("train_fraction must be in [0.5, 0.9)")
    if episodes < causes * 12:
        raise ValueError("episodes is too small for the requested cause count")

    seed_results = [
        _single_seed(
            seed=int(seed),
            episodes=episodes,
            sensors=sensors,
            causes=causes,
            nodes=nodes,
            state_dim=state_dim,
            routes_per_node=routes_per_node,
            noise=noise,
            train_fraction=train_fraction,
            sensor_threshold=sensor_threshold,
        )
        for seed in seeds
    ]

    predictive_accuracy = np.array(
        [item["predictive"]["heldout_accuracy"] for item in seed_results], dtype=float
    )
    hebbian_accuracy = np.array(
        [item["hebbian_v0"]["heldout_accuracy"] for item in seed_results], dtype=float
    )
    frozen_accuracy = np.array(
        [item["frozen"]["heldout_accuracy"] for item in seed_results], dtype=float
    )
    surprise_ratios = np.array(
        [item["predictive"]["surprise_late_over_early"] for item in seed_results], dtype=float
    )

    predictive_median = float(np.median(predictive_accuracy))
    hebbian_median = float(np.median(hebbian_accuracy))
    frozen_median = float(np.median(frozen_accuracy))
    wins_hebbian = int(np.sum(predictive_accuracy > hebbian_accuracy))
    wins_frozen = int(np.sum(predictive_accuracy > frozen_accuracy))
    all_equal_budget = all(
        bool(item["event_accounting"]["same_external_sensory_budget"])
        for item in seed_results
    )

    aggregate = {
        "predictive_median_accuracy": predictive_median,
        "hebbian_median_accuracy": hebbian_median,
        "frozen_median_accuracy": frozen_median,
        "predictive_vs_hebbian_median_margin": float(predictive_median - hebbian_median),
        "predictive_vs_frozen_median_margin": float(predictive_median - frozen_median),
        "predictive_vs_hebbian_wins": wins_hebbian,
        "predictive_vs_frozen_wins": wins_frozen,
        "predictive_median_surprise_late_over_early": float(np.median(surprise_ratios)),
        "all_external_sensory_budgets_matched": bool(all_equal_budget),
    }

    # The frozen v1 science gate is intentionally defined for the default
    # 12-seed suite. Smaller seed sets remain useful deterministic smoke tests.
    required_wins = min(V1_MIN_PAIRED_WINS, len(seed_results))
    pass_gate = (
        aggregate["predictive_vs_hebbian_median_margin"] >= V1_MIN_MEDIAN_MARGIN
        and aggregate["predictive_vs_frozen_median_margin"] >= V1_MIN_MEDIAN_MARGIN
        and wins_hebbian >= required_wins
        and wins_frozen >= required_wins
        and aggregate["predictive_median_surprise_late_over_early"] <= V1_SURPRISE_DECLINE_RATIO
        and all_equal_budget
    )

    return {
        "configuration": {
            "seeds": [int(seed) for seed in seeds],
            "seed_count": len(seed_results),
            "episodes_per_seed": episodes,
            "sensors": sensors,
            "causes": causes,
            "nodes": nodes,
            "state_dim": state_dim,
            "routes_per_node": routes_per_node,
            "noise": noise,
            "sensor_threshold": sensor_threshold,
            "train_fraction": train_fraction,
            "episode_internal_ticks": 3,
            "fast_state_reset_between_episodes": True,
            "frozen_min_median_margin": V1_MIN_MEDIAN_MARGIN,
            "frozen_min_paired_wins": V1_MIN_PAIRED_WINS,
            "frozen_surprise_decline_ratio": V1_SURPRISE_DECLINE_RATIO,
        },
        "seeds": seed_results,
        "aggregate": aggregate,
        "verdict": "PASS_PREDICTION_RESIDUAL_GATE" if pass_gate else "FAIL_PREDICTION_RESIDUAL_GATE",
        "claim_boundary": {
            "labels_used_for_learning": False,
            "common_world_tape_within_seed": True,
            "common_initialization_within_seed": True,
            "global_loss_used": False,
            "backpropagation_used": False,
            "world_model_established": False,
            "biological_implementation_established": False,
            "note": "v1 tests whether route-local one-step prediction residual improves held-out cause decoding over both the failed v0 Hebbian learner and a frozen control. Labels are used only by the external evaluator after development.",
        },
    }
