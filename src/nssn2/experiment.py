from __future__ import annotations

import numpy as np

from .network import NSSNNetwork
from .world import make_world_tape


def _encode_episode(net: NSSNNetwork, event: np.ndarray, learn: bool) -> tuple[np.ndarray, int, int]:
    """Run one cause episode through three internal ticks.

    Fast electrical state is reset between v0 episodes while learned route
    access persists. This isolates structural/plastic development from the
    previous episode's accidental fast-state context.
    """
    net.reset_state()
    sensor_count = 0
    recurrent_count = 0

    first = net.step(event, learn=learn)
    sensor_count += first.sensor_events
    recurrent_count += first.recurrent_events

    zero = np.zeros_like(event)
    for _ in range(2):
        step = net.step(zero, learn=learn)
        sensor_count += step.sensor_events
        recurrent_count += step.recurrent_events

    return net.snapshot(), sensor_count, recurrent_count


def _distance_metrics(states: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    within: list[float] = []
    between: list[float] = []
    scale = np.sqrt(states.shape[1])
    for i in range(states.shape[0]):
        for j in range(i + 1, states.shape[0]):
            distance = float(np.linalg.norm(states[i] - states[j]) / scale)
            if labels[i] == labels[j]:
                within.append(distance)
            else:
                between.append(distance)
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    ratio = between_mean / max(within_mean, 1e-12)
    return within_mean, between_mean, float(ratio)


def _centroids(states: np.ndarray, labels: np.ndarray, causes: int) -> np.ndarray:
    centers = np.zeros((causes, states.shape[1]), dtype=float)
    for cause in range(causes):
        selected = states[labels == cause]
        if selected.size == 0:
            raise ValueError("reference split is missing a cause")
        centers[cause] = selected.mean(axis=0)
    return centers


def _predict(states: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(states[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def _route_specialization(access: np.ndarray) -> float:
    vectors = np.asarray(access, dtype=float).reshape(-1, access.shape[-1])
    if vectors.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    similarities = unit @ unit.T
    upper = similarities[np.triu_indices(vectors.shape[0], k=1)]
    return float(np.mean(1.0 - upper))


def _evaluate(
    net: NSSNNetwork,
    reference_events: np.ndarray,
    reference_labels: np.ndarray,
    eval_events: np.ndarray,
    eval_labels: np.ndarray,
    causes: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    reference_states = np.array([_encode_episode(net, event, False)[0] for event in reference_events])
    eval_states = np.array([_encode_episode(net, event, False)[0] for event in eval_events])
    centers = _centroids(reference_states, reference_labels, causes)
    predictions = _predict(eval_states, centers)
    within, between, ratio = _distance_metrics(eval_states, eval_labels)
    metrics = {
        "within_cause_distance": within,
        "between_cause_distance": between,
        "separation_ratio": ratio,
        "heldout_accuracy": _accuracy(predictions, eval_labels),
    }
    return metrics, predictions, eval_states


def run_v0(
    seed: int = 17,
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
    if not (0.5 <= train_fraction < 0.9):
        raise ValueError("train_fraction must be in [0.5, 0.9)")
    if episodes < causes * 12:
        raise ValueError("episodes is too small for the requested cause count")

    tape = make_world_tape(seed=seed, episodes=episodes, sensors=sensors, causes=causes, noise=noise)
    sensory = np.where(tape.events >= sensor_threshold, tape.events, 0.0)

    learner = NSSNNetwork.create(seed=seed + 1000, sensors=sensors, nodes=nodes, state_dim=state_dim, routes_per_node=routes_per_node)
    frozen = NSSNNetwork.create(seed=seed + 1000, sensors=sensors, nodes=nodes, state_dim=state_dim, routes_per_node=routes_per_node)

    learner_sensor_initial = learner.sensor_access.copy()
    learner_recurrent_initial = learner.recurrent_access.copy()
    frozen_sensor_initial = frozen.sensor_access.copy()
    frozen_recurrent_initial = frozen.recurrent_access.copy()
    initial_specialization = _route_specialization(learner.sensor_access)

    train_count = int(round(episodes * train_fraction))
    train_count = min(max(train_count, causes * 4), episodes - causes * 4)

    learner_sensor_events = 0
    frozen_sensor_events = 0
    learner_recurrent_events = 0
    frozen_recurrent_events = 0

    for index in range(train_count):
        _, sensors_l, recurrent_l = _encode_episode(learner, sensory[index], True)
        _, sensors_f, recurrent_f = _encode_episode(frozen, sensory[index], False)
        learner_sensor_events += sensors_l
        frozen_sensor_events += sensors_f
        learner_recurrent_events += recurrent_l
        frozen_recurrent_events += recurrent_f

    # After development, freeze both machines and re-encode the common reference
    # tape. Labels are consulted only here, in the external evaluator.
    reference_events = sensory[:train_count]
    reference_labels = tape.labels[:train_count]
    eval_events = sensory[train_count:]
    eval_labels = tape.labels[train_count:]

    learner_metrics, learner_predictions, _ = _evaluate(
        learner, reference_events, reference_labels, eval_events, eval_labels, causes
    )
    frozen_metrics, frozen_predictions, _ = _evaluate(
        frozen, reference_events, reference_labels, eval_events, eval_labels, causes
    )

    learner_route_change = float(
        np.linalg.norm(learner.sensor_access - learner_sensor_initial)
        + np.linalg.norm(learner.recurrent_access - learner_recurrent_initial)
    )
    frozen_route_change = float(
        np.linalg.norm(frozen.sensor_access - frozen_sensor_initial)
        + np.linalg.norm(frozen.recurrent_access - frozen_recurrent_initial)
    )
    final_specialization = _route_specialization(learner.sensor_access)
    specialization_delta = float(final_specialization - initial_specialization)

    shuffle_rng = np.random.default_rng(seed + 9000)
    shuffled_accuracies = []
    for _ in range(64):
        shuffled = shuffle_rng.permutation(eval_labels)
        shuffled_accuracies.append(_accuracy(learner_predictions, shuffled))
    shuffled_mean = float(np.mean(shuffled_accuracies))
    shuffled_std = float(np.std(shuffled_accuracies))
    chance = 1.0 / causes

    learner_out = {
        **learner_metrics,
        "route_change_norm": learner_route_change,
        "initial_route_specialization": initial_specialization,
        "final_route_specialization": final_specialization,
        "route_specialization_delta": specialization_delta,
    }
    frozen_out = {
        **frozen_metrics,
        "route_change_norm": frozen_route_change,
        "route_specialization": _route_specialization(frozen.sensor_access),
    }

    event_accounting = {
        "learner_sensor_events": int(learner_sensor_events),
        "frozen_sensor_events": int(frozen_sensor_events),
        "learner_recurrent_events": int(learner_recurrent_events),
        "frozen_recurrent_events": int(frozen_recurrent_events),
        "same_external_sensory_budget": learner_sensor_events == frozen_sensor_events,
    }

    pass_gate = (
        learner_metrics["separation_ratio"] > frozen_metrics["separation_ratio"]
        and learner_metrics["heldout_accuracy"] > frozen_metrics["heldout_accuracy"]
        and learner_route_change > 0.05
        and abs(specialization_delta) > 1e-4
        and event_accounting["same_external_sensory_budget"]
        and abs(shuffled_mean - chance) <= 0.12
    )

    return {
        "configuration": {
            "seed": seed,
            "episodes": episodes,
            "train_episodes": train_count,
            "eval_episodes": episodes - train_count,
            "sensors": sensors,
            "causes": causes,
            "nodes": nodes,
            "state_dim": state_dim,
            "routes_per_node": routes_per_node,
            "noise": noise,
            "sensor_threshold": sensor_threshold,
            "episode_internal_ticks": 3,
            "fast_state_reset_between_episodes": True,
        },
        "learner": learner_out,
        "frozen": frozen_out,
        "shuffled": {
            "mean_accuracy": shuffled_mean,
            "std_accuracy": shuffled_std,
            "chance": chance,
            "repeats": 64,
        },
        "event_accounting": event_accounting,
        "verdict": "PASS_DEVELOPING_RESIDENT_STRUCTURE" if pass_gate else "FAIL_DEVELOPING_RESIDENT_STRUCTURE",
        "claim_boundary": {
            "labels_used_for_learning": False,
            "common_world_tape": True,
            "demo_is_evidence": False,
            "world_model_established": False,
            "biological_implementation_established": False,
            "note": "v0 isolates whether local route adaptation improves cause-specific resident responses on a repeated synthetic world. It does not establish object concepts, a predictive world model, or biological realism.",
        },
    }
