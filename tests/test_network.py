import numpy as np

from nssn2.network import NSSNNetwork


def test_network_is_deterministic_sparse_and_bounded():
    first = NSSNNetwork.create(seed=4, sensors=5, nodes=6, state_dim=4, routes_per_node=2)
    second = NSSNNetwork.create(seed=4, sensors=5, nodes=6, state_dim=4, routes_per_node=2)

    assert first.state.shape == (6, 4)
    assert np.array_equal(first.sensor_targets, second.sensor_targets)
    assert np.allclose(first.sensor_access, second.sensor_access)
    assert np.all((first.sensor_access >= 0.0) & (first.sensor_access <= 1.0))
    assert np.allclose(first.sensor_access.sum(axis=2), 1.0)
    assert np.all(np.count_nonzero(first.sensor_access > 0.0, axis=2) <= 2)

    step = first.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), learn=False)
    assert step.sensor_events == 1
    assert step.total_events >= 1
    assert step.node_activity.shape == (6,)
    assert np.all(np.isfinite(first.state))
    assert np.max(np.abs(first.state)) <= 1.0

    touched = set(first.sensor_targets[0].tolist())
    changed = set(np.flatnonzero(np.linalg.norm(first.state, axis=1) > 1e-12).tolist())
    assert changed <= touched


def test_reset_clears_resident_and_pending_state():
    net = NSSNNetwork.create(seed=9, sensors=4, nodes=5, state_dim=4, routes_per_node=2)
    net.step(np.ones(4), learn=False)
    net.reset_state()

    assert np.allclose(net.state, 0.0)
    assert np.allclose(net.pending_events, 0.0)


def test_learning_updates_only_active_sensor_routes_and_preserves_budget():
    net = NSSNNetwork.create(seed=12, sensors=4, nodes=6, state_dim=5, routes_per_node=2)
    before = net.sensor_access.copy()
    support = before > 0.0

    net.step(np.array([1.0, 0.0, 0.0, 0.0]), learn=True)

    assert not np.allclose(net.sensor_access[0], before[0])
    assert np.allclose(net.sensor_access[1:], before[1:])
    assert np.array_equal(net.sensor_access > 0.0, support)
    assert np.all(net.sensor_access >= 0.0)
    assert np.allclose(net.sensor_access.sum(axis=2), 1.0)


def test_learning_false_keeps_route_access_fixed():
    net = NSSNNetwork.create(seed=13, sensors=4, nodes=6, state_dim=5, routes_per_node=2)
    sensor_before = net.sensor_access.copy()
    recurrent_before = net.recurrent_access.copy()

    net.step(np.ones(4), learn=False)
    net.step(np.zeros(4), learn=False)

    assert np.allclose(net.sensor_access, sensor_before)
    assert np.allclose(net.recurrent_access, recurrent_before)


def test_predictive_learning_remembers_local_expectation_and_reduces_repeat_surprise():
    net = NSSNNetwork.create(
        seed=21,
        sensors=4,
        nodes=6,
        state_dim=5,
        routes_per_node=2,
        learning_mode="predictive",
    )
    event = np.array([1.0, 0.0, 0.0, 0.0])
    before = net.sensor_access.copy()

    net.reset_state()
    first = net.step(event, learn=True)
    after_first = net.sensor_access.copy()

    net.reset_state()
    second = net.step(event, learn=True)
    after_second = net.sensor_access.copy()

    first_delta = np.linalg.norm(after_first[0] - before[0])
    second_delta = np.linalg.norm(after_second[0] - after_first[0])

    assert first.mean_surprise > 0.0
    assert second.mean_surprise < first.mean_surprise
    assert first_delta > 0.0
    assert second_delta < first_delta
    assert np.allclose(net.sensor_prediction[1:], 0.0)
    assert np.array_equal(net.sensor_access > 0.0, before > 0.0)
    assert np.allclose(net.sensor_access.sum(axis=2), 1.0)
