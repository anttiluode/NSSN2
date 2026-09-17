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
