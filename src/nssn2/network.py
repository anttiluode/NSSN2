from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NetworkStep:
    sensor_events: int
    recurrent_events: int
    total_events: int
    node_activity: np.ndarray
    emitted: np.ndarray
    mean_surprise: float = 0.0


@dataclass
class NSSNNetwork:
    state: np.ndarray
    base_ops: np.ndarray
    readout: np.ndarray
    sensor_targets: np.ndarray
    sensor_access: np.ndarray
    recurrent_targets: np.ndarray
    recurrent_access: np.ndarray
    pending_events: np.ndarray
    sensor_prediction: np.ndarray
    recurrent_prediction: np.ndarray
    learning_mode: str = "hebbian"
    active_gain: float = 0.16
    active_threshold: float = 0.22
    active_slope: float = 0.08
    publish_threshold: float = 0.22
    learning_rate: float = 0.12
    prediction_rate: float = 0.5

    @classmethod
    def create(
        cls,
        seed: int,
        sensors: int,
        nodes: int,
        state_dim: int,
        routes_per_node: int,
        learning_mode: str = "hebbian",
    ) -> "NSSNNetwork":
        if min(sensors, nodes, state_dim, routes_per_node) <= 0:
            raise ValueError("all dimensions must be positive")
        if routes_per_node > nodes:
            raise ValueError("routes_per_node cannot exceed node count")
        if learning_mode not in {"hebbian", "predictive"}:
            raise ValueError("learning_mode must be 'hebbian' or 'predictive'")

        rng = np.random.default_rng(seed)

        base_ops = np.empty((nodes, state_dim, state_dim), dtype=float)
        for node in range(nodes):
            raw = rng.normal(0.0, 0.24, size=(state_dim, state_dim))
            raw += np.eye(state_dim) * rng.uniform(0.45, 0.70)
            radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
            base_ops[node] = raw * (0.78 / max(radius, 1e-9))

        readout = rng.normal(size=(nodes, state_dim))
        readout /= np.linalg.norm(readout, axis=1, keepdims=True) + 1e-12

        sensor_targets = np.empty((sensors, routes_per_node), dtype=int)
        for sensor in range(sensors):
            sensor_targets[sensor] = rng.choice(nodes, size=routes_per_node, replace=False)

        recurrent_targets = np.empty((nodes, routes_per_node), dtype=int)
        for source in range(nodes):
            candidates = np.array([n for n in range(nodes) if n != source], dtype=int)
            if candidates.size >= routes_per_node:
                recurrent_targets[source] = rng.choice(candidates, size=routes_per_node, replace=False)
            else:
                recurrent_targets[source] = rng.choice(nodes, size=routes_per_node, replace=False)

        sensor_access = cls._make_sparse_access(rng, (sensors, routes_per_node), state_dim)
        recurrent_access = cls._make_sparse_access(rng, (nodes, routes_per_node), state_dim)

        return cls(
            state=np.zeros((nodes, state_dim), dtype=float),
            base_ops=base_ops,
            readout=readout,
            sensor_targets=sensor_targets,
            sensor_access=sensor_access,
            recurrent_targets=recurrent_targets,
            recurrent_access=recurrent_access,
            pending_events=np.zeros(nodes, dtype=float),
            sensor_prediction=np.zeros_like(sensor_access),
            recurrent_prediction=np.zeros_like(recurrent_access),
            learning_mode=learning_mode,
        )

    @staticmethod
    def _make_sparse_access(
        rng: np.random.Generator,
        prefix_shape: tuple[int, int],
        state_dim: int,
    ) -> np.ndarray:
        out = np.zeros((*prefix_shape, state_dim), dtype=float)
        support_size = min(2, state_dim)
        for index in np.ndindex(prefix_shape):
            support = rng.choice(state_dim, size=support_size, replace=False)
            weights = rng.uniform(0.2, 1.0, size=support_size)
            weights /= weights.sum()
            out[index][support] = weights
        return out

    def reset_state(self) -> None:
        self.state[...] = 0.0
        self.pending_events[...] = 0.0

    def snapshot(self) -> np.ndarray:
        return self.state.reshape(-1).copy()

    def _sigmoid(self, value: np.ndarray) -> np.ndarray:
        z = np.clip((value - self.active_threshold) / self.active_slope, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _adapt_route(
        self,
        access: np.ndarray,
        target_state: np.ndarray,
        amplitude: float,
        prediction: np.ndarray | None = None,
    ) -> float:
        support = access > 0.0
        if not np.any(support) or amplitude <= 1e-12:
            return 0.0

        if self.learning_mode == "predictive":
            if prediction is None:
                raise ValueError("predictive learning requires route-local prediction state")
            residual = target_state - prediction
            desired = np.abs(residual[support]) + 0.03
            surprise = float(np.linalg.norm(residual[support]) / np.sqrt(np.count_nonzero(support)))
            prediction[...] = (
                (1.0 - self.prediction_rate) * prediction
                + self.prediction_rate * target_state
            )
        else:
            desired = np.abs(target_state[support]) + 0.03
            surprise = 0.0

        desired /= desired.sum()
        eta = min(0.5, self.learning_rate * float(amplitude))
        updated = (1.0 - eta) * access[support] + eta * desired
        updated = np.maximum(updated, 1e-12)
        updated /= updated.sum()
        access[support] = updated
        access[~support] = 0.0
        return surprise

    def step(self, sensor_event: np.ndarray, learn: bool = True) -> NetworkStep:
        sensor_event = np.asarray(sensor_event, dtype=float)
        if sensor_event.shape != (self.sensor_targets.shape[0],):
            raise ValueError("sensor_event has wrong shape")
        if np.any(sensor_event < 0.0):
            raise ValueError("sensor events must be non-negative")

        previous_pending = self.pending_events.copy()
        node_inputs = np.zeros_like(self.state)
        active_sensor_channels = int(np.count_nonzero(sensor_event > 1e-12))
        recurrent_sources = int(np.count_nonzero(previous_pending > 1e-12))

        for sensor, amplitude in enumerate(sensor_event):
            if amplitude <= 1e-12:
                continue
            for route, target in enumerate(self.sensor_targets[sensor]):
                node_inputs[target] += float(amplitude) * self.sensor_access[sensor, route]

        for source, amplitude in enumerate(previous_pending):
            if amplitude <= 1e-12:
                continue
            for route, target in enumerate(self.recurrent_targets[source]):
                node_inputs[target] += float(amplitude) * self.recurrent_access[source, route]

        transported = np.einsum("nij,nj->ni", self.base_ops, self.state)
        gate = self._sigmoid(np.abs(self.state))
        local_active = self.active_gain * gate * np.tanh(2.0 * self.state)
        self.state[...] = np.tanh(transported + node_inputs + local_active)

        surprises: list[float] = []
        if learn:
            for sensor, amplitude in enumerate(sensor_event):
                if amplitude <= 1e-12:
                    continue
                for route, target in enumerate(self.sensor_targets[sensor]):
                    surprises.append(
                        self._adapt_route(
                            self.sensor_access[sensor, route],
                            self.state[target],
                            float(amplitude),
                            self.sensor_prediction[sensor, route],
                        )
                    )
            for source, amplitude in enumerate(previous_pending):
                if amplitude <= 1e-12:
                    continue
                for route, target in enumerate(self.recurrent_targets[source]):
                    surprises.append(
                        self._adapt_route(
                            self.recurrent_access[source, route],
                            self.state[target],
                            float(amplitude),
                            self.recurrent_prediction[source, route],
                        )
                    )

        projected = np.abs(np.sum(self.readout * self.state, axis=1))
        node_activity = np.linalg.norm(self.state, axis=1) / np.sqrt(self.state.shape[1])
        emitted = np.clip(np.maximum(projected, node_activity) - self.publish_threshold, 0.0, 1.0)
        self.pending_events[...] = emitted

        return NetworkStep(
            sensor_events=active_sensor_channels,
            recurrent_events=recurrent_sources,
            total_events=active_sensor_channels + recurrent_sources,
            node_activity=node_activity.copy(),
            emitted=emitted.copy(),
            mean_surprise=float(np.mean(surprises)) if surprises else 0.0,
        )
