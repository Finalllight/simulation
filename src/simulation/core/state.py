"""Kinematic state data structures for recording simulation output."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class KinematicState:
    """Snapshot of a single body at one timestep."""

    time: float
    body_name: str
    position: np.ndarray  # (3,) world-frame x, y, z
    velocity: np.ndarray  # (3,) world-frame linear velocity
    acceleration: np.ndarray  # (3,) world-frame linear acceleration
    orientation: np.ndarray  # (4,) quaternion w, x, y, z


@dataclass
class KinematicTimeseries:
    """Time series of kinematic data for a single body."""

    body_name: str
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    position: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    acceleration: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    def append(self, state: KinematicState) -> None:
        self.time = np.append(self.time, state.time)
        self.position = np.vstack([self.position, state.position.reshape(1, 3)])
        self.velocity = np.vstack([self.velocity, state.velocity.reshape(1, 3)])
        self.acceleration = np.vstack([self.acceleration, state.acceleration.reshape(1, 3)])

    def __len__(self) -> int:
        return len(self.time)
