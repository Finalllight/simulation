"""Kinematic data recorder for capturing simulation time series."""

import os
from typing import TYPE_CHECKING

import numpy as np

from simulation.core.state import KinematicState, KinematicTimeseries

if TYPE_CHECKING:
    from simulation.core.engine import SimulationEngine


class KinematicRecorder:
    """Records position, velocity, acceleration for tracked bodies during simulation.

    Usage:
        recorder = KinematicRecorder(engine)
        recorder.add_tracked_body("vehicle_0")
        recorder.reset()
        for _ in range(100):
            engine.step()
            recorder.record_frame()
        ts = recorder.get_timeseries("vehicle_0")
    """

    def __init__(self, engine: "SimulationEngine", record_interval: int = 1):
        self._engine = engine
        self._record_interval = record_interval
        self._tracked_bodies: set[str] = set()
        self._buffer: dict[str, list[KinematicState]] = {}
        self._step_counter: int = 0

    def add_tracked_body(self, body_name: str) -> None:
        """Register a body for kinematic recording."""
        self._tracked_bodies.add(body_name)
        if body_name not in self._buffer:
            self._buffer[body_name] = []

    def record_frame(self) -> None:
        """Snapshot all tracked bodies. Call after each engine.step()."""
        if self._step_counter % self._record_interval != 0:
            self._step_counter += 1
            return

        t = self._engine.time
        for body_name in self._tracked_bodies:
            state = KinematicState(
                time=t,
                body_name=body_name,
                position=self._engine.body_position(body_name),
                velocity=self._engine.body_velocity(body_name),
                acceleration=self._engine.body_acceleration(body_name),
                orientation=self._engine.body_orientation(body_name),
            )
            self._buffer[body_name].append(state)
        self._step_counter += 1

    def get_timeseries(self, body_name: str) -> KinematicTimeseries:
        """Convert the recorded buffer for a body into a KinematicTimeseries."""
        states = self._buffer.get(body_name, [])
        ts = KinematicTimeseries(body_name=body_name)
        for s in states:
            ts.append(s)
        return ts

    def get_all_timeseries(self) -> dict[str, KinematicTimeseries]:
        """Return timeseries for all tracked bodies."""
        return {name: self.get_timeseries(name) for name in self._tracked_bodies}

    def reset(self) -> None:
        """Clear all recorded data and reset step counter."""
        for name in self._buffer:
            self._buffer[name].clear()
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self, output_dir: str) -> dict[str, str]:
        """Export all tracked bodies to CSV files. Returns {body_name: filepath}."""
        os.makedirs(output_dir, exist_ok=True)
        filepaths: dict[str, str] = {}
        for body_name in self._tracked_bodies:
            ts = self.get_timeseries(body_name)
            filepath = os.path.join(output_dir, f"{body_name}.csv")
            self._write_csv(ts, filepath)
            filepaths[body_name] = filepath
        return filepaths

    def export_numpy(self, output_dir: str) -> dict[str, str]:
        """Export all tracked bodies to .npy files. Returns {body_name: filepath}."""
        os.makedirs(output_dir, exist_ok=True)
        filepaths: dict[str, str] = {}
        for body_name in self._tracked_bodies:
            ts = self.get_timeseries(body_name)
            data = np.column_stack(
                [
                    ts.time,
                    ts.position,
                    ts.velocity,
                    ts.acceleration,
                ]
            )
            filepath = os.path.join(output_dir, f"{body_name}.npy")
            np.save(filepath, data)
            filepaths[body_name] = filepath
        return filepaths

    @staticmethod
    def _write_csv(ts: KinematicTimeseries, filepath: str) -> None:
        header = (
            "time,"
            "pos_x,pos_y,pos_z,"
            "vel_x,vel_y,vel_z,"
            "acc_x,acc_y,acc_z"
        )
        rows = []
        for i in range(len(ts.time)):
            row = (
                f"{ts.time[i]:.6f},"
                f"{ts.position[i, 0]:.6f},{ts.position[i, 1]:.6f},{ts.position[i, 2]:.6f},"
                f"{ts.velocity[i, 0]:.6f},{ts.velocity[i, 1]:.6f},{ts.velocity[i, 2]:.6f},"
                f"{ts.acceleration[i, 0]:.6f},{ts.acceleration[i, 1]:.6f},{ts.acceleration[i, 2]:.6f}"
            )
            rows.append(row)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("\n".join(rows) + "\n")

    @property
    def tracked_bodies(self) -> set[str]:
        return self._tracked_bodies.copy()
