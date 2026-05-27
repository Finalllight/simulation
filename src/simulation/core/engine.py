"""MuJoCo simulation engine wrapper (headless, no rendering)."""

import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np


class SimulationEngine:
    """Thin wrapper around MuJoCo MjModel + MjData for headless physics simulation."""

    def __init__(self, xml_string: str, mesh_dir: str = ""):
        self._mesh_dir = str(mesh_dir) if mesh_dir else tempfile.mkdtemp()
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._body_ids: dict[str, int] = {}
        self._joint_ids: dict[str, int] = {}
        self._actuator_ids: dict[str, int] = {}
        self._step_count: int = 0

        self._compile(xml_string)

    def _compile(self, xml_string: str) -> None:
        xml_bytes = xml_string.encode("utf-8")
        if self._mesh_dir:
            tmp_path = os.path.join(self._mesh_dir, "_scene.xml")
            Path(self._mesh_dir).mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(xml_string)
            self._model = mujoco.MjModel.from_xml_path(tmp_path)
        else:
            self._model = mujoco.MjModel.from_xml_string(xml_bytes)

        self._data = mujoco.MjData(self._model)
        self._cache_ids()

    def _cache_ids(self) -> None:
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                self._body_ids[name] = i
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._joint_ids[name] = i
        for i in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._actuator_ids[name] = i

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

    def step_n(self, n: int) -> None:
        """Advance the simulation by n timesteps."""
        for _ in range(n):
            self.step()

    def body_position(self, name: str) -> np.ndarray:
        """World-frame position (x, y, z) of a body."""
        return self._data.xpos[self._body_id(name)].copy()

    def body_velocity(self, name: str) -> np.ndarray:
        """World-frame linear velocity (vx, vy, vz) of a body."""
        bid = self._body_id(name)
        # 6D velocity: linear (3) + angular (3)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, bid)
        lin_vel = jacp @ self._data.qvel
        return lin_vel

    def body_acceleration(self, name: str) -> np.ndarray:
        """World-frame linear acceleration of a body, computed via Jacobian + qacc."""
        bid = self._body_id(name)
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, bid)
        lin_acc = jacp @ self._data.qacc
        return lin_acc

    def body_orientation(self, name: str) -> np.ndarray:
        """Quaternion (w, x, y, z) of a body's orientation."""
        return self._data.xquat[self._body_id(name)].copy()

    def joint_position(self, name: str) -> float:
        """Current joint position (qpos)."""
        jid = self._joint_ids[name]
        qpos_addr = self._model.jnt_qposadr[jid]
        return float(self._data.qpos[qpos_addr])

    def joint_velocity(self, name: str) -> float:
        """Current joint velocity (qvel)."""
        jid = self._joint_ids[name]
        dof_addr = self._model.jnt_dofadr[jid]
        return float(self._data.qvel[dof_addr])

    def set_position_target(self, actuator_name: str, target: float) -> None:
        """Set the position target for a position actuator."""
        aid = self._actuator_ids[actuator_name]
        self._data.ctrl[aid] = target

    def set_velocity_target(self, actuator_name: str, target: float) -> None:
        """Set the velocity target for a velocity actuator."""
        aid = self._actuator_ids[actuator_name]
        self._data.ctrl[aid] = target

    def get_body_names(self) -> list[str]:
        return list(self._body_ids.keys())

    def get_joint_names(self) -> list[str]:
        return list(self._joint_ids.keys())

    def get_actuator_names(self) -> list[str]:
        return list(self._actuator_ids.keys())

    def forward(self) -> None:
        """Recompute kinematics (positions, velocities) from current qpos/qvel.

        Call this after manually setting qpos or qvel values."""
        mujoco.mj_forward(self._model, self._data)

    def reset(self) -> None:
        """Reset simulation data to initial state."""
        mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0

    def _body_id(self, name: str) -> int:
        if name not in self._body_ids:
            raise KeyError(f"Body '{name}' not found. Available: {list(self._body_ids.keys())}")
        return self._body_ids[name]

    @property
    def time(self) -> float:
        return self._data.time

    @property
    def dt(self) -> float:
        return self._model.opt.timestep

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data
