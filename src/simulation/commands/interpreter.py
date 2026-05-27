"""Command interpreter: translates high-level commands into MuJoCo actuator targets."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from simulation.commands.types import Command, CommandResult, CommandType

if TYPE_CHECKING:
    from simulation.core.engine import SimulationEngine
    from simulation.core.state import KinematicRecorder
    from simulation.grid.manager import GridManager
    from simulation.models.group import VehicleGroup
    from simulation.models.transport_vehicle import TransportVehicleModel

logger = logging.getLogger(__name__)


class CommandInterpreter:
    """Translates high-level human commands into physics engine control signals.

    Handles:
      - Target expansion (group → member vehicle IDs)
      - Grid coordinate computation and validation
      - Actuator target setting
      - Stepping until completion or timeout
      - Recording kinematic data during execution
    """

    # Direction vectors for move commands
    _DIRECTION_VECTORS: dict[CommandType, tuple[int, int]] = {
        CommandType.MOVE_FORWARD: (0, 1),
        CommandType.MOVE_BACKWARD: (0, -1),
        CommandType.MOVE_LEFT: (-1, 0),
        CommandType.MOVE_RIGHT: (1, 0),
    }

    def __init__(
        self,
        engine: "SimulationEngine",
        grid: "GridManager",
        vehicle_models: dict[str, "TransportVehicleModel"],
        recorder_class: type = None,
        groups: dict[str, "VehicleGroup"] | None = None,
    ):
        self._engine = engine
        self._grid = grid
        self._vehicle_models = vehicle_models
        self._groups: dict[str, VehicleGroup] = groups or {}
        self._recorder_class = recorder_class
        self._position_tolerance: float = 0.02  # meters
        self._max_steps: int = 5000  # timeout: dt=0.005 * 5000 = 25s simulated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, command: Command) -> CommandResult:
        """Dispatch a command to the appropriate handler."""
        handlers = {
            CommandType.MOVE_FORWARD: self._handle_move,
            CommandType.MOVE_BACKWARD: self._handle_move,
            CommandType.MOVE_LEFT: self._handle_move,
            CommandType.MOVE_RIGHT: self._handle_move,
            CommandType.LIFT_UP: self._handle_lift,
            CommandType.LIFT_DOWN: self._handle_lift,
            CommandType.STOP: self._handle_stop,
            CommandType.FORM_GROUP: self._handle_form_group,
            CommandType.DISSOLVE_GROUP: self._handle_dissolve_group,
            CommandType.MOVE_TO_CELL: self._handle_move_to_cell,
        }
        handler = handlers.get(command.type)
        if handler is None:
            return CommandResult(False, command, f"Unknown command type: {command.type}")

        try:
            return handler(command)
        except Exception as e:
            logger.exception(f"Command failed: {command}")
            return CommandResult(False, command, str(e))

    # ------------------------------------------------------------------
    # Move handler
    # ------------------------------------------------------------------
    def _handle_move(self, command: Command) -> CommandResult:
        dx, dy = self._DIRECTION_VECTORS.get(command.type, (0, 0))
        expanded = self._expand_target(command.target)

        target_positions: dict[str, tuple[int, int]] = {}
        target_world: dict[str, tuple[float, float]] = {}

        for vid in expanded:
            current = self._grid.get_position(vid)
            if current is None:
                return CommandResult(False, command, f"Vehicle '{vid}' not on grid")
            new_cell = (current[0] + dx, current[1] + dy)
            if not self._grid.is_within_bounds(*new_cell):
                return CommandResult(
                    False, command,
                    f"Vehicle '{vid}' move to {new_cell} is out of bounds"
                )
            if self._grid.is_occupied(*new_cell, exclude=vid):
                return CommandResult(
                    False, command,
                    f"Cell {new_cell} is occupied"
                )
            target_positions[vid] = new_cell
            target_world[vid] = self._grid.cell_to_world(*new_cell)

        return self._execute_move(command, target_world, target_positions)

    # ------------------------------------------------------------------
    # Lift handler
    # ------------------------------------------------------------------
    def _handle_lift(self, command: Command) -> CommandResult:
        amount = command.params.get("amount", 0.05)
        if command.type == CommandType.LIFT_DOWN:
            amount = -amount

        expanded = self._expand_target(command.target)
        model = self._vehicle_models[command.target]
        max_lift = model.params.max_lift

        targets: dict[str, float] = {}
        for vid in expanded:
            current_z = self._engine.joint_position(model.z_joint)
            new_z = max(0.0, min(max_lift, current_z + amount))
            targets[vid] = new_z

        return self._execute_lift(command, targets)

    # ------------------------------------------------------------------
    # Stop handler
    # ------------------------------------------------------------------
    def _handle_stop(self, command: Command) -> CommandResult:
        expanded = self._expand_target(command.target)
        for vid in expanded:
            model = self._vehicle_models.get(vid)
            if model is None:
                continue
            # Hold current position by setting target to current
            self._engine.set_position_target(
                model.x_actuator, self._engine.joint_position(model.x_joint)
            )
            self._engine.set_position_target(
                model.y_actuator, self._engine.joint_position(model.y_joint)
            )
            self._engine.set_position_target(
                model.z_actuator, self._engine.joint_position(model.z_joint)
            )
        return CommandResult(True, command)

    # ------------------------------------------------------------------
    # Move to cell handler
    # ------------------------------------------------------------------
    def _handle_move_to_cell(self, command: Command) -> CommandResult:
        cx = command.params.get("cell_x")
        cy = command.params.get("cell_y")
        if cx is None or cy is None:
            return CommandResult(False, command, "Missing cell_x or cell_y parameter")

        expanded = self._expand_target(command.target)
        target_world: dict[str, tuple[float, float]] = {}
        target_positions: dict[str, tuple[int, int]] = {}

        for vid in expanded:
            if not self._grid.is_within_bounds(cx, cy):
                return CommandResult(False, command, f"Cell ({cx}, {cy}) out of bounds")
            if self._grid.is_occupied(cx, cy, exclude=vid):
                return CommandResult(False, command, f"Cell ({cx}, {cy}) is occupied")
            target_positions[vid] = (cx, cy)
            target_world[vid] = self._grid.cell_to_world(cx, cy)

        return self._execute_move(command, target_world, target_positions)

    # ------------------------------------------------------------------
    # Group handlers
    # ------------------------------------------------------------------
    def _handle_form_group(self, command: Command) -> CommandResult:
        from simulation.models.group import VehicleGroup

        member_ids = command.params.get("member_ids", [])
        if len(member_ids) < 2:
            return CommandResult(False, command, "Need at least 2 members to form a group")

        # Check no member is already in a group
        for gid, group in self._groups.items():
            for mid in member_ids:
                if mid in group.member_ids:
                    return CommandResult(
                        False, command,
                        f"Vehicle '{mid}' is already in group '{gid}'"
                    )

        group_id = command.target
        relative_offsets = {}
        positions = {}
        for vid in member_ids:
            pos = self._grid.get_position(vid)
            if pos is None:
                return CommandResult(False, command, f"Vehicle '{vid}' not on grid")
            positions[vid] = pos

        anchor_id = member_ids[0]
        anchor_pos = positions[anchor_id]
        for vid in member_ids:
            relative_offsets[vid] = (
                positions[vid][0] - anchor_pos[0],
                positions[vid][1] - anchor_pos[1],
            )

        group = VehicleGroup(group_id, member_ids, relative_offsets)
        self._groups[group_id] = group
        return CommandResult(True, command)

    def _handle_dissolve_group(self, command: Command) -> CommandResult:
        group_id = command.target
        if group_id not in self._groups:
            return CommandResult(False, command, f"Group '{group_id}' not found")
        del self._groups[group_id]
        return CommandResult(True, command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _expand_target(self, target: str) -> list[str]:
        """If target is a group_id, return all member vehicle IDs."""
        group = self._groups.get(target)
        if group is not None:
            return list(group.member_ids)
        return [target]

    def _execute_move(
        self,
        command: Command,
        target_world: dict[str, tuple[float, float]],
        target_positions: dict[str, tuple[int, int]],
    ) -> CommandResult:
        """Set position targets, step until arrival, return result with timeseries."""
        if self._recorder_class is None:
            return self._execute_move_fast(command, target_world, target_positions)

        recorder = self._recorder_class(self._engine)
        for vid in target_world:
            model = self._vehicle_models[vid]
            recorder.add_tracked_body(model.body_name)
            recorder.add_tracked_body(model.lift_body_name)

        recorder.reset()
        self._set_move_targets(target_world)

        arrived = self._wait_for_arrival_move(target_world, recorder)

        if not arrived:
            return CommandResult(False, command, "Move timed out")
        self._update_occupancy(target_positions)
        return CommandResult(
            True, command,
            timeseries=recorder.get_all_timeseries()
        )

    def _execute_move_fast(
        self,
        command: Command,
        target_world: dict[str, tuple[float, float]],
        target_positions: dict[str, tuple[int, int]],
    ) -> CommandResult:
        """Execute a move without recording (fast path)."""
        self._set_move_targets(target_world)
        self._wait_for_arrival_simple(target_world)
        self._update_occupancy(target_positions)
        return CommandResult(True, command)

    def _execute_lift(
        self, command: Command, targets: dict[str, float]
    ) -> CommandResult:
        for vid, new_z in targets.items():
            model = self._vehicle_models.get(vid)
            if model:
                self._engine.set_position_target(model.z_actuator, new_z)

        recorder = None
        if self._recorder_class is not None:
            recorder = self._recorder_class(self._engine)
            for vid in targets:
                model = self._vehicle_models.get(vid)
                if model:
                    recorder.add_tracked_body(model.body_name)
                    recorder.add_tracked_body(model.lift_body_name)
            recorder.reset()

        arrived = self._wait_for_completion_lift(targets, recorder)

        if not arrived:
            return CommandResult(False, command, "Lift timed out")

        if recorder:
            return CommandResult(
                True, command, timeseries=recorder.get_all_timeseries()
            )
        return CommandResult(True, command)

    def _set_move_targets(self, target_world: dict[str, tuple[float, float]]) -> None:
        for vid, (wx, wy) in target_world.items():
            model = self._vehicle_models[vid]
            self._engine.set_position_target(model.x_actuator, wx)
            self._engine.set_position_target(model.y_actuator, wy)

    def _wait_for_arrival_move(
        self,
        target_world: dict[str, tuple[float, float]],
        recorder,
    ) -> bool:
        for _ in range(self._max_steps):
            self._engine.step()
            recorder.record_frame()

            all_arrived = True
            for vid, (tx, ty) in target_world.items():
                pos = self._engine.body_position(vid)
                if np.hypot(pos[0] - tx, pos[1] - ty) > self._position_tolerance:
                    all_arrived = False
                    break

            if all_arrived:
                return True
        return False

    def _wait_for_arrival_simple(self, target_world: dict[str, tuple[float, float]]) -> bool:
        for _ in range(self._max_steps):
            self._engine.step()
            all_arrived = True
            for vid, (tx, ty) in target_world.items():
                pos = self._engine.body_position(vid)
                if np.hypot(pos[0] - tx, pos[1] - ty) > self._position_tolerance:
                    all_arrived = False
                    break
            if all_arrived:
                return True
        return False

    def _wait_for_completion_lift(
        self, targets: dict[str, float], recorder=None
    ) -> bool:
        for _ in range(self._max_steps):
            self._engine.step()
            if recorder is not None:
                recorder.record_frame()
            all_arrived = True
            for vid, target_z in targets.items():
                model = self._vehicle_models[vid]
                current_z = self._engine.joint_position(model.z_joint)
                if abs(current_z - target_z) > self._position_tolerance:
                    all_arrived = False
                    break
            if all_arrived:
                return True
        return False

    def _update_occupancy(self, target_positions: dict[str, tuple[int, int]]) -> None:
        for vid, new_cell in target_positions.items():
            self._grid.vacate(vid)
            self._grid.occupy(vid, *new_cell)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def groups(self) -> dict[str, "VehicleGroup"]:
        return self._groups

    @property
    def position_tolerance(self) -> float:
        return self._position_tolerance

    @position_tolerance.setter
    def position_tolerance(self, value: float) -> None:
        self._position_tolerance = value
