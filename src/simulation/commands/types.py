"""Command data types for the simulation system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from simulation.core.state import KinematicTimeseries


class CommandType(Enum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    LIFT_UP = "lift_up"
    LIFT_DOWN = "lift_down"
    STOP = "stop"
    FORM_GROUP = "form_group"
    DISSOLVE_GROUP = "dissolve_group"
    MOVE_TO_CELL = "move_to_cell"


@dataclass
class Command:
    """High-level operation command for a vehicle or group."""

    type: CommandType
    target: str  # vehicle_id or group_id
    params: dict[str, Any] = field(default_factory=dict)
    # Example params:
    #   {"amount": 0.1} for LIFT_UP / LIFT_DOWN
    #   {"cell_x": 5, "cell_y": 10} for MOVE_TO_CELL
    #   {"member_ids": ["v0", "v1", "v2"]} for FORM_GROUP


@dataclass
class CommandResult:
    """Result of executing a command, including kinematic timeseries."""

    success: bool
    command: Command
    error_message: str = ""
    timeseries: dict[str, KinematicTimeseries] = field(default_factory=dict)
