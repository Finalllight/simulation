"""Abstract base class for all device models."""

from abc import ABC, abstractmethod


class DeviceModel(ABC):
    """Protocol that every device type must implement.

    Each device generates its own MJCF XML fragments (body subtree + actuators)
    and declares its naming conventions for joints, actuators, and mesh assets.
    """

    def __init__(self, device_id: str):
        self.device_id = device_id

    @abstractmethod
    def body_xml(self) -> str:
        """Generate the <body> subtree MJCF XML for this device."""
        ...

    @abstractmethod
    def actuator_xml(self) -> str:
        """Generate the <actuator> entries MJCF XML for this device."""
        ...

    @abstractmethod
    def joint_names(self) -> list[str]:
        """All joint names belonging to this device."""
        ...

    @abstractmethod
    def actuator_names(self) -> list[str]:
        """All actuator names belonging to this device."""
        ...

    @abstractmethod
    def mesh_assets(self) -> list[str]:
        """Mesh file paths needed by this device. Empty list if using primitives."""
        ...

    @abstractmethod
    def init_joint_positions(self) -> dict[str, float]:
        """Initial qpos values for each joint at simulation start."""
        ...
