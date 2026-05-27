"""MJCF XML scene builder. Generates complete MuJoCo XML from grid config + devices."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.models.base import DeviceModel


@dataclass
class GridConfig:
    """Configuration for the rail grid floor."""

    x_cells: int = 20
    y_cells: int = 20
    cell_size: float = 1.0  # meters per cell
    rail_width: float = 0.05  # visual-only rail thickness
    world_origin: tuple[float, float] = (0.0, 0.0)  # bottom-left corner of grid


class SceneBuilder:
    """Builds a complete MJCF XML scene string from grid config and device models."""

    def __init__(self, grid_config: GridConfig | None = None):
        self._grid = grid_config or GridConfig()
        self._devices: list[DeviceModel] = []

    def add_device(self, model: "DeviceModel") -> None:
        self._devices.append(model)

    def add_devices(self, models: list["DeviceModel"]) -> None:
        self._devices.extend(models)

    def build_xml(self) -> str:
        parts = [
            self._header(),
            self._compiler_options(),
            self._option_block(),
            self._assets_block(),
            self._worldbody_open(),
            self._ground_geom(),
            self._rail_geometry(),
            self._device_bodies(),
            self._worldbody_close(),
            self._actuator_open(),
            self._device_actuators(),
            self._actuator_close(),
            self._footer(),
        ]
        return "\n".join(parts)

    def _header(self) -> str:
        return '<mujoco model="grid_transport_system">'

    def _compiler_options(self) -> str:
        return '  <compiler angle="radian" autolimits="true"/>'

    def _option_block(self) -> str:
        return '  <option timestep="0.005" gravity="0 0 -9.81" iterations="50" tolerance="1e-8"/>'

    def _assets_block(self) -> str:
        lines = ["  <asset>"]
        for dev in self._devices:
            # Meshes handled via asset manager — placeholders for now
            pass
        lines.append("  </asset>")
        return "\n".join(lines) if len(lines) > 2 else ""

    def _worldbody_open(self) -> str:
        return "  <worldbody>"

    def _ground_geom(self) -> str:
        half_w = (self._grid.x_cells * self._grid.cell_size) / 2
        half_h = (self._grid.y_cells * self._grid.cell_size) / 2
        return f'    <geom name="ground" type="plane" size="{half_w} {half_h} 0.1" rgba="0.85 0.85 0.85 1"/>'

    def _rail_geometry(self) -> str:
        """Generate visual-only rail geometry for the grid."""
        lines = []
        cell = self._grid.cell_size
        ox, oy = self._grid.world_origin
        cols = self._grid.x_cells
        rows = self._grid.y_cells

        total_w = cols * cell
        total_h = rows * cell

        # Horizontal rails (run along X axis) at each Y interval
        for r in range(rows + 1):
            y = oy + r * cell
            x_center = ox + total_w / 2
            lines.append(
                f'    <geom name="rail_h_{r}" type="box"'
                f' pos="{x_center:.3f} {y:.3f} 0.01"'
                f' size="{total_w / 2:.3f} 0.015 0.01"'
                f' rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>'
            )

        # Vertical rails (run along Y axis) at each X interval
        for c in range(cols + 1):
            x = ox + c * cell
            y_center = oy + total_h / 2
            lines.append(
                f'    <geom name="rail_v_{c}" type="box"'
                f' pos="{x:.3f} {y_center:.3f} 0.01"'
                f' size="0.015 {total_h / 2:.3f} 0.01"'
                f' rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>'
            )

        return "\n".join(lines)

    def _device_bodies(self) -> str:
        lines = []
        for dev in self._devices:
            lines.append(dev.body_xml())
        return "\n".join(lines)

    def _worldbody_close(self) -> str:
        return "  </worldbody>"

    def _actuator_open(self) -> str:
        return "  <actuator>"

    def _device_actuators(self) -> str:
        lines = []
        for dev in self._devices:
            lines.append(dev.actuator_xml())
        return "\n".join(lines)

    def _actuator_close(self) -> str:
        return "  </actuator>"

    def _footer(self) -> str:
        return "</mujoco>"

    @property
    def grid_config(self) -> GridConfig:
        return self._grid
