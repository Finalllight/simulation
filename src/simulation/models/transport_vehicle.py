"""Grid-based four-wheel transport vehicle model for MuJoCo."""

from dataclasses import dataclass, field
import math

from simulation.models.base import DeviceModel


@dataclass
class TransportVehicleParams:
    """Physical and performance parameters for a transport vehicle."""

    device_id: str
    chassis_mass: float = 120.0  # kg
    chassis_dims: tuple[float, float, float] = (0.9, 0.7, 0.35)  # (lx, ly, lz) meters
    wheel_radius: float = 0.06  # meters
    wheel_width: float = 0.04  # meters
    max_speed: float = 1.2  # m/s
    max_acceleration: float = 0.6  # m/s^2
    max_lift: float = 0.4  # meters
    lift_speed: float = 0.08  # m/s
    initial_grid_pos: tuple[int, int] = (0, 0)  # (cx, cy) on the grid
    initial_lift: float = 0.0  # meters
    cell_size: float = 1.0  # meters per grid cell (set externally)

    # Servo gains
    xy_kp: float = 1000.0
    xy_kv: float = 600.0
    z_kp: float = 20000.0
    z_kv: float = 900.0

    def __post_init__(self):
        if self.xy_kp <= 0 or self.xy_kv <= 0:
            self.xy_kp = self.chassis_mass * 100
            self.xy_kv = 2.0 * math.sqrt(self.xy_kp)

    def initial_world_pos(self) -> tuple[float, float]:
        """World (x, y) for the center of the initial grid cell."""
        cx, cy = self.initial_grid_pos
        cs = self.cell_size
        return (cx * cs + cs / 2, cy * cs + cs / 2)


class TransportVehicleModel(DeviceModel):
    """MuJoCo model of a grid-based four-wheel transport vehicle.

    Kinematic structure:
      vehicle body (slide X + slide Y)
        └── lift body (slide Z)
              └── wheels × 4 (passive hinges)
    """

    def __init__(self, params: TransportVehicleParams):
        super().__init__(params.device_id)
        self.params = params

    # ------------------------------------------------------------------
    # Naming conventions
    # ------------------------------------------------------------------
    def _n(self, suffix: str) -> str:
        return f"{self.device_id}_{suffix}"

    @property
    def body_name(self) -> str:
        return self.device_id

    @property
    def lift_body_name(self) -> str:
        return self._n("lift")

    @property
    def x_joint(self) -> str:
        return self._n("rail_x")

    @property
    def y_joint(self) -> str:
        return self._n("rail_y")

    @property
    def z_joint(self) -> str:
        return self._n("lift_z")

    @property
    def x_actuator(self) -> str:
        return self._n("act_x")

    @property
    def y_actuator(self) -> str:
        return self._n("act_y")

    @property
    def z_actuator(self) -> str:
        return self._n("act_z")

    # ------------------------------------------------------------------
    # DeviceModel interface
    # ------------------------------------------------------------------
    def body_xml(self) -> str:
        p = self.params
        lx, ly, lz = p.chassis_dims

        r = p.wheel_radius
        ww = p.wheel_width
        # Wheel positions relative to chassis center
        hx = lx / 2 - r * 0.8
        hy = ly / 2 - r * 0.8

        lines = [
            f'    <body name="{self.body_name}" pos="0 0 0.25">',
            f'      <joint name="{self.x_joint}" type="slide" axis="1 0 0"/>',
            f'      <joint name="{self.y_joint}" type="slide" axis="0 1 0"/>',
            f'      <inertial pos="0 0 0" mass="{p.chassis_mass:.2f}"'
            f' diaginertia="{self._inertia(p.chassis_mass, lx, ly, lz)}"/>',
            f'      <geom name="{self._n("chassis")}" type="box"'
            f' size="{lx/2:.4f} {ly/2:.4f} {lz/2:.4f}"'
            f' rgba="0.25 0.35 0.85 1"/>',
            # Lift body
            f'      <body name="{self.lift_body_name}" pos="0 0 {lz/2:.4f}">',
            f'        <joint name="{self.z_joint}" type="slide" axis="0 0 1"'
            f' limited="true" range="0 {p.max_lift:.4f}"/>',
            f'        <inertial pos="0 0 0" mass="10" diaginertia="0.5 0.5 0.5"/>',
            f'        <geom name="{self._n("platform")}" type="box"'
            f' size="{lx/2:.4f} {ly/2:.4f} 0.02" rgba="0.40 0.55 0.90 1"/>',
            # Four wheels
            self._wheel_xml("fl", hx, hy, r, ww),
            self._wheel_xml("fr", hx, -hy, r, ww),
            self._wheel_xml("bl", -hx, hy, r, ww),
            self._wheel_xml("br", -hx, -hy, r, ww),
            f'      </body>',
            f'    </body>',
        ]
        return "\n".join(lines)

    def _wheel_xml(self, tag: str, px: float, py: float, r: float, w: float) -> str:
        return (
            f'        <geom name="{self._n(f"wheel_{tag}")}" type="cylinder"'
            f' pos="{px:.4f} {py:.4f} {-r:.4f}"'
            f' size="{r:.4f} {w:.4f}" rgba="0.15 0.15 0.15 1"/>'
        )

    def actuator_xml(self) -> str:
        p = self.params
        lines = [
            f'    <position name="{self.x_actuator}" joint="{self.x_joint}"'
            f' kp="{p.xy_kp:.0f}" kv="{p.xy_kv:.0f}"/>',
            f'    <position name="{self.y_actuator}" joint="{self.y_joint}"'
            f' kp="{p.xy_kp:.0f}" kv="{p.xy_kv:.0f}"/>',
            f'    <position name="{self.z_actuator}" joint="{self.z_joint}"'
            f' kp="{p.z_kp:.0f}" kv="{p.z_kv:.0f}"/>',
        ]
        return "\n".join(lines)

    def joint_names(self) -> list[str]:
        return [
            self.x_joint,
            self.y_joint,
            self.z_joint,
        ]

    def actuator_names(self) -> list[str]:
        return [self.x_actuator, self.y_actuator, self.z_actuator]

    def mesh_assets(self) -> list[str]:
        return []

    def init_joint_positions(self) -> dict[str, float]:
        wx, wy = self.params.initial_world_pos()
        return {
            self.x_joint: wx,
            self.y_joint: wy,
            self.z_joint: self.params.initial_lift,
        }

    @staticmethod
    def _inertia(mass: float, lx: float, ly: float, lz: float) -> str:
        """Diagonal inertia for a box: 1/12 * m * (ly^2+lz^2, lx^2+lz^2, lx^2+ly^2)."""
        ix = (1 / 12) * mass * (ly**2 + lz**2)
        iy = (1 / 12) * mass * (lx**2 + lz**2)
        iz = (1 / 12) * mass * (lx**2 + ly**2)
        return f"{ix:.4f} {iy:.4f} {iz:.4f}"
