"""Physical property extraction from parsed CAD geometry."""

from pathlib import Path

import numpy as np

from simulation.cad.parser import CADParser, ParsedGeometry


class PropertyExtractor:
    """Extracts physical parameters from CAD geometry for device model construction."""

    def __init__(self, default_density: float = 7850.0):
        self._default_density = default_density  # steel, kg/m³

    def extract_from_geometry(
        self,
        geometry: ParsedGeometry,
        material_density: float | None = None,
        mass_override: float | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Extract mass (kg), COM (3,), and inertia (3,3) from ParsedGeometry.

        If mass_override is set, inertia is scaled proportionally.
        """
        density = material_density or self._default_density

        if mass_override is not None:
            mass = mass_override
            scale = mass_override / (geometry.volume * density) if geometry.volume > 0 else 1.0
            inertia = geometry.inertia_tensor * density * scale
        else:
            mass = geometry.volume * density
            inertia = geometry.inertia_tensor * density

        return mass, geometry.center_of_mass.copy(), inertia

    def dimensions_from_geometry(self, geometry: ParsedGeometry) -> tuple[float, float, float]:
        """Bounding-box dimensions: (length_x, length_y, height_z)."""
        bbox = geometry.bounding_box
        dx = bbox[0, 1] - bbox[0, 0]
        dy = bbox[1, 1] - bbox[1, 0]
        dz = bbox[2, 1] - bbox[2, 0]
        return (dx, dy, dz)

    def diaginertia_from_box(
        self, mass: float, lx: float, ly: float, lz: float
    ) -> tuple[float, float, float]:
        """Compute diagonal inertia for a uniform box: 1/12 * m * (ly²+lz², lx²+lz², lx²+ly²)."""
        ix = (1.0 / 12.0) * mass * (ly**2 + lz**2)
        iy = (1.0 / 12.0) * mass * (lx**2 + lz**2)
        iz = (1.0 / 12.0) * mass * (lx**2 + ly**2)
        return (ix, iy, iz)

    def build_vehicle_params(
        self,
        cad_file: str,
        motion_yaml: str,
        device_id: str,
        initial_grid_pos: tuple[int, int] = (0, 0),
    ) -> "TransportVehicleParams":
        """Full pipeline: CAD + motion YAML → TransportVehicleParams.

        Motion YAML values override CAD-derived values where specified.
        """
        from simulation.models.transport_vehicle import TransportVehicleParams

        motion = CADParser.parse_motion_params(motion_yaml)
        parser = CADParser()
        geometry = parser.parse(cad_file)

        density = motion.get("material_density", self._default_density)
        mass, com, inertia = self.extract_from_geometry(
            geometry,
            material_density=density,
            mass_override=motion.get("chassis_mass"),
        )
        dims = self.dimensions_from_geometry(geometry)

        return TransportVehicleParams(
            device_id=device_id,
            chassis_mass=motion.get("chassis_mass", mass),
            chassis_dims=(
                motion.get("chassis_dims", {}).get("length", dims[0]),
                motion.get("chassis_dims", {}).get("width", dims[1]),
                motion.get("chassis_dims", {}).get("height", dims[2]),
            ),
            wheel_radius=motion.get("wheel_radius", 0.06),
            wheel_width=motion.get("wheel_width", 0.04),
            max_speed=motion.get("max_speed", 1.2),
            max_acceleration=motion.get("max_acceleration", 0.6),
            max_lift=motion.get("max_lift", 0.4),
            lift_speed=motion.get("lift_speed", 0.08),
            initial_grid_pos=initial_grid_pos,
        )
