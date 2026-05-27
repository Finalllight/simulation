"""CAD file parser: STEP/STL → ParsedGeometry (trimesh)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ParsedGeometry:
    """Normalized geometry representation extracted from a CAD file."""

    mesh: "trimesh.Trimesh"
    volume: float
    center_of_mass: np.ndarray  # (3,) in geometry-local frame
    inertia_tensor: np.ndarray  # (3,3) in geometry-local frame
    bounding_box: np.ndarray  # (3,2) min and max corners
    source_format: str  # "step" or "stl"


class CADParser:
    """Parse CAD files (STEP/STL) into ParsedGeometry with mass properties."""

    def parse(self, filepath: str) -> ParsedGeometry:
        suffix = Path(filepath).suffix.lower()
        if suffix in (".step", ".stp"):
            return self._parse_step(filepath)
        elif suffix in (".stl",):
            return self._parse_stl(filepath)
        else:
            raise ValueError(f"Unsupported CAD format: {suffix}")

    def _parse_step(self, filepath: str) -> ParsedGeometry:
        try:
            import pythonocc-core  # noqa: F401
        except ImportError:
            raise ImportError(
                "STEP parsing requires pythonocc-core. "
                "Install with: pip install pythonocc-core"
            )
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepGProp import brepgprop_VolumeProperties
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods_Face
        from OCC.Core.gp import gp_Pnt

        import trimesh

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(filepath))
        if status != 1:
            raise RuntimeError(f"Failed to read STEP file: {filepath}")
        reader.TransferRoots()
        shape = reader.OneShape()

        # Compute mass properties (assume density=1, caller scales)
        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        volume = props.Mass()
        com = props.CentreOfMass()
        com = np.array([com.X(), com.Y(), com.Z()])
        inertia = props.MatrixOfInertia()
        inertia_3x3 = np.array(
            [
                [inertia.Value(1, 1), inertia.Value(1, 2), inertia.Value(1, 3)],
                [inertia.Value(2, 1), inertia.Value(2, 2), inertia.Value(2, 3)],
                [inertia.Value(3, 1), inertia.Value(3, 2), inertia.Value(3, 3)],
            ]
        )
        bbox = self._compute_bbox_occ(shape)

        # Tessellate
        mesh = BRepMesh_IncrementalMesh(shape, 0.5)
        mesh.Perform()
        vertices, faces = self._extract_triangles_occ(shape)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.remove_unreferenced_vertices()

        return ParsedGeometry(
            mesh=tri_mesh,
            volume=volume,
            center_of_mass=com,
            inertia_tensor=inertia_3x3,
            bounding_box=bbox,
            source_format="step",
        )

    def _parse_stl(self, filepath: str) -> ParsedGeometry:
        import trimesh

        mesh = trimesh.load(filepath, file_type="stl")
        if isinstance(mesh, trimesh.Scene):
            # Combine all geometries in the scene
            meshes = []
            for name, geom in mesh.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()

        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()

        density = 1.0  # caller scales by material density
        mass_props = mesh.mass_properties(density=density)
        volume = mesh.volume or mass_props.get("volume", 0.0)
        com = mesh.center_mass
        inertia = mesh.moment_inertia

        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        bbox = np.column_stack([bbox_min, bbox_max])

        return ParsedGeometry(
            mesh=mesh,
            volume=float(volume),
            center_of_mass=com,
            inertia_tensor=inertia,
            bounding_box=bbox,
            source_format="stl",
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def parse_motion_params(yaml_path: str) -> dict:
        """Load motion parameter YAML into a dict."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _compute_bbox_occ(shape) -> np.ndarray:
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.Bnd import Bnd_Box

        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])

    @staticmethod
    def _extract_triangles_occ(shape) -> tuple[np.ndarray, np.ndarray]:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopoDS import topods_Face
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopLoc import TopLoc_Location

        vertices_list: list[list[float]] = []
        faces_list: list[list[int]] = []
        vertex_index = 0

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = topods_Face(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, location)
            if triangulation is not None:
                # Get face nodes
                nodes = triangulation.Nodes()
                triangles = triangulation.Triangles()

                # Map node indices to global vertex indices
                node_map = {}
                for i in range(1, triangulation.NbNodes() + 1):
                    pnt = nodes.Value(i)
                    pnt_transformed = pnt.Transformed(location.Transformation())
                    vertices_list.append(
                        [pnt_transformed.X(), pnt_transformed.Y(), pnt_transformed.Z()]
                    )
                    node_map[i] = vertex_index
                    vertex_index += 1

                for i in range(1, triangulation.NbTriangles() + 1):
                    tri = triangles.Value(i)
                    n1, n2, n3 = tri.Get()
                    faces_list.append(
                        [node_map[n1], node_map[n2], node_map[n3]]
                    )

            explorer.Next()

        return np.array(vertices_list, dtype=np.float32), np.array(
            faces_list, dtype=np.int32
        )
