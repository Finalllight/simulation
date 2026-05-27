"""Mesh asset management for MuJoCo model loading."""

import os
import shutil
import tempfile
from pathlib import Path


class MeshAssetManager:
    """Copies referenced mesh files into a flat directory that MuJoCo can resolve.

    MuJoCo resolves <mesh file="..."> paths relative to the XML file location.
    By placing all meshes alongside the generated XML, path resolution is trivial.
    """

    def __init__(self, mesh_dir: str | None = None):
        self.mesh_dir = mesh_dir or tempfile.mkdtemp(prefix="sim_meshes_")
        os.makedirs(self.mesh_dir, exist_ok=True)

    def register(self, mesh_path: str) -> str:
        """Copy a mesh file to mesh_dir and return its basename.

        The basename can be used directly in MJCF: <mesh name="X" file="basename.obj"/>
        """
        src = Path(mesh_path)
        dst = Path(self.mesh_dir) / src.name
        if not dst.exists() or os.path.getmtime(str(src)) > os.path.getmtime(str(dst)):
            shutil.copy2(str(src), str(dst))
        return src.name

    def register_all(self, paths: list[str]) -> list[str]:
        return [self.register(p) for p in paths]

    def cleanup(self) -> None:
        """Remove the mesh directory and all its contents."""
        if os.path.isdir(self.mesh_dir):
            shutil.rmtree(self.mesh_dir, ignore_errors=True)

    def __del__(self):
        self.cleanup()
