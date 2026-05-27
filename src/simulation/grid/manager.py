"""Grid occupancy tracking and coordinate transforms."""


class GridManager:
    """Manages the rail grid: cell occupancy and coordinate conversions.

    Grid coordinate system:
      - X axis: right (positive X = MOVE_RIGHT)
      - Y axis: forward (positive Y = MOVE_FORWARD)
      - Origin (0, 0) = bottom-left corner of the grid
      - Cell (cx, cy) center is at world: (cx * cell_size + cell_size/2, cy * cell_size + cell_size/2)
    """

    def __init__(
        self,
        x_cells: int,
        y_cells: int,
        cell_size: float = 1.0,
        world_origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.x_cells = x_cells
        self.y_cells = y_cells
        self.cell_size = cell_size
        self.world_origin = world_origin
        self._occupancy: dict[tuple[int, int], str] = {}

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------
    def cell_to_world(self, cx: int, cy: int) -> tuple[float, float]:
        """Convert grid cell coordinates to world XY (cell center)."""
        ox, oy = self.world_origin
        cs = self.cell_size
        return (ox + cx * cs + cs / 2, oy + cy * cs + cs / 2)

    def world_to_cell(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world XY to the nearest grid cell."""
        ox, oy = self.world_origin
        cs = self.cell_size
        cx = int(round((wx - ox - cs / 2) / cs))
        cy = int(round((wy - oy - cs / 2) / cs))
        return (max(0, min(cx, self.x_cells - 1)), max(0, min(cy, self.y_cells - 1)))

    # ------------------------------------------------------------------
    # Bounds checking
    # ------------------------------------------------------------------
    def is_within_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.x_cells and 0 <= cy < self.y_cells

    # ------------------------------------------------------------------
    # Occupancy management
    # ------------------------------------------------------------------
    def is_occupied(self, cx: int, cy: int, exclude: str | None = None) -> bool:
        """Check if a cell has a vehicle. Optionally exclude a vehicle_id."""
        occupant = self._occupancy.get((cx, cy))
        if occupant is None:
            return False
        if exclude is not None and occupant == exclude:
            return False
        return True

    def occupy(self, vehicle_id: str, cx: int, cy: int) -> None:
        """Mark a cell as occupied by a vehicle."""
        self._occupancy[(cx, cy)] = vehicle_id

    def vacate(self, vehicle_id: str) -> None:
        """Remove a vehicle's occupancy entry."""
        to_remove = [pos for pos, vid in self._occupancy.items() if vid == vehicle_id]
        for pos in to_remove:
            del self._occupancy[pos]

    def get_position(self, vehicle_id: str) -> tuple[int, int] | None:
        """Get a vehicle's current grid cell, or None if not tracked."""
        for pos, vid in self._occupancy.items():
            if vid == vehicle_id:
                return pos
        return None

    def all_positions(self) -> dict[str, tuple[int, int]]:
        """Return {vehicle_id: (cx, cy)} for all tracked vehicles."""
        return {vid: pos for pos, vid in self._occupancy.items()}

    def adjacent_cells(self, cx: int, cy: int) -> dict[str, tuple[int, int]]:
        """Return adjacent cells in the four cardinal directions."""
        return {
            "forward": (cx, cy + 1),
            "backward": (cx, cy - 1),
            "left": (cx - 1, cy),
            "right": (cx + 1, cy),
        }
