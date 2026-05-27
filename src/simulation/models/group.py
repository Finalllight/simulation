"""Vehicle group for synchronized multi-vehicle transport."""


class VehicleGroup:
    """Logical grouping of vehicles that move in lockstep for cargo transport.

    Grouping is logical (not a physical constraint): the CommandInterpreter
    ensures all members receive identical velocity/lift targets simultaneously.
    """

    def __init__(
        self,
        group_id: str,
        member_ids: list[str],
        relative_offsets: dict[str, tuple[int, int]] | None = None,
    ):
        self.group_id = group_id
        self._members: list[str] = list(member_ids)
        self._offsets: dict[str, tuple[int, int]] = relative_offsets or {}

    def add_member(self, vehicle_id: str, offset: tuple[int, int] = (0, 0)) -> None:
        """Add a vehicle to the group."""
        if vehicle_id not in self._members:
            self._members.append(vehicle_id)
        self._offsets[vehicle_id] = offset

    def remove_member(self, vehicle_id: str) -> None:
        """Remove a vehicle from the group."""
        if vehicle_id in self._members:
            self._members.remove(vehicle_id)
        self._offsets.pop(vehicle_id, None)

    def get_member_targets(
        self,
        anchor_dx: int,
        anchor_dy: int,
        grid_manager,
    ) -> dict[str, tuple[int, int]]:
        """Compute target grid cells for all members given an anchor move (dx, dy).

        Returns {vehicle_id: (target_cx, target_cy)}.
        """
        anchor_id = self._members[0]
        anchor_pos = grid_manager.get_position(anchor_id)
        anchor_target = (anchor_pos[0] + anchor_dx, anchor_pos[1] + anchor_dy)

        targets: dict[str, tuple[int, int]] = {anchor_id: anchor_target}
        for vid in self._members[1:]:
            off = self._offsets.get(vid, (0, 0))
            targets[vid] = (anchor_target[0] + off[0], anchor_target[1] + off[1])
        return targets

    def get_lift_targets(
        self,
        anchor_dz: float,
        vehicle_models: dict,
        engine,
    ) -> dict[str, float]:
        """Compute lift targets for all members given a lift delta."""
        anchor_id = self._members[0]
        model = vehicle_models[anchor_id]
        current_z = engine.joint_position(model.z_joint)
        max_lift = model.params.max_lift
        new_z = max(0.0, min(max_lift, current_z + anchor_dz))

        targets: dict[str, float] = {anchor_id: new_z}
        for vid in self._members[1:]:
            targets[vid] = new_z
        return targets

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def member_ids(self) -> list[str]:
        return list(self._members)

    @property
    def size(self) -> int:
        return len(self._members)

    @property
    def offsets(self) -> dict[str, tuple[int, int]]:
        return dict(self._offsets)

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(self, vehicle_id: str) -> bool:
        return vehicle_id in self._members
