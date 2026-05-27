#!/usr/bin/env python3
"""End-to-end demo: load grid transport scene, accept interactive commands."""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from simulation.core.engine import SimulationEngine
from simulation.core.scene import GridConfig, SceneBuilder
from simulation.commands.types import Command, CommandType
from simulation.commands.interpreter import CommandInterpreter
from simulation.grid.manager import GridManager
from simulation.models.transport_vehicle import TransportVehicleModel, TransportVehicleParams
from simulation.models.group import VehicleGroup
from simulation.output.recorder import KinematicRecorder


def build_system(config_path: str) -> tuple[SimulationEngine, CommandInterpreter, GridManager]:
    """Build the complete simulation system from a config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Grid
    gcfg = cfg["grid"]
    grid_config = GridConfig(
        x_cells=gcfg["x_cells"],
        y_cells=gcfg["y_cells"],
        cell_size=gcfg["cell_size"],
        world_origin=tuple(gcfg.get("world_origin", [0.0, 0.0])),
    )

    # Scene builder
    builder = SceneBuilder(grid_config)

    # Vehicle models
    vehicle_models: dict[str, TransportVehicleModel] = {}
    for vdata in cfg["vehicles"]:
        params = TransportVehicleParams(
            device_id=vdata["device_id"],
            chassis_mass=vdata.get("chassis_mass", 120.0),
            chassis_dims=tuple(vdata.get("chassis_dims", [0.9, 0.7, 0.35])),
            wheel_radius=vdata.get("wheel_radius", 0.06),
            wheel_width=vdata.get("wheel_width", 0.04),
            max_speed=vdata.get("max_speed", 1.2),
            max_acceleration=vdata.get("max_acceleration", 0.6),
            max_lift=vdata.get("max_lift", 0.4),
            lift_speed=vdata.get("lift_speed", 0.08),
            initial_grid_pos=tuple(vdata.get("initial_grid_pos", [0, 0])),
            initial_lift=vdata.get("initial_lift", 0.0),
            cell_size=grid_config.cell_size,
            xy_kp=vdata.get("xy_kp", 500.0),
            xy_kv=vdata.get("xy_kv", 100.0),
            z_kp=vdata.get("z_kp", 1000.0),
            z_kv=vdata.get("z_kv", 200.0),
        )
        model = TransportVehicleModel(params)
        vehicle_models[params.device_id] = model
        builder.add_device(model)

    # Build XML and engine
    xml = builder.build_xml()
    engine = SimulationEngine(xml)

    # Set initial joint positions and actuator targets
    for model in vehicle_models.values():
        init_qpos = model.init_joint_positions()
        for jname, val in init_qpos.items():
            jid = engine._joint_ids.get(jname)
            if jid is not None:
                qpos_addr = engine.model.jnt_qposadr[jid]
                engine.data.qpos[qpos_addr] = val
        # Lock actuators at initial positions so servos don't pull to zero
        engine.set_position_target(model.x_actuator, init_qpos[model.x_joint])
        engine.set_position_target(model.y_actuator, init_qpos[model.y_joint])
        engine.set_position_target(model.z_actuator, init_qpos[model.z_joint])
    engine.forward()

    # Grid manager and register vehicles
    grid = GridManager(
        x_cells=gcfg["x_cells"],
        y_cells=gcfg["y_cells"],
        cell_size=gcfg["cell_size"],
        world_origin=tuple(gcfg.get("world_origin", [0.0, 0.0])),
    )
    for vdata in cfg["vehicles"]:
        vid = vdata["device_id"]
        pos = tuple(vdata.get("initial_grid_pos", [0, 0]))
        grid.occupy(vid, *pos)

    # Command interpreter
    interpreter = CommandInterpreter(
        engine=engine,
        grid=grid,
        vehicle_models=vehicle_models,
        recorder_class=KinematicRecorder,
    )

    return engine, interpreter, grid


def print_help():
    print("\nCommands:")
    print("  f <id>     - Move vehicle/group forward (+Y)")
    print("  b <id>     - Move vehicle/group backward (-Y)")
    print("  l <id>     - Move vehicle/group left (-X)")
    print("  r <id>     - Move vehicle/group right (+X)")
    print("  u <id>     - Lift up (default 0.05m)")
    print("  d <id>     - Lift down (default 0.05m)")
    print("  s <id>     - Stop vehicle/group")
    print("  g <gid> <v1> <v2> ... - Form group")
    print("  x <gid>    - Dissolve group")
    print("  pos        - Show all vehicle positions")
    print("  groups     - List groups")
    print("  h / help   - This help")
    print("  q / quit   - Exit")
    print()


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    print("Building simulation system...")
    engine, interpreter, grid = build_system(str(config_path))
    print(f"Loaded {len(engine.get_body_names())} bodies, "
          f"{len(engine.get_joint_names())} joints, "
          f"{len(engine.get_actuator_names())} actuators")
    print(f"Timestep: {engine.dt:.4f}s")
    print(f"Grid: {grid.x_cells}x{grid.y_cells}, cell size: {grid.cell_size}m")
    print_help()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd_name = parts[0].lower()

        if cmd_name in ("q", "quit", "exit"):
            break
        elif cmd_name in ("h", "help"):
            print_help()
            continue
        elif cmd_name == "pos":
            positions = grid.all_positions()
            for vid, pos in positions.items():
                eng_pos = engine.body_position(vid)
                print(f"  {vid}: grid={pos}, world=({eng_pos[0]:.2f}, {eng_pos[1]:.2f}, {eng_pos[2]:.4f}), lift={engine.joint_position(f'{vid}_lift_z'):.4f}")
            continue
        elif cmd_name == "groups":
            for gid, group in interpreter.groups.items():
                print(f"  {gid}: {group.member_ids}")
            continue

        # Parse main commands
        command = None
        if cmd_name in ("f", "forward"):
            if len(parts) < 2:
                print("Usage: f <vehicle_id|group_id>")
                continue
            command = Command(CommandType.MOVE_FORWARD, parts[1])
        elif cmd_name in ("b", "backward"):
            if len(parts) < 2:
                print("Usage: b <vehicle_id|group_id>")
                continue
            command = Command(CommandType.MOVE_BACKWARD, parts[1])
        elif cmd_name in ("l", "left"):
            if len(parts) < 2:
                print("Usage: l <vehicle_id|group_id>")
                continue
            command = Command(CommandType.MOVE_LEFT, parts[1])
        elif cmd_name in ("r", "right"):
            if len(parts) < 2:
                print("Usage: r <vehicle_id|group_id>")
                continue
            command = Command(CommandType.MOVE_RIGHT, parts[1])
        elif cmd_name in ("u", "up"):
            if len(parts) < 2:
                print("Usage: u <vehicle_id|group_id> [amount]")
                continue
            amount = float(parts[2]) if len(parts) > 2 else 0.05
            command = Command(CommandType.LIFT_UP, parts[1], {"amount": amount})
        elif cmd_name in ("d", "down"):
            if len(parts) < 2:
                print("Usage: d <vehicle_id|group_id> [amount]")
                continue
            amount = float(parts[2]) if len(parts) > 2 else 0.05
            command = Command(CommandType.LIFT_DOWN, parts[1], {"amount": amount})
        elif cmd_name in ("s", "stop"):
            if len(parts) < 2:
                print("Usage: s <vehicle_id|group_id>")
                continue
            command = Command(CommandType.STOP, parts[1])
        elif cmd_name == "g":
            if len(parts) < 4:
                print("Usage: g <group_id> <v1> <v2> [v3 ...]")
                continue
            command = Command(CommandType.FORM_GROUP, parts[1], {"member_ids": parts[2:]})
        elif cmd_name == "x":
            if len(parts) < 2:
                print("Usage: x <group_id>")
                continue
            command = Command(CommandType.DISSOLVE_GROUP, parts[1])
        else:
            print(f"Unknown command: {cmd_name}")
            continue

        # Execute
        result = interpreter.execute(command)
        if result.success:
            npoints = sum(len(ts) for ts in result.timeseries.values())
            print(f"  OK — {npoints} data points recorded across {len(result.timeseries)} bodies")

            # Export to output dir
            out_dir = Path(__file__).resolve().parents[1] / "output" / f"cmd_{parts[0]}"
            recorder = KinematicRecorder(engine)
            for body_name in result.timeseries:
                recorder.add_tracked_body(body_name)
            # Write the recorded timeseries directly
            csv_paths = _export_result(result, str(out_dir))
            for name, path in csv_paths.items():
                print(f"    → {path}")
        else:
            print(f"  FAILED: {result.error_message}")


def _export_result(result, out_dir: str) -> dict[str, str]:
    """Export CommandResult timeseries to CSV."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    filepaths = {}
    for body_name, ts in result.timeseries.items():
        filepath = os.path.join(out_dir, f"{body_name}.csv")
        header = "time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z"
        rows = [
            f"{ts.time[i]:.6f},{ts.position[i,0]:.6f},{ts.position[i,1]:.6f},{ts.position[i,2]:.6f},"
            f"{ts.velocity[i,0]:.6f},{ts.velocity[i,1]:.6f},{ts.velocity[i,2]:.6f},"
            f"{ts.acceleration[i,0]:.6f},{ts.acceleration[i,1]:.6f},{ts.acceleration[i,2]:.6f}"
            for i in range(len(ts.time))
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + "\n" + "\n".join(rows) + "\n")
        filepaths[body_name] = filepath
    return filepaths


if __name__ == "__main__":
    main()
