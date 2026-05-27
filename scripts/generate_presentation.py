#!/usr/bin/env python3
"""Generate presentation-quality plots and data for group meeting.

Scenario: Single vehicle performs a sequence of operations
  1. Move forward (0,0) → (0,3)  — long straight movement
  2. Turn right (0,3) → (3,3)    — perpendicular movement
  3. Lift platform 0 → 0.3m      — height adjustment
  4. Lift down 0.3 → 0.1m

Then: Two vehicles form a group and move together.

Outputs to presentation/:
  - velocity_profile.png      — velocity over time (accel-cruise-decel)
  - position_trace.png        — position over time for 3 axes
  - lift_profile.png          — Z-axis lift with gravity compensation
  - group_movement.png        — two vehicles moving in sync
  - scenario_data/*.csv       — raw timeseries data
"""

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulation.core.engine import SimulationEngine
from simulation.core.scene import GridConfig, SceneBuilder
from simulation.models.transport_vehicle import TransportVehicleModel, TransportVehicleParams
from simulation.commands.types import Command, CommandType
from simulation.commands.interpreter import CommandInterpreter
from simulation.grid.manager import GridManager
from simulation.output.recorder import KinematicRecorder

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "presentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR / "scenario_data", exist_ok=True)

# Style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "lines.linewidth": 1.8,
        "figure.figsize": (10, 5),
    }
)


def build_engine(grid_cells=6):
    gc = GridConfig(x_cells=grid_cells, y_cells=grid_cells, cell_size=1.0)
    p0 = TransportVehicleParams(
        device_id="v0", initial_grid_pos=(0, 0), cell_size=1.0, max_lift=0.4
    )
    p1 = TransportVehicleParams(
        device_id="v1", initial_grid_pos=(2, 0), cell_size=1.0, max_lift=0.4
    )
    m0 = TransportVehicleModel(p0)
    m1 = TransportVehicleModel(p1)
    builder = SceneBuilder(gc)
    builder.add_device(m0)
    builder.add_device(m1)
    engine = SimulationEngine(builder.build_xml())

    for m in [m0, m1]:
        qp = m.init_joint_positions()
        for jn, v in qp.items():
            engine.data.qpos[engine.model.jnt_qposadr[engine._joint_ids[jn]]] = v
        engine.set_position_target(m.x_actuator, qp[m.x_joint])
        engine.set_position_target(m.y_actuator, qp[m.y_joint])
        engine.set_position_target(m.z_actuator, qp[m.z_joint])
    engine.forward()

    grid = GridManager(x_cells=gc.x_cells, y_cells=gc.y_cells, cell_size=gc.cell_size)
    grid.occupy("v0", 0, 0)
    grid.occupy("v1", 2, 0)

    interp = CommandInterpreter(
        engine=engine,
        grid=grid,
        vehicle_models={"v0": m0, "v1": m1},
        recorder_class=KinematicRecorder,
    )
    return engine, grid, interp, m0, m1


def collect_timeseries(interp, commands):
    """Execute a sequence and collect all timeseries."""
    all_ts = {}
    for cmd in commands:
        r = interp.execute(cmd)
        for name, ts in r.timeseries.items():
            if name in all_ts:
                # Append to existing
                offset = all_ts[name].time[-1] if len(all_ts[name]) > 0 else 0.0
                all_ts[name].time = np.append(
                    all_ts[name].time, offset + ts.time
                )
                all_ts[name].position = np.vstack([all_ts[name].position, ts.position])
                all_ts[name].velocity = np.vstack([all_ts[name].velocity, ts.velocity])
                all_ts[name].acceleration = np.vstack(
                    [all_ts[name].acceleration, ts.acceleration]
                )
            else:
                all_ts[name] = ts
    return all_ts


# ===================================================================
# Scenario 1: Single vehicle long movement + lift
# ===================================================================
print("Running scenario 1: single vehicle movement...")
engine, grid, interp, m0, m1 = build_engine(grid_cells=6)

cmds_single = [
    Command(CommandType.MOVE_FORWARD, "v0"),
    Command(CommandType.MOVE_FORWARD, "v0"),
    Command(CommandType.MOVE_FORWARD, "v0"),
    Command(CommandType.MOVE_RIGHT, "v0"),
    Command(CommandType.MOVE_RIGHT, "v0"),
    Command(CommandType.MOVE_RIGHT, "v0"),
    Command(CommandType.LIFT_UP, "v0", {"amount": 0.3}),
    Command(CommandType.LIFT_DOWN, "v0", {"amount": 0.2}),
]

ts_single = collect_timeseries(interp, cmds_single)
ts_v0 = ts_single["v0"]

# Save CSV
csv_path = OUTPUT_DIR / "scenario_data" / "single_vehicle_scenario.csv"
with open(csv_path, "w") as f:
    f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n")
    for i in range(len(ts_v0.time)):
        f.write(
            f"{ts_v0.time[i]:.6f},"
            f"{ts_v0.position[i,0]:.6f},{ts_v0.position[i,1]:.6f},{ts_v0.position[i,2]:.6f},"
            f"{ts_v0.velocity[i,0]:.6f},{ts_v0.velocity[i,1]:.6f},{ts_v0.velocity[i,2]:.6f},"
            f"{ts_v0.acceleration[i,0]:.6f},{ts_v0.acceleration[i,1]:.6f},{ts_v0.acceleration[i,2]:.6f}\n"
        )
print(f"  Saved {csv_path} ({len(ts_v0)} rows)")

# ===================================================================
# Plot 1: Velocity Profile — showing accel-cruise-decel
# ===================================================================
print("Plotting velocity profile...")

# Find a single movement segment (first forward move)
# We look for the first segment where Y velocity rises and falls
t = ts_v0.time
vy = np.abs(ts_v0.velocity[:, 1])

# Focus on the first 3 forward moves (Y-axis movement)
# Find when Y velocity is active
y_moving = vy > 0.02
if y_moving.any():
    # Get first movement contiguous segment
    segments = []
    in_seg = False
    seg_start = 0
    for i, moving in enumerate(y_moving):
        if moving and not in_seg:
            seg_start = i
            in_seg = True
        elif not moving and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(y_moving) - 1))

# Focus on the first forward-move segment for detailed view
s0, e0 = segments[0] if segments else (0, min(400, len(t)))
pad = 20
s, e = max(0, s0 - pad), min(len(t), e0 + pad)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top: velocity
ax1.plot(
    t[s:e],
    ts_v0.velocity[s:e, 0],
    label="Vx (lateral)",
    color="#3498db",
    alpha=0.6,
)
ax1.plot(
    t[s:e],
    ts_v0.velocity[s:e, 1],
    label="Vy (forward)",
    color="#e74c3c",
    linewidth=2.2,
)
ax1.axhline(y=1.2, color="gray", linestyle="--", alpha=0.5, label="max_speed=1.2 m/s")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title("Single Move Forward — Velocity Profile (acceleration → cruise → deceleration)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Annotate phases
mid_t = t[(s + e) // 2]
peak_v = np.max(np.abs(ts_v0.velocity[s:e, 1]))
peak_idx = np.argmax(np.abs(ts_v0.velocity[s:e, 1])) + s
ax1.annotate(
    f"v_peak={peak_v:.2f} m/s",
    xy=(t[peak_idx], ts_v0.velocity[peak_idx, 1]),
    xytext=(t[peak_idx] + 0.1, ts_v0.velocity[peak_idx, 1] + 0.2),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10,
)

# Bottom: position
ax2.plot(
    t[s:e],
    ts_v0.position[s:e, 1],
    label="Y position",
    color="#2ecc71",
    linewidth=2.2,
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position Y (m)")
ax2.set_title("Position Trace (Y-axis)")
ax2.grid(True, alpha=0.3)

# Mark start and end grid positions
ax2.axhline(y=ts_v0.position[s, 1], color="gray", linestyle=":", alpha=0.5)
ax2.axhline(y=ts_v0.position[e - 1, 1], color="gray", linestyle=":", alpha=0.5)
ax2.annotate(
    f"Start: {ts_v0.position[s,1]:.1f}m",
    xy=(t[s], ts_v0.position[s, 1]),
    fontsize=9,
    color="gray",
)
ax2.annotate(
    f"End: {ts_v0.position[e-1,1]:.1f}m",
    xy=(t[e - 1], ts_v0.position[e - 1, 1]),
    fontsize=9,
    color="gray",
)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "velocity_profile.png", bbox_inches="tight")
plt.close(fig)
print("  Saved velocity_profile.png")

# ===================================================================
# Plot 2: Full Position Trace — all 3 axes + movements
# ===================================================================
print("Plotting position trace...")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

colors = ["#e74c3c", "#2ecc71", "#3498db"]
labels = ["X (lateral)", "Y (forward)", "Z (lift)"]
for ax, dim, color, label in zip(axes, range(3), colors, labels):
    ax.plot(
        ts_v0.time,
        ts_v0.position[:, dim],
        color=color,
        linewidth=1.6,
        label=label,
    )
    ax.set_ylabel(f"{label} (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

axes[2].set_xlabel("Time (s)")
axes[0].set_title("Transport Vehicle — Complete Position Trace (3 moves + lift)")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "position_trace.png", bbox_inches="tight")
plt.close(fig)
print("  Saved position_trace.png")

# ===================================================================
# Plot 3: Lift Profile — gravity compensation
# ===================================================================
print("Plotting lift profile...")

# Use lift body data (v0_lift) for Z-axis movement
ts_lift = ts_single.get("v0_lift", ts_v0)

# Find the liftup segment using Z velocity
vz = np.abs(ts_lift.velocity[:, 2])
z_moving = vz > 0.005
lift_segments = []
in_seg = False
seg_s = 0
for i, m in enumerate(z_moving):
    if m and not in_seg:
        seg_s = i
        in_seg = True
    elif not m and in_seg:
        lift_segments.append((seg_s, i))
        in_seg = False
if in_seg:
    lift_segments.append((seg_s, len(z_moving) - 1))

# Take first lift segment
if lift_segments:
    ls, le = lift_segments[0]
    ls = max(0, ls - 30)
    le = min(len(ts_v0.time), le + 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(
        ts_lift.time[ls:le],
        ts_lift.position[ls:le, 2] - 0.425,  # subtract initial lift body Z offset
        color="#9b59b6",
        linewidth=2.2,
    )
    ax1.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="target=0.3m")
    ax1.set_ylabel("Z Position (m, relative)")
    ax1.set_title("Lift Platform — Z-Axis Height Change (with gravity compensation)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Steady-state error annotation
    final_z = ts_lift.position[le - 1, 2] - 0.425
    ax1.annotate(
        f"Steady-state: {final_z:.3f}m\n(error: {0.3-final_z:.4f}m)",
        xy=(ts_lift.time[le - 1], final_z),
        xytext=(ts_lift.time[le - 80], final_z + 0.03),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        color="#8e44ad",
    )

    ax2.plot(
        ts_lift.time[ls:le],
        ts_lift.velocity[ls:le, 2],
        color="#e67e22",
        linewidth=2.0,
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Z Velocity (m/s)")
    ax2.set_title("Lift Velocity Profile")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lift_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved lift_profile.png")

# ===================================================================
# Scenario 2: Group movement
# ===================================================================
print("\nRunning scenario 2: group movement...")
engine2, grid2, interp2, m2_0, m2_1 = build_engine(grid_cells=6)

cmds_group = [
    Command(CommandType.MOVE_FORWARD, "v0"),
    Command(CommandType.MOVE_FORWARD, "v0"),
    Command(CommandType.MOVE_FORWARD, "v1"),
    Command(CommandType.FORM_GROUP, "g0", {"member_ids": ["v0", "v1"]}),
    Command(CommandType.MOVE_RIGHT, "g0"),
    Command(CommandType.MOVE_RIGHT, "g0"),
    Command(CommandType.MOVE_FORWARD, "g0"),
]

ts_group = collect_timeseries(interp2, cmds_group)

# ===================================================================
# Plot 4: Group movement — two vehicles moving in sync
# ===================================================================
print("Plotting group movement...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# X position
ax1.plot(
    ts_group["v0"].time,
    ts_group["v0"].position[:, 0],
    label="Vehicle 0",
    color="#e74c3c",
    linewidth=2,
)
ax1.plot(
    ts_group["v1"].time,
    ts_group["v1"].position[:, 0],
    label="Vehicle 1",
    color="#3498db",
    linewidth=2,
    linestyle="--",
)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("X Position (m)")
ax1.set_title("Group Movement — X Axis (synchronized)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Highlight group formation point: find when both vehicles first move together in X
group_start_t = None
n = min(len(ts_group["v0"].time), len(ts_group["v1"].time))
for i in range(1, n):
    v0_moving_x = abs(ts_group["v0"].velocity[i, 0]) > 0.05
    v1_moving_x = abs(ts_group["v1"].velocity[i, 0]) > 0.05
    if v0_moving_x and v1_moving_x:
        group_start_t = ts_group["v0"].time[i]
        break

if group_start_t:
    ax1.axvline(x=group_start_t, color="green", linestyle=":", alpha=0.7, label="Group formed")
    ax1.legend()

# Y position
ax2.plot(
    ts_group["v0"].time,
    ts_group["v0"].position[:, 1],
    label="Vehicle 0",
    color="#e74c3c",
    linewidth=2,
)
ax2.plot(
    ts_group["v1"].time,
    ts_group["v1"].position[:, 1],
    label="Vehicle 1",
    color="#3498db",
    linewidth=2,
    linestyle="--",
)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Y Position (m)")
ax2.set_title("Group Movement — Y Axis (synchronized)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Synchronization annotation — compute position difference (use min length)
n_pos = min(len(ts_group["v0"].position), len(ts_group["v1"].position))
diff_xy = np.hypot(
    ts_group["v0"].position[:n_pos, 0] - ts_group["v1"].position[:n_pos, 0],
    ts_group["v0"].position[:n_pos, 1] - ts_group["v1"].position[:n_pos, 1],
)
# After group formation, the relative distance should stay constant
if group_start_t is not None:
    group_mask = ts_group["v0"].time > group_start_t
    if group_mask.any():
        # Use min length for safe indexing
        n_common = min(len(diff_xy), group_mask.sum())
        rel_dist = np.mean(diff_xy[-n_common:]) if n_common > 0 else 2.0
        ax2.text(
            0.5, 0.95,
            f"Relative distance between vehicles: {rel_dist:.2f}m",
            transform=ax2.transAxes,
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "group_movement.png", bbox_inches="tight")
plt.close(fig)
print("  Saved group_movement.png")

# Save group CSV
for vid in ["v0", "v1"]:
    ts = ts_group[vid]
    csv_path = OUTPUT_DIR / "scenario_data" / f"group_{vid}.csv"
    with open(csv_path, "w") as f:
        f.write("time,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z\n")
        for i in range(len(ts.time)):
            f.write(
                f"{ts.time[i]:.6f},"
                f"{ts.position[i,0]:.6f},{ts.position[i,1]:.6f},{ts.position[i,2]:.6f},"
                f"{ts.velocity[i,0]:.6f},{ts.velocity[i,1]:.6f},{ts.velocity[i,2]:.6f},"
                f"{ts.acceleration[i,0]:.6f},{ts.acceleration[i,1]:.6f},{ts.acceleration[i,2]:.6f}\n"
            )
    print(f"  Saved {csv_path} ({len(ts)} rows)")

# ===================================================================
# Summary
# ===================================================================
print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("Files:")
for f in sorted(OUTPUT_DIR.glob("*")):
    if f.is_file():
        print(f"  {f.name}")
for f in sorted((OUTPUT_DIR / "scenario_data").glob("*")):
    print(f"  scenario_data/{f.name}")
