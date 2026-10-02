"""Store a named SO-101 follower pose and move the arm back to it slowly (e.g. the raised start pose).

    uv run python steering/goto_pose.py save start --robot-port /dev/ttyACM0
    uv run python steering/goto_pose.py go start --robot-port /dev/ttyACM0 --dry-run
    uv run python steering/goto_pose.py go start --robot-port /dev/ttyACM0
    uv run python steering/goto_pose.py go start --robot-port /dev/ttyACM0 --speed 5 --tolerance 1.5 --timeout 25

``save NAME`` reads Present_Position once and writes steering/poses/NAME.json (joint names -> degrees,
gripper 0-100, LeRobot >= 0.5 calibration). It writes nothing to the motors and does not touch torque
(it reuses check_state_range.read_robot_state): hold the arm in place by hand while it reads, or have
it held by torque. The serial port can only be open in one process, so stop any teleop first (note that
lerobot-teleoperate disables the follower's torque when it exits: support the arm).

``go NAME`` moves the follower to NAME:
  1. Opens the bus, checks the motors' calibration against the file for ``--robot-id`` (refuses on a
     mismatch, never prompts or calibrates), reads Present_Position and writes it as Goal_Position, so
     that enabling torque holds the arm where it is instead of jumping to a stale goal.
  2. Enables torque directly on the bus, without ``robot.connect()``: its ``configure()`` switches
     torque off for a moment, which would drop a raised arm and snap it back. The motor settings it
     writes persist in the servos, so the arm must have been through ``lerobot-calibrate`` or
     ``lerobot-teleoperate`` at least once (true after step 2 of COMMANDS.md).
  3. Sends goals linearly interpolated from the present pose to NAME at ``--fps`` (30 Hz), at most
     ``--speed`` deg/s on the joint that moves most (all joints arrive together), with
     ``max_relative_target = --max-step`` (5 deg) as a second cap: SOFollower.send_action reads
     Present_Position and clips every goal to within max_relative_target of it.
  4. Keeps sending NAME until every joint is within ``--tolerance`` (1 deg) or ``--timeout`` (15 s,
     counted from the first step) runs out, then prints the per-joint error.
  5. Disconnects with ``disable_torque_on_disconnect=False``: TORQUE STAYS ON and the arm holds NAME
     after the script exits. Whatever opens the port next takes over; any stock LeRobot disconnect
     (lerobot-record, lerobot-teleoperate) disables torque and the arm drops: support it.

Ctrl-C during ``go`` stops the motion: the script reads Present_Position, writes it as the goal (the
arm stops where it is, holding), disconnects with torque on, and exits. Any other error does the same.

SAFETY: keep a hand near the follower's power switch during ``go``. Cutting power is the only stop
that does not depend on this script, the USB link or the motors obeying a goal. Run ``--dry-run``
first: it reads the present pose (read-only, like ``save``) and prints start, end, the largest move,
the number of steps and the duration without sending anything.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from check_state_range import JOINTS, read_robot_state

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

POSES = Path(__file__).parent / "poses"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def pose_path(name: str) -> Path:
    """steering/poses/NAME.json, rejecting names that are not plain file stems."""
    if not NAME_RE.match(name):
        raise SystemExit(f"pose name {name!r}: use letters, digits, '_' or '-'")
    return POSES / f"{name}.json"


def load_pose(name: str) -> np.ndarray:
    """Target in JOINTS order."""
    path = pose_path(name)
    if not path.is_file():
        raise SystemExit(f"no pose {path}; run: goto_pose.py save {name}")
    joints = json.loads(path.read_text())["joints"]
    try:
        return np.asarray([joints[j] for j in JOINTS], dtype=np.float64)
    except KeyError as e:
        raise SystemExit(f"{path}: missing joint {e}") from e


def print_table(rows: dict[str, np.ndarray]) -> None:
    """One column per named vector, one row per joint."""
    print(f"{'joint':<14}" + "".join(f"{k:>10}" for k in rows))
    for i, joint in enumerate(JOINTS):
        print(f"{joint:<14}" + "".join(f"{v[i]:10.2f}" for v in rows.values()))


def plan(start: np.ndarray, target: np.ndarray, speed: float, max_step: float, fps: float) -> np.ndarray:
    """Interpolated goals (n_steps, 6), ending exactly on target; per-step move <= min(speed/fps, max_step)."""
    step = min(speed / fps, max_step)
    n_steps = max(1, math.ceil(float(np.abs(target - start).max()) / step))
    fractions = np.arange(1, n_steps + 1, dtype=np.float64)[:, None] / n_steps
    return start + fractions * (target - start)


def save(args: argparse.Namespace) -> None:
    """Read the present pose (no writes) and store it."""
    path = pose_path(args.name)
    state = read_robot_state(args.robot_port, args.robot_id)
    POSES.mkdir(exist_ok=True)
    record = {
        "name": args.name,
        "robot_id": args.robot_id,
        "units": "degrees (gripper 0-100), LeRobot >= 0.5 calibration",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "joints": {j: round(float(v), 3) for j, v in zip(JOINTS, state, strict=True)},
    }
    existed = path.is_file()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print_table({"saved": state})
    print(f"{'overwrote' if existed else 'wrote'} {path}")


def read_present(robot: SOFollower) -> np.ndarray:
    """Present_Position in JOINTS order."""
    present = robot.bus.sync_read("Present_Position", num_retry=robot.config.num_read_retries)
    return np.asarray([present[j] for j in JOINTS], dtype=np.float64)


def _free_port(robot: SOFollower) -> None:
    """Clear the SDK's busy flag if Ctrl-C landed mid-packet (as MotorsBus.disconnect does)."""
    robot.bus.port_handler.is_using = False
    robot.bus.port_handler.clearPort()


def hold_here(robot: SOFollower) -> np.ndarray:
    """Stop: write Present_Position as the goal so the arm holds where it is."""
    present = read_present(robot)
    robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, present.tolist(), strict=True)))
    return present


def go(args: argparse.Namespace) -> None:
    """Move slowly to the stored pose and leave torque on."""
    target = load_pose(args.name)
    if args.dry_run:
        start = read_robot_state(args.robot_port, args.robot_id)
        goals = plan(start, target, args.speed, args.max_step, args.fps)
        print_table({"start": start, "end": target, "move": target - start})
        duration = len(goals) / args.fps
        print(
            f"\nlargest move {np.abs(target - start).max():.1f} deg; {len(goals)} steps at {args.fps:g} Hz "
            f"= {duration:.1f} s (timeout {args.timeout:g} s); max_relative_target {args.max_step:g} deg"
        )
        if duration >= args.timeout:
            print("the interpolation alone exceeds --timeout: raise --timeout or --speed")
        print("dry run: nothing sent")
        return

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.robot_port,
            id=args.robot_id,
            cameras={},
            max_relative_target=float(args.max_step),  # ensure_safe_goal_position needs a float
            disable_torque_on_disconnect=False,  # leave torque on, also from Robot.__del__
        )
    )
    if not robot.calibration:
        raise SystemExit(f"no calibration file at {robot.calibration_fpath}; run lerobot-calibrate first")

    # 1. Read-only calibration check, then seed Goal_Position with Present_Position.
    robot.bus.connect()
    try:
        if not robot.bus.is_calibrated:
            raise SystemExit(
                f"motor calibration differs from {robot.calibration_fpath}; run lerobot-calibrate"
            )
        start = hold_here(robot)
    finally:
        robot.bus.disconnect(disable_torque=False)

    goals = plan(start, target, args.speed, args.max_step, args.fps)
    duration = len(goals) / args.fps
    print_table({"start": start, "end": target, "move": target - start})
    print(f"\n{len(goals)} steps at {args.fps:g} Hz = {duration:.1f} s (timeout {args.timeout:g} s)")
    if duration >= args.timeout:
        raise SystemExit("the interpolation alone exceeds --timeout: raise --timeout or --speed")

    period = 1.0 / args.fps
    present = start
    outcome = "timeout"
    try:
        # 2. Enable torque WITHOUT robot.connect(): its configure() runs under torque_disabled(), which
        #    would drop a raised arm for a moment and then snap it back at full acceleration. The motor
        #    settings configure() writes persist in the servos from any earlier lerobot-calibrate /
        #    lerobot-teleoperate run on this arm, so they are not rewritten here.
        robot.bus.connect()
        start = hold_here(robot)  # re-seed: the arm may have moved since step 1
        robot.bus.enable_torque()
        print("torque ON. Moving; Ctrl-C stops and holds. Keep a hand near the power switch.")
        t0 = time.perf_counter()
        k = 0
        while True:
            tick = time.perf_counter()
            goal = goals[min(k, len(goals) - 1)]
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, goal, strict=True)})
            k += 1
            present = read_present(robot)
            if k >= len(goals) and np.abs(target - present).max() <= args.tolerance:
                outcome = "reached"
                break
            if tick - t0 >= args.timeout:
                break
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        outcome = "interrupted"
        if robot.bus.is_connected:
            _free_port(robot)
            present = hold_here(robot)
    except BaseException:
        if robot.bus.is_connected:
            _free_port(robot)
            hold_here(robot)
        raise
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)  # torque stays on

    print()
    print_table({"target": target, "present": present, "error": target - present})
    worst = int(np.abs(target - present).argmax())
    print(
        f"\n{outcome}: max |error| {abs(target[worst] - present[worst]):.2f} on {JOINTS[worst]} "
        f"(tolerance {args.tolerance:g})"
    )
    if outcome == "interrupted":
        print("motion stopped where it was.")
    print("TORQUE IS STILL ON: the arm is holding this pose. Support it before anything disables torque.")


def main() -> None:
    """Dispatch save / go."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("save", "store the present pose"), ("go", "move slowly to a stored pose")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("name", help="pose name, stored as steering/poses/NAME.json")
        p.add_argument("--robot-port", required=True, help="so101_follower serial port, e.g. /dev/ttyACM0")
        p.add_argument("--robot-id", default="follower", help="calibration id")
        if name == "go":
            p.add_argument(
                "--max-step",
                type=float,
                default=5.0,
                help="max_relative_target, deg per step (a sanity bound)",
            )
            p.add_argument("--speed", type=float, default=10.0, help="deg/s on the joint that moves most")
            p.add_argument("--tolerance", type=float, default=1.0, help="deg, every joint")
            p.add_argument("--timeout", type=float, default=15.0, help="s, from the first step")
            p.add_argument("--fps", type=float, default=30.0, help="goal rate, Hz")
            p.add_argument("--dry-run", action="store_true", help="read only; print the plan")
    args = parser.parse_args()
    if args.command == "go" and min(args.max_step, args.speed, args.tolerance, args.timeout, args.fps) <= 0:
        parser.error("--max-step, --speed, --tolerance, --timeout and --fps must be positive")
    if args.command == "save":
        save(args)
    else:
        go(args)


if __name__ == "__main__":
    main()
