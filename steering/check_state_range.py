"""Check that an SO-101 joint state lies inside the range the MolmoAct2 checkpoint was trained on.

    uv run python steering/check_state_range.py                                  # Ai2's sample state
    uv run python steering/check_state_range.py --state '0,-90,90,60,0,1'        # arm frame, degrees
    uv run python steering/check_state_range.py --state-json state.json
    uv run python steering/check_state_range.py --robot-port /dev/ttyACM0 --robot-id follower

``lerobot/MolmoAct2-SO100_101-LeRobot`` normalises state and action with QUANTILES (q01 -> -1,
q99 -> +1). The preprocessor then clamps the normalised state to [-1, 1] and the postprocessor clamps
the normalised action to [-1, 1] before unnormalising. A start pose outside [q01, q99] is therefore
seen by the model as if it were on the bound, and the first action cannot land outside the action
range: the arm jumps back inside by at least the distance it was outside. On Ai2's sample pose the
elbow jumps about 8 degrees.

The stats are in the model frame (pre-v0.5 calibration). The arm (LeRobot >= 0.5 calibration,
degrees, gripper 0-100) relates to it as ``state_model = joint_signs * arm + joint_offsets``, with
both vectors read from the checkpoint's config.json. Ranges are shown in the arm frame.

State inputs (arm frame):
  * ``--state``: six comma-separated numbers.
  * ``--state-json``: a JSON list of six numbers, or an object keyed by joint name (``elbow_flex`` or
    ``elbow_flex.pos``), optionally nested under ``observation.state``.
  * ``--robot-port`` / ``--robot-id``: reads Present_Position once from an ``so101_follower`` with no
    cameras. It opens the bus directly instead of calling ``robot.connect()``, because ``connect()``
    re-enables torque (``configure()`` runs under ``torque_disabled()``) and may prompt to calibrate,
    and ``robot.disconnect()`` disables torque (an arm holding a pose would drop). This path only
    pings the motors, reads the calibration registers and the positions, and closes the port with
    ``disable_torque=False``: it writes nothing, so torque stays as it was. It refuses to read if the
    motors' calibration differs from the calibration file for ``--robot-id``.
  * No input: Ai2's model-card sample (``SAMPLE_STATE_MODEL_FRAME`` in molmo_common.py).

Runs in seconds on CPU; no model weights are loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

POLICY_REPO = "lerobot/MolmoAct2-SO100_101-LeRobot"
STATE_STATS_FILE = "policy_preprocessor_step_3_molmoact2_masked_normalizer.safetensors"
ACTION_STATS_FILE = "policy_postprocessor_step_1_molmoact2_masked_unnormalizer.safetensors"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TOLERANCE = 1e-3  # degrees; below this a distance counts as inside


def load_frame(repo: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (joint_signs, joint_offsets) from the checkpoint's config.json."""
    config = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
    signs = np.asarray(config["joint_signs"], dtype=np.float64)
    offsets = np.asarray(config["joint_offsets"], dtype=np.float64)
    if signs.shape != (len(JOINTS),) or offsets.shape != (len(JOINTS),):
        raise ValueError(f"expected {len(JOINTS)} joint signs/offsets, got {signs}, {offsets}")
    return signs, offsets


def load_quantiles(repo: str, filename: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the model-frame (q01, q99) of ``key`` from a processor state file."""
    stats = load_file(hf_hub_download(repo, filename))
    q01, q99 = stats[f"{key}.q01"].astype(np.float64), stats[f"{key}.q99"].astype(np.float64)
    mask = stats.get(f"{key}.mask")
    if mask is not None and not mask.astype(bool).all():
        # Masked dimensions pass through unnormalised and unclamped: they have no trained range.
        q01, q99 = q01.copy(), q99.copy()
        q01[~mask.astype(bool)], q99[~mask.astype(bool)] = -np.inf, np.inf
    return q01, q99


def model_to_arm(values: np.ndarray, signs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Invert ``model = signs * arm + offsets``."""
    return signs * (values - offsets)


def range_to_arm(
    q01: np.ndarray, q99: np.ndarray, signs: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map a model-frame [q01, q99] to the arm frame; a sign flip swaps the bounds."""
    a, b = model_to_arm(q01, signs, offsets), model_to_arm(q99, signs, offsets)
    return np.minimum(a, b), np.maximum(a, b)


def distance_outside(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Distance from ``x`` to [low, high], 0 inside."""
    return np.maximum(low - x, 0.0) + np.maximum(x - high, 0.0)


def parse_state_json(path: str) -> np.ndarray:
    """Read an arm-frame state from a JSON list or a joint-keyed object."""
    data: Any = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "observation.state" in data:
        data = data["observation.state"]
    if isinstance(data, dict):
        try:
            return np.asarray([data[j] if j in data else data[f"{j}.pos"] for j in JOINTS], dtype=np.float64)
        except KeyError as e:
            raise SystemExit(f"{path}: missing joint {e}; expected keys {list(JOINTS)}") from e
    return np.asarray(data, dtype=np.float64)


def read_robot_state(port: str, robot_id: str) -> np.ndarray:
    """Read one arm-frame state from an so101_follower without writing to the motors."""
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SOFollowerRobotConfig

    # disable_torque_on_disconnect=False: if anything below fails with the port open, Robot.__del__
    # calls robot.disconnect(), which would otherwise switch torque off and drop a held arm.
    robot = make_robot_from_config(
        SOFollowerRobotConfig(port=port, id=robot_id, cameras={}, disable_torque_on_disconnect=False)
    )
    if not robot.calibration:
        raise SystemExit(f"no calibration file at {robot.calibration_fpath}; run lerobot-calibrate first")
    # Bus only: robot.connect() would enable torque, robot.disconnect() would disable it.
    try:
        robot.bus.connect()
        if not robot.bus.is_calibrated:
            raise SystemExit(
                f"motor calibration differs from {robot.calibration_fpath}; run lerobot-calibrate "
                "(this script does not write calibration)"
            )
        observation = robot.get_observation()
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
    return np.asarray([observation[f"{j}.pos"] for j in JOINTS], dtype=np.float64)


def sample_state(signs: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Ai2's model-card sample state, converted to the arm frame."""
    from molmo_common import SAMPLE_STATE_MODEL_FRAME  # heavy (torch); only needed here

    return model_to_arm(SAMPLE_STATE_MODEL_FRAME.astype(np.float64), signs, offsets)


def fmt_range(low: float, high: float) -> str:
    """Format a closed interval."""
    return f"[{low:8.2f}, {high:8.2f}]"


def main() -> None:
    """Print, per joint, the state against the checkpoint's trained state and action ranges."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--state",
        help="arm frame, 'a,b,c,d,e,f' (degrees, gripper 0-100); write --state=-10,... if it starts with -",
    )
    source.add_argument("--state-json", help="JSON list of six values or object keyed by joint name")
    source.add_argument("--robot-port", help="so101_follower serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--robot-id", default="follower", help="calibration id (with --robot-port)")
    parser.add_argument("--repo", default=POLICY_REPO)
    args = parser.parse_args()

    signs, offsets = load_frame(args.repo)
    if args.state is not None:
        arm, label = np.asarray([float(v) for v in args.state.split(",")], dtype=np.float64), "--state"
    elif args.state_json is not None:
        arm, label = parse_state_json(args.state_json), args.state_json
    elif args.robot_port is not None:
        arm, label = read_robot_state(args.robot_port, args.robot_id), f"robot on {args.robot_port}"
    else:
        arm, label = sample_state(signs, offsets), "Ai2 sample (molmo_common.SAMPLE_STATE_MODEL_FRAME)"
    if arm.shape != (len(JOINTS),) or not np.isfinite(arm).all():
        raise SystemExit(f"expected {len(JOINTS)} finite joint values, got {arm.tolist()}")

    s_low, s_high = range_to_arm(
        *load_quantiles(args.repo, STATE_STATS_FILE, "observation.state"), signs, offsets
    )
    a_low, a_high = range_to_arm(*load_quantiles(args.repo, ACTION_STATS_FILE, "action"), signs, offsets)
    s_out = distance_outside(arm, s_low, s_high)
    a_out = distance_outside(arm, a_low, a_high)

    print(f"state source: {label}")
    print(f"checkpoint:   {args.repo}  (arm frame; model = {signs.tolist()} * arm + {offsets.tolist()})")
    print()
    header = f"{'joint':<14}{'state':>9}  {'state q01..q99':<22}{'out':>6}  {'action q01..q99':<22}{'out':>6}  flag"
    print(header)
    print("-" * len(header))
    outside = []
    for i, joint in enumerate(JOINTS):
        flags = []
        if s_out[i] > TOLERANCE:
            flags.append("STATE")
        if a_out[i] > TOLERANCE:
            flags.append("ACTION")
        if flags:
            outside.append(joint)
        print(
            f"{joint:<14}{arm[i]:9.2f}  {fmt_range(s_low[i], s_high[i]):<22}{s_out[i]:6.2f}  "
            f"{fmt_range(a_low[i], a_high[i]):<22}{a_out[i]:6.2f}  {'OUTSIDE ' + '+'.join(flags) if flags else 'ok'}"
        )
    print()
    print("STATE: the model sees this joint clamped to the bound. ACTION: the first action is clamped")
    print("to the action range, so the joint must move at least 'out' degrees on it.")
    print()
    if not outside:
        print("inside the trained range")
    else:
        jump = float(a_out.max())
        if jump > TOLERANCE:
            worst = JOINTS[int(a_out.argmax())]
            print(
                f"OUTSIDE on {', '.join(outside)}: expect a jump of at least {jump:.1f}° on the first action "
                f"(largest on {worst})"
            )
        else:
            print(
                f"OUTSIDE on {', '.join(outside)} (state range only): the model sees a clamped state, "
                "but the action range contains it, so no forced jump"
            )


if __name__ == "__main__":
    main()
