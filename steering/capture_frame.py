"""Capture one SO-101 observation (two camera frames + joint state) as inputs for frame_diagnostic.py.

    uv run python steering/capture_frame.py --robot-port /dev/ttyACM0 --cam0 0 --cam1 2 --out steering/results/frames/rig1
    uv run python steering/capture_frame.py --robot-port /dev/ttyACM0 --robot-id follower \
        --cam0 /dev/video0 --cam1 /dev/video2 --out steering/results/frames/rig1 --warmup 30
    uv run python steering/frame_diagnostic.py --cam0 steering/results/frames/rig1/cam0.png \
        --cam1 steering/results/frames/rig1/cam1.png --state steering/results/frames/rig1/state.json --tag rig1

Writes ``<out>/cam0.png``, ``<out>/cam1.png`` (RGB, 640x480) and ``<out>/state.json`` (a JSON list of
six floats: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper; arm frame,
degrees, gripper 0-100), which is what frame_diagnostic.py's ``--cam0/--cam1/--state`` read. Then prints
the state and runs check_state_range.py on it.

The arm is never moved and torque is never changed. As in check_state_range.py, the script does not
call ``robot.connect()`` (which runs ``configure()`` under ``torque_disabled()`` and so re-enables
torque, and may prompt to calibrate) nor ``robot.disconnect()`` (which disables torque). It opens the
motor bus with ``robot.bus.connect()``, checks the calibration, connects the two cameras itself with
``robot.cameras[...].connect()``, reads one ``robot.get_observation()``, then closes the cameras and the
bus with ``disable_torque=False``. The robot is built with ``disable_torque_on_disconnect=False`` so
that ``Robot.__del__`` (which calls ``disconnect()`` if everything is still connected after an error)
cannot drop an arm that is holding a pose either. A held arm (e.g. after goto_pose.py go) stays held.

Cameras: ``--cam0`` / ``--cam1`` are OpenCV ``index_or_path`` values (an integer index or a device
path such as /dev/video2; ``lerobot-find-cameras opencv`` lists them), opened at 640x480 and 30 fps,
RGB. After connecting, ``--warmup`` fresh frames are read and discarded from each camera so auto
exposure and the capture buffers settle before the saved frame (lock_camera.sh fixes exposure).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from check_state_range import JOINTS, main as check_state_range_main
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SOFollowerRobotConfig

CAMERAS = ("cam0", "cam1")
WIDTH, HEIGHT, FPS = 640, 480, 30


def index_or_path(value: str) -> int | Path:
    """OpenCV accepts an integer device index or a path."""
    return int(value) if value.isdigit() else Path(value)


def capture(
    port: str, robot_id: str, cams: dict[str, int | Path], warmup: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (arm-frame state in JOINTS order, {cam name: RGB uint8 HxWx3}) without writing to the motors."""
    camera_configs = {
        name: OpenCVCameraConfig(index_or_path=src, fps=FPS, width=WIDTH, height=HEIGHT)
        for name, src in cams.items()
    }
    # disable_torque_on_disconnect=False: Robot.__del__ calls disconnect() when bus and cameras are all
    # connected; with the default it would switch torque off.
    robot = make_robot_from_config(
        SOFollowerRobotConfig(
            port=port, id=robot_id, cameras=camera_configs, disable_torque_on_disconnect=False
        )
    )
    if not robot.calibration:
        raise SystemExit(f"no calibration file at {robot.calibration_fpath}; run lerobot-calibrate first")
    try:
        robot.bus.connect()  # pings the motors; writes nothing
        if not robot.bus.is_calibrated:
            raise SystemExit(
                f"motor calibration differs from {robot.calibration_fpath}; run lerobot-calibrate "
                "(this script does not write calibration)"
            )
        for name, cam in robot.cameras.items():
            cam.connect()  # includes the config's warmup_s (1 s) of background reads
            for _ in range(warmup):
                cam.async_read(timeout_ms=1000)  # waits for a fresh frame
            print(f"{name}: connected ({cams[name]}), {warmup} warmup frames discarded")
        observation = robot.get_observation()  # Present_Position + each camera's latest frame
    finally:
        for cam in robot.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
    state = np.asarray([observation[f"{j}.pos"] for j in JOINTS], dtype=np.float64)
    images = {name: np.asarray(observation[name]) for name in cams}
    return state, images


def main() -> None:
    """Capture, save, print, and range-check one observation."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot-port", required=True, help="so101_follower serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--robot-id", default="follower", help="calibration id")
    parser.add_argument("--cam0", required=True, type=index_or_path, help="OpenCV index or path")
    parser.add_argument("--cam1", required=True, type=index_or_path, help="OpenCV index or path")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument("--warmup", type=int, default=15, help="frames discarded per camera before capture")
    args = parser.parse_args()

    state, images = capture(
        args.robot_port, args.robot_id, {"cam0": args.cam0, "cam1": args.cam1}, max(args.warmup, 0)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise SystemExit(f"{name}: expected an RGB uint8 HxWx3 frame, got {image.dtype} {image.shape}")
        # The camera's color_mode defaults to RGB (BGR -> RGB already done in OpenCVCamera), which is
        # what PIL expects; cv2.imwrite would need BGR.
        Image.fromarray(image).save(args.out / f"{name}.png")
        print(f"wrote {args.out / f'{name}.png'}  ({image.shape[1]}x{image.shape[0]})")
    state_path = args.out / "state.json"
    state_path.write_text(json.dumps([float(v) for v in state]) + "\n")
    print(f"wrote {state_path}")
    print()
    for joint, value in zip(JOINTS, state, strict=True):
        print(f"  {joint:<14}{value:9.2f}")
    print()

    # Same check as `check_state_range.py --state-json <out>/state.json`, run in-process.
    sys.argv = [sys.argv[0], "--state-json", str(state_path)]
    try:
        check_state_range_main()
    except Exception as e:  # the capture is already saved; do not lose it to a Hub/network error
        print(f"range check failed ({type(e).__name__}: {e}); rerun:")
        print(f"  uv run python steering/check_state_range.py --state-json {state_path}")


if __name__ == "__main__":
    main()
