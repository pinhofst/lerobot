"""Step 2 servo health check on a short teleop recording (no video decoding).

    uv run python steering/servo_check.py --repo-id <user>/<teleop_dataset> [--root PATH]

Per joint, on the follower's measured state and the leader's commanded action:
  * dead:    std of the commanded action < 1% of the joint's full scale, or the follower barely
             tracks a leader that is moving (tracking correlation < 0.9; not applied to the
             gripper, which stalls on grasped objects by design).
  * jittery: RMS second difference of the follower state (frame-to-frame acceleration, the
             high-frequency part a smooth teleop motion does not have) > 3x the median across joints.

The plan phrased "dead" as variance < 1% of range; variance is in squared units, so this uses std.
Raw variance is also a poor jitter signal (a joint that moves a lot has large variance), hence the
second-difference criterion.
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets import LeRobotDataset

# SO-101 with the current default use_degrees=True: arm joints in degrees (a nominal 180 deg of
# usable travel), gripper still in [0, 100]. For datasets recorded with use_degrees=False the arm
# joints are in [-100, 100]: pass --full-scale 200 200 200 200 200 100.
DEFAULT_FULL_SCALE = [180, 180, 180, 180, 180, 100]


def main() -> None:
    """Print per-joint dead/jitter statistics for a teleop dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--full-scale", type=float, nargs=6, default=DEFAULT_FULL_SCALE)
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    data = dataset.hf_dataset.select_columns(["episode_index", "action", "observation.state"]).with_format(
        "numpy"
    )
    episodes = np.asarray(data["episode_index"])
    action = np.stack(data["action"]).astype(np.float64)
    state = np.stack(data["observation.state"]).astype(np.float64)
    names = dataset.meta.features["action"]["names"] or [f"j{i}" for i in range(action.shape[1])]
    full_scale = np.asarray(args.full_scale)

    accel = []
    for ep in np.unique(episodes):
        s = state[episodes == ep]
        if len(s) > 2:
            accel.append(np.diff(s, n=2, axis=0))
    jitter = np.sqrt(np.mean(np.square(np.concatenate(accel)), axis=0))
    jitter_ratio = jitter / np.median(jitter)

    print(f"{dataset.num_episodes} episodes, {len(action)} frames @ {dataset.fps} fps")
    print(f"{'joint':<16}{'std/scale':>10}{'track r':>9}{'track rms':>10}{'jitter':>8}{'x med':>7}  flags")
    for j, name in enumerate(names):
        std_frac = action[:, j].std() / full_scale[j]
        r = np.corrcoef(action[:, j], state[:, j])[0, 1] if action[:, j].std() > 0 else float("nan")
        track = np.sqrt(np.mean(np.square(state[:, j] - action[:, j])))
        flags = []
        if std_frac < 0.01:
            flags.append("DEAD? (leader barely moved this joint — exercise it and re-record)")
        elif not r >= 0.9 and "gripper" not in name:  # a grasped object stalls the gripper by design
            flags.append("DEAD? (follower does not track leader)")
        if jitter_ratio[j] > 3:
            flags.append("JITTERY")
        print(
            f"{name:<16}{std_frac:>10.3f}{r:>9.3f}{track:>10.2f}{jitter[j]:>8.3f}{jitter_ratio[j]:>7.2f}  "
            + ", ".join(flags)
        )


if __name__ == "__main__":
    main()
