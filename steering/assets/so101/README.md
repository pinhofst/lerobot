# SO-101 follower URDF (pinned copy)

Source: TheRobotStudio/SO-ARM100, `Simulation/SO101/so101_new_calib.urdf`, downloaded 2026-10-02 at
commit `5f6d2b876a53a4872e405b991dd925556c9e38a4` (main HEAD then; the last commit touching the URDF
is `385e8d7c68e24945df6c60d9bd68837a4b7411ae`, 2025-07-02, "fix(urdf): fixing multiple issues related
to the last URDF update (#117)").

```bash
REV=5f6d2b876a53a4872e405b991dd925556c9e38a4
BASE=https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/$REV/Simulation/SO101
curl -sSfO $BASE/so101_new_calib.urdf
mkdir -p assets
for f in $(grep -o 'filename="assets/[^"]*"' so101_new_calib.urdf | sort -u | sed 's/filename="assets\///;s/"//'); do
  curl -sSf -o assets/$f $BASE/assets/$f
done
```

- `so101_new_calib.urdf` (sha256 `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`):
  "new calibration", each joint's zero at the middle of its range, the convention of LeRobot >= 0.5
  (`use_degrees=True`). Used by `steering/so101_fk.py`.
- `assets/*.stl`: the 13 meshes the URDF references (16 MB). Not needed for FK. `so101_fk.py` reads
  `base_so101_v2.stl` only to find the base underside (z = -0.0024 m in `base_link`); the moving-jaw
  fingertip offset in `so101_fk.py` was measured from `moving_jaw_so101_v1.stl`. `*.stl` is gitignored
  repo-wide, so re-run the loop above on a fresh clone.

Upstream notes: the gripper is a revolute joint in the URDF (-10..100 deg), whereas LeRobot reports it
as 0 (closed) .. 100 (open); upstream says this mapping "is not yet reflected" in the URDF.
