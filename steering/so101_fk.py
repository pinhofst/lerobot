"""Forward kinematics of the SO-101 follower from TheRobotStudio's so101_new_calib.urdf, numpy only.

    uv run python steering/so101_fk.py                      # self-test: rest, median and zero poses
    uv run python steering/so101_fk.py --state=0,-33,34,58,-11,9  # one arm-frame state

The URDF (and its meshes, used only for viewing) is pinned in steering/assets/so101/ (see the README
there). The joint origins (xyz, rpy) and axes are parsed with xml.etree and composed with numpy, so no
kinematics extra (placo) is needed.

Joint mapping (LeRobot >= 0.5 calibration, ``use_degrees=True``, the so101_follower default):
  * The five arm joints are read in degrees with zero at the middle of the range swept during
    ``lerobot-calibrate`` (motors_bus.py: ``(raw - (min + max) / 2) * 360 / 4095``). so101_new_calib's
    zero is also "the middle of each joint's range" (upstream README), and LeRobot's own
    ``RobotKinematics`` users (examples/phone_to_so100, examples/so100_to_so100_EE,
    examples/isaac_teleop_to_so101) pass the calibrated degrees straight to this URDF as radians with no
    sign flip or offset. This module does the same: ``q_urdf = deg2rad(q_arm)``, same joint names.
    Assumption: the arm's swept range is centred where the CAD range is centred; a lopsided sweep
    during calibration shifts that joint's zero by half the asymmetry (typically a few degrees).
  * The gripper is RANGE_0_100 (0 closed, 100 open), not degrees. It does not move the tip frame
    (``gripper_frame_link`` hangs off the fixed jaw, ``gripper_link``); for drawing the moving jaw it is
    mapped linearly onto the URDF jaw limits [-10 deg, 100 deg]. That mapping is a guess (display only).

Frames: everything is in ``base_link`` (the robot base frame), metres. In the zero pose x points
forward (away from the base, along the arm), z up. ``base_link``'s origin is 2.4 mm above the underside
of the base mesh (base_so101_v2.stl spans z = -0.0024..0.0696 m), so a table the base stands directly on
is the plane z = -0.0024; the rig's actual mounting (clamp, plate) is not recorded. ``gripper_frame_link``
sits between the jaws about 6 mm short of the fixed jaw's end.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STEERING = Path(__file__).parent
ASSETS = STEERING / "assets" / "so101"
DEFAULT_URDF = ASSETS / "so101_new_calib.urdf"
POSES = STEERING / "poses"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINTS[:5]
BASE_LINK = "base_link"
TIP_LINK = "gripper_frame_link"  # LeRobot's RobotKinematics target frame, between the jaw tips
GRIPPER_LINK = "gripper_link"  # wrist-roll output, carries the fixed jaw
JAW_LINK = "moving_jaw_so101_v1_link"
# Fingertip of the moving jaw in JAW_LINK's frame (m): the moving_jaw_so101_v1.stl vertex farthest
# from the jaw axis, measured once from the pinned mesh. Display only.
JAW_TIP_IN_JAW = np.array([-0.0114, -0.0820, 0.0189])
# Skeleton drawn by the viewers: one point per joint origin, then the tip.
SKELETON = (
    ("base", BASE_LINK),
    ("shoulder_pan", "shoulder_link"),
    ("shoulder_lift", "upper_arm_link"),
    ("elbow_flex", "lower_arm_link"),
    ("wrist_flex", "wrist_link"),
    ("wrist_roll", GRIPPER_LINK),
    ("tip", TIP_LINK),
)


@dataclass(frozen=True)
class Joint:
    """One URDF joint: parent -> child with a fixed origin and, if revolute, an axis and limits."""

    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray  # unit 3-vector (zeros for fixed)
    lower: float
    upper: float


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.asarray([float(v) for v in (text or default).split()], dtype=np.float64)


def parse_urdf(path: Path = DEFAULT_URDF) -> dict[str, Joint]:
    """Joints of a URDF keyed by child link name (each link has one parent joint)."""
    root = ET.parse(path).getroot()
    out: dict[str, Joint] = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = _floats(o.get("xyz") if o is not None else None, "0 0 0")
        rpy = _floats(o.get("rpy") if o is not None else None, "0 0 0")
        origin = np.eye(4)
        origin[:3, :3] = rpy_matrix(*rpy)
        origin[:3, 3] = xyz
        a = j.find("axis")
        axis = _floats(a.get("xyz") if a is not None else None, "1 0 0")
        kind = j.get("type", "fixed")
        norm = float(np.linalg.norm(axis))
        axis = np.zeros(3) if kind == "fixed" or not norm else axis / norm
        lim = j.find("limit")
        lower = float(lim.get("lower", "-inf")) if lim is not None else -np.inf
        upper = float(lim.get("upper", "inf")) if lim is not None else np.inf
        parent, child = j.find("parent"), j.find("child")
        if parent is None or child is None:
            raise ValueError(f"{path}: joint {j.get('name')} has no parent/child")
        joint = Joint(
            j.get("name", ""), kind, parent.get("link", ""), child.get("link", ""), origin, axis, lower, upper
        )
        out[joint.child] = joint
    return out


def _axis_angle(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N, 4, 4) rotations about a fixed unit axis by angles q (N,) (Rodrigues)."""
    n = len(q)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = np.eye(3) + s * k + (1 - c) * (k @ k)
    return out


class SO101FK:
    """Batched FK: arm-frame LeRobot states (N, 6) -> link frames in base_link, metres."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        """Parse the URDF once."""
        self.urdf_path = Path(urdf_path)
        self.joints = parse_urdf(self.urdf_path)
        jaw = self.joints[JAW_LINK]
        self.jaw_limits = (jaw.lower, jaw.upper)

    def arm_to_urdf(self, arm: np.ndarray) -> np.ndarray:
        """(N, 6) LeRobot arm frame (deg, gripper 0-100) -> (N, 6) URDF joint angles (rad)."""
        arm = np.atleast_2d(np.asarray(arm, dtype=np.float64))
        q = np.deg2rad(arm)
        lo, hi = self.jaw_limits
        q[:, 5] = lo + np.clip(arm[:, 5], 0.0, 100.0) / 100.0 * (hi - lo)
        return q

    def _chain(self, link: str) -> list[Joint]:
        chain = []
        while link != BASE_LINK:
            j = self.joints[link]
            chain.append(j)
            link = j.parent
        return chain[::-1]

    def frames(self, arm: np.ndarray, links: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """{link: (N, 4, 4) pose in base_link} for every link (or ``links``) at arm-frame states."""
        q = self.arm_to_urdf(arm)
        n = len(q)
        cache: dict[str, np.ndarray] = {BASE_LINK: np.tile(np.eye(4), (n, 1, 1))}

        def pose(link: str) -> np.ndarray:
            if link not in cache:
                j = self.joints[link]
                t = pose(j.parent) @ j.origin
                if j.kind != "fixed":
                    t = t @ _axis_angle(j.axis, q[:, JOINTS.index(j.name)])
                cache[link] = t
            return cache[link]

        wanted = links if links is not None else (BASE_LINK, *self.joints)
        return {link: pose(link) for link in wanted}

    def tip(self, arm: np.ndarray) -> np.ndarray:
        """(N, 3) gripper tip (gripper_frame_link origin) in base_link, metres."""
        return self.frames(arm, (TIP_LINK,))[TIP_LINK][:, :3, 3]

    def skeleton(self, arm: np.ndarray) -> np.ndarray:
        """(N, len(SKELETON) + 1, 3) joint origins base..tip, then the moving-jaw fingertip."""
        f = self.frames(arm, (*[link for _, link in SKELETON], JAW_LINK))
        pts = [f[link][:, :3, 3] for _, link in SKELETON]
        jaw_tip = f[JAW_LINK] @ np.append(JAW_TIP_IN_JAW, 1.0)
        pts.append(jaw_tip[:, :3])
        return np.stack(pts, axis=1)


def _stl_vertices(path: Path) -> np.ndarray:
    """Vertices of a binary STL (N, 3), in the mesh's own units (metres for these meshes)."""
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].reshape(-1, 3).astype(np.float64)


def base_mesh_z_range(urdf_path: Path = DEFAULT_URDF) -> tuple[float, float] | None:
    """(z_min, z_max) of base_link's visual meshes in base_link, or None if the meshes are missing."""
    root = ET.parse(urdf_path).getroot()
    link = next(lk for lk in root.findall("link") if lk.get("name") == BASE_LINK)
    zs = []
    for vis in link.findall("visual"):
        mesh, o = vis.find("geometry/mesh"), vis.find("origin")
        if mesh is None:
            continue
        path = Path(urdf_path).parent / mesh.get("filename", "")
        if not path.is_file():
            return None
        t = np.eye(4)
        t[:3, :3] = rpy_matrix(*_floats(o.get("rpy") if o is not None else None, "0 0 0"))
        t[:3, 3] = _floats(o.get("xyz") if o is not None else None, "0 0 0")
        v = _stl_vertices(path) @ t[:3, :3].T + t[:3, 3]
        zs.append(v[:, 2])
    if not zs:
        return None
    z = np.concatenate(zs)
    return float(z.min()), float(z.max())


def load_pose(path: Path) -> np.ndarray:
    """Arm-frame state (6,) from a steering/poses/*.json file."""
    joints = json.loads(Path(path).read_text())["joints"]
    return np.asarray([joints[j] for j in JOINTS], dtype=np.float64)


def describe(fk: SO101FK, name: str, arm: np.ndarray) -> str:
    """One line: tip xyz, horizontal reach and height."""
    p = fk.tip(arm)[0]
    reach = float(np.hypot(p[0], p[1]))
    return (
        f"{name:<8} tip = ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m   reach {reach:.3f} m   "
        f"height {p[2]:.3f} m   joints {np.round(arm, 1).tolist()}"
    )


def self_test(fk: SO101FK) -> None:
    """Print rest, median and zero poses with a plausibility verdict."""
    home, median = load_pose(POSES / "home.json"), load_pose(POSES / "molmo_median.json")
    zero = np.zeros(6)
    print(f"URDF: {fk.urdf_path}")
    zr = base_mesh_z_range(fk.urdf_path)
    if zr is not None:
        print(
            f"base mesh z range in base_link: [{zr[0]:.4f}, {zr[1]:.4f}] m (underside = table if mounted flat)"
        )
    for name, arm in (("rest", home), ("median", median), ("zero", zero)):
        print(describe(fk, name, arm))
    ph, pm = fk.tip(home)[0], fk.tip(median)[0]
    rh, rm = np.hypot(*ph[:2]), np.hypot(*pm[:2])
    checks = {
        "rest folded: tip within 0.20 m of the base axis": rh < 0.20,
        "rest low: tip below 0.15 m": ph[2] < 0.15,
        "median in front: x > 0.15 m, |y| < x": pm[0] > 0.15 and abs(pm[1]) < pm[0],
        "median raised above the table: z > 0.02 m": pm[2] > 0.02,
        "median reaches farther than rest": rm > rh + 0.05,
    }
    for label, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not all(checks.values()):
        raise SystemExit("FK self-test failed")


def main() -> None:
    """CLI: self-test or one state."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--state", help="arm frame 'pan,lift,elbow,wflex,wroll,gripper' (deg, 0-100)")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = parser.parse_args()
    fk = SO101FK(args.urdf)
    if args.state is None:
        self_test(fk)
        return
    arm = np.asarray([float(v) for v in args.state.split(",")], dtype=np.float64)
    print(describe(fk, "state", arm))
    f = fk.frames(arm, (TIP_LINK, GRIPPER_LINK))
    np.set_printoptions(precision=4, suppress=True)
    print(f"{TIP_LINK} in base_link:\n{f[TIP_LINK][0]}")
    print(f"{GRIPPER_LINK} in base_link:\n{f[GRIPPER_LINK][0]}")


if __name__ == "__main__":
    main()
