"""Run ``lerobot-rollout`` in-process with a torque-safe SO-101 connect and a per-tick recording.

    uv run python steering/rollout.py --tag median_rtc <the usual lerobot-rollout arguments>
    uv run python steering/plot_run.py steering/results/runs/<stamp>_median_rtc

Wrapper-only flags (stripped before forwarding): ``--tag NAME`` (default ``run``),
``--allow-torque-blip`` (see patch a) and ``--no-plot`` (skip the automatic plot_run on the run dir
after the run, normal exit or Ctrl-C). ``--robot.max_relative_target=DEG`` is required: the cap is
per control step at 30 Hz, so 4 deg per step is about 120 deg/s. Every other argument is passed
unchanged to ``lerobot.scripts.lerobot_rollout.main``. If ``--robot.disable_torque_on_disconnect`` is not given,
``--robot.disable_torque_on_disconnect=false`` is added, so the arm keeps holding its pose when the
rollout exits (support it before anything else disables torque).

Patch a: torque-safe connect (``SOFollower.configure``)
    Stock ``SOFollower.connect()`` calls ``configure()``, which rewrites the motor settings inside
    ``bus.torque_disabled()``: a raised arm drops for a moment and snaps back. The patched configure
    first reads, without writing, every EEPROM setting configure() would write: Operating_Mode ==
    POSITION, P/I/D == the config's coefficients, Return_Delay_Time == 0, Phase bit 4 clear, and the
    gripper's Max_Torque_Limit 500 / Protection_Current 250 / Overload_Torque 25. It also checks
    ``bus.is_calibrated``. If all match (they persist in the servos once lerobot-calibrate or
    lerobot-teleoperate has run), it does what steering/goto_pose.py does: Goal_Position :=
    Present_Position (skipped when every motor already has torque on, e.g. holding a goto_pose.py
    pose, so the goal is not lowered to the sagged present position), rewrites the two RAM settings (Acceleration, Maximum_Acceleration = 254) only
    if they differ (RAM writes do not need torque off), and calls ``bus.enable_torque()``. Torque is
    never switched off. The read is tried 3 times. If anything differs or cannot be read, it exits
    (SystemExit listing the mismatches) without touching torque; only with ``--allow-torque-blip``
    does it print a WARNING that torque will blip and run the original configure().
    connect()'s only other torque-off path is ``calibrate()``, run when the motors' calibration
    differs from the file. That would prompt and may switch torque off, so the patched
    ``SOFollower.calibrate`` refuses with an error instead (run lerobot-calibrate separately).

Patch b: recording, into steering/results/runs/<YYYYmmdd-HHMMSS>_<tag>/
    ticks.npz   one row per ``SOFollower.send_action`` call: ``t`` (s, perf_counter from the start
                of the recording), ``t_wall`` (epoch s), ``state`` (Present_Position from the last
                get_observation, 6 joints), ``action_policy`` (the action passed to send_action,
                i.e. after the robot_action_processor, before the max_relative_target clamp),
                ``action_sent`` (what send_action returned: the goal actually written, clipped by
                ensure_safe_goal_position), ``clipped`` (bool per joint, |policy - sent| > 1e-4, the
                same threshold as ensure_safe_goal_position's warning) and ``phase`` (0 before the
                control loop, 1 inside it, 2 teardown, e.g. return_to_initial_position).
    chunks.npz  one row per ``policy.predict_action_chunk`` call (sync engine via select_action,
                RTC engine directly): ``t_start``, ``duration`` (s), ``chunk_norm`` (normalised
                model output, (n, steps, 6)), ``chunk_arm`` (the same chunk clamped to [-1, 1],
                unnormalised and converted to the arm frame, as the postprocessor does; computed at
                the end of the run) and ``inference_delay`` (RTC only, else -1).
    frames/     one RGB JPEG per camera every 0.5 s, taken from the observation dict, written by a
                background thread. Named ``<index>_<t>s_<camera>.jpg``.
    meta.json   argv (as typed and as forwarded), tag, task, fps, policy path, robot settings, the
                connect report (fast path or fallback, and why), counts and recording errors.

Recording never stops the control loop: every hook is wrapped in try/except and drops the record on
error (counted in meta.json). The arrays are written when lerobot-rollout returns (Ctrl-C included).

SAFETY: as with plain lerobot-rollout, keep a hand near the follower's power switch.
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

RUNS = Path(__file__).parent / "results" / "runs"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FRAME_PERIOD_S = 0.5
CLIP_EPS = 1e-4  # ensure_safe_goal_position's own threshold
MAX_LOGGED_ERRORS = 5

logger = logging.getLogger("steering.rollout")


# ---------------------------------------------------------------------------
# argv handling
# ---------------------------------------------------------------------------


def split_argv(argv: list[str]) -> tuple[str, list[str], dict[str, bool]]:
    """Strip the wrapper flags, require ``--robot.max_relative_target``, add the torque default.

    Wrapper flags: ``--tag``, ``--allow-torque-blip``, ``--no-plot``.
    """
    tag, rest, i = "run", [], 0
    flags = {"allow_torque_blip": False, "no_plot": False}
    while i < len(argv):
        arg = argv[i]
        if arg in ("--allow-torque-blip", "--no-plot"):
            flags[arg[2:].replace("-", "_")] = True
            i += 1
            continue
        if arg == "--tag":
            if i + 1 >= len(argv):
                raise SystemExit("--tag needs a value")
            tag, i = argv[i + 1], i + 2
            continue
        if arg.startswith("--tag="):
            tag = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    if not tag or not all(c.isalnum() or c in "_-" for c in tag):
        raise SystemExit(f"--tag {tag!r}: use letters, digits, '_' or '-'")
    is_help = any(a in ("-h", "--help") for a in rest)
    if not is_help and not any(a.startswith("--robot.max_relative_target=") for a in rest):
        raise SystemExit(
            "refusing to run without --robot.max_relative_target=DEG. The cap is per control step at "
            "30 Hz, so --robot.max_relative_target=4 (4 deg per step) allows about 120 deg/s per joint."
        )
    if not any(a.startswith("--robot.disable_torque_on_disconnect") for a in rest):
        rest.append("--robot.disable_torque_on_disconnect=false")
    return tag, rest, flags


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Thread-safe in-memory log of ticks, chunks and frames; written once at the end."""

    def __init__(self, run_dir: Path, meta: dict[str, Any]) -> None:
        """Create the run dir and start the frame-writer thread."""
        self.run_dir = run_dir
        self.frames_dir = run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.t0 = time.perf_counter()
        self.lock = threading.Lock()
        self.phase = 0
        self.last_state: np.ndarray | None = None
        self.ticks: dict[str, list] = {
            k: [] for k in ("t", "t_wall", "state", "action_policy", "action_sent", "clipped", "phase")
        }
        self.chunks: dict[str, list] = {
            k: [] for k in ("t_start", "duration", "chunk_norm", "inference_delay")
        }
        self.frames: list[dict[str, Any]] = []
        self.next_frame_t = 0.0
        self.errors: dict[str, int] = {}
        self.policy_path: str | None = None
        self.saved = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=8)
        self.frame_thread = threading.Thread(target=self._frame_writer, name="frame-writer", daemon=True)
        self.frame_thread.start()

    def now(self) -> float:
        """Seconds since the recorder started (perf_counter)."""
        return time.perf_counter() - self.t0

    def error(self, where: str, exc: BaseException) -> None:
        """Count a dropped record; log the first few."""
        n = self.errors.get(where, 0) + 1
        self.errors[where] = n
        if n <= MAX_LOGGED_ERRORS:
            logger.warning("recording: dropped a %s record (%r)", where, exc)

    # -- hooks ---------------------------------------------------------------

    def on_observation(self, obs: dict[str, Any], cameras: dict[str, Any]) -> None:
        """Keep the latest joint state; queue a frame per camera every FRAME_PERIOD_S."""
        state = np.asarray([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float64)
        t = self.now()
        with self.lock:
            self.last_state = state
            save_frames = t >= self.next_frame_t
            if save_frames:
                self.next_frame_t = t + FRAME_PERIOD_S
        if save_frames:
            for cam in cameras:
                img = obs.get(cam)
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    try:
                        self.frame_queue.put_nowait((t, cam, img.copy()))
                    except queue.Full:
                        self.errors["frame_queue_full"] = self.errors.get("frame_queue_full", 0) + 1

    def on_send(self, t: float, t_wall: float, action: dict[str, Any], sent: dict[str, Any]) -> None:
        """Record one control tick."""
        policy = np.asarray([float(action[f"{j}.pos"]) for j in JOINTS], dtype=np.float64)
        out = np.asarray([float(sent[f"{j}.pos"]) for j in JOINTS], dtype=np.float64)
        with self.lock:
            state = self.last_state if self.last_state is not None else np.full(len(JOINTS), np.nan)
            self.ticks["t"].append(t)
            self.ticks["t_wall"].append(t_wall)
            self.ticks["state"].append(state)
            self.ticks["action_policy"].append(policy)
            self.ticks["action_sent"].append(out)
            self.ticks["clipped"].append(np.abs(policy - out) > CLIP_EPS)
            self.ticks["phase"].append(self.phase)

    def on_chunk(self, t_start: float, duration: float, chunk: Any, kwargs: dict[str, Any]) -> None:
        """Record one policy inference call."""
        arr = chunk.detach().float().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        delay = kwargs.get("inference_delay")
        with self.lock:
            self.chunks["t_start"].append(t_start)
            self.chunks["duration"].append(duration)
            self.chunks["chunk_norm"].append(arr)
            self.chunks["inference_delay"].append(-1 if delay is None else int(delay))

    def _frame_writer(self) -> None:
        from PIL import Image

        while True:
            item = self.frame_queue.get()
            if item is None:
                return
            t, cam, img = item
            try:
                name = f"{len(self.frames):05d}_{t:08.3f}s_{cam}.jpg"
                Image.fromarray(img).save(self.frames_dir / name, quality=85)
                self.frames.append({"file": f"frames/{name}", "t": round(t, 4), "camera": cam})
            except Exception as e:  # noqa: BLE001
                self.error("frame", e)

    # -- output --------------------------------------------------------------

    def save(self) -> None:
        """Write ticks.npz, chunks.npz and meta.json (idempotent)."""
        if self.saved:
            return
        self.saved = True
        try:
            self.frame_queue.put(None, timeout=1.0)
            self.frame_thread.join(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            self.error("frame_flush", e)
        with self.lock:
            ticks = {k: list(v) for k, v in self.ticks.items()}
            chunks = {k: list(v) for k, v in self.chunks.items()}
        n_j = len(JOINTS)
        np.savez_compressed(
            self.run_dir / "ticks.npz",
            t=np.asarray(ticks["t"], dtype=np.float64),
            t_wall=np.asarray(ticks["t_wall"], dtype=np.float64),
            state=np.asarray(ticks["state"], dtype=np.float64).reshape(-1, n_j),
            action_policy=np.asarray(ticks["action_policy"], dtype=np.float64).reshape(-1, n_j),
            action_sent=np.asarray(ticks["action_sent"], dtype=np.float64).reshape(-1, n_j),
            clipped=np.asarray(ticks["clipped"], dtype=bool).reshape(-1, n_j),
            phase=np.asarray(ticks["phase"], dtype=np.int8),
            joints=np.asarray(JOINTS),
        )
        chunk_norm = _stack_padded(chunks["chunk_norm"], n_j)
        out = {
            "t_start": np.asarray(chunks["t_start"], dtype=np.float64),
            "duration": np.asarray(chunks["duration"], dtype=np.float64),
            "chunk_norm": chunk_norm,
            "inference_delay": np.asarray(chunks["inference_delay"], dtype=np.int64),
        }
        if self.policy_path and len(chunk_norm):
            try:
                from plot_run import chunk_norm_to_arm

                out["chunk_arm"] = chunk_norm_to_arm(chunk_norm, self.policy_path)
            except Exception as e:  # noqa: BLE001
                self.error("chunk_arm", e)
        np.savez_compressed(self.run_dir / "chunks.npz", **out)
        self.meta.update(
            {
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "n_ticks": len(ticks["t"]),
                "n_ticks_in_loop": int(sum(p == 1 for p in ticks["phase"])),
                "n_chunks": len(chunks["t_start"]),
                "n_frames": len(self.frames),
                "frames": sorted(self.frames, key=lambda f: f["t"]),
                "recording_errors": self.errors,
            }
        )
        (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str) + "\n")
        print(
            f"\nrecorded {len(ticks['t'])} ticks, {len(chunks['t_start'])} chunks, {len(self.frames)} frames"
        )
        print(f"run dir: {self.run_dir}")
        print(f"plot:    uv run python steering/plot_run.py {self.run_dir}")


def _stack_padded(arrays: list[np.ndarray], n_j: int) -> np.ndarray:
    """(n, max_steps, n_j) float32, NaN-padded if chunk lengths differ."""
    if not arrays:
        return np.zeros((0, 0, n_j), dtype=np.float32)
    steps = max(a.shape[0] for a in arrays)
    dim = max(a.shape[1] for a in arrays)
    out = np.full((len(arrays), steps, dim), np.nan, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0], : a.shape[1]] = a
    return out


REC: Recorder | None = None


# ---------------------------------------------------------------------------
# Patch a: torque-safe connect
# ---------------------------------------------------------------------------

CONNECT_REPORT: dict[str, Any] = {}
ALLOW_TORQUE_BLIP = False  # set by --allow-torque-blip
SETTINGS_READ_ATTEMPTS = 3


def _settings_mismatches(robot: Any) -> list[str]:
    """Read-only comparison of the EEPROM settings configure() would write; [] when all match."""
    from lerobot.motors.feetech import OperatingMode

    bus, cfg = robot.bus, robot.config
    retry = cfg.num_read_retries
    if not bus.is_calibrated:
        return ["motor calibration differs from the calibration file"]
    expected: dict[str, dict[str, int]] = {
        "Operating_Mode": dict.fromkeys(bus.motors, OperatingMode.POSITION.value),
        "P_Coefficient": dict.fromkeys(bus.motors, cfg.position_p_coefficient),
        "I_Coefficient": dict.fromkeys(bus.motors, cfg.position_i_coefficient),
        "D_Coefficient": dict.fromkeys(bus.motors, cfg.position_d_coefficient),
        "Return_Delay_Time": dict.fromkeys(bus.motors, 0),
    }
    if "gripper" in bus.motors:
        expected["Max_Torque_Limit"] = {"gripper": 500}
        expected["Protection_Current"] = {"gripper": 250}
        expected["Overload_Torque"] = {"gripper": 25}
    bad = []
    for reg, want in expected.items():
        got = bus.sync_read(reg, list(want), normalize=False, num_retry=retry)
        bad += [f"{m}.{reg}={got[m]} (want {v})" for m, v in want.items() if int(got[m]) != int(v)]
    phase = bus.sync_read("Phase", normalize=False, num_retry=retry)
    bad += [
        f"{m}.Phase bit4 set ({p:#x})"
        for m, p in phase.items()
        if bus.motors[m].model == "sts3215" and p & 0x10
    ]
    return bad


def _patch_connect() -> None:
    from lerobot.robots.so_follower import so_follower as sof

    original_configure = sof.SOFollower.configure

    @functools.wraps(original_configure)
    def configure(self: Any) -> None:
        bad: list[str] = []
        for attempt in range(1, SETTINGS_READ_ATTEMPTS + 1):
            try:
                bad = _settings_mismatches(self)
                break
            except Exception as e:  # noqa: BLE001
                bad = [f"could not read the motor settings ({SETTINGS_READ_ATTEMPTS} attempts): {e!r}"]
                logger.warning("settings read attempt %d/%d failed: %r", attempt, SETTINGS_READ_ATTEMPTS, e)
        if bad:
            CONNECT_REPORT.update({"path": "refused", "mismatches": bad})
            listing = "\n  ".join(bad)
            if not ALLOW_TORQUE_BLIP:
                raise SystemExit(
                    "\nREFUSING TO CONNECT: the stock configure() would switch torque off (a raised arm "
                    f"drops). Motor settings differ from what configure() writes, or could not be read:\n  "
                    f"{listing}\nFix the settings (e.g. lerobot-calibrate / lerobot-teleoperate with the arm "
                    "supported), or rerun with --allow-torque-blip while supporting the arm."
                )
            CONNECT_REPORT["path"] = "fallback_configure"
            print(
                "\n"
                + "!" * 78
                + "\nWARNING: motor settings differ from what configure() writes:\n  "
                + listing
                + "\n--allow-torque-blip given: running the stock configure(): TORQUE WILL BLIP OFF. "
                + "SUPPORT THE ARM NOW.\n"
                + "!" * 78
                + "\n",
                flush=True,
            )
            time.sleep(2.0)
            original_configure(self)
            return
        bus, retry = self.bus, self.config.num_read_retries
        torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=retry)
        already_on = all(int(v) == 1 for v in torque.values())
        present = None
        if not already_on:
            present = bus.sync_read("Present_Position", num_retry=retry)
            bus.sync_write("Goal_Position", present)  # holding target = where the arm is
        ram = {}
        for reg in ("Acceleration", "Maximum_Acceleration"):
            try:
                got = bus.sync_read(reg, normalize=False, num_retry=retry)
                for m, v in got.items():
                    if int(v) != 254:
                        bus.write(reg, m, 254)
                        ram[f"{m}.{reg}"] = int(v)
            except Exception as e:  # noqa: BLE001
                ram[reg] = f"not set: {e!r}"
        bus.enable_torque()
        CONNECT_REPORT.update(
            {
                "path": "torque_safe",
                "torque_already_on": already_on,
                "seeded_goal": None
                if present is None
                else {m: round(float(v), 3) for m, v in present.items()},
                "ram_rewritten_from": ram,
            }
        )
        how = "torque already on, goal kept" if already_on else "goal := present"
        print(f"torque-safe connect: settings match, {how}, torque enabled (no blip)", flush=True)

    def calibrate(self: Any) -> None:
        raise RuntimeError(
            f"{self}: motor calibration differs from {self.calibration_fpath}. steering/rollout.py does not "
            "calibrate (it would prompt and may switch torque off): run lerobot-calibrate first."
        )

    sof.SOFollower.configure = configure  # type: ignore[method-assign]
    sof.SOFollower.calibrate = calibrate  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Patch b: recording hooks
# ---------------------------------------------------------------------------


def _patch_recording(rollout_module: Any) -> None:
    from lerobot.robots.so_follower import so_follower as sof

    original_obs = sof.SOFollower.get_observation
    original_send = sof.SOFollower.send_action

    @functools.wraps(original_obs)
    def get_observation(self: Any) -> Any:
        obs = original_obs(self)
        if REC is not None:
            try:
                REC.on_observation(obs, self.cameras)
            except Exception as e:  # noqa: BLE001
                REC.error("observation", e)
        return obs

    @functools.wraps(original_send)
    def send_action(self: Any, action: Any) -> Any:
        t = REC.now() if REC is not None else 0.0
        t_wall = time.time()
        sent = original_send(self, action)
        if REC is not None:
            try:
                REC.on_send(t, t_wall, action, sent)
            except Exception as e:  # noqa: BLE001
                REC.error("tick", e)
        return sent

    sof.SOFollower.get_observation = get_observation  # type: ignore[method-assign]
    sof.SOFollower.send_action = send_action  # type: ignore[method-assign]

    original_build = rollout_module.build_rollout_context

    @functools.wraps(original_build)
    def build_rollout_context(cfg: Any, shutdown_event: Any) -> Any:
        ctx = original_build(cfg, shutdown_event)
        try:
            _wrap_policy(ctx.policy.policy)
        except Exception as e:  # noqa: BLE001
            if REC is not None:
                REC.error("context", e)
        rec = REC
        if rec is not None:
            try:
                rec.policy_path = str(cfg.policy.pretrained_path) if cfg.policy else None
            except Exception as e:  # noqa: BLE001
                rec.error("meta.policy_path", e)
            rec.meta["connect"] = CONNECT_REPORT
            robot_cfg = getattr(cfg, "robot", None)
            fields: dict[str, Any] = {
                "task": lambda: cfg.dataset.single_task if cfg.dataset else cfg.task,
                "fps": lambda: cfg.fps,
                "duration_cfg": lambda: cfg.duration,
                "policy_path": lambda: rec.policy_path,
                "inference_type": lambda: getattr(cfg.inference, "type", None),
                "strategy_type": lambda: getattr(cfg.strategy, "type", None),
                "return_to_initial_position": lambda: cfg.return_to_initial_position,
                "initial_position": lambda: ctx.hardware.initial_position,
            }
            robot_fields: dict[str, Any] = {
                "type": lambda: robot_cfg.type,
                "port": lambda: getattr(robot_cfg, "port", None),
                "id": lambda: robot_cfg.id,
                "max_relative_target": lambda: getattr(robot_cfg, "max_relative_target", None),
                "disable_torque_on_disconnect": lambda: getattr(
                    robot_cfg, "disable_torque_on_disconnect", None
                ),
                "cameras": lambda: {k: repr(v) for k, v in (getattr(robot_cfg, "cameras", {}) or {}).items()},
            }
            for key, get in fields.items():
                try:
                    rec.meta[key] = get()
                except Exception as e:  # noqa: BLE001
                    rec.error(f"meta.{key}", e)
            robot_meta: dict[str, Any] = {}
            for key, get in robot_fields.items():
                try:
                    robot_meta[key] = get()
                except Exception as e:  # noqa: BLE001
                    rec.error(f"meta.robot.{key}", e)
            rec.meta["robot"] = robot_meta
        return ctx

    rollout_module.build_rollout_context = build_rollout_context

    original_create = rollout_module.create_strategy

    @functools.wraps(original_create)
    def create_strategy(strategy_cfg: Any) -> Any:
        strategy = original_create(strategy_cfg)
        run, teardown = strategy.run, strategy.teardown

        def run_wrapped(ctx: Any) -> Any:
            _set_phase(1)
            try:
                return run(ctx)
            finally:
                _set_phase(2)

        def teardown_wrapped(ctx: Any) -> Any:
            _set_phase(2)
            return teardown(ctx)

        strategy.run = run_wrapped
        strategy.teardown = teardown_wrapped
        return strategy

    rollout_module.create_strategy = create_strategy


def _set_phase(phase: int) -> None:
    if REC is not None:
        REC.phase = phase


def _wrap_policy(policy: Any) -> None:
    """Instance-level wrap of predict_action_chunk: select_action (sync) and RTC both look it up on self."""
    original = policy.predict_action_chunk

    @functools.wraps(original)
    def predict_action_chunk(batch: Any, **kwargs: Any) -> Any:
        t_start = REC.now() if REC is not None else 0.0
        actions = original(batch, **kwargs)
        if REC is not None:
            try:
                REC.on_chunk(t_start, REC.now() - t_start, actions, kwargs)
            except Exception as e:  # noqa: BLE001
                REC.error("chunk", e)
        return actions

    policy.predict_action_chunk = predict_action_chunk


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Patch, record, run lerobot-rollout, save."""
    global REC
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__)
        print("--- lerobot-rollout options follow ---\n", flush=True)

    global ALLOW_TORQUE_BLIP
    tag, forwarded, flags = split_argv(argv)
    ALLOW_TORQUE_BLIP = flags["allow_torque_blip"]
    from lerobot.scripts import lerobot_rollout

    _patch_connect()
    if not any(a in ("-h", "--help") for a in argv):
        _patch_recording(lerobot_rollout)
        run_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{tag}"
        REC = Recorder(
            run_dir,
            {
                "tag": tag,
                "argv": sys.argv,
                "forwarded_argv": forwarded,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "t0_wall": time.time(),
                "frame_period_s": FRAME_PERIOD_S,
            },
        )
        atexit.register(REC.save)
        print(f"recording to {run_dir}", flush=True)

    sys.argv = [sys.argv[0], *forwarded]
    try:
        lerobot_rollout.main()
    finally:
        if REC is not None:
            saved = False
            try:
                REC.save()
                saved = True
            except Exception as e:  # noqa: BLE001
                print(f"recording: could not save ({e!r})", file=sys.stderr)
            if saved and not flags["no_plot"]:
                _plot(REC.run_dir)


def _plot(run_dir: Path) -> None:
    """Run plot_run on the run dir; never raises."""
    try:
        if not (run_dir / "ticks.npz").is_file():
            return
        from plot_run import plot_run

        plot_run(run_dir)
        print(f"plots:   {run_dir / 'plots'}", flush=True)
    except (Exception, SystemExit) as e:  # noqa: BLE001  (plot_run raises SystemExit on an empty run)
        print(f"plotting failed ({e!r}); run: uv run python steering/plot_run.py {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
