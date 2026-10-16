"""Client for the hosted MolmoAct2 API on cgpa0 (SoC cluster), and a command line to test access.

The server (run by Haoyu, branch ``steering-cgpa0`` of the fork) serves the stock two-camera checkpoint
``lerobot/MolmoAct2-SO100_101-LeRobot`` in fp32 behind HTTPS:

- ``GET /health``: no token; up or not, GPU memory, queue length.
- ``GET /info``: token; checkpoint, expected inputs, limits, latency so far.
- ``POST /predict``: token; one observation (task, 6 joint readings in the so101_follower frame, cam0 and
  cam1 images) -> 30 x 6 absolute joint targets in the same frame, plus ``first_step_jump_deg``,
  ``max_step_deg``, ``clamped_entries`` and ``chunk_ms``.

Nothing site-specific or secret is in git:

- the token comes from ``$MOLMOACT2_TOKEN`` or ``~/.molmoact2_token`` (keep it mode 600) and is never printed;
- the URL and the certificate's SHA-256 fingerprint come from flags, ``$MOLMOACT2_URL`` /
  ``$MOLMOACT2_FINGERPRINT``, or ``steering/remote/config.json`` (git-ignored);
- the certificate is pinned to ``steering/remote/cgpa0.crt`` (git-ignored), saved once by ``fetch-cert``
  after its fingerprint matches.

    uv run python steering/remote_api.py fetch-cert   # once: save the server certificate, fingerprint checked
    uv run python steering/remote_api.py health
    uv run python steering/remote_api.py info
    uv run python steering/remote_api.py sample       # Ai2's sample frame, seed 29, vs our local bf16 chunk
    uv run python steering/remote_api.py frame steering/results/runs/<run> --wrist-as-both

From a laptop off the cluster network the server is only reachable through the SoC VPN or an SSH tunnel
(``ssh -N -L <port>:<server ip>:<port> <soc-id>@<login node>``; then ``--url https://127.0.0.1:<port>``, which the
certificate also lists).

The client refuses a chunk (``UnsafeChunk``) whose first action is more than 30 deg from the state sent or
whose consecutive steps move a joint more than 15 deg, as the server's own client does. It never drives
hardware; an arm-side per-step cap is still required.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

STEERING = Path(__file__).resolve().parent
REMOTE_DIR = STEERING / "remote"
CONFIG_FILE = REMOTE_DIR / "config.json"
DEFAULT_CERT = REMOTE_DIR / "cgpa0.crt"
TOKEN_FILE = Path("~/.molmoact2_token").expanduser()
RESULTS = STEERING / "results" / "remote_api"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = slice(0, 5)  # joints in degrees; the gripper is 0-100
MAX_FIRST_JUMP_DEG = 30.0
MAX_STEP_DEG = 15.0
N_ACTIONS = 30  # 1 s at 30 Hz
MAX_RETRY_WAIT_S = 30.0
CONNECT_TIMEOUT_S = 10.0
LOCAL_REFERENCE = STEERING / "results" / "latency_bfloat16_graph-on_seed29.npy"


class ApiError(RuntimeError):
    """The server answered with an error status; ``detail`` says what to fix."""

    def __init__(self, status: int, detail: str):
        """Keep the HTTP status and the server's detail."""
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class UnsafeChunk(RuntimeError):  # noqa: N818  (same name as the server bundle's api_client.py)
    """A chunk that must not be sent to an arm (first-action jump or per-step move too large)."""


@dataclass
class Prediction:
    """One /predict response, plus the round trip measured here."""

    actions: np.ndarray  # (30, 6), so101_follower frame: degrees for joints 0-4, gripper 0-100
    first_step_jump_deg: float | None
    max_step_deg: float | None
    clamped_entries: int | None
    chunk_ms: float | None
    round_trip_ms: float
    extra: dict[str, Any] = field(default_factory=dict)  # any other response fields


def _encode_image(image: np.ndarray | str | Path) -> str:
    """A file (sent as stored) or an HxWx3 uint8 RGB array (sent as lossless PNG), base64."""
    if isinstance(image, (str, Path)):
        return base64.b64encode(Path(image).read_bytes()).decode("ascii")
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 uint8 RGB, got {arr.dtype} {arr.shape}")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _clean_token(token: str | None) -> str | None:
    """One printable line, or refuse without echoing it (a stray newline would land in an error message)."""
    if not token:
        return None
    token = token.strip()
    if not token.isprintable() or any(c.isspace() for c in token):
        raise ValueError("the token must be a single line without spaces; check ~/.molmoact2_token")
    return token


def _retry_after_s(value: str | None) -> float:
    """Retry-After as seconds or an HTTP date, clamped to [1, MAX_RETRY_WAIT_S]."""
    if not value:
        return 1.0
    try:
        wait = float(value)
    except ValueError:
        try:
            wait = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            wait = 1.0
    return min(max(wait, 1.0), MAX_RETRY_WAIT_S)


def check_chunk(
    actions: np.ndarray,
    state: np.ndarray,
    max_first_jump_deg: float = MAX_FIRST_JUMP_DEG,
    max_step_deg: float = MAX_STEP_DEG,
) -> tuple[float, float]:
    """Raise UnsafeChunk unless the chunk is finite, 30 x 6 and within both limits (arm joints, degrees)."""
    if state.shape != (len(JOINTS),) or not np.all(np.isfinite(state)):
        raise UnsafeChunk(f"cannot check a chunk against a malformed state {state}")
    if actions.shape != (N_ACTIONS, len(JOINTS)) or not np.all(np.isfinite(actions)):
        raise UnsafeChunk(
            f"malformed chunk: shape {actions.shape}, finite {bool(np.all(np.isfinite(actions)))}"
        )
    first = float(np.max(np.abs(actions[0, ARM] - state[ARM])))
    step = float(np.max(np.abs(np.diff(actions[:, ARM], axis=0))))
    if first > max_first_jump_deg:
        raise UnsafeChunk(f"first action is {first:.1f} deg from the state (limit {max_first_jump_deg})")
    if step > max_step_deg:
        raise UnsafeChunk(f"a step moves a joint {step:.1f} deg (limit {max_step_deg})")
    return first, step


class MolmoAct2Client:
    """Thin client for the cgpa0 MolmoAct2 API: certificate-pinned HTTPS, bearer token, 503 retries."""

    def __init__(
        self,
        url: str,
        token: str | None,
        cert: str | Path,
        timeout_s: float = 30.0,
        retries: int = 3,
        max_first_jump_deg: float = MAX_FIRST_JUMP_DEG,
        max_step_deg: float = MAX_STEP_DEG,
    ):
        """Pin the certificate on a session; the token is kept only in memory and in the header."""
        if urlparse(url).scheme != "https":
            raise ValueError(f"refusing a non-https URL (the token would travel in plaintext): {url}")
        self.url = url.rstrip("/")
        self.cert = str(cert)
        self.timeout_s = timeout_s
        self.retries = retries
        self.max_first_jump_deg = max_first_jump_deg
        self.max_step_deg = max_step_deg
        self.session = requests.Session()
        self.session.verify = self.cert
        self.session.trust_env = (
            False  # $REQUESTS_CA_BUNDLE / $CURL_CA_BUNDLE / proxies must not replace the pin
        )
        self._token = _clean_token(token)

    def _request(self, method: str, route: str, auth: bool, body: dict | None = None) -> tuple[dict, float]:
        headers = {}
        if auth:
            if not self._token:
                raise ApiError(401, f"no token: set $MOLMOACT2_TOKEN or write it to {TOKEN_FILE} (chmod 600)")
            headers["Authorization"] = f"Bearer {self._token}"
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            resp = self.session.request(
                method,
                self.url + route,
                json=body,
                headers=headers,
                timeout=(CONNECT_TIMEOUT_S, self.timeout_s),
                verify=self.cert,
            )
            ms = (time.perf_counter() - t0) * 1e3
            if resp.status_code == 503 and attempt < self.retries:
                wait = _retry_after_s(resp.headers.get("Retry-After"))
                print(f"[remote] busy (503), retrying in {wait:.0f} s", file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                raise ApiError(resp.status_code, str(detail)[:500])
            return resp.json(), ms
        raise AssertionError("unreachable")

    def health(self) -> dict:
        """GET /health (no token)."""
        return self._request("GET", "/health", auth=False)[0]

    def info(self) -> dict:
        """GET /info (token)."""
        return self._request("GET", "/info", auth=True)[0]

    def predict(
        self,
        task: str,
        state: np.ndarray | list[float],
        images: dict[str, np.ndarray | str | Path],
        seed: int | None = None,
        num_steps: int | None = None,
        check: bool = True,
    ) -> Prediction:
        """One chunk for one observation. ``images`` must hold exactly cam0 and cam1."""
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (len(JOINTS),) or not np.all(np.isfinite(state)):
            raise ValueError(f"state must be {len(JOINTS)} finite numbers, got {state}")
        if set(images) != {"cam0", "cam1"}:
            raise ValueError(f"the server needs exactly cam0 and cam1, got {sorted(images)}")
        body: dict[str, Any] = {
            "task": task,
            "state": [float(x) for x in state],
            "images": {k: _encode_image(v) for k, v in images.items()},
        }
        if seed is not None:
            body["seed"] = int(seed)
        if num_steps is not None:
            body["num_steps"] = int(num_steps)
        out, ms = self._request("POST", "/predict", auth=True, body=body)
        actions = np.asarray(out["actions"], dtype=np.float64)
        if check:
            check_chunk(actions, state, self.max_first_jump_deg, self.max_step_deg)
        known = {"actions", "first_step_jump_deg", "max_step_deg", "clamped_entries", "chunk_ms"}
        return Prediction(
            actions=actions,
            first_step_jump_deg=out.get("first_step_jump_deg"),
            max_step_deg=out.get("max_step_deg"),
            clamped_entries=out.get("clamped_entries"),
            chunk_ms=out.get("chunk_ms"),
            round_trip_ms=ms,
            extra={k: v for k, v in out.items() if k not in known},
        )


# ----------------------------------------------------------------------------- settings
def load_settings(args: argparse.Namespace) -> dict[str, str | None]:
    """URL and fingerprint: flag, then environment, then steering/remote/config.json."""
    cfg: dict[str, Any] = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    url = args.url or os.environ.get("MOLMOACT2_URL") or cfg.get("url")
    fingerprint = args.fingerprint or os.environ.get("MOLMOACT2_FINGERPRINT") or cfg.get("fingerprint")
    if not url:
        sys.exit(f"no URL: pass --url, set $MOLMOACT2_URL or put {{'url': ...}} in {CONFIG_FILE}")
    return {"url": url, "fingerprint": fingerprint}


def load_token() -> str | None:
    """$MOLMOACT2_TOKEN, else ~/.molmoact2_token (warns if others can read it)."""
    token = os.environ.get("MOLMOACT2_TOKEN")
    if token:
        return token
    if TOKEN_FILE.exists():
        if TOKEN_FILE.stat().st_mode & 0o077:
            print(f"[remote] warning: {TOKEN_FILE} is readable by others; chmod 600 it", file=sys.stderr)
        return TOKEN_FILE.read_text()
    return None


def _norm_fp(fp: str) -> str:
    return fp.replace(":", "").replace(" ", "").upper()


def fetch_cert(url: str, fingerprint: str | None, out: Path) -> None:
    """Save the server's certificate if its SHA-256 matches the published fingerprint."""
    if not fingerprint:
        sys.exit(
            "fetch-cert needs the published fingerprint (--fingerprint, $MOLMOACT2_FINGERPRINT or config.json)"
        )
    u = urlparse(url)
    try:
        pem = ssl.get_server_certificate((u.hostname, u.port or 443), timeout=10)
    except OSError as e:  # timeout, refused, unreachable
        sys.exit(
            f"cannot reach {u.hostname}:{u.port} ({e}): SoC VPN, or the SSH tunnel and --url https://127.0.0.1:<port>"
        )
    got = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest().upper()
    if got != _norm_fp(fingerprint):
        sys.exit(
            f"fingerprint MISMATCH, certificate not saved:\n  expected {_norm_fp(fingerprint)}\n  got      {got}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pem)
    print(f"fingerprint matches; saved {out}")


# ----------------------------------------------------------------------------- test inputs
def sample_observation() -> tuple[str, np.ndarray, dict[str, Path]]:
    """Ai2's model-card sample (cam0 = top, cam1 = side), state in the so101_follower frame."""
    from huggingface_hub import hf_hub_download
    from molmo_common import BASE_REPO, SAMPLE_STATE_MODEL_FRAME, SAMPLE_TASK, load_config, model_to_arm_frame

    state = model_to_arm_frame(SAMPLE_STATE_MODEL_FRAME, load_config(device="cpu"))
    images = {
        "cam0": Path(hf_hub_download(BASE_REPO, "assets/sample_realsense_top_rgb.png")),
        "cam1": Path(hf_hub_download(BASE_REPO, "assets/sample_realsense_side_rgb.png")),
    }
    return SAMPLE_TASK, state.astype(np.float64), images


def run_observation(run_dir: Path, offset_s: float) -> tuple[str, np.ndarray, Path, float]:
    """A recorded wrist frame ``offset_s`` after the first policy tick, with the state of the next tick."""
    meta = json.loads((run_dir / "meta.json").read_text())
    ticks = np.load(run_dir / "ticks.npz")
    policy = np.flatnonzero(ticks["phase"] == 1)
    if not len(policy):
        sys.exit(f"{run_dir.name}: no policy ticks")
    target = float(ticks["t"][policy[0]]) + offset_s
    t_ticks = ticks["t"]

    def next_tick(t: float) -> int:
        return min(int(np.searchsorted(t_ticks, t)), len(t_ticks) - 1)

    # only frames captured by a tick's own observation (its tick follows within 10 ms), so the state is current
    frames = [
        f
        for f in meta["frames"]
        if f.get("camera", "cam0") == "cam0" and 0 <= t_ticks[next_tick(f["t"])] - f["t"] <= 0.010
    ]
    if not frames:
        sys.exit(f"{run_dir.name}: no frame with a tick right after it")
    frame = min(frames, key=lambda f: abs(f["t"] - target))
    i = next_tick(frame["t"])
    return str(meta["task"]), ticks["state"][i].astype(np.float64), run_dir / frame["file"], float(frame["t"])


def summarise(pred: Prediction, state: np.ndarray) -> str:
    """Timing, the server's safety numbers, and the first and last action against the state."""
    a = pred.actions
    lines = [
        f"chunk {a.shape}: server {pred.chunk_ms} ms, round trip {pred.round_trip_ms:.0f} ms",
        f"first_step_jump_deg {pred.first_step_jump_deg}, max_step_deg {pred.max_step_deg}, "
        f"clamped_entries {pred.clamped_entries}",
        "        " + " ".join(f"{j[:8]:>9}" for j in JOINTS),
        "state   " + " ".join(f"{v:9.2f}" for v in state),
        "a[0]    " + " ".join(f"{v:9.2f}" for v in a[0]),
        "a[29]   " + " ".join(f"{v:9.2f}" for v in a[-1]),
    ]
    if pred.extra:
        lines.append(f"other fields: {sorted(pred.extra)}")
    return "\n".join(lines)


def save(kind: str, record: dict[str, Any]) -> Path:
    """Write one result as JSON under results/remote_api/ (never the token)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}.json"
    path.write_text(
        json.dumps(record, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    )
    return path


def pred_record(pred: Prediction) -> dict[str, Any]:
    """A Prediction as plain JSON-able fields."""
    return {
        "actions": pred.actions,
        "first_step_jump_deg": pred.first_step_jump_deg,
        "max_step_deg": pred.max_step_deg,
        "clamped_entries": pred.clamped_entries,
        "chunk_ms": pred.chunk_ms,
        "round_trip_ms": round(pred.round_trip_ms, 1),
        "extra": pred.extra,
    }


# ----------------------------------------------------------------------------- commands
def cmd_sample(client: MolmoAct2Client, args: argparse.Namespace) -> None:
    """Ai2's sample twice at one seed (determinism), then against our local bf16 chunk at seed 29."""
    task, state, images = sample_observation()
    print(f"Ai2 sample frame, seed {args.seed}, state {np.round(state, 2).tolist()}")
    a = client.predict(task, state, images, seed=args.seed)
    print(summarise(a, state))
    b = client.predict(task, state, images, seed=args.seed)
    same = float(np.max(np.abs(a.actions - b.actions)))
    print(f"same seed twice: max|diff| = {same:.6f} (the server promises bit-identical)")
    record: dict[str, Any] = {"kind": "sample", "seed": args.seed, "state": state, "first": pred_record(a)}
    record["repeat_max_abs_diff"] = same
    if LOCAL_REFERENCE.exists() and args.seed == 29:
        ref = np.load(LOCAL_REFERENCE).astype(np.float64)
        d = np.abs(a.actions - ref)
        print(
            f"vs our local bf16 seed-29 chunk ({LOCAL_REFERENCE.name}): arm joints mean {d[:, ARM].mean():.3f} deg, "
            f"max {d[:, ARM].max():.3f} deg; gripper max {d[:, 5].max():.3f}"
        )
        print(
            "  (small = same noise and weights, fp32 vs bf16; several degrees = the noise is drawn differently)"
        )
        record["vs_local_bf16"] = {
            "mean_arm_deg": float(d[:, ARM].mean()),
            "max_arm_deg": float(d[:, ARM].max()),
        }
    if args.save:
        print(f"saved {save('sample', record)}")


def cmd_frame(client: MolmoAct2Client, args: argparse.Namespace) -> None:
    """One recorded wrist frame, sent as both cameras."""
    task, state, frame, t = run_observation(args.run_dir, args.offset)
    if not args.wrist_as_both:
        sys.exit(
            "the server needs cam0 (third-person) and cam1; our runs have only the wrist camera. "
            "Pass --wrist-as-both to send the wrist frame as both (offline, the same frame twice moved "
            "Ai2's chunk by 1.4 deg RMS against two real views: a test of access, not of the rig)."
        )
    print(f"{args.run_dir.name}: frame {frame.name} (t {t:.2f} s), task {task!r}")
    task = args.task or task
    pred = client.predict(task, state, {"cam0": frame, "cam1": frame}, seed=args.seed)
    print(summarise(pred, state))
    if args.save:
        rec = {
            "kind": "frame",
            "run": args.run_dir.name,
            "frame": frame.name,
            "task": task,
            "seed": args.seed,
        }
        rec.update(state=state, prediction=pred_record(pred), wrist_as_both=True)
        print(f"saved {save('frame', rec)}")


def main() -> None:
    """Command line: fetch-cert, health, info, sample, frame."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="server URL (default: $MOLMOACT2_URL or steering/remote/config.json)")
    ap.add_argument("--cert", type=Path, default=DEFAULT_CERT, help="pinned server certificate (PEM)")
    ap.add_argument("--fingerprint", help="published SHA-256 fingerprint, for fetch-cert")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout, s")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-cert", help="save the server certificate after checking its fingerprint")
    sub.add_parser("health", help="GET /health (no token)")
    sub.add_parser("info", help="GET /info (token)")
    sp = sub.add_parser(
        "sample", help="Ai2's sample frame: determinism and a comparison with our local chunk"
    )
    sp.add_argument("--seed", type=int, default=29)
    sp.add_argument("--save", action="store_true", help=f"write the result to {RESULTS}")
    fp = sub.add_parser("frame", help="one recorded frame from a steering/results/runs/ folder")
    fp.add_argument("run_dir", type=Path)
    fp.add_argument("--offset", type=float, default=0.0, help="seconds after the first policy tick")
    fp.add_argument("--wrist-as-both", action="store_true", help="send the wrist frame as cam0 and cam1")
    fp.add_argument("--task", help="override the run's task")
    fp.add_argument("--seed", type=int)
    fp.add_argument("--save", action="store_true", help=f"write the result to {RESULTS}")
    args = ap.parse_args()

    settings = load_settings(args)
    if args.cmd == "fetch-cert":
        fetch_cert(settings["url"], settings["fingerprint"], args.cert)
        return
    if not args.cert.exists():
        sys.exit(
            f"no certificate at {args.cert}: run `fetch-cert` first (or pass --cert cgpa0.crt from the bundle)"
        )
    try:
        client = MolmoAct2Client(settings["url"], load_token(), args.cert, timeout_s=args.timeout)
    except ValueError as e:  # non-https URL, malformed token (never echoed)
        sys.exit(str(e))
    try:
        if args.cmd == "health":
            print(f"health: {client.health()}")
        elif args.cmd == "info":
            print(json.dumps(client.info(), indent=1))
        elif args.cmd == "sample":
            print(f"health: {client.health()}")
            cmd_sample(client, args)
        elif args.cmd == "frame":
            cmd_frame(client, args)
    # most specific first: ConnectTimeout and SSLError are both ConnectionErrors
    except requests.exceptions.ConnectTimeout:
        sys.exit(
            "connection timed out: no network path (SoC VPN, or the SSH tunnel and --url https://127.0.0.1:<port>)"
        )
    except requests.exceptions.ReadTimeout:
        sys.exit(
            f"the server accepted the connection but sent no answer within {args.timeout:.0f} s (busy or stuck)"
        )
    except requests.exceptions.SSLError as e:
        sys.exit(f"certificate check failed ({e}); use the pinned certificate with a host name it lists")
    except requests.exceptions.ConnectionError as e:
        sys.exit(f"cannot connect ({type(e).__name__}): is the VPN up, or the SSH tunnel still running?")
    except (ApiError, UnsafeChunk) as e:
        sys.exit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
