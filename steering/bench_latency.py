"""Step 3: per-chunk latency and peak VRAM for MolmoAct2-SO100_101 on Ai2's sample frame (no robot).

    uv run python steering/bench_latency.py --cuda-graph on
    uv run python steering/bench_latency.py --cuda-graph off

Run each setting in its own process so peak-memory counters are not shared.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from molmo_common import (
    REFERENCE_SEED,
    SAMPLE_TASK,
    load_config,
    load_policy,
    load_sample_observation,
    preprocess,
)

RESULTS = Path(__file__).parent / "results"


def gib(n_bytes: int) -> float:
    """Bytes to GiB, rounded."""
    return round(n_bytes / 2**30, 3)


def nvidia_smi_used_mib() -> int:
    """Device memory in use as nvidia-smi reports it (all processes, incl. display)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def timed(fn):
    """Run ``fn`` between CUDA syncs; return its result and the wall time in ms."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def main() -> None:
    """Measure load, warm-up and per-chunk latency and VRAM for one CUDA-graph setting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-graph", choices=["on", "off"], required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    device = "cuda"
    smi_before = nvidia_smi_used_mib()
    config = load_config(dtype=args.dtype, cuda_graph=args.cuda_graph == "on", device=device)
    t0 = time.perf_counter()
    policy, preprocessor, postprocessor = load_policy(config)
    load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    after_load = {
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()

    observation = load_sample_observation(config)

    def run(seed: int, *, model_only_batch=None):
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            batch = model_only_batch or preprocess(preprocessor, observation, SAMPLE_TASK, device)
            chunk = policy.predict_action_chunk(batch, generator=generator)
            return postprocessor(chunk) if model_only_batch is None else chunk

    warmup_ms = [timed(lambda s=s: run(1000 + s))[1] for s in range(args.warmup)]
    e2e_ms, model_ms = [], []
    chunk = None
    with torch.inference_mode():
        fixed_batch = preprocess(preprocessor, observation, SAMPLE_TASK, device)
    for i in range(args.iters):
        chunk, ms = timed(lambda i=i: run(i))
        e2e_ms.append(ms)
        model_ms.append(timed(lambda i=i: run(i, model_only_batch=fixed_batch))[1])
    smi_peak = nvidia_smi_used_mib()
    # Full chunk at a fixed seed, independent of --iters, for verify_local_checkpoint.py.
    reference = torch.as_tensor(run(REFERENCE_SEED)).squeeze(0).float().cpu().numpy()

    chunk = torch.as_tensor(chunk).squeeze(0).float().cpu()
    n_actions = int(chunk.shape[0])
    median_e2e = statistics.median(e2e_ms)
    result = {
        "setting": {"dtype": args.dtype, "cuda_graph": args.cuda_graph, "iters": args.iters},
        "machine": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "load_s": round(load_s, 1),
        "vram_after_load": after_load,
        "vram_peak_during_inference": {
            "max_allocated_gib": gib(torch.cuda.max_memory_allocated()),
            "max_reserved_gib": gib(torch.cuda.max_memory_reserved()),
            "nvidia_smi_used_mib": smi_peak,
            "nvidia_smi_used_before_load_mib": smi_before,
        },
        "warmup_ms": [round(x, 1) for x in warmup_ms],
        "e2e_ms": {
            "median": round(median_e2e, 1),
            "p90": round(sorted(e2e_ms)[int(0.9 * len(e2e_ms)) - 1], 1),
            "min": round(min(e2e_ms), 1),
            "max": round(max(e2e_ms), 1),
        },
        "model_only_ms_median": round(statistics.median(model_ms), 1),
        "chunk_actions": n_actions,
        "amortised_control_hz": round(n_actions / (median_e2e / 1000), 1),
        "state_arm_frame": [round(float(x), 2) for x in observation["observation.state"]],
        "first_action": [round(float(x), 2) for x in chunk[0]],
        "last_action": [round(float(x), 2) for x in chunk[-1]],
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"latency_{args.dtype}_graph-{args.cuda_graph}.json"
    out.write_text(json.dumps(result, indent=2))
    np.save(RESULTS / f"latency_{args.dtype}_graph-{args.cuda_graph}_seed{REFERENCE_SEED}.npy", reference)
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
