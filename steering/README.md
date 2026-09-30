# steering/ — MolmoAct2 on the SO-101

Status on 30 Sep 2026: the software side of steps 0–3 is done and measured. Steps 2, 2b, 4 and 5 are
waiting on the arm. The checkpoint fits a 16 GB laptop GPU, at 361 ms per 30-action chunk, but only
with the workarounds below.

Shareable summary for colleagues: https://claude.ai/artifact/En5od24WhQKE6LD37nTahs (private until shared from its Share menu; generated from this file and RUNLOG, republished when they change).

This folder holds all experiment code. `src/` is unchanged, so rebasing on upstream stays cheap.

| File | What it is |
|---|---|
| `README.md` | This page: setup, decisions, status. Start here. |
| `RUNLOG.md` | Measurements and every deviation from the plan. Nothing in it is tuned. |
| `COMMANDS.md` | Copy-paste commands for the arm steps (ports, calibration, cameras, rollout, pilot). |
| `molmo_common.py` | Loads the checkpoint on current `main` (config translation, streamed weights). |
| `bench_latency.py` | Step 3: latency and VRAM per chunk, no robot. |
| `frame_diagnostic.py` | Step 5 precursor: do predictions separate by instruction on a fixed frame? |
| `servo_check.py` | Step 2: dead or jittery joints from a short teleop recording. |
| `lock_camera.sh` | Step 2b: lock exposure, white balance, gain and focus with `v4l2-ctl`. |
| `make_local_checkpoint.py` | Builds the bf16 checkpoint that `lerobot-rollout` can load on 16 GB. |
| `verify_local_checkpoint.py` | Checks that checkpoint against the streamed load (identical actions). |
| `results/` | Raw JSON behind every number in `RUNLOG.md`. |

## Setup

Linux, NVIDIA GPU with at least 16 GB (see RUNLOG for the margins), Python ≥ 3.12,
[uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:pinhofst/lerobot.git && cd lerobot
git switch steering
git remote add upstream https://github.com/huggingface/lerobot.git   # base: tag steering-base = d8a09caa
uv sync --locked --extra molmoact2 --extra core_scripts --extra feetech --extra async
sudo apt install ffmpeg v4l-utils          # torchcodec video decoding; camera locking

uv run python steering/make_local_checkpoint.py   # first run downloads ~44 GB, writes a 12 GB checkpoint
uv run python steering/bench_latency.py --cuda-graph on
```

- Run everything through `uv run`. A separately `pip install`-ed `lerobot` shadows the fork's
  commands without any warning.
- `uv sync` removes extras you leave out, so always pass all four.
- No Hugging Face account is needed: the models are public, and every recording command in `COMMANDS.md` keeps datasets on disk.
- Loading needs about 24 GB of free host RAM.

## Decisions

Newest first. "Plan" is the plan artifact the three of us work from.

| Date | Decision | Why |
|---|---|---|
| 30 Sep | Pen-pilot frames start with the arm **raised over the middle of the workspace**, not at rest | From rest, the first 30 actions are a generic lift under every instruction, so a rest-pose frame cannot show which pen the model targets (RUNLOG, frame diagnostic) |
| 30 Sep | No Hugging Face account; datasets stay local (`--dataset.push_to_hub=false`, repo id `local/<name>`, stored under `~/.cache/huggingface/lerobot/local/`) | Both checkpoints are public and download without an account. An account only helps to move a dataset to a rented GPU through the Hub (plan 1b-C), and `rsync` does that too |
| 30 Sep | Upstream bugs (config load, 16 GB OOM) are **not** reported for now; keep the workaround in `steering/` | Keeps `src/` identical to upstream |
| 30 Sep | Camera pose stays open until the rig is built | The one published pose (low horizontal + overhead) differs from the plan's front camera at 20–30° (RUNLOG deviation 5) |
| 29 Sep | Run inference in bf16 with CUDA graphs on | 361 ms vs 445 ms per chunk for the same memory; fp32 would not fit |
| 29 Sep | Load the checkpoint the way a Hub id loads it (no continuous-training mask) | Matches Ai2's native inference; the local-path mask changes actions by up to 3.7° |
| 29 Sep | Go ahead on a 16.3 GB laptop GPU despite the plan's "< 16 GB → stop" gate | Ai2 report bf16 under 16 GB; measured peak is 12.2 GiB allocated |
| 29 Sep | Uninstall the PyPI `lerobot` from the conda base env | It shadowed the fork's CLI entry points |

## Protocol (plan Appendices E and G)

Adopted as written:
- **Grounding is scored separately from task success.** Grounding has three outcomes at the first
  sustained gripper contact: *compliant* (the object satisfies the constraint), *violating* (an
  excluded object), and *null* (no sustained contact). Task is success or failure. Report grounding
  compliance = compliant / (compliant + violating), null rate = null / N, and task success = success / N.
- **Log the end-effector position at first contact**, not just the category. That lets the
  instruction conditions be plotted as spatial distributions.
- **Demonstrations randomise which pen the operator picks**, and the choice is recorded per episode
  (a colour or position habit would otherwise be learned and inherited by the experiment).
- **Camera controls locked, daylight excluded, a fixed marker in frame**, checked at the start of
  every session.
- **Camera pose chosen by measurement**: several rig configurations × the same scene × the three
  instructions on recorded frames, keeping the configuration with the most language-driven
  separation. Losing configurations are recorded too.
- CAG (counterfactual action guidance) goes into related work as the closest inference-time competitor.

Proposed changes (need your OK):
1. **Make the pass condition directional.** Appendix E reads "colours close to each other, far from
   null" as "language does nothing". On Ai2's frame that is exactly the pattern, yet language
   triples the chance of moving. It gates motion without choosing a target. Pass condition instead:
   among moving seeds, the blue-vs-red shoulder_pan difference has a 95% CI excluding 0, and its sign
   flips when the pens swap sides (`frame_diagnostic.py`, `mover_direction`).
2. **Pose-selection guard rails.**
   - Keep the arm pose (raised, see Decisions) and the scene identical across camera configurations.
   - Use two counterbalanced pen layouts.
   - Include Ai2's table-height + overhead pair as one of the configurations.
   - Confirm the winner on a fresh layout. Taking the best of several noisy measurements otherwise
     rewards luck.
3. **Keep capture at 640×480.** The model squashes every frame to 378×378 (RUNLOG, model input
   format), so a 16:9 camera would be distorted differently from the 4:3 training rigs.

Still to decide, before the first pilot rollout:
- [ ] **What counts as "sustained contact"**, fixed in advance. Proposal: the gripper is commanded
  closed, and its measured position stays at least N units above its empty-closed value for ≥ 10
  frames (0.33 s at 30 fps). Contacts without a grasp are labelled from the overhead video, by a
  rule written down beforehand.
- [ ] **How to get the end-effector position.** Proposal: record joint states (already in every
  dataset) and compute forward kinematics offline (SO-101 URDF, LeRobot's `kinematics` extra), so
  nothing is added to the control loop.
- [ ] **Whether CAG becomes a baseline, not only related work.** It needs two forward passes
  mixed at every flow step inside the action expert: about 2× model time (≈ 630 ms per chunk here)
  and a change to the sampling loop.

## Status

| Step | State | Where |
|---|---|---|
| 0 Fork, branch, pin | Done: branch `steering`, tag `steering-base` = `d8a09caa` | — |
| 1 Machine check | Done: RTX 5080 Laptop 16.3 GB, driver 580, cu128 wheels, Python 3.13 | RUNLOG |
| 2 Install, ports, calibrate, teleop, servos | Install done; arm steps pending | COMMANDS §2 |
| 2b Camera rig | Pending; lock script ready | COMMANDS §2b |
| 3 Latency and VRAM | Done: 361 ms/chunk, 12.2 GiB peak | RUNLOG |
| 4 First rollout | Pending; checkpoint built and verified | COMMANDS §4 |
| 5 Pilot | Precursor run on Ai2's frame; pen frame and rollouts pending | RUNLOG, COMMANDS §5 |
| 6 Report | Ongoing in RUNLOG | RUNLOG |

## Open items

- [ ] Decide the camera pose once the rig is up (compare both poses on the frame diagnostic).
- [ ] If `lerobot-find-port` misbehaves: `sudo apt remove brltty`.
