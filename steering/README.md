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
| `check_state_range.py` | Is the arm's state inside the checkpoint's trained range? Outside it the model sees a clipped state and the first action jumps. Reads the robot without changing torque. |
| `lock_camera.sh` | Step 2b: lock exposure, white balance, gain and focus with `v4l2-ctl`. |
| `make_local_checkpoint.py` | Builds the bf16 checkpoint that `lerobot-rollout` can load on 16 GB. |
| `verify_local_checkpoint.py` | Checks that checkpoint against the streamed load (identical actions). |
| `results/` | Raw data behind the RUNLOG tables. A few one-off checks (the OOM test, the mask comparison, host RAM) are recorded only in RUNLOG. |

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
| 30 Sep | Lighting: lock cameras per session (`lock_camera.sh`, 10 ms exposure for 50 Hz mains), a quick look for banding, no flicker tooling | The policy is probably fairly robust to lighting; the risk is to our small blue-vs-red and activation differences. If lighting looks suspicious, test offline with synthetic brightness and white-balance shifts on a fixed frame |
| 30 Sep | CAG as a complementary offline test, final-action variant only (`frame_diagnostic.py --cag-weights`); mixing inside the flow sampler rejected | Cheap and needs no change to `src/`. The core of the project is steering, not CAG. First result: CAG amplifies reach, not target choice (RUNLOG) |
| 30 Sep | End-effector position by forward kinematics from joint states, live or offline (both cheap); one quick table-mark check that it is not badly wrong | Only needs to say which pen was contacted |
| 30 Sep | Contact and target labelled by a person from video only; fixed episode length 20 s, so a null outcome is well defined | Gripper current and position gaps are unreliable for thin pens. Write the labelling rule down before the first rollout, and label without knowing the instruction |
| 30 Sep | Frame tests inform the arm pilot, they do not gate it; directional readout (`mover_direction`) adopted | Arm time is available anyway. "Moves more often" is not "moves to the named pen" |
| 30 Sep | Camera-pose selection keeps few configurations | GPU time is cheap here; rebuilding the rig is the expensive part |
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

Amendments agreed on 30 Sep (see Decisions):
1. **Directional readout.** Among moving seeds, compare where blue and red go (`mover_direction`
   in `frame_diagnostic.py`). Appendix E's "colours close to each other, far from null" test misreads
   the bimodal case. The frame test informs the pilot and does not gate it.
2. **Pose selection, kept small.** Same raised arm pose and scene for each configuration, the pens
   swapped for a second layout, Ai2's pose as one candidate. Few configurations, because rebuilding
   the rig is what costs time.
3. **Capture at 640×480**, because the model squashes every frame to 378×378.
4. **Contact is labelled by a person from video**, with a rule written before the first rollout,
   ideally without knowing which instruction was given. Episodes last 20 s. No contact within
   20 s is null.
5. **End-effector position by forward kinematics** from the joint states, with a quick check against
   a few table marks to catch a badly wrong calibration.
6. **CAG as a complementary offline test** (final-action variant, `--cag-weights`).

Still open:
- [ ] The written video-labelling rule for "first sustained contact" (what counts, which frame).
- [ ] The raised start pose, picked on arm day with teleop.

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
