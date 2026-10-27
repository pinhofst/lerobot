Status on 3 Oct 2026: MolmoAct2 runs zero-shot on the SO-101 from one wrist camera, at 29.9 Hz with RTC.
It reaches pens reliably but rarely grasps them. With two pens, the named colour won 3 of 3 runs.
Offline, on the same frames, the colour word turns the arm towards the named pen in 6 of 8 scenes,
but the pre-registered verdict is still "unclear" (RUNLOG). The checkpoint fits a 16 GB laptop GPU
only with the workarounds below.

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
| `fast_load.py` | Loads the local bf16 checkpoint in ~25 s instead of ~110 s (used by `rollout.py`). |
| `goto_pose.py`, `poses/` | Saves a joint pose and moves the arm to it, without dropping torque. `molmo_median.json` is the start pose, `home.json` the rest pose. |
| `capture_frame.py` | Saves camera frames and the joint state, with the range check. Never touches torque. |
| `rollout.py` | Wraps `lerobot-rollout` for every policy run: torque-safe connect, required per-step cap, GPU check, recording to `results/runs/`, plots and a replay video. |
| `plot_run.py`, `plot_chunks3d.py`, `runs_table.py` | Per-run plots, a 3-D view of the chunks (forward kinematics), and the table of runs (`results/runs.csv`). |
| `so101_fk.py`, `assets/so101/` | SO-101 forward kinematics from the pinned URDF. |
| `camera_count_test.py` | Offline: one view vs two views on Ai2's frame. |
| `prompt_counterfactual.py` | Offline: frames from recorded runs, re-run with each prompt and the same seeds (`results/prompt_counterfactual/`). |
| `label_provisional.py`, `plan_consistency.py` | Provisional machine labels for the runs, and the RTC plan-consistency analysis (`results/ANALYSIS_2026-10-02.md`). |
| `hover_causes.py` | Offline: why consecutive chunks disagree while hovering (noise vs observation), and the remedies (`results/hover_causes/`). |
| `results_page.py` | Builds the results dashboard `results/results_page.html` (published at https://claude.ai/artifact/DgeXSsn5hB4fpZUqVDK3TJ). Re-run it, then republish, when results change. |
| `CHUNK_CONSISTENCY_NOTES.md` | Literature and LeRobot code notes on chunk disagreement and RTC. |
| `remote_api.py` | Client and access test for the hosted MolmoAct2 API on cgpa0 (two cameras, fp32; COMMANDS "Hosted MolmoAct2 API"). Endpoint and certificate in git-ignored `remote/`. |
| `sync_runs.py` | Uploads `results/runs/`, `server_runs/` and `analyses/` to the private dataset `pinhofst10d/so-101-steering` (`--dry-run`, `--download` on a new machine). |
| `steer_hooks.py`, `steer_patch.py`, `steer_barrier.py` | Activation steering for `rollout.py` (`--steer-vector`, `--steer-barrier`; see "Steering modes"). Standalone ports of the steervla hooks and the trained barrier. |
| `vectors/`, `barriers/` | Steering files with JSON sidecars (each README says what they were fitted on). Primary: `*_so101` (calibrated offline on the SO-101 1cam checkpoint); secondary: the **MolmoAct2-LIBERO** files. |
| `test_steer_hooks.py`, `test_steer_barrier.py` | CPU tests of the steering code (toy policy, no checkpoint). |
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
| 16 Oct | Recordings and new results go to the private Hugging Face dataset `pinhofst10d/so-101-steering` (organisation members only), uploaded with `sync_runs.py`; git keeps code and docs. Supersedes the 30 Sep "no Hugging Face account" entry. Results already tracked in `results/` stay | Run folders existed only on this laptop; the fork is public and the wrist videos show the lab; an organisation is the only free way to share a private repo |
| 3 Oct | The colleague's interface uses the same LeRobot RTC as `lerobot-rollout`, computed server-side (his protocol carries no previous chunk) | Same chunking on both paths, so runs stay comparable |
| 3 Oct | Deploy through the colleague's interface (his client, dashboard and console, with our MolmoAct2 server), keeping our core features: torque-safe connect, a required per-step cap, the median start pose, per-tick and per-chunk recording in `rollout.py`'s format, and the automatic plots and replay | His interface is easier to deploy; our features are what make runs safe and analysable |
| 2 Oct | All policy runs on the arm go through `steering/rollout.py`: torque-safe connect (refuses rather than dropping the arm), a required per-step cap, every run recorded to `results/runs/` and plotted. Run folders are not committed by default; share one with `git add -f` | Plain `lerobot-rollout` drops a raised arm at connect and records nothing |
| 2 Oct | Start rollouts from the checkpoint's median training pose (`poses/molmo_median.json`: pan 3°, lift −33°, elbow 34°, wrist_flex 58°), moved there by software (`goto_pose.py`) | From the folded rest pose the state is clipped, and the first rollout never left rest |
| 2 Oct | One wrist camera for now, with the one-view checkpoint copy (`checkpoints/…-1cam`); add a fixed third-person camera before the pen pilot | It is the only camera. Wrist-only is about 1.7% of the SO training episodes (paper), and the view used changes behaviour (offline test) |
| 2 Oct | Use the colleague's calibration (`so101_follower` / `so101_leader`); verified against the motors and by teleop | Same arms; LeRobot 0.6.1 uses the convention MolmoAct2 expects |
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
- [ ] The written video-labelling rule for "first sustained contact". Draft, to approve:
  1. Label from the overhead camera (cam1), with the side camera for occlusions, without the
     instruction visible (episodes shuffled, file names hidden).
  2. *Contact* = a gripper jaw touches a pen, and the pen visibly moves or the jaws close on it.
     Brushing past without moving it does not count.
  3. *Sustained* = the contact lasts at least 10 frames (0.33 s) or ends in a grasp.
  4. The first sustained contact decides the outcome: compliant (the named pen), violating (the
     other pen), or null if there is none within the 20 s episode. Later contacts are ignored.
  5. Record the frame index of that first contact (for the end-effector position) and a
     confidence (sure / unsure). Unsure episodes get a second labeller.
- [ ] A fixed start pose for the counterbalanced two-pen block (the median training pose is used so far).
- [x] **Offline prompt comparison on recorded runs** (done 2 Oct, `prompt_counterfactual.py`; red/green
  pens rather than blue/red): the colour word steers in 6 of 8 scenes; the pre-registered verdict is "unclear".

## Steering modes

`rollout.py` can edit the VLM's hidden states at one decoder block in the prompt pass of every policy call
(`steer_hooks.py`, `steer_patch.py`). There are two modes. `--steer-vector` and `--steer-barrier` are mutually
exclusive: giving both exits with an error.

| | Constant vector (`--steer-vector`) | Barrier (`--steer-barrier`) |
|---|---|---|
| What is added | The same fixed vector, alpha·v, at every chosen token | u = k·ReLU(τ − ⟨a,h⟩)·a: k times the smallest edit that brings ⟨a,h⟩ up to τ. A small trained network reads the layer's states and gives the direction a (unit) and the threshold τ per token |
| When it acts | Always, on every call | In principle only where ⟨a,h⟩ < τ (zero edit elsewhere) |
| Strength | `--steer-k` (alpha = k × the sidecar's `alpha_unit_value`) or `--steer-alpha` (absolute) | `--steer-k` scales the edit (ControlLLM's form), continuously from 0. k = 0 is no edit; k = 1 is the trained module |
| File | `vectors/<name>.npy` + `.json` sidecar | `barriers/<name>.pt` (module with its frozen description bank) + `.json` sidecar |
| Files | Primary: `const_so101` (calibrated offline on SO-101 real frames, 1cam, pens; k = 1). Secondary: `const_r3_L06` (trained), `word_probe_L09`, `green_meandiff_L06` (hand-fitted); all MolmoAct2-LIBERO | Two versions to compare on the robot, both targeting **red**, k = 1:<br>**calibrated (SO-101 real frames)**: `barrier_so101.pt`<br>**uncalibrated (LIBERO sim only)**: `barrier_r3_L06.pt` |

- The constant vector covers both the hand-fitted vectors and the trained constant vector. It is simple, predictable
  and always on.
- The barrier is the AdaBarrier method (ControlLLM / barriersteer), adapted to MolmoAct2 (`steer_barrier.py`). It reads
  only the instruction tokens (`positions = task`, the " pick up the pen" span located with the policy's
  tokenizer; it fails loudly if the span is not found). The run's `--task` is checked when the policy loads, so an
  empty task (the null condition) is refused before the run starts.
- `--steer-k 0` with a barrier adds an exact zero (bitwise the unsteered policy), but it still locates the span and
  counts the calls. The edit norm is k times the trained one (SO-101 smoke test below).
- steervla's `steering_wrist` sim k sweep used the other form, k scaling the threshold (u = ReLU(k·τ − ⟨a,h⟩)·a). Its
  chosen k = 0.75 therefore does not carry over numerically to this `--steer-k`. At k = 1 the two forms are the same,
  so the calibrated k = 1 holds as is.
- meta.json `steering` records the file and its sha256, k, layer, positions, the decoded task span, n_calls and
  n_fires (must be equal), and, for the barrier, the mean zero-edit fraction and the mean edit norm. If the policy is
  not the one the file was trained on, it prints a warning and records `policy_mismatch: true`.

### Calibration (27 Oct; offline proxy, no real-robot runs yet)

`barrier_so101` and `const_so101` start from the LIBERO round-3 modules and were calibrated **offline** in steervla
(`runs/steering_so101`). The checkpoint was `lerobot/MolmoAct2-SO100_101-LeRobot`, bf16, one-view copy (wrist
`cam0`). The data were recorded real SO-101 frames with two pens. The target was velocity matching to the same frozen
policy prompted "pick up the red pen", from the students "pick up the pen" and "pick up the green pen". On held-out
real frames:
- The calibrated barrier reaches 12% of the no-edit loss at k = 1, and beats the calibrated constant (barrier /
  constant loss ratio 0.65).
- k = 1 was the pre-registered choice for both modules.

The two barrier versions on the 8 held-out real runs (20 two-pen frames). Both target **RED**. "Gap" is the gap to the
red-pen teacher, with unsteered = 100%. Heading counts the frames where the predicted motion heads the same way as
the teacher's; the teacher moves +7.5 mm toward red.

| barrier | gap | heading matches red, neutral prompt | heading matches red, "green pen" prompt |
|---|---|---|---|
| none | 100% | 9/20 | 1/20 |
| uncalibrated (LIBERO sim only), k = 1 | 47% | 17/20 (+3.5 mm toward red) | 18/20 (+9.4 mm) |
| calibrated (SO-101 real frames), k = 1 | 12% | 20/20 (+7.6 mm) | 20/20 (+7.5 mm) |

Deploy-path smoke test of the calibrated barrier (GPU, 27 Oct). Setup:
- `steer_patch` + `steer_hooks` + `steer_barrier`, with the 1cam checkpoint loaded as `prompt_counterfactual` does,
  CUDA graph on;
- one recorded frame (`20261002-170142_median_rtc_cap6_no_color_prompt_2_pens_1`, frame 00003) with
  "pick up the pen";
- seeds 0-7 of `predict_action_chunk`.

Results:
- The span was " pick up the pen" (4 tokens), with n_fires == n_calls and no mismatch warning.
- k = 0 was bitwise the unsteered chunk.
- The mean edit norm was 17.7 at k = 0.5 and 35.4 at k = 1.
- At k = 1 the chunk moved by a mean of 9.5° per joint and step from the unsteered one (shoulder_lift 27°); at
  k = 0.5 it moved 9.0°.
- At k = 1 the output was bitwise steervla `steering_so101`'s own hook on the same frame and noise.
- The uncalibrated barrier (`barrier_r3_L06`), same frame and noise:
  - the span was found, n_fires == n_calls, k = 0 was bitwise;
  - the mismatch warning fired and meta.json recorded `policy_mismatch: true`;
  - the mean edit norm was 12.4 at k = 0.5 and 24.9 at k = 1;
  - the chunk moved by a mean of 5.9° from the unsteered one at k = 1 (shoulder_lift 14.7°) and 1.5° at k = 0.5;
  - the output was bitwise steervla's hook with `barrier_r3.pt`.

These are offline proxies: loss and heading on recorded frames. They say nothing yet about grasps, closed-loop
drift, markers or two cameras. The barrier still never switches off (zero-edit fraction 0, as in LIBERO), so it also
edits under the red prompt.

Status (27 Oct):
- In LIBERO sim, the trained barrier and the trained constant vector performed the same in every setting.
- On SO-101 (offline, above), the calibrated barrier beats the calibrated constant.
- So far the barrier never switches off in practice, in LIBERO and on SO-101 frames.

```bash
# barrier (primary): k = 0 baseline first, then 0.5, then 1 (the calibrated k), same scene and pose
uv run python steering/rollout.py --tag bar_k0 ... --steer-barrier steering/barriers/barrier_so101.pt --steer-k 0
# constant vector (calibrated, k = 1)
uv run python steering/rollout.py --tag const_k1 ... --steer-vector steering/vectors/const_so101.npy --steer-k 1
```

Full commands: COMMANDS "Steered rollouts".

## Status

| Step | State | Where |
|---|---|---|
| 0 Fork, branch, pin | Done: branch `steering`, tag `steering-base` = `d8a09caa` | — |
| 1 Machine check | Done: RTX 5080 Laptop 16.3 GB, driver 580, cu128 wheels, Python 3.13 | RUNLOG |
| 2 Install, ports, calibrate, teleop, servos | Done 2 Oct (the colleague's calibration; teleop OK) | COMMANDS §2 |
| 2b Camera rig | One wrist camera, locked (20 ms exposure); third-person camera still to add | COMMANDS §2b |
| 3 Latency and VRAM | Done: 361 ms/chunk, 12.2 GiB peak | RUNLOG |
| 4 First rollout | Done 2 Oct: 20 recorded runs from the median pose with RTC. Reaches reliably, grasps rarely (2 of 12 single-pen runs) | RUNLOG, `results/ANALYSIS_2026-10-02.md` |
| 5 Pilot | 8 two-pen runs (named colour 3 of 3, n tiny) and the offline prompt comparison (verdict "unclear"). Next: a counterbalanced block, blind human labels | RUNLOG, `results/prompt_counterfactual/SUMMARY.md` |
| 6 Report | Ongoing in RUNLOG | RUNLOG |

## Open items

- [ ] **Single side-view camera** (16 Oct): record wrist and side together in the same counterbalanced scenes,
  then compare wrist-only, side-only and wrist+side offline (colour effect on pan, seeds moving under the empty
  prompt) before side-only rollouts. A low side view may also show the gripper-to-pen height behind the empty closes.
- [ ] **MolmoAct2 behind the colleague's dashboard** (3 Oct). The server and wrappers are in
  `~/Documents/so101_smolvla/so101_smolvla/molmoact2/` (his repo, new folder only, not committed; see
  its README). They are tested in dry-run without the arm, and the server records to `results/server_runs/`
  (git-ignored). Per-tick client recording in `rollout.py`'s format, a required step cap, and keep-torque on
  exit are in place. Server-side LeRobot RTC is on by default (h = 10; `--rtc-execution-horizon 20` for the A/B),
  bit-exact against LeRobot's own RTC call. Everything is dry-run tested only; nothing has run on the arm.
- [ ] Decide the camera pose once the rig is up (compare both poses on the frame diagnostic).
- [ ] If `lerobot-find-port` misbehaves: `sudo apt remove brltty`.
