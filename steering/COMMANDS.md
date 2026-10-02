# Arm-side commands (steps 2, 2b, 4, 5)

No Hugging Face account is needed. The models are already cached, and every recording
command here passes `--dataset.push_to_hub=false` and uses a `local/<name>` repo id. Datasets land
in `~/.cache/huggingface/lerobot/local/`.

If `lerobot-find-port` finds nothing, or the port appears and then vanishes, run
`sudo apt remove brltty`. It grabs the USB-serial chips SO-101 boards use.

## This rig (2 Oct 2026)

| Device | Stable path |
|---|---|
| Follower (calibration `so101_follower`, verified against the motors) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00` |
| Leader (`so101_leader`) | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79018576-if00` |
| **Wrist camera** (Sonix USB2.0_CAM1; checked from a frame: gripper jaws in view) | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0` |
| Not a rig camera: Kingcome FHD WebCam, the laptop's built-in webcam (black frame, shutter closed) | `/dev/v4l/by-id/usb-Kingcome_FHD_WebCam_200901010001-video-index0` |

Camera settings used on 2 Oct (wrist camera): manual exposure 200 (20 ms; 10 ms was too dark, mean
brightness ~50/255), gain 0, 50 Hz, **auto white balance kept on**. This camera's manual white balance
leaves a green cast at every temperature from 2800 to 6500 K, unlike its auto mode, so it is not used
for now. Revisit before the pen pilot. To apply: `steering/lock_camera.sh <wrist path> 200`, then
`v4l2-ctl -d <wrist path> --set-ctrl=white_balance_automatic=1`. Check with `--list-ctrls` after the
first lerobot run, because OpenCV may reset them.

Use these paths in place of `/dev/ttyACM0` / `/dev/ttyACM1` below: the `ttyACM` numbers can swap on replug.

Always go through `uv run` so the fork's code runs, not a stray install. Replace ports and
camera paths with what steps 2.1–2.2 report. Joint values are in **degrees**
(`use_degrees=True` is the SO-101 default, and the MolmoAct2 frame correction assumes degrees).

## 2 · Install check, ports, cameras, calibration, teleop

```bash
uv sync --locked --extra molmoact2 --extra core_scripts --extra feetech --extra async

uv run lerobot-find-port                       # once per arm: unplug when prompted
uv run lerobot-find-cameras opencv             # saves a frame per camera to outputs/captured_images/
ls -l /dev/v4l/by-id/                          # stable per-device names; use these, not indices,
                                               # so cam0/cam1 cannot swap after a replug

uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
uv run lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader

uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader
```

### Servo health (dead / jittery joints)

Record ~2 minutes of teleop that moves **every** joint through its range (a joint you never
move reads as dead), locally only, then analyse:

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader \
  --dataset.repo_id=local/servo_check --dataset.no_stamp=true --dataset.push_to_hub=false \
  --dataset.num_episodes=3 --dataset.episode_time_s=40 --dataset.single_task="exercise every joint"

uv run python steering/servo_check.py --repo-id local/servo_check
```

## 2b · Camera rig

```bash
sudo apt install v4l-utils
v4l2-ctl -d /dev/v4l/by-id/<cam0> --list-ctrls        # find this camera's control names
steering/lock_camera.sh /dev/v4l/by-id/<cam0> 156 4600 0 0
steering/lock_camera.sh /dev/v4l/by-id/<cam1> 156 4600 0 0
```

Re-run the lock after every replug and reboot. Check `--list-ctrls` again after the first
lerobot command has opened the camera, because some UVC cameras reset their controls when opened.
Record the final values in `RUNLOG.md`.

Ai2's only published SO-101 sample (`Beegbrain/pick_lemon_and_drop_in_bowl`, episode 0)
uses a **top** view and a **side** view by name, but the names are swapped: "top" is a
table-height horizontal view and "side" is overhead (RUNLOG deviation 5). The model card says camera order does not matter
for this checkpoint. Before settling on the plan's front-facing cam0, look at that dataset's
views (https://huggingface.co/spaces/lerobot/visualize_dataset?path=/Beegbrain/pick_lemon_and_drop_in_bowl).

## 4 · First rollout

`lerobot-rollout --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot` fails on this commit
for two reasons: the Hub config predates #4249, and the stock loader runs out of memory on 16 GB
(RUNLOG deviations 1 and 8). Use the local bf16 re-save, already built and verified in
`steering/checkpoints/`:

```bash
uv run python steering/make_local_checkpoint.py      # only if steering/checkpoints/ is missing

uv run lerobot-rollout \
  --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --robot.max_relative_target=5 \
  --robot.cameras='{
      front: {type: opencv, index_or_path: /dev/v4l/by-id/<cam0>, width: 640, height: 480, fps: 30},
      side:  {type: opencv, index_or_path: /dev/v4l/by-id/<cam1>, width: 640, height: 480, fps: 30}
  }' \
  --rename_map='{"observation.images.front": "observation.images.cam0", "observation.images.side": "observation.images.cam1"}' \
  --task="pick up the red cube" --duration=30
```

**One camera (current rig: a single wrist webcam).** Use the one-view checkpoint copy
`steering/checkpoints/MolmoAct2-SO100_101-LeRobot-1cam`: the same weights, with the input step set to
expect only `cam0`. Name the camera `cam0`; no `--rename_map` is needed. Tested offline on Ai2's frame
(`results/camera_count_test.json`): it runs, and it is a little faster (~266 ms per chunk). Which view
is used matters, though, and a wrist view is untested.

```bash
uv run python steering/rollout.py --tag onecam \
  --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot-1cam \
  --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00 --robot.id=so101_follower \
  --robot.max_relative_target=5 \
  --robot.cameras='{cam0: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0, width: 640, height: 480, fps: 30}}' \
  --task="pick up the red cube" --duration=30
```

The colleague's calibration ids are `so101_follower` / `so101_leader` (files in
`~/.cache/huggingface/lerobot/calibration/`). Use them wherever these commands say `follower` / `leader`.

Before the first rollout, and for the raised start pose used for pen frames, check that the arm
starts inside the range the checkpoint was trained on. Outside it, the first action is a jump back
into range, about 8° on the elbow for Ai2's rest pose.

```bash
uv run python steering/check_state_range.py --robot-port /dev/ttyACM0 --robot-id follower
# or offline: --state 'pan,lift,elbow,wrist_flex,wrist_roll,gripper' (degrees, arm frame)
```

`max_relative_target=5` caps each step at 5° per joint. The policy runs at 30 Hz, so that is
still up to 150°/s. Start at 2–3 if you want the first run slower. The joint-frame correction
is already inside the checkpoint's pre/post-processors, so do not add it again. If the arm moves
the wrong way, check the calibration and `use_degrees` first.

Step 3 measured about 360 ms per 30-action chunk. The synchronous loop (the default) pauses for that
long at the start of every chunk, about once a second. Add `--inference.type=rtc` to hide it once
the basic run looks sane.

### Recorded rollouts (steering/rollout.py)

**From now on, `steering/rollout.py` is the ONLY way to run a policy on the arm**: it records every
run, plots it, and connects torque-safely. Do not call `lerobot-rollout` directly.

`steering/rollout.py` runs `lerobot-rollout` in-process with the same arguments, plus wrapper-only
`--tag`, `--allow-torque-blip` and `--no-plot`. It refuses to start without
`--robot.max_relative_target=...`: the cap is per step at 30 Hz, so 4° per step ≈ 120°/s.
It connects without the torque blip when the servos already hold the settings `configure()`
would write (and keeps the goal of an arm already holding a pose with torque on); otherwise it exits
with the list of mismatches, without touching torque. Only with `--allow-torque-blip` does it warn and
run the stock `configure()`, and then the arm sags: support it. It also
records every tick (state, policy and sent command, clipping), every inference call with its chunk,
and a camera frame every 0.1 s (10 fps) to `steering/results/runs/<stamp>_<tag>/`. It adds
`--robot.disable_torque_on_disconnect=false`, so the arm keeps holding its pose after the run.
See the docstring for details.

First move to the most in-distribution start pose (torque-safe, torque stays on), then run, then plot:

```bash
uv run python steering/goto_pose.py go molmo_median \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00 --robot-id so101_follower

uv run python steering/rollout.py --tag median_rtc \
  --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot-1cam \
  --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00 --robot.id=so101_follower \
  --robot.max_relative_target=4 \
  --robot.cameras='{cam0: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0, width: 640, height: 480, fps: 30}}' \
  --inference.type=rtc \
  --task="pick up the red cube" --duration=30

# plots are made automatically at the end (normal exit or Ctrl-C; --no-plot skips); to redo them:
uv run python steering/plot_run.py steering/results/runs/<stamp>_median_rtc   # path is printed at the end
```

`plot_run.py` writes `plots/joints.png`, `chunks.png`, `cadence.png` and `frames.png`, and prints a summary:
Hz, % of ticks clipped per joint, travel, the start and end pose, and whether the start was inside
the trained range. `uv run python steering/plot_run.py --self-test` checks the plotting without the arm.
At the end the stock teardown still returns the arm to its start pose over 3 s
(`--return_to_initial_position=false` to skip).

#### Looped episodes (one model load)

`--strategy.type=episodic` loads the model once and runs `--dataset.num_episodes` episodes. Each one is a
policy phase of up to `episode_time_s`, then a reset of up to `reset_time_s`. No reset follows the last
episode. It also writes a LeRobot dataset, and the repo name must start with `rollout_`. The reset moves
the arm back to the pose it had at connect **in 1 s**, which is fast. So start from the median pose and
keep a hand near the arm and the power switch:

```bash
uv run python steering/goto_pose.py go molmo_median \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00 --robot-id so101_follower

uv run python steering/rollout.py --tag loop_cube \
  --strategy.type=episodic \
  --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot-1cam \
  --robot.type=so101_follower --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114089-if00 --robot.id=so101_follower \
  --robot.max_relative_target=6 \
  --robot.cameras='{cam0: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0, width: 640, height: 480, fps: 30}}' \
  --inference.type=rtc \
  --dataset.repo_id=local/rollout_cube --dataset.single_task="pick up the red cube" \
  --dataset.num_episodes=5 --dataset.episode_time_s=25 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false
```

- Keys: right arrow (or `n`) ends the current episode or reset early. Left arrow (or `r`) discards the
  episode and re-records it. Escape (or `q`) stops the session.
- One prompt per session: `--dataset.single_task` is used for every episode. To change the
  instruction, start a new session.
- In the run dir, `plots/` has the whole-session plots, `episodes.png` (all episodes overlaid,
  time from each episode's start), `ep<k>/` (joints, chunks, frames for attempt k) and a per-episode
  table at the top of `summary.txt`. The videos are `replay.mp4` (the whole session) and
  `replay_ep<k>.mp4` (attempt k, its policy phase plus reset). `k` counts attempts, so a discarded
  attempt keeps its own number, and `meta.json["episodes"]` maps each one to its dataset episode
  index, or marks it `discarded`.
- The dataset lands in `~/.cache/huggingface/lerobot/local/rollout_cube_<timestamp>/`.

## 5 · Pilot: pens on one frame, then on the arm

Frame precursor, once a frame of the pen layout exists:

```bash
# grab one frame per camera from the rig (e.g. from outputs/captured_images/), and the arm's
# current joint state as a JSON list of 6 numbers in degrees
uv run python steering/frame_diagnostic.py --tag pens_layoutA \
  --cam0 cam0.png --cam1 cam1.png --state state.json \
  --condition blue="pick up the blue pen" --condition red="pick up the red pen" --condition null=""
```

Take the frame with the arm **raised over the middle of the workspace**, not at rest, and inside
the trained range (`check_state_range.py`). From rest,
the first 30 actions are a generic lift, and no instruction changes the pan direction (RUNLOG,
frame diagnostic). Use `--seeds 128` and read the `movers` block, not just the mean-based ratios.
Repeat with the colours swapped (layout B). The hardware pilot only makes sense if the moving
seeds head towards the named pen in both layouts: in `mover_direction`, the `blue_vs_red`
`mover_pan_diff_ci95` excludes 0 in both layouts, with opposite signs.

### Capturing a pen frame

**Support the arm by hand whenever a command connects to it.** LeRobot's `robot.connect()` briefly
switches torque off while it configures the motors (`lerobot-teleoperate`, `lerobot-rollout`), so a
raised arm sags for a moment. `goto_pose.py go` avoids this: it enables torque without reconfiguring,
which works once the arm has been through calibration or teleop. Quitting teleop or rollout switches torque
off and the arm drops. So hold the arm when saving a pose after teleop, and at the start of a rollout.

```bash
uv run python steering/goto_pose.py save raised        # once: teleop/hold the arm there, then save
uv run python steering/goto_pose.py go raised          # slow move (2°/step), torque stays on, holding
uv run python steering/capture_frame.py --out frames/layoutA --robot-port /dev/ttyACM0 --robot-id follower \
  --cam0 /dev/v4l/by-id/<cam0> --cam1 /dev/v4l/by-id/<cam1>
# writes cam0.png, cam1.png, state.json and prints the trained-range check
```

### Pilot rollouts (fixed 20 s episodes)

The `episodic` strategy records fixed-length episodes and, with no leader arm connected, returns
the arm to its startup pose between episodes. So send it to the raised pose first, then launch
**while holding the arm in place** until the first episode starts, because the startup pose is
read after connecting. Check the first reset returns to the raised pose. If it returns to a sagged
pose, restart the session. One
session per instruction × layout; the instruction is in the dataset name and task.

```bash
uv run python steering/goto_pose.py go raised
uv run python steering/rollout.py --tag pilot_pens_blue_A --strategy.type=episodic \
  --policy.path=steering/checkpoints/MolmoAct2-SO100_101-LeRobot \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --robot.max_relative_target=5 \
  --robot.cameras='{front: {type: opencv, index_or_path: /dev/v4l/by-id/<cam0>, width: 640, height: 480, fps: 30},
                    side:  {type: opencv, index_or_path: /dev/v4l/by-id/<cam1>, width: 640, height: 480, fps: 30}}' \
  --rename_map='{"observation.images.front": "observation.images.cam0", "observation.images.side": "observation.images.cam1"}' \
  --dataset.repo_id=local/rollout_pens_blue_A --dataset.single_task="pick up the blue pen" \
  --dataset.num_episodes=10 --dataset.episode_time_s=20 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false
```

- Keys: right arrow ends the episode early, left arrow discards and re-records, Escape stops.
- Rollout dataset names must start with `rollout_`.
- For the null condition, use `--dataset.single_task=""`. The config accepts it, but the dataset
  writer has not been tried with an empty task. If it errors, note it and skip null on the arm (the
  frame test covers it).
- Videos of every episode are saved in the dataset, which is what the contact labelling uses.

### Choosing the camera pose (Appendix G, with the README guard rails)

For a few rig configurations (keep it to 3–4 in total, since rebuilding the rig is the expensive part:
for example the plan's front pose, Ai2's table-height + overhead pair, and one variant of either), with the arm raised in the same pose and the same
pens:

```bash
uv run python steering/frame_diagnostic.py --tag pose_<config>_A --seeds 128 \
  --cam0 cam0.png --cam1 cam1.png --state state.json \
  --condition blue="pick up the blue pen" --condition red="pick up the red pen" --condition null=""
# then swap the pens (layout B) and repeat with --tag pose_<config>_B
```

Keep the configuration whose blue-vs-red pan difference is largest while passing in both layouts.
Confirm it on a fresh layout before building the rig around it. Keep every losing configuration's
JSON in `results/`.

### Recording demonstrations

Randomise which pen the operator picks, for example with a coin flip or a printed random sequence,
and write the choice per episode. `lerobot-record` takes one task string per session
(`--dataset.single_task`), but `--resume=true` appends to the same dataset and each episode keeps
the task it was recorded with. So record short sessions in the randomised order, one target colour
each, all into one dataset:

```bash
# first session creates the dataset; later ones add --resume=true (same repo id, no stamp)
uv run lerobot-record <robot and teleop flags as in §2, camera flags as in §4> \
  --dataset.repo_id=local/pens_demos --dataset.no_stamp=true --dataset.push_to_hub=false \
  --dataset.single_task="pick up the blue pen" --dataset.num_episodes=5 [--resume=true]
```

`--dataset.num_episodes` counts the episodes to add in that session, not the dataset total.
