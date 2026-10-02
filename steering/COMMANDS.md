# Arm-side commands (steps 2, 2b, 4, 5)

No Hugging Face account is needed. The models are already cached, and every recording
command here passes `--dataset.push_to_hub=false` and uses a `local/<name>` repo id. Datasets land
in `~/.cache/huggingface/lerobot/local/`.

If `lerobot-find-port` finds nothing, or the port appears and then vanishes, run
`sudo apt remove brltty`. It grabs the USB-serial chips SO-101 boards use.

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
uv run lerobot-rollout --strategy.type=episodic \
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
