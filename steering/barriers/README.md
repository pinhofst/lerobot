# Steering barriers

Trained barrier modules for `--steer-barrier` (README "Steering modes"; format in `steer_barrier.py`). Each
`<name>.pt` holds the module and its frozen description bank, so loading needs no model pass. `<name>.json` is the
sidecar: layer, positions, `k_default`, `policy_checkpoint`, `trained_on`, the bank, and the source checkpoint with
its sha256.

Two versions for the real-robot comparison. Both target **red** and are used at k = 1 (COMMANDS "Steered
rollouts"). Offline on 8 held-out real runs: gap to the red teacher (unsteered = 100%) 47% uncalibrated vs 12%
calibrated; heading matches red on 17/20 vs 20/20 frames (neutral prompt) and 18/20 vs 20/20 ("green pen" prompt);
unsteered 9/20 and 1/20.

**Calibrated (SO-101 real frames):** `barrier_so101.{pt,json}` (13.7 MB): the barrier calibrated offline in steervla `runs/steering_so101`
on the one-view SO-101 checkpoint (`lerobot/MolmoAct2-SO100_101-LeRobot`, bf16, wrist `cam0` only) with recorded
real frames of two pens, teacher "pick up the red pen"; SO-101 description bank inside. Block 6, task span, k = 1.
Offline on held-out frames: 12% of the no-edit loss at k = 1, the red teacher's heading on 20/20 two-pen frames
(neutral and "green" prompts), better than the calibrated constant (loss ratio 0.65). Offline proxy only, no
real-robot runs yet (README "Calibration").

    uv run python steering/rollout.py --tag bar_k0 ... --steer-barrier steering/barriers/barrier_so101.pt --steer-k 0

**Uncalibrated (LIBERO sim only):** `barrier_r3_L06.{pt,json}` (13.7 MB): the round-3 barrier of steervla's `steering_dial`, block 6, task-span tokens,
3.41 M parameters. Trained on **MolmoAct2-LIBERO** in simulation (two cameras, agentview + wrist; red and green
cubes) by on-policy velocity matching to the same frozen policy prompted "pick up the red cube". It pushes toward
**red**. LIBERO test: hard red 79% under the neutral prompt and 77% under "pick up the green cube", the same as the
trained constant vector (`../vectors/const_r3_L06.npy`). Its zero-edit fraction was 0%: it never switched off.

**Not fitted on this repo's checkpoint** (`MolmoAct2-SO100_101`): it prints a mismatch warning and may not
transfer. Run `--steer-k 0` first, then 0.25, 0.5, 1. `--steer-k` scales the edit: u = k·ReLU(τ − ⟨a,h⟩)·a, k = 1 is
the trained module.

    uv run python steering/rollout.py --tag bar_k0 ... --steer-barrier steering/barriers/barrier_r3_L06.pt --steer-k 0

To add another module (same format), export the steervla checkpoint:

    python steering/steer_barrier.py --src <steervla ckpt.pt> --out steering/barriers/<name>.pt --layer 6 \
      --policy-checkpoint allenai/MolmoAct2-SO100_101 --trained-on "..." --src-id "steervla:<path>"
