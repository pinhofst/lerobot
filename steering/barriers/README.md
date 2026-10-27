# Steering barriers

Trained barrier modules for `--steer-barrier` (README "Steering modes"; format in `steer_barrier.py`). Each
`<name>.pt` holds the module and its frozen description bank, so loading needs no model pass. `<name>.json` is the
sidecar: layer, positions, `k_default`, `policy_checkpoint`, `trained_on`, the bank, and the source checkpoint with
its sha256.

`barrier_r3_L06.{pt,json}` (13.7 MB): the round-3 barrier of steervla's `steering_dial`, block 6, task-span tokens,
3.41 M parameters. Trained on **MolmoAct2-LIBERO** in simulation (two cameras, agentview + wrist; red and green
cubes) by on-policy velocity matching to the same frozen policy prompted "pick up the red cube". It pushes toward
**red**. LIBERO test: hard red 79% under the neutral prompt and 77% under "pick up the green cube", the same as the
trained constant vector (`../vectors/const_r3_L06.npy`). Its zero-edit fraction was 0%: it never switched off.

**Not fitted on this repo's checkpoint** (`MolmoAct2-SO100_101`): it prints a mismatch warning and may not
transfer. Run `--steer-k 0` first, then a small k.

    uv run python steering/rollout.py --tag bar_k0 ... --steer-barrier steering/barriers/barrier_r3_L06.pt --steer-k 0

To add an SO-101 version (same format), export the steervla checkpoint:

    python steering/steer_barrier.py --src <steervla ckpt.pt> --out steering/barriers/<name>.pt --layer 6 \
      --policy-checkpoint allenai/MolmoAct2-SO100_101 --trained-on "..." --src-id "steervla:<path>"
