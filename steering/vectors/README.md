# Steering vectors

`word_probe_L09.{npy,json}`: red-vs-green colour direction fitted on **MolmoAct2-LIBERO** in simulation
(two cubes, agentview + wrist cameras). Block 9, text tokens, +alpha = toward red, -alpha = toward green.
Validation-selected strength k = 0.25 (alpha = k * 48.9). In LIBERO it raised red picks from 9/50 to 38/50.

**Not fitted on this repo's checkpoint** (`MolmoAct2-SO100_101`). It may not transfer; run `--steer-k 0` first.

    uv run python steering/rollout.py --tag steer_red \
      --steer-vector steering/vectors/word_probe_L09.npy --steer-k 0.25 --robot.max_relative_target=4 ...
