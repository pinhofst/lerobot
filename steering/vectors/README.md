# Steering vectors

**Primary:** `const_so101.{npy,json}`: the constant vector calibrated offline in steervla `runs/steering_so101` on
the one-view SO-101 checkpoint (`lerobot/MolmoAct2-SO100_101-LeRobot`, bf16, wrist `cam0`), two pens, teacher "pick up
the red pen". Block 6, task-span tokens, +k = toward red. `--steer-k 1` (the calibrated k) adds scale·w, norm 54.0
(bitwise steervla's `const_so101_k1.npy`). Offline it is behind the calibrated barrier (loss ratio 0.65). Offline
proxy only, no real-robot runs yet.

    --steer-vector steering/vectors/const_so101.npy --steer-k 1

`const_so101_green.{npy,json}`: the same for the **green** target (teacher "pick up the green pen"): +k = toward green,
`--steer-k 1` adds norm 45.3 (bitwise steervla's `const_so101_green_k1.npy`). Held-out loss 0.00175 (no edit 0.0111,
green barrier 0.00140).

**Safety:** `const_never_red_v2.{npy,json}` and `const_never_green.{npy,json}`, both at `--steer-k 1.25`: "never touch
the red / green pen" (README "Safety"). Never-red v2 is the best safety file: under 3-fold run-level CV, 7 of 9
red-only runs hold (no descent, no grip), both pens → green on 18–20/20 frames, green alone still picked. Never-green
is anecdotal (one test run, one pen position). Offline proxy only.

    --steer-vector steering/vectors/const_never_red_v2.npy --steer-k 1.25

The files below were fitted on **MolmoAct2-LIBERO** (secondary).

`word_probe_L09.{npy,json}`: red-vs-green colour direction fitted on **MolmoAct2-LIBERO** in simulation
(two cubes, agentview + wrist cameras). Block 9, text tokens, +alpha = toward red, -alpha = toward green.
Validation-selected strength k = 0.25 (alpha = k * 48.9). In LIBERO it raised red picks from 9/50 to 38/50.

**Not fitted on this repo's checkpoint** (`MolmoAct2-SO100_101`). It may not transfer; run `--steer-k 0` first.

    uv run python steering/rollout.py --tag steer_red \
      --steer-vector steering/vectors/word_probe_L09.npy --steer-k 0.25 --robot.max_relative_target=4 ...

`green_meandiff_L06.{npy,json}`: separate green direction, fitted on LIBERO as mean(green prompt) - mean(ambiguous
prompt), block 6, text tokens. **Sign is opposite to the red file: +alpha = toward GREEN.** Validation-selected k = 0.25
(alpha = k * sidecar alpha_unit_value). LIBERO test (50 layouts, "pick up the cube"): green picks 76% -> 86%
(p=0.18 vs unsteered), the same counts as the flipped red vector (-0.25 on `word_probe_L09`); both match the
"pick up the green cube" prompt (86%). Not tested on the SO-101 checkpoint.

    --steer-vector steering/vectors/green_meandiff_L06.npy --steer-k 0.25

`const_r3_L06.{npy,json}`: the trained constant vector of steervla's `steering_dial` (round 3), the baseline arm next
to the barrier (`../barriers/barrier_r3_L06.pt`). Block 6, **task-span tokens** (the instruction only; the tokenizer
is loaded from the policy). Trained on **MolmoAct2-LIBERO** (two cameras, cubes) by on-policy velocity matching to
the policy prompted "pick up the red cube": +k = toward red. Stored as w with `alpha_unit_value` = the module scale,
so `--steer-k 1` adds exactly the trained vector (norm 49.9). LIBERO test: the same as the barrier in every setting
(hard red 78% under the neutral prompt and 78% under the green prompt; barrier 79% and 77%). Not tested on the SO-101 checkpoint.

    --steer-vector steering/vectors/const_r3_L06.npy --steer-k 1

`const_m1_L06_green.{npy,json}`: the green counterpart (steervla `steering_dial` mirror `m_const_r1`, one refresh
round), next to `../barriers/barrier_m1_L06_green.pt`. Trained on **MolmoAct2-LIBERO** toward the policy prompted
"pick up the green cube": +k = toward green; `--steer-k 1` adds the trained vector (norm 38.5). LIBERO validation: 74% green on hard layouts vs 77% for the fixed −L6
vector (missed the gate; no LIBERO test). Offline on the SO-101 held-out frames, zero-shot: green-teacher loss 0.00517
(no edit 0.0111).

    --steer-vector steering/vectors/const_m1_L06_green.npy --steer-k 1
