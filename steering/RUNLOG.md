# Run log (step 6 record)

Measurements only. Nothing here has been tuned to look better.

## Versions and commits

| | |
|---|---|
| Fork | `origin` = github.com/pinhofst/lerobot, branch `steering` |
| Upstream base | `steering-base` = `d8a09caa` (upstream `main`, 29 Sep 2026). Includes the MolmoAct2 fix `c13d79e6` (#4249, 21 Aug 2026) |
| Reference remote | `allenai` → allenai/lerobot `molmoact2-hf-inference` @ `28b4f721` (not checked out) |
| Env | `uv sync --locked --extra molmoact2 --extra core_scripts --extra feetech --extra async` |
| torch / CUDA wheels | 2.11.0+cu128 (sm_120 in arch list) |
| transformers / peft | 5.5.4 / 0.21.0 |
| Policy checkpoint | `lerobot/MolmoAct2-SO100_101-LeRobot` @ `a93b5fcd` (26 Jun 2026) |
| Base checkpoint pulled by it | `allenai/MolmoAct2-SO100_101` (pulled because it is the policy's `checkpoint_path`) |

## Machine (step 1)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 **Laptop** GPU (mobile), 16,303 MiB, about 445 MiB used by the display |
| Driver | 580.178.04 (CUDA 13.0 capable). LeRobot's cu128 wheels need ≥ 570.86 |
| Python | 3.13.9 (anaconda base interpreter, isolated `.venv`) |
| RAM / disk | 30 GiB / 1.3 TB free |

The VRAM gate ("under 16 GB → stop") is borderline. 15.9 GiB is below 16 GiB. We went ahead
because Ai2 report bf16 inference under 16 GB. Measured peak is in step 3.

## Latency and VRAM (step 3)

Measured on Ai2's sample frame and task, arm disconnected, bf16, 5 warm-up calls and 30 timed calls,
`chunk_size = n_action_steps = 30`, flow solver at the checkpoint default. Script: `bench_latency.py`,
raw data: `results/latency_bfloat16_graph-*.json`.

| | CUDA graph on | CUDA graph off |
|---|---|---|
| End-to-end per chunk, median (p90) | **361 ms** (370) | 445 ms (454) |
| Model only (`predict_action_chunk`), median | 272 ms | 350 ms |
| Preprocessing (end to end minus model) | ~89 ms | ~95 ms |
| First call (graph capture / warm-up) | 2.2 s | 1.0 s |
| Amortised control rate, 30 / latency | 83 Hz | 67 Hz |
| VRAM after load (allocated) | 11.28 GiB | 11.28 GiB |
| VRAM peak (allocated / reserved) | 12.15 / 13.37 GiB | 12.08 / 13.24 GiB |
| nvidia-smi used at peak (incl. context and display) | 14.4 GB of 16.3 | 14.1 GB |
| Host RAM peak (load) | 23.9 GB of 30 | 24.5 GB |
| Load time (cache warm) | 107 s | 106 s |

Reading:
- It fits, with about 1.5–2 GB of VRAM to spare. CUDA graphs save about 80 ms and cost almost no memory at bf16.
- It is 2× slower than the 179 ms H100 figure, and much faster than the plan's ~1 s guess for a mobile GPU.
- 83 Hz amortised is above the 30 Hz control rate, but the default synchronous `lerobot-rollout`
  loop blocks for the whole ~360 ms at the start of every 30-step chunk, which is a visible pause
  every second. Use `--inference.type=rtc` (MolmoAct2 continuous supports RTC) or the async
  server/client.
- About 90 ms per chunk goes to CPU preprocessing (image processing and tokenisation), not the GPU.
  It is a cheap target if latency matters later.
- Loading peaks at about 24 GB of host RAM. Close other heavy programs before loading.
- **The stock loader runs out of memory on this GPU** (tested: fails at 14.65 GiB allocated). See deviation 8.
  The numbers above use `molmo_common.load_policy`, which streams the weights in one tensor at a time.
  `steering/checkpoints/MolmoAct2-SO100_101-LeRobot` (bf16 re-save, 12.0 GB) loads through the
  stock path with an 11.3 GiB peak and gives identical actions (`verify_local_checkpoint.py`,
  max |diff| 0.00°).
- Sanity: the arm-frame state is [-0.5, -99.1, 91.4, 60.6, -3.6, 1.1] and the first predicted action is
  [-1.8, -96.1, 83.6, 60.4, -5.6, 0.0]. That is close to the current pose, as expected, so the
  calibration inversion is consistent.

## Camera mapping

_pending (arm steps)_

## Frame diagnostic on Ai2's sample frame (step 5 precursor, stand-in for the pen frame)

This is **not** the pen test. It is the same method run on the only frame available today
(Ai2's sample: apple, lemon, strawberry, peach and a red bowl; the arm at rest; the lemon is the
trained target). Instructions use Ai2's training template with only the noun changed ("Move the arm
towards the X, grasp it, lift it up, and drop it into the red bowl."), plus an empty instruction.
Every condition uses the same seeds. Script: `frame_diagnostic.py`, raw data: `results/frame_diagnostic_ai2_sample*.json`.

| Instruction | Seeds that start moving (> 5° in 30 steps) | Mean movement of those seeds, pan / lift / elbow (°) |
|---|---|---|
| lemon (trained) | **77 / 128 (60%)** | -0.5 / +26.6 / -28.2 |
| apple | 38 / 128 (30%) | +1.5 / +20.5 / -18.3 |
| strawberry | 40 / 128 (31%) | -0.4 / +21.8 / -19.6 |
| peach | 35 / 128 (27%) | -0.7 / +15.8 / -12.4 |
| empty | 9 / 128 (7%) | -3.5 / +3.8 / -2.0 |

Reading:
- **The prediction is bimodal per seed.** The arm either holds still or starts a lift/reach. The
  instruction changes *how likely it is to start* in the first second: trained object > any named
  object > no instruction. So the model is not ignoring language.
- **There is no evidence of target-directed motion within one chunk.** The seeds that move all do
  the same lift. Mean shoulder_pan change is at most 1.5°, with no consistent sign, even though the
  objects sit at clearly different bearings. From a rest pose, the first 30 actions (1 s) come
  before the arm commits to a side.
- Mean-based statistics (whole-chunk RMS, the separation ratio, Cohen's d) mostly measure how often
  the arm starts. They are misleading here: ratios < 1 and permutation p < 0.05 at the same time.
  Use the moving fraction and the direction of the moving seeds.
- **Consequence for the plan's precursor (Appendix E).** A single frame taken with the arm at rest
  cannot tell "follows the colour" apart from "reaches the same way regardless". For the pen frame,
  take the frame with the arm already raised over the middle of the workspace, so the first chunk
  has to choose a side. Or take several frames partway through a reach. Also report the moving
  fraction per condition.

## Pilot table (step 5)

_pending (arm steps)_

## Deviations from the documented setup / the plan

1. **The Hub checkpoint config does not load on the pinned commit.** #4249 removed
   `enable_lora_vlm`, `enable_lora_action_expert`, `train_action_expert_only` and `model_dtype`
   from `MolmoAct2Config`, but `lerobot/MolmoAct2-SO100_101-LeRobot/config.json` still has them,
   so draccus raises `DecodingError`. `lerobot-rollout --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot`
   fails as written. Workaround: `steering/molmo_common.translate_legacy_config` maps
   them to `train_mode_vlm="fft"`. If left unset, the new default `"lora"` would wrap the VLM in
   untrained adapters. No upstream issue found as of 29 Sep 2026.
2. **Inference path depends on whether `--policy.path` is local.** `_saved_policy_action_mode`
   reads `action_mode` from `<pretrained_path>/config.json` only for local directories. When it
   finds it, it adds a continuous-training encoder attention mask at inference. A Hub id never
   triggers it, so the same checkpoint runs two different inference paths. We use the Hub-id
   behaviour, which matches Ai2's native inference. The local copy made by
   `make_local_checkpoint.py` drops `action_mode` to keep it that way. Measured effect of the mask
   on the sample frame (32 seeds): max 3.7°, mean 0.24° per action, same moving fraction (59%). It is
   small but not zero, so it is a confound when comparing runs.
3. **The download is about 44 GB, not about 12 GB.** Both repos hold fp32 weights
   (5.44 B params, 21.8 GB each), and loading needs both: the base builds the model, then the
   LeRobot `model.safetensors` is loaded on top. VRAM at bf16 is still about 11 GB of weights.
4. The plan's "horizon 10" latency anchor: this checkpoint predicts **30** actions per chunk.
   Ai2's `num_steps=10` is the flow-solver step count, not the action horizon. Control rate
   is reported as 30 / latency.
5. **Reference camera pose.** Ai2's sample input comes from `Beegbrain/pick_lemon_and_drop_in_bowl`
   (SO-100, 640×480 @ 30 fps, codebase v2.1, keys `realsense_top` and `realsense_side`). The two
   sample PNGs are labelled the wrong way round. `sample_realsense_top_rgb.png` is a horizontal
   view from roughly table height, with the arm entering from the left. `sample_realsense_side_rgb.png`
   is an overhead view looking down at the table, with the gripper entering from the top edge.
   The model card says camera order does not matter. So the one published pose is a low
   horizontal view plus an overhead view. Neither matches the plan's cam0 at 20–30° elevation.
   `bench_latency.py` / `frame_diagnostic.py` feed the file named "top" as cam0. Both views side by side:
   `results/ai2_sample_views.png`.
6. A PyPI `lerobot 0.6.1` in anaconda base shadowed the fork's CLI entry points. It has been uninstalled.
7. `torchcodec` is installed but fails to load (`libavutil.so.60` missing, no system FFmpeg),
   so video decoding falls back to PyAV. Fix: `sudo apt install ffmpeg`.
8. **Stock loading needs more than 16 GB of VRAM.** `PreTrainedPolicy._load_as_safetensor` calls
   `safetensors.torch.load_model(device="cuda:0")`, which puts the whole fp32 `model.safetensors`
   (21.8 GB) on the GPU before copying it into the bf16 model. Tested: OOM at 14.65 GiB allocated on
   this card. Ai2's "under 16 GB" figure is for their own loader, not LeRobot's. Workaround: stream
   the weights (`molmo_common`), and for `lerobot-rollout` use the bf16 re-save from
   `make_local_checkpoint.py`. Worth reporting upstream together with deviation 1.
