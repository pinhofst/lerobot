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
| nvidia-smi used at peak (incl. context and display) | 14,419 of 16,303 MiB | 14,127 MiB |
| Host RAM peak (load) | 23.9 GB of 30 | 24.5 GB |
| Load time (cache warm) | 107 s | 106 s |

A rerun on 30 Sep (graph on) gave a median of **321 ms** (p90 360 ms), model only 278 ms, and the same
peak memory (`results/latency_bfloat16_graph-on_rerun-2026-09-30.json`). Treat per-chunk latency as
about 320–360 ms on this machine. Preprocessing was ~43 ms in the rerun against ~89 ms in the first
run, so the CPU share varies with machine load.

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
  stock path with an 11.3 GiB peak and gives identical actions (`verify_local_checkpoint.py`: the full
  30×6 seed-29 chunk matches the streamed load exactly, max |diff| 0.000000, including the 159 of 180
  entries not at the clamp bounds; `results/verify_local_checkpoint_seed29.npy`).
- Not a calibration check (corrected after review): the first predicted action,
  [-1.8, -96.1, 83.6, 60.4, -5.6, 0.0], has shoulder_lift and elbow_flex pinned at the
  postprocessor's clamp bounds. Ai2's rest-pose state (model frame 189.1 / 181.4) lies above the
  checkpoint's q99 for both joints, so the output is clamped. It says nothing about the arm-frame
  inversion. That inversion is verified by code review instead: it is the exact inverse of
  `MolmoAct2StateFrameTransformStep`.

## Camera mapping and the one-camera setup (2 Oct)

Rig on 2 Oct: **one camera, on the wrist** (Sonix USB2.0_CAM1, `cam0`). The second USB camera the
laptop lists (Kingcome FHD) is its built-in webcam. The LeRobot checkpoint is packaged for two views
and raises an error if one is missing, so a one-view copy is used:
`steering/checkpoints/MolmoAct2-SO100_101-LeRobot-1cam` (same weights; `molmoact2_pack_inputs.image_keys
= [cam0]`). `lerobot-rollout` accepts it, because the robot's cameras only need to be a subset of the
policy's (`rollout/context.py`).

**Offline test on Ai2's frame** (`camera_count_test.py`, `results/camera_count_test.json`, 64 shared seeds):

| Input | Seeds moving | Movers pan / lift / elbow (°) | RMS from two views (°) | Per chunk |
|---|---|---|---|---|
| Two views | 41 / 64 | −0.5 / +26.2 / −27.8 | — | ~316 ms |
| Same frame twice (table-height) | 35 / 64 | 0.0 / +28.9 / −34.2 | 1.4 | ~319 ms |
| One view: table-height | 28 / 64 | +0.1 / +23.4 / −27.8 | 2.2 | ~266 ms |
| One view: overhead | 4 / 64 | +0.5 / +9.9 / −20.3 | 4.0 | ~265 ms |

One view mostly makes the arm move less often. The view chosen matters a lot, and a wrist view was
not testable offline.

**What the paper says** (`MOLMO_PAPER_NOTES.md`):
- About 11% of SO-100/101 training episodes have a single camera (Table 23).
- Only about 1.7% of episodes (39 single-camera datasets) name that camera wrist, gripper or hand.
- Every SO-100 evaluation in the paper used wrist + third-person (Sec. 6.2).
- Views enter as "Image 1, Image 2…", with camera order shuffled per episode (Sec. 4.3.1), so the
  cam0/cam1 slots carry no meaning.

So one camera is in distribution, but **wrist-only is rare**: expect weaker behaviour, which is not
by itself a sign of a fault. Add a fixed third-person camera before the pen pilot.

**First rollout, 2 Oct 15:44** (one view, `--task="pick up the red pen"`, 20 s, `max_relative_target=3`,
synchronous inference, started from the folded rest pose):
- It ran end to end, but **the arm did not leave the rest pose**.
- The model kept commanding elbow 83.6°, the edge of the trained range, because the clipped state
  tells it the elbow is there. The 3° cap turned that into present − 3°, and the elbow stayed at
  about 94.6°. A position error capped at 3° is likely too small for the elbow servo to unfold
  the arm against its load.
- Effective loop rate 15.5 Hz, against a 30 Hz target: the synchronous loop waits ~300 ms per chunk.
- Nothing was recorded besides the terminal log.

Next: start from the checkpoint's median training pose (`poses/molmo_median.json`, moved there with
`goto_pose.py`), with a larger cap, `--inference.type=rtc`, and recording.

## How action chunking runs here (read from LeRobot's code, 2 Oct)

- **Each inference predicts 30 actions (1.0 s at 30 Hz)**, from one pass of the vision-language model
  plus 10 flow-matching steps. Nothing is ever averaged across chunks; there is no ACT-style
  temporal ensembling anywhere.
- **Sync mode** (the default): plays all 30 actions open-loop, then stalls about 0.4 s on the control
  thread for the next chunk. That gives a ~1.4 s cycle and is where the first run's 15.5 Hz came
  from. This matches the paper's open-loop setup, except for the stall.
- **RTC mode** (`--inference.type=rtc`, defaults `execution_horizon=10`, `max_guidance_weight=10`,
  `queue_threshold=30`):
  - Inference runs back to back, about one chunk per 0.47 s (d ≈ 14 steps).
  - Each new chunk *replaces* the old one: its first d actions are dropped, and roughly indices
    14–27 actually run. That is about 14 open-loop actions per observation, each executed 0.47–0.9 s
    after the observation was taken.
  - Guidance pulls indices 0–9 of the new chunk towards the old plan, but those are exactly the
    dropped ones. **At these defaults, guidance shapes only discarded actions.** Executed actions are
    pulled towards the old plan only indirectly, so seam smoothing is weak.
  - The original RTC method weights all indices below d and fades out over the overlap. A larger
    `--inference.rtc.execution_horizon` (about 20, within the 16 actions left in the queue)
    would move guidance onto executed actions. Untested.
  - The model-side delay is a running maximum that is never reset, so one latency spike raises it
    for the rest of the session.
- **Noise:** fresh and unseeded for every chunk (`per_episode_seed=False`), so runs are not
  reproducible. Compare distributions over runs, not runs pair by pair.
- **For analysis:** in RTC runs only about `chunk[k][d:2d]` was executed. Match `ticks.action_policy`
  against the chunk rows to find it. Seam continuity is `chunk_k[d]` against the last executed action
  of chunk k−1.

## Frame diagnostic on Ai2's sample frame (step 5 precursor, stand-in for the pen frame)

This is **not** the pen test. It is the same method run on the only frame available so far
(Ai2's sample: apple, lemon, strawberry, peach and a red bowl; the arm at rest; the lemon is the
trained target). Instructions use Ai2's training template with only the noun changed ("Move the arm
towards the X, grasp it, lift it up, and drop it into the red bowl."). The null condition is an
empty task, which the processor turns into "The task is to . The setup is …", itself out of
distribution. Every condition uses the same 128 noise seeds.

Script: `frame_diagnostic.py`. Raw data: `results/frame_diagnostic_ai2_sample_cag.json` and
`_chunks.npz`, which hold the raw, clamped and unclamped chunks. Revised on 30 Sep after an
independent code review; see the corrections at the end of this section.

A seed counts as moving when shoulder_pan, shoulder_lift, elbow_flex or wrist_flex changes by more
than 5° between the first and last action of the chunk (29 steps, about 1 s). Wrist_roll is
excluded, because roll twitches alone were being counted as reaches.

| Instruction | Seeds moving | Mean movement of those seeds, pan / lift / elbow (°) |
|---|---|---|
| lemon (trained) | **77 / 128 (60%)** | −0.5 / +26.6 / −28.2 |
| apple | 36 / 128 (28%) | +1.7 / +21.5 / −19.3 |
| strawberry | 40 / 128 (31%) | −0.4 / +21.8 / −19.6 |
| peach | 26 / 128 (20%) | 0.0 / +20.2 / −16.5 |
| empty | 2 / 128 (2%) | too few to average |

Pan difference among moving seeds (bootstrap 95% CI): lemon − apple −2.2° (−2.9 to −1.5), apple −
strawberry +2.2° (1.5 to 2.9), apple − peach +1.7° (0.9 to 2.6), lemon − strawberry −0.1° (−0.3 to
0.2). The empty condition has too few movers for any comparison.

**The start pose sits outside the trained range, and the model never sees it.** Ai2's rest pose
has shoulder_lift and elbow_flex beyond the checkpoint's 99th percentile (arm frame: lift 3.9° and
elbow 8.3° outside the *state* range, 3.0° and 7.8° outside the slightly wider *action* range;
`check_state_range.py`). The preprocessor clips the normalised *state* to the
trained range (`molmoact2_clamp_normalized`) before the model sees it. So the model believes the
elbow is at the edge of the range, 7.8° from where it really is, and plans from there. The first
action therefore jumps about 8° (elbow 91.4° → 83.6°). "Holding still" means staying at that edge.
The postprocessor's clamp on the *actions*, which also clips to the trained range, barely matters.
Raw outputs go at most 1.7% past the bound, and clamping changes actions by at most 1.2°. Clamped
and unclamped readouts give the same moving counts (except peach, 26 vs 27) and the same pan
differences. The tables report the clamped values, which are what the robot would receive.

Reading:
- **The instruction mainly gates whether the arm moves.** 60% of seeds move for the trained object,
  20–31% for the other named objects, 2% for the empty instruction. So the model is not ignoring
  language, and without it the arm mostly stays put. E reports that π0.5 reaches for the
  training-task object even without language. The numbers are not comparable (one frame and 1 s
  here, whole LIBERO episodes there), but they point to a different profile. Repeat on the pen frame.
- **Target direction is at most weak within one chunk.** Moving seeds do broadly the same lift
  (+20° to +27° on shoulder_lift). Apple differs from the rest by about 2° of pan, and the CI excludes 0.
  But strawberry, on the same side of the table as apple, does not differ from lemon. So the small
  signal that exists does not follow the objects' bearings.
- Mean-based statistics (whole-chunk RMS, separation ratio, Cohen's d, still in the JSON) mostly
  measure how often the arm starts, so they mislead here. Use `movers` and `mover_direction`.
- **Consequence for the plan's precursor (Appendix E).** A frame with the arm at rest cannot tell
  "follows the named object" apart from "reaches the same way regardless". Also, starting outside
  the trained range makes the first action a correction back into range. Take the pen frame with
  the arm raised over the middle of the workspace, inside the trained range.

**Checked against Appendix E.** E's filter is: "if the two colour conditions produce chunks that are
closer to each other than either is to the null, language is doing nothing". This frame shows that
pattern, yet language clearly matters (2% → 20–60% moving). The pattern means language *gates
motion* but does not *select the target*, which is a Level-1 failure, not "no effect". The readout
adopted instead is directional (`mover_direction`).

**CAG, final-action variant, on the same frame** (`--cag-weights 1 1.5 2 3`, same 128 seeds;
a = a_null + w·(a_instr − a_null), mixed on the raw normalised output before the postprocessor).
At w = 1 the guided chunks equal the plain ones exactly.

| w | Seeds moving: lemon / apple / strawberry / peach | Lift of lemon movers | Pan diff apple − strawberry (95% CI) | Pan diff lemon − strawberry (95% CI) | Largest step within chunk | Jump from state to first action |
|---|---|---|---|---|---|---|
| 1 | 77 / 36 / 40 / 26 | +27° | +2.2° (1.5 to 2.9) | −0.1° (−0.3 to 0.2) | 3.5° | 7.9° |
| 1.5 | 79 / 39 / 42 / 31 | +39° | +2.9° (1.9 to 4.0) | −0.2° (−0.6 to 0.1) | 5.1° | 8.0° |
| 2 | 81 / 41 / 43 / 36 | +50° | +3.7° (2.3 to 5.1) | −0.4° (−0.9 to 0.0) | 6.7° | 8.2° |
| 3 | 89 / 48 / 47 / 42 | +69° | +4.7° (2.8 to 6.7) | −0.6° (−1.4 to 0.1) | 10.0° | 8.7° |

Reading: CAG amplifies what the instruction already changes, which on this frame is mainly *whether
and how far* the arm moves. Lift scales roughly with w. The one clear pan difference (apple −
strawberry) grows more slowly, about 2× at w = 3, and the near-zero one drifts from −0.1° to −0.6°
with its CI still touching 0. It does not create target selection. The steps grow with w. On this
frame the first action is already about 8° from the state, because of the out-of-range start. Any
arm test of CAG therefore keeps `--robot.max_relative_target` on.

**Corrections made on 30 Sep after review:**
- Wrist_roll was dropped from the moving test. The empty condition went from 9 to 2 movers, and
  peach from 35 to 26.
- The pan CIs quoted earlier came from an older run and have been replaced.
- "Pan grows in proportion to w" was wrong: it grows more slowly.
- An intermediate reading that "holding still is mostly the clamp" was wrong. The clamp changes
  actions by at most 1.2°.
- The earlier latency sanity check ("first action close to the current pose") was reading the edge
  of the trained range, not validating the calibration inversion.

## Model input format (Appendix G check)

Measured from the preprocessed batch, not assumed (`results/model_input_view.png`):

- Each camera frame is **squashed to 378×378**: `crop_mode="resize"`, bilinear, `antialias=False`.
  There is no crop and no letterbox, so nothing at the edges is lost. A 640×480 frame is scaled by
  0.59 horizontally and 0.79 vertically, so objects look 25% narrower relative to their height than
  in the camera image. A 16:9 source would be distorted differently from the 4:3 training rigs, so
  keep 640×480.
- 27×27 patches of 14 px, pooled 2×2 → **196 visual tokens per camera**, 392 for two cameras,
  against 97 text, state and special tokens (489 total).
- One visual token covers **47×36 px** of the 640×480 frame, and one 14-px patch covers 24×18 px. An
  attribute needs to span at least a patch to be represented reliably. Colour on a pen body likely
  survives, while a cap or a clip only a few pixels wide probably does not, and `antialias=False`
  makes thin details alias rather than blur. This is G's framing-scale concern, in numbers.
- The `224×224` shape in the checkpoint config is nominal: nothing in the LeRobot pipeline resizes
  to it.

## Rollouts on 2 Oct (median start pose, RTC, one wrist view)

Full analysis: `results/ANALYSIS_2026-10-02.md`. Per-run table: `results/runs.csv`. Labels so far are
**provisional machine labels** (gripper stalls plus a look at each contact sheet,
`results/labels_provisional.csv`), not the protocol's blind human labelling.

**Setup.** 20 recorded runs: one with a 4° cap, 19 with 6°. 12 single-pen runs and 8 two-pen runs.

**Control loop.**
- 29.7–29.9 Hz in every run.
- Inference median 377–463 ms per chunk; 11–14 steps dropped per chunk.
- The two runs with the live display were the slowest (≈ +40–60 ms per chunk, n = 2). Keep it off.

**Single pen.**
- The arm reaches the pen in every run, but only 2 of 12 runs grasped, late (22 s, 27.5 s).
- 35 empty closes in total.
- Empty closes happen at the same computed tip depth as the grasps. The pen probably sits beyond or
  below the fingertips, which a wrist view cannot show.

**Two pens:**

| Prompt | Runs | Outcome |
|---|---|---|
| "red pen" | 1 (red left) | went to red, grasped, lifted 8.7 cm |
| "green pen" | 2 (green left; green **right**) | went to green both times, grasped |
| "the pen" / "any colour" | 5 | 3 parked or hovered without approaching (all with green on the left); 2 went to green (on the right), one push |

So the named colour won 3 of 3, including once with the pen on the right. The n is tiny. Two
cases were never run: red prompt with red on the right, and green prompt with green on the left
(more than once).

**RTC plan consistency** (`plan_consistency.py`):
- The tick-to-chunk alignment comes from matching each tick's command to the chunk rows: 100% of
  ticks match.
- The guided steps (0–9) agree with the old plan to 0.02–0.24°, but they are the dropped ones.
- The executed, unguided overlap disagrees by 0.3–6° (2–20 mm at the tip).
- Hovering coincides with larger disagreement in all 8 runs that have both kinds of window
  (median ≈ 1.8×). Successive plans disagree, the command jumps at each switch, and the 6° cap clips
  the jumps.
- This confirms the chunking note above with data. Next: RTC A/B with `execution_horizon` 10 vs ≈ 20.

**Forward kinematics.** The tip computes 0.1–1.6 cm below the assumed table. The mat dents by a few
mm, so ≈ 1 cm is likely a calibration or mount offset. Check against table marks.

## Why chunks disagree while hovering (offline, 3 Oct)

`hover_causes.py`; full numbers in `results/hover_causes/SUMMARY.md`. Literature and LeRobot code notes
are in `CHUNK_CONSISTENCY_NOTES.md`.

**Setup.**
- Hover windows from 9 runs (63 consecutive-chunk pairs) and moving windows from 8 runs (56 pairs).
- Each chunk re-sampled offline from its recorded frame and state. The sampler is bit-exact against stock
  inference, with or without RTC.
- Metric: RMS disagreement in degrees over the executed overlap, arm joints 0–4 (plan_consistency's
  "unguided").

| Readout (median) | Hover | Moving |
|---|---|---|
| Measured on the arm | 5.4° | 2.2° |
| Offline with RTC h10, against the recorded previous chunk | 6.0° | 2.5° |
| Two seeds, same frame (noise only; ≈ 3.0° / 1.4° over the full overlap) | 2.7° | 1.0° |
| Same noise, consecutive frames (observation change only) | 7.4° | 3.4° |
| Fresh noise, consecutive frames | 7.7° | 3.6° |

**Reading.**
- **Observation sensitivity dominates; sampling noise is minor.** Holding the noise fixed barely lowers the
  disagreement (7.4 vs 7.7°). The fixed-noise change across frames exceeds the same-frame seed spread in
  9 of 9 hover runs.
- A frame only 0.1 s later already moves the plan by 3.7°.
- Hovering is the same mechanism, about 2× larger.
- wrist_roll carries the most. Its distribution is wide but mostly unimodal (bimodal at 7 of 72 hover
  frames). The whole distribution shifts from one observation to the next.

**Remedies, offline** (hover):

| Remedy | Effect | Cost |
|---|---|---|
| One noise sample per episode | none (7.8 vs 7.7°) | — |
| Lower noise temperature (0.7, 0.5) | none across frames | — |
| Best-of-16 closest to the previous plan | −27% (−34% on top of RTC h10); wrist_roll seam 10.0 → 4.6° | +68 ms per chunk; biases the plan by up to ≈ 2.3° (wrist_roll) |
| RTC `execution_horizon` 20 instead of 10 | 6.0 → 1.0°; wrist_roll seam 4.0 → 0.2° | commits to the old plan (new steps shift 3.2°); partly by construction, since it guides the measured steps |

Image and state changes are not yet separated.

**Code notes.**
- LeRobot's `execution_horizon` is where guidance ends. With an inference delay of 12–14 steps and h = 10,
  no executed step is guided (`modeling_rtc.py:256`, checked).
- `per_episode_seed` draws new noise every chunk, so it is not a fixed noise sample.
- The colleague's client sends no previous chunk and its seams have no guidance at all.

**Next on the arm:** A/B RTC h20 vs h10 on the same scenes, then best-of-16 on top.

## Offline prompt comparison on the two-pen runs (2 Oct)

`prompt_counterfactual.py`; full numbers in `results/prompt_counterfactual/SUMMARY.md`.

**Setup.**
- 8 two-pen rollouts (wrist view only), 4 per layout.
- 4 frames each, at +0/1/2/3 s.
- The saved frame and state, re-run with 5 prompts × the same 64 seeds, so only the words differ.
- Bit-exact against stock `predict_action_chunk` at batch 1; within 0.4° when the seeds are batched.
- "Colour effect" = pan of the "red pen" seeds minus pan of the "green pen" seeds, signed so that +
  means towards the red pen. Movers only; resampled over runs, then seeds.

**Results** (frames up to +1 s, before the arm commits):

| Readout | Value |
|---|---|
| Colour effect, all runs | **+6.0°** (95% CI −1.9 to +11.9), positive on 13/16 frames |
| Red pen on the left | +10.1° (8.5 to 12.1), 8/8 frames positive |
| Red pen on the right | +1.9° (−12.0 to +13.8); the spread is all from no_color_1/2 |
| First frame per run | positive in 6/8 runs (+3 to +18°); no_color_1 −3.1°, no_color_2 −20.8° |
| Neutral prompts ("any colour", "the pen", "") | drift +2 to +3° to the image right in both layouts, colour-blind |
| Seeds moving (≤ +1 s) | 86–96% for every prompt, "" included; on Ai2's frame "" moved 2% |
| The arm's own first chunk vs the offline seeds | 37th–91st percentile (median 59th) |

**Reading.**
- The pre-registered rule says **unclear**: the run-level CI includes 0.
- The percentile CIs are approximate, because there are only 4–8 runs. A t-interval over the per-run means is
  wider: about −3.6 to +15.6 overall, and 6.8 to 13.5 with the red pen on the left. The verdict is unchanged.
- In plain terms, the colour word picks the pen in 6 of 8 scenes, in both layouts.
- "Position dominates" is not supported.
- It is not robust across scenes: in the two earliest scenes the words did not select the pen.
- On the arm's own frames, swapping the colour word sends the median seed towards the other pen
  (red_1 with "green": −26 mm; green_2 with "red": +34 mm).
- green_2 had the green pen on the right, and the arm went right.

**Limits.**
- One wrist view.
- Pan only.
- 8 scenes.
- No RTC guidance offline.
- "" fills an unseen template.

Next: more counterbalanced scenes from one fixed start pose, especially with green on the left.

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
9. **Camera controls cannot be set "in the OpenCV camera config"** (Appendix G). LeRobot's
   `OpenCVCameraConfig` has no exposure, white-balance, gain or focus fields. They are locked out of
   band with `v4l2-ctl` (`lock_camera.sh`) and must be re-checked after the first lerobot command
   opens the camera.
10. **Appendix E's single-frame filter misreads the bimodal case.** See the frame-diagnostic
    section: "colours close to each other, far from null" means language gates motion without
    selecting a target. The pass condition is directional (the pan difference among moving seeds).
