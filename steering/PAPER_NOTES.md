# MolmoAct2 paper: key insights for the SO-101 steering project

Source: Fang, Duan et al., "MolmoAct2: Action Reasoning Models for Real-World Deployment",
arXiv:2605.02881v2 (v1 4 May 2026, v2 8 May 2026), HTML version https://arxiv.org/html/2605.02881v2,
read in full including Appendices A to E. Fetched 2026-10-02.
Compared against: Ai2 blog https://allenai.org/blog/molmoact2 (5 May 2026, updated 28 May 2026)
and the model card https://huggingface.co/allenai/MolmoAct2-SO100_101 (its README, `norm_stats.json`,
`processor_config.json`, `chat_template.jinja`, `modeling_molmoact2.py`).

Tags used below:
- [paper] the arXiv paper. Section, table and figure numbers are from the v2 HTML.
- [card] the Hugging Face model card or files in that repo.
- [blog] the Ai2 blog.
- [ours] our own analysis. One analysis is of the released dataset manifest
  `allenai/MolmoAct2-SO100_101-Dataset` (1,220 repos; `meta/info.json` fetched for 1,205 of them,
  plus all 1,220 `tasks_annotated.parquet` files). The other is of the released model code.
  The HTML has no page numbers, so references are to sections, tables and figures.

## What matters for us

- Wrist-only input is rare in training. About 11% of SO-100/101 episodes have a single camera
  (Table 23: 230 datasets, 4,159 episodes). The paper never says which camera that is. In the
  released manifest, only 39 single-camera datasets (630 episodes, about 1.7%) have a camera
  explicitly named wrist, gripper or hand [ours]. The paper evaluates only with "wrist and a
  single allocentric view" (Sec. 6.2). So one camera is in distribution, but wrist-only is a
  thin edge of it.
- Camera order was randomised per episode for the SO-100/101 fine-tune, "rather than imposing a
  fixed view naming convention" (Sec. 4.3.1). Views are serialised as "Image 1", "Image 2"
  and so on, with no slots and no padding (App. A.1). The card says "random camera order is
  acceptable". So the cam0/cam1 slots in the LeRobot packaging carry no meaning.
- Language reaches the action expert only through the VLM's per-layer keys and values. Each of
  the 36 expert blocks cross-attends to the projected K/V of the matching VLM layer (Sec. 4.2.1,
  Eq. 6-7, Fig. 4). There is no other path for text. In the released code, the image tokens come
  before the text, and attention is causal. So the image-position K/V do not
  depend on the instruction. Only the text-position K/V do [ours, code].
- The instruction is a free-form sentence written by Qwen3.5-27B, roughly 5 to 25 words long.
  The relabel prompt says "Pay attention to the color and position of the objects" (Sec. 3.4,
  App. D.2). Colour words are everywhere: 75.5% of SO episodes contain one, and "red/blue/
  yellow pen" all occur [ours]. Negative instructions are essentially absent (0.09%, all
  incidental) [ours]. The paper never mentions negation.
- The paper has no test of target selection among same-category distractors on SO-100. It has
  no counterfactual-instruction test and no no-language ablation. The only language test is
  rephrasing, on fine-tuned bimanual YAM (Table 9). Our finding that the instruction "gates
  motion, not target" is therefore not contradicted by anything in the paper.
- State and action are normalised with 1-99 percentile statistics, then each state value is
  "uniformly discretized into one of 256 state tokens" (Sec. 4.1.2). A value outside q01-q99
  saturates at the end bins. The paper says nothing about calibration, start poses or joint
  ranges. The card's stats put the median at about 124 degrees for shoulder_lift and elbow_flex
  (`norm_stats.json`), which hints at a calibration convention that differs from current LeRobot
  [ours, to verify].
- The SO-100/101 checkpoint outputs 30-step chunks (30 Hz, 1 s) from 10 Euler flow steps. The
  chunks are run open-loop and the seams between them are not smoothed (App. A.2, App. E).
  Depth reasoning is disabled for this checkpoint [card].
- The quoted "179 ms" is derived from the paper: 10 / 55.79 Hz, at horizon 10 on LIBERO with one
  H100 (Sec. 6.8, Fig. 8). The blog says "about 180 ms". No latency is given for the 30-step
  SO checkpoint.

## Q1. Camera setups in the SO-100/101 training mixture

Paper facts:
- Table 23 (App. D.1) gives camera count per dataset (datasets / episodes / frames):
  1 cam: 230 / 4,159 / 2.26M. 2 cams: 835 / 24,151 / 13.05M. 3 cams: 151 / 9,235 / 4.11M.
  4 cams: 6 / 514 / 342.1K. The caption says "Most demonstrations are recorded at 30 Hz with two
  camera streams". By episodes: 1 cam 10.9%, 2 cams 63.5%, 3 cams 24.3%, 4 cams 1.4%.
- How often a wrist camera appears: not stated in the paper. The relabel prompt assumes one may
  exist: "only say the arm is grasping an object if this is shown in the wrist camera when the
  gripper is closed" (App. D.2).
- Handling: "Multi-camera robot episodes are serialized as multiple image inputs, and the camera
  order is randomized at the episode level during pre-training" (App. A.1). "Multi-camera
  observations are identified by image-index text" (App. A.1). Each camera is one image, so a
  1-camera episode is 1 image and a 3-camera episode is 3 images. Padding to a fixed count,
  dropping cameras and random camera subsets are not stated. Nothing suggests any of them.
- Randomisation: pre-training randomises camera order "to prevent the model from relying on a
  fixed camera slot" (Sec. 4.1.2). The SO-100/101 fine-tune also randomises it at the episode
  level (Sec. 4.3.1). Camera count was not randomised; it follows each dataset.
- Contrast with other embodiments: YAM uses a fixed order of top, left, right. DROID uses a fixed
  exterior-then-wrist order, and one variant samples one of the two exterior cameras
  (Sec. 4.3.1). "Other evaluation fine-tunes" use "fixed metadata camera order".
- MolmoAct2-Think depth targets come from "the first policy view" (Sec. 5.1). That does not apply
  to us.

From the released manifest [ours]:
- The camera-count distribution is 226 / 824 / 149 / 6 datasets over 1,205 repos, which matches
  Table 23.
- Camera names are mostly LeRobot defaults: "laptop" in 541 repos, "phone" in 491, "wrist" in
  162, then side, above, top, front. The most common pair is ("laptop", "phone"), in 368 repos.
  Those names do not say where the camera was mounted.
- 286 repos (38% of episodes) have at least one camera explicitly named wrist, gripper or hand.
  That is a lower bound for wrist presence.
- Single-camera repos are mostly "laptop" (104 repos, 1,440 episodes). "wrist" accounts for 26
  repos and 415 episodes. Explicit wrist-only covers 39 repos and 630 episodes (1.7% of all).
  Some "laptop", "phone" or "arm" single cameras may also be wrist-mounted; we cannot tell from
  the names.
- Native resolution is 480x640 for about 96% of camera streams.

Is wrist-only input in distribution? A single image is in distribution: about 11% of episodes.
Wrist-only is at most a few percent of episodes, and the paper never evaluates it. Every
reported SO-100 result used wrist plus a third-person camera (Sec. 6.2). Our observation that
one view "moves less often" fits this.

## Q2. Image preprocessing

- [paper] "each camera observation is represented as a single resized crop rather than a tiled
  high-resolution image" (App. A.1, also Sec. 4.1.2 and 4.3.1). Table 15 lists the image
  encoder as SigLIP2 with image size 384x384, patch 14, 27 layers and 380M parameters. The
  connector pools 2x2 patches for images and 3x3 for video, using features from the third-to-last
  and ninth-from-last ViT layers (App. A.1).
- [paper] Whether the resize keeps or distorts the aspect ratio is not stated.
- [card] `processor_config.json` has `crop_mode: "resize"`, size 378x378 (27 x 14), mean and std
  0.5, `pooling_size [2,2]`. That agrees with our measured 378x378 squash. With 480x640
  sources, the image is stretched non-uniformly in training too, so the squash is in
  distribution.
- Number of image tokens: not stated in the paper. From the code [ours], 27x27 patches, padded
  and pooled 2x2, give 14x14 = 196 visual tokens per view, plus image start/end markers
  (`use_single_crop_col_tokens: false`).
- [paper] Training-time augmentation, applied to all images and turned off at inference
  (App. B.1): RandomCrop to 95% then resize back, RandomRotation of 5 degrees, ColorJitter
  (brightness 0.2, contrast and saturation 0.8-1.2, hue 0.05), and GaussianBlur with p=0.2.
  Note the hue jitter of 0.05: colour identity is mostly preserved but not exactly.

## Q3. The SO-100/101 data and its language

- [paper] Size: "1,222 public LeRobot datasets contributed by 377 users", "38,059 robot
  demonstration episodes, 19.8M frames, and approximately 184 hours" (Sec. 3.2, Table 21).
  1,660 entries were collected and TOPReward removed 438 of them (App. D.1). By robot type
  (Table 22): SO-100 has 921 datasets and 31,101 episodes (141.7 h). SO-101 has 299 datasets and
  6,898 episodes (41.6 h). MOSS has 2 datasets. So SO-101 is only about 18% of episodes, and
  every reported evaluation used an SO-100.
- [paper] Frame rate: 30 Hz for 1,114 datasets. Some use 20, 25 or 60 Hz (Table 23). All final
  datasets have 6-D action and 6-D state.
- [paper] Filtering, in four stages: "(i) structural validity checks ..., (ii) removal of
  eval-style datasets, (iii) license/codebase eligibility checks, and (iv) a final TOPReward
  quality gate" (Sec. 3.2). The gate keeps a dataset if the "mean TOPReward over the last 3
  sampled episodes" is above a threshold set from human-audited good datasets.
- [paper] Re-annotation is done by Qwen3.5-27B, an open VLM (Sec. 3.4). It gets 12 equally
  spaced frames from each non-frozen camera plus the original instruction, at temperature 0.1
  (App. D.2). The length target is "randomly sampled from a right tailed power distribution,
  with a minimum value of 5 and a maximum value of 25" words.
- [paper] Key prompt phrases: "generate a clear imperative instruction describing all of the
  actions performed by the robot arm", "Pay attention to the color and position of the
  objects", "Start directly with an action verb".
- [paper] No fixed templates are used. Diversity comes from the random word target. Unique
  SO instructions rise from 707 (1.5%) to 16,205 (34%) in Table 24. Table 21 gives 12,818
  unique (33.7%) instead, an internal inconsistency in the paper.
- [paper] Fig. 13 shows the verb shift after relabelling. "put" almost disappears. "grasp",
  "place" and "move" dominate, about 2-3 x 10^4 episodes each.
- [paper] Example instructions in Fig. 14: "Move the arm to grasp the red battery, lift it, and
  place it into the black bin"; "Pick yellow nut, place in blue cup. Pick gray nut, place in
  green cup." The card sample is "Move the arm towards the lemon, grasp it, lift it up, and
  drop it into the red bowl."
- Colour and attribute words: yes, often. In the released annotations [ours], 75.5% of 37,459
  episodes contain a basic colour word. The most common are blue (8.3k), red (7.8k), green
  (6.6k) and yellow (4.9k). "pen" appears in 707 episodes across 34 repos. Colour-plus-pen
  phrases appear: yellow pen 92, blue pen 64, black pen 24, white pen 16, red pen 12.
  140 repos use the same noun with two or more colours, e.g. "blue cubes to the blue dish and
  red cubes to the red dish". The annotator was told to mention colour, so many colour words
  describe a lone object rather than pick one out from look-alikes.
- Negative instructions: not stated in the paper. In the annotations [ours], 33 episodes (0.09%)
  match not, avoid or without, and all are incidental: "without resetting", "avoid knocking
  over other objects", "ensuring not to damage the connector". There are none of the form
  "pick up the pen that is not red".
- [card] At inference, `normalize_language=True` lowercases the task and strips trailing
  punctuation "to match training preprocessing". App. B.1 says raw task strings are normalised
  and embodiment details are moved into the setup field.

## Q4. Zero-shot evaluation on SO-100/101 and other arms

- [paper] SO-100 setup (Sec. 6.2): "we use the SO-100 robot, retaining the wrist camera and
  adding a third-person external camera with a randomly initialized position". Objects are
  novel and environments are out of distribution. Five pick-and-place tasks, 15 trials each,
  "partial credit awarded for near-successful executions".
- [paper] Scoring for SO-100 (Table 7, Table 20): 0.25 for reach, 0.5 for pickup, 1.0 for
  successful placement, averaged over 15 trials.
- [paper] SO-100 results (Table 7 = Table 20), listed as SmolVLA / pi0-SO100/101 / MolmoAct2:
  Fork on plate 3.3 / 30.0 / 70.0. Stack blocks 5.0 / 6.7 / 20.0. Tissues in basket 0.0 / 20.0 /
  73.3. Pen on notebook 3.3 / 80.0 / 86.7. Block in box 0.0 / 90.0 / 33.3. Average 2.3 / 45.3 /
  56.7. The pi0 baseline is one the authors fine-tuned on their own SO mixture; Table 20 calls
  it "DePi 0".
- [paper] Fig. 12 shows the SO-100 scenes from a third-person side view. Each has a single
  target object, e.g. a yellow pen and a notebook. We see no same-category distractors.
- Camera pose contradiction inside the paper. Sec. 6.2 says "camera poses are randomly
  initialized without conforming to any dataset visual distribution". App. C.4 says "using a
  fixed, pre-initialized camera viewpoint", and Table 20's caption says "fixed initial camera
  position". The most consistent reading is that the external camera was placed once at an
  arbitrary pose and then held fixed for all trials and models. That reading is our
  interpretation; the paper does not say it. For DROID, App. C.2 also says "Camera positions are
  held fixed across all three models".
- [paper] DROID real-world zero-shot, wrist plus one exterior camera, 5 tasks x 15 trials
  (3 positions x 5) (Table 6, Table 18): MolmoAct2-DROID 87.1% average, MolmoBot 48.4%,
  pi0.5-DROID 45.2%. One task is "Put the red cube inside the tape roll", at 93.3%.
- [paper] Simulation zero-shot with DROID: MolmoSpaces average 37.7 vs pi0.5-DROID 34.5
  (Table 4). MolmoBot benchmark 20.6 vs 10.0 (Table 5). The MolmoBot set includes "PnP Color"
  (17.2 oracle / 8.8 at end) and "Pick Rand.-Cam." (15.4).

## Q5. Language-following and grounding evidence

- [paper] Rephrasing (Sec. 6.5, Table 9): bimanual YAM, with a separate fine-tune per task on 4
  tasks, "we rephrase the language instruction in three different ways". The Language column
  reads 60.35 for MolmoAct2 (the row is labelled MolmoAct2-Think in the table), 51.25 for
  OpenVLA-OFT and 26.15 for pi0.5. Trials: "20 trials, 5 trials for each perturbation" per
  task. The wording is ambiguous.
- [paper] Distractors (Table 9): "add unseen distractor objects while keeping the original
  in-distribution spatial layout". 54.10 vs 48.30 for OpenVLA-OFT, the narrowest margin
  ("its advantage is narrowest on Distractor"). These are fine-tuned single-task policies, not
  zero-shot, and the distractors are not same-category look-alikes.
- [blog] The blog names "rephrased instructions, shifted object positions, distractor objects in
  the scene, and object substitutions" but gives no numbers. Object substitution does not
  appear in the paper's Table 9.
- Counterfactual instructions (same scene, different target): not stated in the paper.
- Ablations without language or with shuffled language: not stated. Some fine-tunes use no
  language annotations (LIBERO and "other evaluation fine-tunes", Sec. 4.3.1), but that is
  about relabelling, not removing the instruction.
- Indirect grounding evidence: the backbone is trained on referring and pointing (RefSpatial,
  RoboPoint), on CLEVR "compositional attribute-relation reasoning" and on GRiD-3D frames of
  reference (Sec. 2.1). Molmo2-ER improves LIBERO-Long with discrete actions from 77.6% to
  83.6% over Molmo2 (Table 10). Those are VLM-level results. None of them shows that the
  action expert selects a target from the instruction.

## Q6. Action and state representation

- [paper] Control: absolute joint pose for SO-100/101 (Sec. 4.3.1, Table 2). The prompt carries
  `<control_start>absolute joint pose<control_end>`. [card] The setup string is "single
  so100/so101 robotic arm in molmoact2".
- [paper] Normalisation: "continuous action and state dimensions are normalized with 1-99
  percentile statistics"; grippers "are treated separately ... when they are represented as
  binary or narrow-range" (Sec. 4.1.2). [card] `norm_mode: q01_q99`, `normalize_gripper: true`.
- [paper] State goes into the prompt as discrete tokens, one of `<state_0>` to `<state_255>` per
  dimension, "appended to the prompt before the action target" (Sec. 4.1.2, App. A.1). There is
  no continuous state input. The expert sees state only through the K/V of these tokens.
- [paper] Actions are padded to 32 dimensions (Single-Arm layout `[A1..An, G1, 0, ...]`,
  App. B.2). The expert has max horizon 30 (Table 15). The SO-100/101 checkpoint uses "a
  30-step action chunk for the 30 Hz control rate" (Sec. 4.3.1). [card] `action_horizon 30`,
  `n_action_steps 30`.
- [paper] Flow matching uses a rectified-flow interpolation `x_t = (1-t) eps + t a`, target
  `a - eps` (Eq. 1). Inference uses Euler steps; "The released checkpoints use N=10 inference
  steps" (App. A.2). Training uses K=4 flow samples per chunk in post-training and K=8 in
  fine-tuning.
- [paper] There are two modes. Discrete mode decodes FAST tokens (2048-token vocabulary, 1 s of
  32-D action). Continuous mode uses the flow expert. Continuous is the default: discrete is
  "3.94x" slower (Sec. 6.8). [card] Discrete mode is "exposed for parity and debugging" and
  needs `allenai/MolmoAct2-FAST-Tokenizer`.
- [paper] Both heads are co-trained in fine-tuning (L_LM + L_flow, Eq. 9). The ground-truth
  discrete action span is masked from the expert (Sec. 4.2.2).
- [paper] Depth: MolmoAct2-Think predicts a 10x10 grid of 128-way VQ depth codes and
  regenerates only the cells whose RGB changes (Sec. 5). It was fine-tuned only for LIBERO in
  the paper. [card] "Depth reasoning is disabled for this checkpoint".

## Q7. Inference latency and hardware

- [paper] Sec. 6.8 and Fig. 8: "end-to-end action-generation latency on LIBERO using a single
  H100 GPU and an action horizon of 10". The metric is "amortized control rate as action
  horizon divided by latency".
- [paper] Continuous path: 23.02 Hz original, 27.39 Hz with caching, 55.79 Hz with CUDA Graphs.
  As latency per chunk, that is about 434, 365 and 179 ms. The 179 ms figure is our
  arithmetic; the paper reports Hz.
- [paper] Think (with depth): 8.04, 9.72 and 12.71 Hz, so about 787 ms. Discrete path: 14.17 Hz,
  about 706 ms.
- "Horizon 10" means the LIBERO checkpoint's 10-step chunk at 10 Hz (Sec. 4.3.1), so it is one
  chunk per call. The SO checkpoint emits 30 steps per call. Its latency is not stated, but the
  expert cost grows with chunk length while the VLM prefill does not.
- [paper] App. E: "the 55.79 Hz number in Sec 6.8 is amortized chunk throughput, not closed-loop
  reactivity".
- [blog] "about 180 ms in the base model and 790 ms in MolmoAct 2 with adaptive depth
  reasoning, versus 6,700 ms in MolmoAct (running in the LIBERO benchmark environment with 1
  NVIDIA H100)".
- [card] Ai2's reported experiments used float32 inference, about 26 GB with CUDA graphs. bf16
  fits under 16 GB and "usually does not hurt performance much". The paper does not state the
  precision used for Sec. 6.8.

## Q8. The backbone (Molmo2-ER)

- [paper] Molmo2-ER is fine-tuned from the "Molmo2-4B mid-training checkpoint" (Sec. 2.2),
  citing the Qwen3 technical report. Table 15 lists the LLM as 4.0B, 36 layers, width 2560,
  32 heads, 8 KV heads and vocabulary 151,936. The vision encoder is SigLIP2 (Sec. 4.1.2).
  The string "Qwen3-4B" never appears in the paper, but the specs match Qwen3-4B.
- [paper] Training: 3.3M embodied-reasoning samples, "specialize-then-rehearse". Stage 1 is
  20K steps; stage 2 is 1.5K steps at a 50/50 embodied/general mix (Sec. 2.2).
- [paper] Results (Table 3, Sec. 6.1): average 63.8 over 13 benchmarks. The runner-up is GR-ER
  1.5 Thinking at 61.3 ("+2.5 points"). The base Molmo2 scores 46.8 ("17 points").
- [paper] "9 of 13": Sec. 2 says it "outperforms every open-weight baseline as well as the
  strongest closed-source models, including Gemini Robot-ER 1.5 Thinking and GPT-5, on 9 of
  13". Recounting Table 3 [ours]:
  - vs GR-ER 1.5 Thinking: wins 9 of 13.
  - vs GPT-5: wins 8 of 13 (loses ERQA, EmbSpatial, MindCube, SAT, OpenEQA).
  - vs GR-ER 1.5 non-thinking: wins 11 of 13.
  - Best of all open-weight models: 9 of 13, which is what the bold marks in Table 3 show.
  - Best of every model in the table: only 5 of 13 (Point-Bench, RefSpatial tie, BLINK,
    CV-Bench, VSI-Bench). Qwen3-VL-4B beats it on RoboSpatial-Point (62.3 vs 32.0) and
    Where2Place (63.0 vs 54.0).
- [paper] Effect on actions: Molmo2-ER vs Molmo2 on LIBERO-Long with discrete actions only is
  83.6% vs 77.6% (Table 10).

## Q9. Start poses, joint ranges and calibration

- Start poses, home or rest pose, initial joint distributions: not stated in the paper.
- Calibration convention (degrees vs range_m100_100, the LeRobot version of the community
  data): not stated in the paper. The paper only says actions and states are "6-D" (Table 23)
  and normalised with q01-q99. The card says "`state` is the raw robot state, and actions are
  returned in robot scale".
- [card] `norm_stats.json`, state q01 / q50 / q99 per joint:
  - shoulder_pan: -41.9 / 3.1 / 48.3
  - shoulder_lift: 43.7 / 123.2 / 185.3
  - elbow_flex: 38.4 / 124.4 / 173.1
  - wrist_flex: 5.7 / 57.9 / 91.8
  - wrist_roll: -63.5 / -11.0 / 42.9
  - gripper: 0.9 / 9.2 / 44.1
  - The raw min/max spans about -270 to +270 on several joints.
- [card] The sample state is [-0.53, 189.14, 181.41, 60.64, -3.60, 1.10], from
  `Beegbrain/pick_lemon_and_drop_in_bowl` frame 0.
- [ours, hypothesis] Medians near 120-125 degrees for shoulder_lift and elbow_flex, and the
  +/-270 tails, suggest the corpus mixes calibration conventions. The bulk looks like the older
  LeRobot SO-100 convention rather than the current zero-centred one. If our calibration
  differs, the clipped rest pose may be a convention offset, not just an unusual pose. Check
  this by converting a known pose, e.g. the card sample, between conventions before any
  steering work.
- Mechanism [paper]: after q01-q99 scaling, state is binned into 256 tokens, so values outside
  the range collapse to the end tokens. Whether the code clips before binning is not stated in
  the paper. We measured the clipping ourselves.

## Q10. Stated limitations and failure modes

- [paper] App. E, open-loop chunks: "motion smoothness across chunk boundaries is not enforced";
  "visible velocity or acceleration discontinuities at the seams"; the policy "cannot react
  within-chunk to perturbations".
- [paper] App. E, embodiment scope: zero-shot only on YAM, SO-100/101 and DROID Franka.
  "MolmoAct2 is not a universal controller."
- [paper] Sec. 6.5: weakest on spatial variation (26.25%), "indicating room for improvement on
  fine-grained spatial generalization". Distractors give the narrowest margin.
- [paper] Table 4: articulated "Open" is weak (9.5 vs 22.7 for pi0.5). Table 5 caption: the gap
  between oracle success and success at the end reflects "the inability of policies to
  determine when a specified task is already completed, e.g. by repeatedly picking up an object
  which has already been placed correctly".
- [paper] SO-100 per-task weaknesses (Table 7): Stack blocks at 20.0, and Block in box at 33.3
  vs 90.0 for pi0.
- [blog] Same two limitations as App. E, plus "transitions between batches can look jerky".
- [card] Validate outputs before running on hardware; bound actions by speed, workspace and
  torque limits.

## Q11. Internal representations and where language enters the action head

What the paper says:
- Conditioning: "Each action-expert block cross-attends to the corresponding VLM layer's keys and
  values, after lightweight learned projections" (Sec. 4.2.1). The expert has 36 blocks to match
  36 VLM layers. Each block runs self-attention over the action chunk, then cross-attention to
  the VLM K/V, then an MLP. Each branch is modulated by time through AdaRMS shift, scale and
  gate (Eq. 3-5).
- Shapes: the VLM has "8 KV heads with head dimension 128", so 1024-D per token. That is
  projected to the expert width of 768, split into 8 heads of 96 (App. A.2).
- What the K/V cover: "the prompt, images, state tokens, and non-target robot text" (App. A.2).
  The context "contains the task, visual observations, setup/control descriptors, and discrete
  state tokens" (Sec. 4.2.1). The discrete action-target span is masked out.
- Cross-attention vs hidden-state conditioning (Table 11, LIBERO): per-layer KV 95.9%, per-head
  per-layer KV 94.8%, final hidden state 94.0%. The paper's argument is that this "exposes the
  attention state used by the VLM itself" (Sec. 1).
- Gradients: post-training uses knowledge insulation (K/V detached, Sec. 4.2.2). Fine-tuning,
  which produced our SO checkpoint, does not: "gradients from the flow loss are allowed to
  update the VLM through the action-expert conditioning path" (Sec. 4.3.1). So the SO
  checkpoint's VLM K/V were shaped partly by the action loss.
- Fine-tuning tunes the LM head and final norm but "do[es] not tune the added-token input
  embeddings" (Sec. 4.3.1).
- Inference: within a chunk, the VLM context is fixed across the 10 flow steps. The projected
  cross-attention K/V are cached and reused (Sec. 4.3.2).
- Probing, attention maps, representation analyses, interpretability of language: none in the
  paper. Its interpretability claim is limited to the predicted depth tokens of MolmoAct2-Think.

From the released code [ours]:
- `chat_template.jinja` puts all images first ("Image 1<|image|>Image 2<|image|>...") and the
  text after them.
- The prompt text (`modeling_molmoact2.py`, around line 1341) is: "The task is to {task}. The
  setup is {setup}. The current state of the robot is {state tokens}. The expected control mode
  is {control}. Given these, what action should the robot take to complete the task?", followed
  by the assistant prefix and `<action_output>`.
- The LM is causal. The paper says visual tokens "forward-attend to one another" (App. A.1). In
  the HF code, that image-to-image bidirectional mask (`token_type_ids_mask_function`) is added
  in the `generate` mask path. The plain forward seems to build a standard causal mask. We have
  not checked which one the action path uses. Either way, no image token can see text that
  comes after it.
- Consequence: the K/V at image positions, in every layer, do not depend on the instruction. The
  instruction can only affect the expert in two ways:
  1. the K/V of the instruction tokens themselves;
  2. the K/V of every later text token (setup, state, control, question, assistant prefix,
     `<action_output>`), which attend to both image and instruction.
- The expert takes the VLM `past_key_values` from one prefill: post-RoPE, post-k_norm keys and
  values for all valid positions. It applies `context_k_proj` and `context_v_proj` (one pair of
  linear layers shared by all 36 layers), then `context_norm`, then each block's own k_norm.
- Local LeRobot code: `src/lerobot/policies/molmoact2/modeling_molmoact2.py`. The KV is
  extracted near line 1816 (`_extract_kv_states`), and the per-layer projection is near line
  1458.

Implications for steering hooks [ours]:
- The natural hook is the VLM residual stream at text positions from the instruction onward.
  An edit at layer l changes the K/V of layers l+1 to 36 at those positions, and the expert
  reads every layer. Because image K/V are fixed by the image alone, the choice of target
  object has to be carried by the text-position K/V, or formed inside the expert's
  cross-attention.
- A cheaper, expert-only hook: edit the cached per-layer K/V, or the projected K/V, before the
  flow loop. The VLM runs once, and the edit applies to all 10 flow steps.
- For negative constraints, attention masking is a mechanistic alternative. Down-weight the
  expert's cross-attention to the image tokens covering the forbidden object (14x14 tokens per
  view).
- For readout, the discrete head decodes the same VLM state, and the paper shows it works
  (Table 13). Both heads can be read to check whether a steer changed the VLM's "intent" or
  only the expert.

## Claims in our plan, checked

| Claim | Verdict | Evidence |
|---|---|---|
| SO-100/101 zero-shot eval used "randomly initialised camera poses" | Contradicted within the paper | Sec. 6.2 says "randomly initialized"; App. C.4 and Table 20 say "fixed, pre-initialized camera viewpoint". Likely an arbitrary pose, then held fixed. |
| 179 ms per chunk on an H100 at horizon 10 | Confirmed (derived) | 10 / 55.79 Hz = 179 ms (Sec. 6.8, Fig. 8). LIBERO 10-step chunk, with CUDA Graphs. Blog: "about 180 ms". Not measured for the 30-step SO checkpoint. |
| Backbone Molmo2-ER = Qwen3-4B + SigLIP2 | Confirmed in substance | SigLIP2 named (Sec. 4.1.2). LLM is "Molmo2-4B" with Qwen3 cited and Table 15 specs that match Qwen3-4B. "Qwen3-4B" is never written out. |
| 63.8% average over 13 embodied-reasoning benchmarks | Confirmed | Table 3, Sec. 6.1. |
| Beats GPT-5 / Gemini Robotics-ER 1.5 on 9 of 13 | Partly confirmed | 9/13 vs GR-ER 1.5 Thinking and vs all open-weight models. Only 8/13 vs GPT-5. Best overall on 5/13. |
| Checkpoint packaged for two views (cam0, cam1) | Not found in paper | The paper uses a variable number of views in random order. Two views is a LeRobot packaging choice. The card's example uses two. |
| One view runs but moves less | Consistent, not tested | One camera is 11% of episodes; wrist-only is at most a few percent; the eval always used two cameras. |
| Instruction gates motion, not target | Not found | No counterfactual or same-category distractor test in the paper. |
| Image preprocessing is a 378x378 squash | Confirmed (card) | Paper: "single resized crop", 384 in Table 15. Card: `crop_mode resize`, 378x378. |
| Rest pose outside the 1st-99th percentile state stats, clipped | Mechanism confirmed | 1-99 percentile normalisation then 256-bin state tokens (Sec. 4.1.2). Start poses and calibration are not stated. |
