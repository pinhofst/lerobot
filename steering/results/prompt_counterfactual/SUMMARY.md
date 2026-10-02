# Offline counterfactual prompts on the two-pen runs (2 Oct)

**Question.** With the wrist frame and joint state held fixed, does the colour word change where MolmoAct2 heads?
**Setup.** `steering/prompt_counterfactual.py`. 8 two-pen runs, 4 per layout: red pen on the left in red_1, green_2, no_color_3 and no_color_4; green on the left in no_color_1, no_color_2, green_1 and no_color_5. Frames at +0/1/2/3 s after the first policy observation (32 frames, both pens segmented; `contact_sheet.jpg`). 5 prompts ("pick up the red pen", "…green pen", "…a pen of any color", "…the pen", "") × the same 64 seeds. The 1-cam checkpoint is sampled on its RTC first-chunk path: batch 1 is bit-exact against stock `predict_action_chunk`, and the 16-seed batch differs by ≤0.4°. One model load, 4 min.
**Direction.** From the recordings, the image shifts by −6.7 px per degree of pan (CI −9.4 to −4.3), so +pan turns the wrist view to the right. "Colour effect" = (red − green) pan among moving seeds, signed so that + means towards the red pen. CIs resample runs, then seeds. With only 4–8 runs these percentile CIs are approximate and too narrow (review, 3 Oct): a t-interval over per-run means gives about −3.6 to +15.6 for the headline and 6.8 to 13.5 for red-left. The verdict is unchanged.

## Numbers (frames ≤ +1 s, before the arm commits; by +2–3 s the prompts converge)
- **(i) Colour-following.** Mean **+6.0° of pan towards the red pen (95% CI −1.9 to +11.9)**, positive on 13 of 16 frames. First frame only: +5.8° (−3.0 to +12.7), 6 of 8 runs.
  - First frame per run: red_1 +13.0, green_2 +12.8, no_color_3 +3.3, no_color_4 +8.8, green_1 +15.1, no_color_5 +17.5, **no_color_1 −3.1, no_color_2 −20.8**. Every per-frame CI excludes 0.
  - By layout: red-left **+10.1° (8.5 to 12.1)**, 8 of 8 frames positive; red-right +1.9° (−12.0 to +13.8), 5 of 8. The red-right spread comes entirely from no_color_1/2.
  - Image-right pan, red-left layout: "red" −6.0° (−9.5 to −3.5), i.e. left towards red; "green" +4.1° (2.0 to 6.3), right towards green. Red-right layout: "red" −0.2°, "green" −2.0°, both with wide CIs.
  - Tip by FK, movers: up to 45 mm laterally towards the named pen (green_1, no_color_5).
- **(ii) Position bias.** Neutral prompts drift slightly right in both layouts: "any" +3.4° (1.6 to 5.6), "the pen" +2.1° (0.5 to 4.1), "" +2.6° (0.9 to 4.6). Towards red: +0.7 to +1.1°, every CI spans 0. So there is a small learned rightward drift, unrelated to colour and well below the colour effect.
- **(iii) Gating.** Seeds moving: red 96%, green 95%, any 93%, pen 86%, "" 89%. On the first frame only: 92–94% for colour prompts, 76% for "the pen" and 80% for "". The words barely gate motion here, unlike on Ai2's frame, where "" moved 2%.

## The chunk the arm actually predicted (first frame, its own prompt; `on_arm.png`)
- The first on-arm chunk falls at the **37th–91st percentile (median 59th)** of the 64 offline seeds for its own prompt. Offline sampling on the saved JPEG frames reproduces the arm.
- Lateral tip motion towards the named pen, chunks 1–5:
  - red_1: +21, +25, +24, +12, +2 mm.
  - green_1: +44, +44, +23, +22, −12 mm.
  - green_2: +4, +5, +3, −5, +6 mm.
- Swapping the colour word on those same frames sends the median seed towards the other pen:
  - red_1 with "green": −26 mm, towards green.
  - green_1 with "red": +7 mm, towards red.
  - green_2 with "red": +34 mm, towards red.
- green_2 had the green pen on the **right**, and both the arm and the offline seeds headed right. So on the arm's own frames the word, not the side, picks the direction.

## Verdict
- **Pre-registered rule: "unclear".** It needs the run-resampled CI to exclude 0 and both layouts to be positive.
- **In plain language:** the colour word steers the target in 6 of 8 scenes, in both layouts, by 3–18° of pan, so it is not position alone. "Position/learned direction dominates" is not supported: the neutral drift is about 2–3°.
- **It is not robust across scenes.** In the two earliest scenes (no_color_1/2) the words do not select the pen. In no_color_2, "red" heads to the green pen and every other prompt heads to red. Those scenes had the arm 3–6° more elbow-extended and the pens closer and larger, with the green pen's body running between the jaws. This is a post-hoc observation, not a tested cause.

## Limits
- One wrist view.
- A 1 s chunk, read through pan only (no 3-D projection onto the pens).
- 8 scenes; frames within a run are correlated, so runs are the unit.
- JPEG (q85) frames, with the state taken from the next tick (≤33 ms).
- No RTC guidance offline, which matters for later chunks.
- The "" prompt is the out-of-distribution template.
- Pen position is the cap centroid.
- Next: more counterbalanced scenes at a fixed start pose, especially green-left.
