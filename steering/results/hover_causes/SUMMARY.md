# Hover chunk disagreement: noise or observation? (offline, 3 Oct 2026)

`hover_causes.py`; outputs in `results/hover_causes/` (results.json, causes.png, remedies.png,
joints.png, wroll_hist.png). 1-cam checkpoint, RTC config as on the arm, sampler bit-exact against
stock `predict_action_chunk` (no prefix, RTC h10, RTC h20; batch 1). GPU time about 12 min.

**Data.** Hover windows from the 9 runs that have them, one window per run: 8 consecutive executed
chunks from the middle of the longest hover segment (63 pairs). "Other" moving windows from 8 of the
same runs (56 pairs). Each chunk's observation is the recorded wrist frame nearest its `t_start`
(≤ 0.06 s), with the state of the first tick after it, and the run's own task string.

**Metric.** D = RMS over steps j in [10, 30−q) and arm joints 0–4 of chunk_k[j] − chunk_p[j+q], in
degrees, arm frame. This is plan_consistency's unguided arm RMS. Medians over pairs; 64 seeds per
frame (32 for the variants, 16 for RTC).

## Causes (hover / other)

| Readout | Hover | Other |
|---|---|---|
| Measured on the arm | 5.4 | 2.2 |
| Offline RTC h10 on frame k vs the recorded chunk p | 6.0 (arm value at the 37th percentile) | 2.5 (47th) |
| Offline unguided vs the recorded chunk p | 7.5 | 3.1 |
| **H1:** two seeds, same frame (unguided / RTC h10) | 2.7 / 2.7 | 1.0 / 1.2 |
| H1, fair baseline: seed spread over both chunks' overlap steps (review, 3 Oct) | ≈ 3.0 | ≈ 1.4 |
| Fresh noise, frame p → k, unguided | 7.7 | 3.6 |
| Same seed, index-aligned | 7.8 | 3.5 |
| **H2:** same noise, tick-aligned (x0 shifted by q) | 7.4 | 3.4 |
| H2 control: same noise, frame k vs the frame 0.1 s later | 3.7 (seed spread on the same steps: 2.7) | — |

**Verdict: H2 (observation sensitivity) dominates. H1 is a minor part.**
- Holding the noise fixed removes almost nothing (7.4 vs 7.7). The same x0 does not give the same sample on a new frame, though: on a frame 0.1 s later it already differs by 3.1°. So the fairer statement is the noise-only baseline (about 3.0° of the 7.7° in hover), not a quadrature split.
- Fixed-noise temporal change is larger than same-frame seed spread in 9/9 hover runs
  (4.8–14.8° vs 2.4–3.3°).
- The offline single step reproduces the arm's disagreement, so it is in the policy, not in
  execution.
- Hover is not a different mechanism: both terms are larger than in moving windows (about 2.1× for the noise baseline, 2.2× across frames).
- Not separated: image vs state change, or genuine replanning after the arm deviated from its plan.
  The 0.1 s control suggests at least part of it is sensitivity.

**Joints (hover, measured / seed spread / fixed noise):**

| Joint | Measured | Seed spread | Fixed noise |
|---|---|---|---|
| pan | 2.4 | 1.2 | 3.3 |
| lift | 4.2 | 1.8 | 5.1 |
| elbow | 3.3 | 1.6 | 3.6 |
| wflex | 3.8 | 1.9 | 4.4 |
| wroll | **4.8** | **2.8** | **7.0** |

**Multimodality:**
- The wrist_roll endpoint is bimodal (ΔBIC > 10, both weights > 0.1, separation > 2 SD) at only
  7/72 hover frames (mode gap about 7.6°), against 1/64 other frames.
- Its spread is wide: SD 3.9° vs 1.1°.
- What jumps is the whole per-frame distribution between consecutive observations (cap6_1: chunk 30
  spans −66..−58°, chunk 31 spans −42..−22°).

## Remedies (hover; other in brackets)

**(a) One noise sample per episode:** 7.8 vs 7.7 fresh [3.5 vs 3.6]. No gain. LeRobot's
`per_episode_seed` is a per-episode generator stream, not a fixed sample.

**(b) Temperature 0.7 / 0.5:** 7.4 / 7.7 [3.6 / 3.4]. No gain across frames.
- Same-frame spread falls 2.7 → 1.8 → 1.2.
- Plan-mean shift 0.35 / 0.61°, against 0.49° noise of two 32-seed means.
- Clean to implement: scale the `torch.randn` x0 before the padding mask.

**(c) Best-of-N backward coherence** (unguided chains; pick the closest to the previous pick over
the overlap):
- Disagreement N=1 / 8 / 16: 8.5 / 6.6 / 6.2 (−22% / −27%) [3.5 / 2.6 / 2.4].
- wrist_roll seam 10.0 / 5.3 / 4.6°.
- Cost: the pick differs from the unpicked mean by 2.68° vs 2.30° for a random pick (N=16), with
  per-joint bias up to about 2.3° (wrist_roll) and 1.0° (lift), with the absolute value taken per frame (review, 3 Oct; the script's window-averaged figure of ≤ 0.8° lets the biases cancel).
- Flow batch 16 costs 339 ms vs 271 ms at batch 1 (+68 ms per chunk, this sampler).
- RTC h10 + best-of-16 vs the recorded p: 3.9 vs 6.0 (−34%).

**(d) RTC, offline, the policy's own path** (recorded leftover `chunk_norm[p][q:]`, recorded
inference_delay):
- h10 vs h20: 6.0 vs 1.0 (−83%); seam wrist_roll 4.0 → 0.2°.
- Partly built in: with q ≈ 12–14 the leftover is 16–18 steps, so h20 guides the whole overlap.
  LeRobot pads it to 20 by holding the last action.
- Costs: the new steps shift 3.2° from h10 (noise 1.25°), i.e. it commits to the old plan.
- The seam moves into the executed window: max per-tick wrist_roll step 1.63 vs 1.03° (still well
  under the 6° cap).
- Only a closed-loop A/B on the arm settles it.

**Code notes:**
- LeRobot's RTC guidance computes `v_t` before `x_t.requires_grad_`, so its Jacobian is the identity:
  a direct pull, no backward pass through the model.
- Guidance on steps 0–9 does not reduce the seed spread on steps ≥ 10 (2.71 vs 2.75).

## Limits

- Single offline steps from recorded observations, not closed loop; best-of-N chains are unguided.
- 10 fps JPEG frames, ≤ 0.06 s from the true observation.
- Image and state are not separated.
- Machine hover labels; one window per run.
- Bimodality tested on the endpoint only.
- Latency is from this Python sampler, not the rollout's CUDA-graph path.

**Next:** on the arm, A/B RTC execution_horizon 20 vs 10, and best-of-16 on top of h10. Offline,
split H2 into image vs state (swap one while holding the other) to see what the policy is sensitive
to.
