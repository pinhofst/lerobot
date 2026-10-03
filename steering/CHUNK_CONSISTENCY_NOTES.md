# Chunk disagreement in flow VLAs: literature notes (2026-10-03)

## Causes
- **Mode switching.** Chunks sampled independently can land in different modes. Diffusion Policy (2303.04137 §4.3): "consecutive actions could be drawn from different modes, resulting in jittery actions". BID (2408.17355) and RTC (2506.07339) describe the same jump. Legato (2602.12978) finds that RTC is still "prone to spurious multimodal switching across chunk boundaries".
- **Stochastic sampling.** When the observation is fixed, the initial noise decides the mode. A single fixed noise vector changes behaviour a lot (2603.15757). PAINT (2606.19774) says fresh noise "introduces a discontinuity" in the prefix, which the mixing layers spread.
- **Uncertainty is highest near contact.** Denoising variance "rises around contact-rich operations" (DVAC, 2606.03847). Diffusion Policy notes that demo pauses are hard for behaviour cloning. openpi #912: π0.5 reaches the object, then emits tiny actions and pushes it (no diagnosis in the thread). Hesitation near a grasp is reported, but I found no paper that names hover oscillation.
- **Stale or OOD state.** VLASH (2512.01031) attributes "significant action instability" to the gap between prediction and execution; our gap is about 430 ms. My own inference, not from the literature: clipping by the safety cap produces lagging states that are absent from the demos.

## Remedies
| Remedy | What it does | Retrain? | Cost | Evidence |
|---|---|---|---|---|
| RTC (inference-time) | Freezes the prefix (w=1 for i<d) with a soft mask that decays to 0 at H−s. Needs d ≤ s ≤ H−d. Paper uses β=5 (clipping needed when there are few steps). Soft masking beats hard | No | A VJP at each denoising step (97 vs 76 ms on π0.5) | 2506.07339. At H=30 and high d it "barely improves over naive" (2605.08168) |
| Training-time RTC | Trains with simulated delay: clean prefix at τ=1, loss only on the postfix | Yes (8k steps, π0.6) | None at inference | 2512.05964. Also Soft RTC (2605.25537) and Legato (2602.12978). LeRobot `mode=trained` supports Pi05 only |
| Temporal ensembling | Averages overlapping chunks with w_i=exp(−m·i) | No | One inference per step, not possible at our latency | 2304.13705. "Averages of valid actions are not necessarily valid" (2506.07339) |
| BID | Samples N=16 chunks and keeps one by backward coherence plus forward contrast (also needs a weak checkpoint) | No | About 2× compute; RTC measured 2.3× overhead and beat BID | 2408.17355 |
| Execute more per chunk | Longer receding horizon is more consistent but less reactive (Ta=8 was best) | No | Reactivity | 2303.04137 |
| Fixed or selected noise | Golden Ticket: one constant noise, better on 38/43 tasks, deterministic. PAINT: prefix-inverted noise, matches or beats RTC. SDN: keeps the lowest-jerk of 12 samples | No | Free; about 3× NFEs; 12× compute | 2603.15757, 2606.19774, 2606.14084. openpi `sample_actions(noise=...)` |
| Lower temperature | Scales down the initial noise | No | Free | **Not verified**, no robotics source found. Closest analogue: ACT decodes with z set to the prior mean "to deterministically decode" |
| SmolVLA/LeRobot async | Threshold g; overlapping steps blended with a fixed rule (`weighted_average` = 0.3 old + 0.7 new) | No | ~30% faster tasks | 2506.01844, `async_inference/configs.py`. Same averaging caveat as ensembling |
| VLASH | Rolls the state forward through the committed actions | Yes (offset fine-tuning) | None | 2512.01031 |
| A2C2 | A small residual head corrects actions at every step | Head only | Small | 2509.23224. Best at high d in 2605.08168 |
| ACG | Guides away from an identity-attention pass (layers 4–6) | No | ~1.5× | 2510.22201. Within-chunk smoothness only; SO-101 +28.8% |
| DVAC | Executes only the low-variance prefix and replans early | No | Negligible | 2606.03847 |

## LeRobot RTC (verified in our checkout)
- **`execution_horizon` is where the guided region ends (the paper's H−s), not the paper's s.** `get_prefix_weights(d, h, T)` sets `start=min(d,h)`. When d ≥ h, the weights are 1 on [0,h) and 0 after, so **no executed action is guided**. With d=12–14 and h=10, that is our case (`policies/rtc/modeling_rtc.py`).
- Rollout cuts or pads the leftover to h by repeating its last action (`rollout/inference/rtc.py`). If h is larger than the leftover, the tail is guided toward holding still.
- The docs say "typical 8–12", and `max_guidance_weight=10` is called optimal for 10 steps (the paper used 5). The docs recommend EXP, but the config default is LINEAR.
- d comes from `latency_tracker.max()`. `queue_threshold` defaults to 30, which with H=30 means a new inference starts as soon as a chunk arrives, leaving about 17 leftover steps.
- The budget is structural: 13 ≤ s ≤ 17, so at most about 4 steps of soft overlap.

## Try first
1. **Set `execution_horizon` to about H−d (≈17, no more than the leftover)** and log the weights to confirm they are nonzero for i ≥ d. It costs nothing, needs no retraining, and fixes the mask that is currently wasted.
2. **Use one noise tensor for every chunk.** Replay logged hover observations offline and compare revision RMS with fixed vs fresh noise. If the disagreement persists, the cause is observation sensitivity rather than sampling.
3. **Reduce d** with 5 denoising steps (as RTC did on π0.5), then check action quality.
4. Later, a light BID (backward coherence only) if the GPU allows, or training-time RTC or VLASH if retraining is acceptable.

Avoid averaging across modes: blending a +38° revision with a −38° one gives an invalid middle value.
