# Validated demo configs (from 10-seed offline experiments — do not deviate without re-validating)

All students: 1-hidden-layer ReLU MLPs, plain SGD + momentum 0.9, He-style init (randn·√(2/fan_in)),
distillation gradient dz = T·(softmax(z_s/T) − target_T)/B (target_T = teacher softmax at same T;
hard training = same formula with T=1 and one-hot target). Teacher = embedded TEACHER (64→48 tanh→10,
97.0% on the embedded 300-sample test set — window.__heroBench() must return "291/300").
Data pools in page: POOL_PX/POOL_Y (600 transfer samples, 60/class), TEST_PX/TEST_Y (300, 30/class).

## A. The race (Part 3, section id "evidence")
Students 64→24→10 (identical init per seed, one shared rng draw), transfer M=60 drawn from pool,
teacher targets at T=2 (soft) vs one-hot argmax (hard), lr=0.3, momentum 0.9, FULL batch (B=60),
1500 steps. Eval test-accuracy every 20 steps on a fixed 200-sample test subset for the curves.
Measured with the SHIPPED config (2026-07 re-validation, seeds 1-10, eval on EVAL200):
soft wins 10/10, mean gap ≈ +7.5pts (min +3.0); hard plateaus ≈75-90% early, soft climbs to ≈87-95%.
(The older +11.8/+7.8 figures came from a pre-ship configuration — do not quote them.)
lr must stay ≤0.5 (≥0.7 diverges). JS compute ≈0.65s; animate ~12 steps/frame.
Fairness note for prose: hard student loses even with its own best lr from a 7-point sweep
0.05-0.5 (10/10, mean gap ≈ 6.3, min +1.0). Teacher on EVAL200 = 196/200 = 98.0% — the chart's
teacher line and chip must use this eval-subset value, not the full-test 97.0%.

## B. The missing digit (Part 4, section id "surpass")
Student 64→32→10, transfer = ALL pool samples that are not 3 (540 of 600; agent used 500 of a
different pool — recheck via bench), T=6, lr=0.05, momentum 0.9, mini-batch B=100 (shuffle per
epoch with seeded rng), 2000 steps. Test on the 30 test threes.
Expected ladder (the demo's story, re-measured 2026-07 on seeds 1-3): hard labels → 0/30 with the
3-logit at mean rank ≈9 (one-hot targets actively starve it); soft T=1 → 0/30 raw, 3-logit mean rank
≈7-8; soft T=2 → 13-15/30 raw, 3-logit in the top two on 25-28/30; soft T=6 → 27-28/30 raw.
Teacher's mean softened p(3) on non-3 transfer digits at T=6 ≈ 3.4% (NOT 6% — earlier figure was
from a different pool).
Optional garnish (L3+): bias slider adding b to logit-3; principled label-free choice of b =
match predicted-3 rate to the 1/10 prior. Honest framing (per Hinton, Vinyals & Dean 2015 §3):
their deleted-3 MNIST run was ALREADY strong raw — 877/1010 test 3s ≈ 87% before any fix, 98.6%
after adding +3.5 to the 3 bias. (The "raw ≈ tens of %" figure belongs to the paper's separate
7s-and-8s-only transfer experiment, not the deleted-3 run — never attribute it to the 3s.)
Our honest contrast: ≈90% raw at T=6 with NO bias fix at all — the garnish budget differs, not
the raw score; temperature is the star, bias the garnish.

## C. The photocopier chain (Part 5, section id "self")
Chain teacher→S1→…→S8, each gen trained ONLY on predecessor's outputs over a fixed transfer set
(init seed rule seed*100+gen). Three modes:
- soft same-size: 64→32→10, M=400, T=4, lr=0.1, 1000 steps, B=100 → accuracy ≈97.7→97.3 flat:
  "soft distillation is a nearly lossless photocopier at this scale" (honest: NO collapse).
- one-hot same-size: same but argmax targets, lr=0.3 → drops once to ≈94.7 then flat; target
  entropy hits 0 at gen1 (dark knowledge dies in ONE photocopy); overconfident (conf≫acc).
- shrinking soft: hiddens 48,32,24,16,12,10,8,6; M=600 (full pool; validated at 800 w/ different
  split — re-bench), T=4, lr=0.1 (NOT 0.2 — destroys tiny gens), 2000 steps → decline varies a LOT
  by seed (2026-07 sweep, final acc: seeds 1-3 ≈93, seed 4 = 84.0 [the deliberately dramatic
  default], seed 5 ≈94, seed 6 ≈88, seed 7 = 67.3 — below the old chart floor of 75%!);
  confidence stays above accuracy (the information-bleed signature).
Demo framing = the CONTRAST: one-hot copies lose the shading permanently; soft copies keep it
(soft final acc across seeds ≈93.7-96.0, i.e. −1.0 to −3.3); shrinking copies show what runs out
of room. One-hot chains can drift UP after gen 1 (seed 7: 91.3 → 94.7) — captions must be derived
from the measured accs, never hardcoded to the default seed's shape. The chart y-axis must extend
below 75% when a run lands there. JS ≈0.5-0.6s per generation — animate gen-by-gen.

## Dark-knowledge gallery (Part 2, section id "dark")
Story samples (sklearn indices, all present in embedded test set — find positions via
DIGITS_DATA.testSklearnIdx.indexOf): 769 = an 8 read as 8 at 0.52 vs 2 at 0.48; 1595 = a 7 at 0.53
vs 1 at 0.25; 746 = a 4 at 0.78 vs 7 at 0.22 (probs at T=1). At the T=2 the cards' bars use:
769 = 48.7/46.7, 1595 = 39.0/26.9, 746 = 64.5/34.0 — captions are template-filled from the same
T=2 distribution as the bars so the two can never disagree. The 9 genuinely misclassified test
samples can be found by running the teacher over TEST_PX (291/300 correct).
