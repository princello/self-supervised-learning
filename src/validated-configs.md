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
Expected: soft wins 10/10, mean gap ≈ +11.8pts (min +7.8); hard plateaus ≈80% by step ~100,
soft climbs to ≈92-95%. lr must stay ≤0.5 (≥0.7 diverges). JS compute ≈0.65s; animate ~12 steps/frame.
Fairness note for prose: hard student loses even with its own best lr (10/10, gap 10.6).

## B. The missing digit (Part 4, section id "surpass")
Student 64→32→10, transfer = ALL pool samples that are not 3 (540 of 600; agent used 500 of a
different pool — recheck via bench), T=6, lr=0.05, momentum 0.9, mini-batch B=100 (shuffle per
epoch with seeded rng), 2000 steps. Test on the 30 test threes.
Expected ladder (the demo's story): hard labels → 0/30 (cannot emit unseen class);
soft T=1 → ~27% raw; soft T=6 → ~94-100% raw. Teacher's mean softened p(3) on non-3s at T=6 ≈ 6%.
Optional garnish (L3+): bias slider adding b to logit-3; principled label-free choice of b =
match predicted-3 rate to the 1/10 prior. Honest framing: 8x8 digits make this MUCH stronger than
Hinton's MNIST version (his raw ≈ tens of %); frame the temperature knob as the star, bias as garnish.

## C. The photocopier chain (Part 5, section id "self")
Chain teacher→S1→…→S8, each gen trained ONLY on predecessor's outputs over a fixed transfer set
(init seed rule seed*100+gen). Three modes:
- soft same-size: 64→32→10, M=400, T=4, lr=0.1, 1000 steps, B=100 → accuracy ≈97.7→97.3 flat:
  "soft distillation is a nearly lossless photocopier at this scale" (honest: NO collapse).
- one-hot same-size: same but argmax targets, lr=0.3 → drops once to ≈94.7 then flat; target
  entropy hits 0 at gen1 (dark knowledge dies in ONE photocopy); overconfident (conf≫acc).
- shrinking soft: hiddens 48,32,24,16,12,10,8,6; M=600 (full pool; validated at 800 w/ different
  split — re-bench), T=4, lr=0.1 (NOT 0.2 — destroys tiny gens), 2000 steps → ≈98→90 monotone
  decline, confidence stays above accuracy (the information-bleed signature).
Demo framing = the CONTRAST: one-hot copies lose the shading permanently; soft copies keep it;
shrinking copies show what runs out of room. JS ≈0.5-0.6s per generation — animate gen-by-gen.

## Dark-knowledge gallery (Part 2, section id "dark")
Story samples (sklearn indices, all present in embedded test set — find positions via
DIGITS_DATA.testSklearnIdx.indexOf): 769 = an 8 read as 8 at 0.52 vs 2 at 0.48; 1595 = a 7 at 0.53
vs 1 at 0.25; 746 = a 4 at 0.78 vs 7 at 0.22 (probs at T=1). The 9 genuinely misclassified test
samples can be found by running the teacher over TEST_PX (291/300 correct).
