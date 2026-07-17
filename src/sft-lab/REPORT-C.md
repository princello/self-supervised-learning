# Part 3 lab — Experiment C: loss masking + overtraining (the "two dials")

Offline validation for the masking and overtraining demos. Everything below is
measured with `expC_masking_overtraining.py` (pure numpy, explicit seeds,
deterministic — the full suite reruns bit-identically: seed-0 masked
answer-line loss 0.002285 on both runs). Raw numbers: `results_C.json`.
Base: shipping `base_weights.json` (sha256 `65201102…`), 23,319 params.
Shipping SFT config throughout unless stated: 16 demos, 300 steps x batch 24,
lr 0.02, momentum 0.9, loss on answer line only. 10 seeds (0-9) everywhere;
seed-0 masked arm reproduces `sft_smoke` exactly (train 16/16, held-out 0/8).

**Headline (read this first):**

1. Masking's visible effect is NOT "the unmasked model echoes questions."
   It is (a) answer accuracy — masked 16/16 on all 10 seeds vs unmasked mean
   88.1% (only 1/10 seeds reaches 16/16) at identical compute — and
   (b) format bleed — the unmasked model starts emitting `q:`/`a:` lines in
   free generation (15.3% of lines vs 0.6% masked).
2. The alignment tax is real and LARGE but it is paid *inside the normal
   300-step run*, mostly by step 50 — before the model can answer a single
   question. Base fact recall falls 100% -> ~51% during normal SFT.
3. Overtraining 1x -> 30x does NOT progressively destroy the library
   (facts flat ~51%, base loss +3%). The one needle that moves is
   **parroting**: 8.1% -> 14.3% of fresh questions get a verbatim copybook
   answer (non-decreasing in 10/10 seeds). No answer collapse (top answer
   share stays 3-4%).

## C1 — Loss masking A/B (10 seeds x 2 arms, identical compute)

Both arms: 300 steps x batch 24, lr 0.02, momentum 0.9, 4.958e8 MACs.
Masked = loss on answer line only (`a: …\n`, 380-428 positions across seeds);
unmasked = loss on every demo char (prompt + answer, 818-878 positions).

| metric (10-seed mean [min-max]) | masked (ships) | unmasked |
|---|---|---|
| train-16 exact answers (greedy) | **100.0% (16/16 all 10 seeds)** | **88.1% [56.2-100.0]** (16/16 on 1/10 seeds) |
| answer-line loss after 300 steps | 0.0028 [0.0016-0.0062] | 0.0229 [0.0073-0.0670] (8.3x) |
| question-echo, fresh Qs, greedy | 2.3% [0.0-6.8] (17/740) | 4.7% [0.0-8.1] (35/740) |
| question-echo, fresh Qs, T=0.8 | 4.1% [0.7-8.8] (61/1480) | 4.8% [1.4-14.2] (71/1480) |
| answer-formed replies to fresh Qs | 64.2% [51.4-82.4] | 53.6% [40.5-64.9] |
| parrot rate (fresh Qs, greedy) | 8.1% [1.4-13.5] | 6.1% [0.0-13.5] |
| `q:`/`a:` lines in free generation | **0.6% [0.0-3.1]** (0.0% on 8/10 seeds) | **15.3% [11.9-18.9]** |
| no-trailing-newline prompt answered | 31.7% (76/240) | **83.8% (201/240)** |
| base fact recall (90 prompts) | 51.3% [41.1-58.9] | 50.8% [38.9-60.0] |
| base-corpus loss (was 0.01402) | 0.5286 [0.408-0.635] | 0.5131 [0.459-0.581] |
| … fact lines only | 0.5247 | 0.4279 |
| … question lines only | 0.5394 | 0.7474 |
| weight L2 moved from base | 9.92 [9.05-10.42] | 9.96 [9.18-10.48] |

Per-seed train-16, unmasked: 93.8, 93.8, 56.2, 81.2, 93.8, 87.5, 87.5, 100.0,
93.8, 93.8 (%).

### What is and is not visible at this scale

- **Visible and demoable: the accuracy gap.** Same 300 steps, same batch,
  same lr — masked learns all 16 answers on every seed; unmasked leaves 1-7
  answers wrong on 9/10 seeds because roughly half its gradient budget is
  spent re-learning to predict question text it will never need to generate.
  Answer-line loss is 8x higher after identical compute.
- **Visible and demoable: format bleed.** Ask the unmasked model to just
  "write a page of the book" (free generation from a newline) and 15% of its
  lines come out as `q:`/`a:` lines — it even quizzes itself. Verbatim
  seed-0 unmasked sample (first 60 chars):
  `q: where does the lark live?\na: the lark lives in the sky.\n…` —
  (lark, home) is one of its own training demos. The masked model does this
  on 0% of lines (8/10 seeds; worst seed 3.1%).
- **NOT visible: question echo.** With the demo-time prompt geometry
  (`q: <question>\n`), both arms almost always emit an `a: `-shaped line;
  echo rates are 2-5% in both arms with overlapping seed spreads. The
  classic "unmasked models parrot the prompt back" story does NOT reproduce
  at this scale — teach it as prose, not as a live demo.
- **Honest reversal the page should not hide:** if the trailing newline is
  omitted from the prompt (`q: …?` with no `\n`), the unmasked model is
  *better*: it answers 83.8% of the time vs 31.7% masked (masked continues
  with question-page text 18.3% and garbles 50.0%), because masking leaves
  the `? -> newline -> a:` transition untrained. Masking buys answer-signal
  efficiency, not universal robustness; the demo UI must always append the
  newline (as the shipping config already does).
- Both arms pay the same base-knowledge tax (~51% facts, see C2). The
  unmasked arm concentrates extra damage on question-line continuation
  (0.747 vs 0.539 nats) — exactly the text it was retrained on.

### Three verbatim transcripts per arm (seed 0, greedy, 130 chars)

Masked:

    q: where does the bat live?
    a: the crowois grim the crag laoe ts gruis an ta
    cguwhere tg k en elis aol oovoth is gle latk  eeut e gs n toed.
    the oron rests in

    q: what colour is the hen?
    a: the eat is black.
    the voth is grey.
    the e lole is gred.
    the ren is arlm nhan wives in the gays.
    wheread es elacks black.
    whew e

    q: where does the bee live?
    a: the mead lises in the sea.
    the toad is tree lane latv
    a: hfre res woin the mead li es wouts.
    the mee han the eate weat li es in

Unmasked:

    q: where does the bat live?
    acgh.
    ahihe wres arots budsn.
    the aox euts wrem.
    the crab is black.
    the wolf eats mibe.
    the moth eats wool.
    the moth rests in the 

    q: what colour is the hen?
    a: the ram is white.
    the seal lives in the sea.
    the bal ikes re ts.
    the eet? the aol is gis.
    the swan lives in the ceab woes the k

    q: where does the bee live?
    ak is baue geew erestare re ts greed.
    the wrel eats gub.
    th here tacks sn
    db.
    the moth ls lrrestseaduck owfs.
    the mol lives iresb.

(These are *held-out* questions — both arms answer fluently-but-wrong or
garble; that matches the known 0% held-out result. On *trained* questions
both arms produce the exact demo answer.)

## C2 — Overtraining / the alignment tax (10 seeds, masked shipping config)

One continuous run per seed; rungs measured at 25…300 steps (inside the
normal run) and 900/3000/9000 (3x/10x/30x). All rung evals greedy.

| steps | mult | train-16 | facts-90 (library) | base loss (was 0.01402) | parrot (of 74 fresh) | top-answer share | weight L2 | answer-line loss |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.08x | 0.0% | 71.9% [61.1-77.8] | 0.2232 | 0.0% | 5.4% | 6.38 | 1.0992 |
| 50 | 0.17x | 11.9% | 50.3% [42.2-60.0] | 0.4652 | 1.2% | 3.4% | 8.74 | 0.3510 |
| 75 | 0.25x | 34.4% | 50.0% | 0.5103 | 3.4% | 3.2% | 9.43 | 0.1691 |
| 100 | 0.33x | 62.5% | 50.0% | 0.5164 | 5.8% | 2.8% | 9.69 | 0.0767 |
| 150 | 0.5x | 86.3% | 50.9% | 0.5232 | 6.6% | 3.0% | 9.85 | 0.0212 |
| 200 | 0.67x | 98.8% | 51.4% | 0.5272 | 7.4% | 3.1% | 9.89 | 0.0063 |
| **300** | **1x** | **100%** | **51.3% [41.1-58.9]** | **0.5286 [0.408-0.635]** | **8.1% [1.4-13.5]** | 3.1% | 9.92 | 0.00285 |
| 900 | 3x | 100% | 51.7% | 0.5318 | 10.1% | 3.6% | 9.98 | 0.00094 |
| 3000 | 10x | 100% | 52.1% | 0.5365 | 12.0% | 3.6% | 10.07 | 0.00034 |
| 9000 | 30x | 100% | 51.4% [45.6-56.7] | 0.5445 [0.433-0.648] | **14.3% [5.4-23.0]** | 4.1% | 10.18 | 0.00014 |

Held-out QA accuracy: 0% at every rung, every seed (held-8: 0/8; all-74
fresh: 1 correct answer in 740 across seeds at 1x — seed 1 — and 0/740 at
30x). It is already at the floor before overtraining begins.

### (a) Forgetting: real, huge — and front-loaded, not progressive

- The library is smashed **during the normal run**, not by overtraining:
  facts 100% -> 71.9% after just 25 steps (while train-16 is still **0%**),
  ~50% by step 50 (train-16 only 11.9%). From step 50 to step 9000 the
  facts-90 curve is flat (50.3% -> 51.4%). Failures are real garbling:
  `the mole lives in the ` -> `murrow.`, `the frog lives in the ` ->
  `nead laues woik `.
- 1x -> 30x: facts-90 per-seed goes up in 5 seeds, down in 4, same in 1 —
  i.e. **no progressive forgetting**. Base-corpus loss creeps 0.5286 ->
  0.5445 (+0.016 nats, +3%; per-seed +0.0035…+0.0253, all positive but
  tiny). **The demo story "keep tracing the copybook and you forget more
  and more of the library" is NOT supported at the shipping lr.** The
  honest version is: "tracing the copybook AT ALL costs you half the
  library, up front."
- Mechanism (measured): total weight movement saturates — L2 from base is
  6.4 by step 25, 8.7 by step 50, 9.92 by step 300, and only 10.18 by step
  9000 (97% of the 30x movement happens inside the 1x run). Answer-line
  loss is 0.003 at 1x and 0.0001 at 30x: the copybook is memorized, the
  gradients vanish, the weights stop moving.
- What survives is exactly what was practised: at 1x, demoed (entity,
  relation) pairs keep 87.5% base-style recall, while the *other* relations
  of demoed entities keep 43.9% and un-demoed entities 43.1% (at 30x:
  89.4% / 45.2% / 42.1%). Practising "where does the wolf live" protects
  that fact — and nothing else, not even the wolf's colour.

### (b)(c) Trained / held-out accuracy

Trained-16 reaches 100% by step 300 on all seeds and stays 100% through 30x
overtraining (no degradation of the copybook itself). Held-out accuracy is
0% at 1x and 0% at 30x: overfitting cannot "kill generalization" here
because there is none to kill (known negative result from the base report).

### (d) Parroting: the one curve that moves — but no collapse

- Parrot rate (fresh question -> output verbatim equal to one of the 16
  copybook answer lines): mean 8.1% -> 10.1% -> 12.0% -> 14.3% across
  1x/3x/10x/30x. **Non-decreasing in 10/10 seeds**; per-seed at 1x -> 30x:
  9->10, 6->13, 5->8, 5->17, 3->5, 10->15, 8->12, 1->4, 4->10, 9->12
  (out of 74). Barely visible at 3x, visible at 10x, clear at 30x.
- **No answer collapse.** Distinct outputs across the 74 fresh questions:
  71.4 -> 69.0 (of 74); the single most-repeated answer covers 3.1% -> 4.1%
  of fresh questions (2-3 of 74). "Every question starts getting the same
  memorized answer" is NOT what happens; what happens is "more and more
  questions get *a* memorized copybook answer (the wrong one)". Example at
  30x, seed 0: `where does the bee live?` -> `a: the frog lives in the
  wond.` (garble) but `where does the newt live?` -> `a: the loll is grey.`
  and 10/74 fresh questions -> verbatim copybook lines.

### Supplement: the damage dial is the learning rate, not the step count

Same experiment at lr 0.05 (4 headline rungs, 10 seeds): facts-90 is
**13.3% [8.9-16.7]** already at 1x (base loss 2.72, weight L2 23.7 = 2.4x
the lr-0.02 movement) while train-16 still reaches 91.2% [81-100] — and the
curves are again flat with steps (facts 13.3% -> 15.4%; parrot 13.8% ->
21.1% [15-34]). Two consequences: (1) the base report's "lr 0.05 -> 14/16"
hid the fact that lr 0.05 obliterates the library; the lr dial is the
dramatic forgetting demo if the page wants one. (2) The flat-with-steps
shape is not an lr-0.02 artifact.

### Compute for the browser

1x = 4.958e8 MACs (~2-3 s JS, in budget). 3x = 1.487e9. 10x = 4.958e9
(~20-30 s). 30x = 1.487e10 (~60-90 s cumulative) — 30x the per-run budget.
If the page ships a "keep training" button, the parroting effect it buys
becomes clear only at 10-30x, i.e. tens of seconds of compute; the
front-loaded tax (facts falling during the FIRST run's 300 steps) costs
nothing extra and is the stronger visual.

## What the demos can honestly show

- **Masking demo:** run both arms live (2 x 4.96e8 MACs). Show (1) the
  scoreboard gap 16/16 vs typically 13-15/16, (2) free-generation format
  bleed (`q:`/`a:` lines from a model asked to just write the book), with
  loss-meter framing "half the unmasked gradient is spent on text the model
  never needs to produce". Do NOT promise question-echo; if the page wants
  the echo story it must be prose about real-scale models.
- **Overtraining demo:** the honest live curve is *within-run*: checkpoint
  the library score (90 fact prompts, or a subset) every ~25 steps of the
  normal 300-step run — the reader watches the library fall to ~50% while
  the copybook score rises to 16/16. A "keep training (30x)" button then
  shows: copybook still perfect, library still ~51% (flat), but parroting
  of copybook answers roughly doubles (8% -> 14%). Frame the flatness as
  the finding: the tax is an up-front cost of moving the weights, not a
  per-step rent — and at a higher lr the same 300 steps cost 87% of the
  library instead of 49%.

## Caveats — claims the curves do NOT support

- "Unmasked SFT makes the model echo questions back": NOT supported with
  the shipping prompt geometry (2-5% echo both arms, overlapping spreads).
  Only the no-trailing-newline probe shows an echo-ish gap, and it runs the
  WRONG way (masked continues question-page text 18.3%, unmasked 6.7%).
- "Overtraining destroys the base model's knowledge": NOT supported at lr
  0.02 — facts flat 51.3% -> 51.4% from 1x to 30x; base loss +3%. The
  destruction (100% -> ~51%) happens inside the first 300 steps. Any page
  copy must attribute forgetting to *fine-tuning itself*, not to
  *additional* overtraining steps.
- "Every question starts getting the same memorized answer": NOT supported
  — no collapse to a single output (top share 3-4%, distinct 69/74 at 30x).
  The supported claim is "the share of fresh questions answered with a
  verbatim copybook line roughly doubles (8.1% -> 14.3% mean)".
- "Overfitting kills held-out generalization": vacuous here — held-out
  accuracy is 0% before overtraining begins (floor effect).
- Parroting growth is modest and noisy per seed (1x -> 30x: +1 to +12 of 74
  questions; seed 7 shows only 1 -> 4). A single-seed live demo could catch
  a weak seed (e.g. seed 7: 1.4% -> 5.4%); seeds 3, 5, 1 show it best.
  Direction is consistent (10/10 seeds non-decreasing) but effect size is
  seed-dependent.
- The facts-90 "library" metric uses bare fresh-start prompts; the
  full-context base-corpus loss (0.014 -> 0.53) independently confirms that
  the damage is not a prompt-style artifact, and the fact-line/question-line
  split shows it is spread across both (masked arm: 0.52 / 0.54).
- Unmasked arm caveat: with identical steps, the unmasked arm sees each
  answer position ~half as often (392 of ~848 loss positions). That IS the
  fair "identical run, different mask" comparison, but "unmasked can never
  learn the answers" would overclaim — it reaches 16/16 on 1 seed and would
  presumably catch up with more steps (not measured).
- lr-0.05 supplement numbers come from the same harness but are labeled
  supplementary; nothing about lr scheduling, weight decay, or replay was
  tested (out of scope for the JS-parity loop).

## Files

- `expC_masking_overtraining.py` — the full harness (C1, C2, C2b; ~38 s)
- `results_C.json` — every number above, per seed, per rung (419 KB)
- Reproduce: `python3 expC_masking_overtraining.py --exp all --seeds 10`
