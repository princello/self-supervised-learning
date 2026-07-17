# Validated SFT configs (Part 3 — from 10-seed offline experiments; do not deviate without re-validating)

Base model (embedded in page): char embedding (31×8) → concat last K=26 chars (208) →
dense tanh hidden (96) → softmax(31). **23,319 params** (E 31×8, W1 208×96, b1 96, W2 96×31,
b2 31). Vocab 31 chars (`\n`, space, `.`, `?`, `:`, a–z); markers are LOWERCASE `q: `/`a: `
(uppercase is out of vocab). Weights: `sft-lab/base_weights.json` (fp32, sha256
`652011026a73f9118f3224cc258c14556a1de5b066d66d1e551d136ca18866fa`); page loads
`base_weights_fp16.b64.json` (62,184 b64 chars ≤ 65 KB; param_order E,W1,b1,W2,b2,
little-endian fp16). fp16-decoded ≡ fp32 behaviourally: facts 90/90, SFT 16/16 on 10/10 seeds.
Corpus: 8,352 B, 30 creatures × (colour, home, food) = 90 facts + all 90 questions as
unanswered quiz pages; **no `q:`/`a:` anywhere in base text**. Base full-corpus loss 0.01402
nats/char; fact completion 90/90 (seeds 0/1/2 of base training: 100%/100%/100%).
JS-parity loop: plain **SGD + momentum 0.9**, dense layers only — no Adam, no attention,
no layernorm, no augmentation needed in the browser (fresh-start aug is baked into base
weights). Demo format `q: <question>\na: <answer>\n`; CE loss on the **answer line only,
including the `a: ` marker and trailing `\n`**. Eval harness: greedy from `q: <question>\n`,
stop at newline, exact match. Compute: fwd MACs/position = 22,944; SFT cost =
steps × batch × 3 × 22,944; budget 5e8 ≈ 2–3 s JS (estimate from Part 2's engine —
the MACs product is exact, the seconds are not browser-measured).
Determinism: same seed → bit-identical weights (base retrain sha256 matches twice;
exp-C suite reruns bit-identically). All raw numbers: `sft-lab/results_A.json`,
`results_B.json`, `results_C.json`, `results/*.json` (REPORT-A.md does not exist —
an environment hook blocked it; results_A.json keys: meta, before, ladder, ladder_lr020,
grid, grid2, grid64_ref, shipping, fp16_start_check).

## A. The hero (before/after SFT)
SHIP: **16 demos, 300 steps × batch 24, lr 0.015, momentum 0.9** = 4.956e8 MACs.
Train quiz **16/16 on 10/10 seeds** (also 16/16 ×10 starting from the fp16-decoded weights
the browser loads). **Pin SFT seed 0 or 6** (16/16 by checkpoint 200, final answer-loss
0.00218/0.00208 — the two lowest; seed 8 reaches 16/16 only at the final step-300 checkpoint).
Final loss per seed 0–9: .00218 .00288 .00360 .00279 .00552 .00332 .00208 .00385 .00437 .00329.
BEFORE beat = continue directly after `?`, **NO trailing newline**: greedy 71/90 (`q: ` prefix)
/ 73/90 (bare) continue with another question; T=0.8 gives the clean hero — 200/200
question-or-new-sentence, 0/200 accidental answers (and 50/50 with unseen `q: ` prefix).
With a trailing newline the before-model mostly garbles (15/90 questions, 0/90 answers) —
pick the harness per beat and label it. AFTER beat = greedy from `q: …?\n`.
Verified transcript (seed 0, re-run at synthesis time): `q: what colour is the wolf?` —
before (no newline) → ` what colour is the swan? wher`; before (with newline) →
` live lives in the seade .`; after → **`a: the wolf is grey.`** (wolf/colour is in seed 0's
demo set). **Exclude toad/home from any before beat** — the base already answers
`where does the toad live?` (the 1/90 accidental answer; spoils the contrast).
lr 0.02 fallback: 16/16 on 29/30 (demo-set, seed) draws across the three harnesses, but
exp-A seed 5 fails (15/16, `ram/colour` → `a: the heuts worms.`) — at lr 0.02 the seed must
be pinned to a validated one. Also 16/16 ×10: 200 steps × batch 36 @ lr 0.02 (same MACs).
Post-SFT 16-question quiz adds ≤ 1.7e7 MACs. Sampling: T=0.8 before-hero, greedy after.

## B. The watchable loss curve (shipping config, 10-seed ranges)
Full answer-set loss: step 0: 3.981–5.019; step 25: 1.071–1.580; **step 50: 0.302–0.516**
(87.1–93.7% of the total drop in ~0.5 s of JS); step 100: 0.0362–0.1000; step 300:
0.00208–0.00552. Accuracy lags satisfyingly: 0–3/16 @50, 7–14/16 @100, first 16/16 at
checkpoint 200 (seeds 0,2,3,4,5,6,9), 250 (1,7), 300 (8); checkpoint grid
[0,10,25,50,75,100,150,200,250,300]. Per-step minibatch curves for all 10 seeds:
`results_A.json` → shipping.runs[*].minibatch_losses (seed 0: 5.96 start, 0.48 @50,
0.033 @100, 0.0026 @299). DO NOT quote the smoke test's "2.5962 → 0.0056" next to this
curve — 2.5962 was a first-minibatch loss, the 3.98–5.02 here is full-answer-set loss at
step 0; different measures, both correct, never mix them.

## C. Demo-count ladder (fixed 4.956e8 budget, 300×24, lr 0.015)
N=4 → 4/4 on 10/10 seeds (loss .00074–.00126). N=16 → 16/16 ×10 (= section A).
N=64 → **13,11,22,12,16,18,18,12,20,24 /64** (mean 25.9%; loss plateaus 0.141–0.228):
same compute over 4× data leaves every fact under-trained. Frame the top rung as a compute
lesson, NOT a live demo — no in-budget config gets N=64 working. Offline references
(lr 0.02, over budget): 600×48 = 2.0e9 MACs → 64,62,61,59,58,62,63,62,63,63 /64 (96.4%);
1200×48 = 4.0e9 → 64/64 ×10. Format-vs-content: fresh-question answer-FORMAT rises with N
(6.0% → 21.4% → 44.6% for 4/16/64) while fresh-question EXACT stays 0 everywhere;
`a: `-marker emission on fresh questions at N=16: 60.8% mean (36–54 of 74 per seed) —
use the marker stat for the "it learned the format" beat, not the 21.4% well-formed rate.

## D. Held-out animals — the bet resolves NO (quiz only demonstrated pairs)
Held-out exact accuracy is **0 in every run of every config**: entity subsets N=6/10/16 →
0/720, 0/600, 0/420 (10 seeds each); 900-step check (train recovers to .875–1.0) → still 0.0
×10; pairs16 → 0 ×10 for BOTH seen-entity-unseen-pair and unseen-entity (demoing fox/colour
does not help fox/food); every overtraining rung through 30× → 0; N=64 at 4.0e9 MACs → 0.
Attribute-level correctness .008–.031 is AT OR BELOW uniform chance (colour .100, home .053,
food .045). Mechanism: answering requires copying the entity name from the question; a
fixed-window MLP without attention can't learn positional copying (entity-echo ≈ 0: 1/1,740).
**The quiz must use exactly the demonstrated (entity, relation) pairs.**
Knowledge tax on top: bare-prompt recall on held-out entities collapses 100% → 21–44% after
SFT. What it says instead (held-out Qs, means): answer-shaped line .270–.410, names a demoed
entity .17–.24, true fact about the wrong animal .12–.17.

## E. The born liar (pairs16 config, lr 0.02, FAKE2 probes)
Before SFT: 0/24 fake-entity questions get an `a: ` response (0/360 at T=0.8). After:
**`a: ` marker .675 mean, per seed .833,.542,.625,.583,.542,.708,.792,.750,.583,.792**;
fully well-formed fake answer .254 (per seed .250,.208,.333,.083,.292,.333,.333,.167,.250,.292);
marker-but-garbled .421; **echoes the fake name 0/240 (0/1,860 across all configs); refusal
structurally impossible**. MUST use length-matched 3–5-char fakes
(yeti, wisp, dodo, puma, lynx, imp, elf, orc): task-named dragon/robot/ghost mostly fail
(dragon 0/30 marker, robot 1/30, ghost 0/30 — 6 letters exceeds every real name).
Best pinned probes (marker / well-formed over 10 seeds): **`what does the puma eat?` 10/10 &
6/10**; `where does the elf live?` 9/10 & 5/10; `what does the wisp eat?` 9/10 & 5/10;
worst: imp-eat 1/10, orc-eat 2/10. Caption-gold verbatim outputs (quote only with the pinned
seed): seed 1 `what colour is the wisp?` → `a: the wasp is gold.` (wasp's TRUE colour —
nearest-neighbour, don't caption as "made up"); seed 2 `where does the dodo live?` →
`a: the brag lives in the pond.`; seed 0 `what does the lynx eat?` → `a: the mole eats worms.`
(true fact, wrong animal). Liar scaling at fixed budget: marker .675/.825/.908/.950 for
16/30/48/90 demos — but train exact collapses 1.0 / .60–.97 / .29–.50 / .02–.09. Ship 16.

## F. Loss masking A/B (both arms 300×24, lr 0.02 = 4.958e8 MACs each; run both live)
Masked (ships): train **16/16 on 10/10 seeds**, answer-line loss 0.0028 [.0016–.0062].
Unmasked: train **88.1% [56.2–100.0]**, 16/16 on only 1/10 seeds (per-seed %: 93.8, 93.8,
56.2, 81.2, 93.8, 87.5, 87.5, 100.0, 93.8, 93.8); loss 0.0229 [.0073–.0670] = 8.3× —
half its gradient is spent predicting prompt text. Second demoable effect, format bleed:
free generation emits `q:`/`a:` lines **15.3% [11.9–18.9] unmasked vs 0.6% [0.0–3.1] masked**
(0.0% on 8/10 seeds) — the unmasked model even quizzes itself with its own training demos.
NOT demoable: question echo (masked 2.3% [0–6.8] vs unmasked 4.7% [0–8.1] greedy —
overlapping spreads; teach echo as prose about real-scale models). Honest reversal the page
must not hide: with NO trailing newline the unmasked arm answers 83.8% vs masked 31.7%
(masking leaves the `?`→`\n`→`a:` transition untrained) — **the demo UI must always append
the newline**. Both arms pay the same ~51% library tax. Do not claim unmasked "cannot" learn
the answers — fixed-steps A/B only; it would likely catch up with more steps (not measured).

## G. Overtraining and the alignment tax (masked, lr 0.02, rungs to 30×)
The tax is FRONT-LOADED: facts-90 100% → **71.9% [61.1–77.8] after 25 steps** (train still
0/16) → ~50% by step 50 → flat forever (1×: 51.3% [41.1–58.9]; 30×: 51.4% [45.6–56.7];
per-seed 1×→30×: 5 up, 4 down, 1 same). Base-corpus loss 0.01402 → 0.5286 @1× → 0.5445 @30×
(+3%). Mechanism: weight-L2 from base saturates (6.38 @25, 9.92 @300, 10.18 @9000 — 97% of
all movement inside the normal run); answer loss 0.00285 → 0.00014 (gradients vanish).
The ONE metric that moves with overtraining: **parroting** (fresh question → verbatim
copybook answer) 8.1% → 10.1% → 12.0% → 14.3% at 1×/3×/10×/30×, non-decreasing in 10/10
seeds; per-seed of 74: 9→10, 6→13, 5→8, 5→17, 3→5, 10→15, 8→12, 1→4, 4→10, 9→12 —
**pin seeds 3, 5, or 1 for this demo; seeds 7 and 4 barely move**. No answer collapse:
distinct outputs 71.4 → 69.0 of 74, top-answer share 3.1% → 4.1%. What survives at 1×:
demoed pairs 87.5%, other relations of demoed entities 43.9%, undemoed entities 43.1%.
The damage dial is lr, not steps: **lr 0.05 leaves 13.3% [8.9–16.7] of the library at 1×**
(base loss ~2.72, weight-L2 23.7 = 2.4×) while train still hits 91.2% — the base report's
"lr 0.05 → 14/16" hid this. Compute: 30× = 1.487e10 MACs ≈ 60–90 s JS (parroting is only
clear at 10–30×; the front-loaded tax is free — checkpoint the library score during the
normal 300 steps). Replay mixing is a measured NEGATIVE (25% replay: train 13–16/16, facts
46–56/90; 50% fresh-start replay: train 8–14/16) — do not ship replay; gentler lr is the
only validated mitigation (lr .015: facts 46–67/90; lr .01: 59–75 but seed 8 = 15/16).

## Fact-retention after SFT — quote only with lr + harness attached
The "library score" after the 16-demo run depends on lr and demo set:
lr 0.015 (exp-A, ships): 57,59,61,67,47,59,56,63,56,46 /90. lr 0.02: exp-A grid 33–48;
exp-B pairs16 37–47; exp-C 37–53 (mean 51.3%). DO NOT quote a single retention number
without naming the lr and seed — the honest sentence is "roughly half the library survives
the shipping run (about 46–67/90 at lr 0.015)". Any UI that lets readers type base-style
prompts (`the fox is `) into the AFTER model will garble ~half the time; only demoed facts
survive well (87.5%).

## Stories the data does NOT support (merged from all four agents)
1. **Any generalization to un-demoed (entity, relation) pairs.** 0 exact in every seed of
   every config, including 4.0e9-MAC N=64 and 30× overtraining. Never build a beat that
   expects it; quiz demoed pairs only.
2. **"SFT unlocks knowledge the base already had."** Attribute-level accuracy .008–.031 is
   at/below uniform chance (.045–.100) — never caption held-out output as "above chance".
3. **"The knowledge is still in there, only the format is missing."** False after SFT:
   held-out bare recall falls 100% → 21–44%. Only the PRE-SFT base knows all 90 facts.
4. **"It confidently invents facts about the dragon."** dragon = 0/30 `a: ` marker.
   The liar demo requires the FAKE2 length-matched names + a pinned seed + validated probe.
5. **"Hallucinations are always fluent."** Well-formed fake answers 25.4% (per-seed
   .083–.333); 42.1% are `a: ` + garble. The robust claim is the HABIT: 0% → 67.5%
   tries-to-answer, never refuses, never says the animal doesn't exist.
6. **"It made that fact up"** — check the pinned line first: 7.5% of fake probes recite a
   TRUE fact about a different real animal (wisp→wasp is retrieval, not invention).
7. **"Unmasked SFT makes the model echo the question."** 2–5% both arms, overlapping
   spreads; the no-newline probe's gap runs the WRONG way (masked continues question-text
   18.3% vs unmasked 6.7%). Echo is prose about real models, not a demo here.
8. **"Keep training and you forget more and more."** Facts are FLAT 1×→30× (51.3→51.4%,
   direction inconsistent across seeds); base loss +3%. The forgetting (100%→~51%) happens
   inside the first 300 steps — attribute the tax to fine-tuning itself, not extra steps.
9. **"Every question starts getting the same memorized answer."** No collapse (top share
   3–4%, 69/74 distinct at 30×). Supported claim: fresh questions answered with a verbatim
   copybook line roughly double, 8.1% → 14.3% mean (per-seed gains +1 to +12 of 74).
10. **"Overfitting kills held-out generalization."** Vacuous — held-out is 0% before
    overtraining begins (floor effect).
11. **"The base knowledge survives SFT."** The shipping run already halves the library;
    "lr 0.05 works (14/16)" is misleading — it leaves 13% of the library.
12. **Replay mixing as a forgetting fix.** Measured negative in budget: costs demo mastery,
    does not restore facts.
13. **"~2–3 s in the browser."** Estimated from Part 2's measured JS throughput; the MACs
    products (4.956e8 shipping; 1.487e10 at 30×) are exact, the seconds are not.
14. **Mixing loss scales.** Smoke's 2.5962 (first minibatch) vs dynamics' 3.98–5.02
    (full answer set at step 0) are different measures — quote either, labelled, never both.
