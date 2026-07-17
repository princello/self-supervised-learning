# Part 3 lab — Experiment B: generalization + hallucination (the two money demos)

Offline validation for the "held-out animals" and "born liar" sections.
Everything below is measured (`expB_generalization.py`, raw numbers in
`results_B.json`), seeded (SFT seeds 0-9 per config), pure numpy, and
reproducible in ~6 s total. Base model = shipping `base_weights.json`
(sha256 `652011...`). SFT config everywhere unless stated: SGD + momentum
0.9, lr 0.02, **300 steps x batch 24**, loss on the answer line only
(incl. the `a: ` marker), greedy decode stopped at newline.
**Compute per SFT run: 300 x 24 x 3 x 22,944 = 495,590,400 MACs = 4.956e8
(<= 5e8 budget).** The 900-step check is 1.487e9 (offline-only, 3x budget).

## Headline (what the page can honestly say)

1. **Held-out entities: the bet resolves NO, decisively.** SFT on 6/10/16
   entities (all 3 attribute types each), then quiz the remaining entities:
   **exact-answer accuracy is 0/720, 0/600, 0/420 — 0.0% in every one of 30
   runs, every seed, every subset size, every attribute type.** Even
   attribute-only correctness (right attribute word, any sentence) is
   0.8-3.1% — *below* uniform chance (colour 10.0%, home 5.3%, food 4.5%).
   The "base already knows it, SFT just unlocks it" hypothesis is dead at
   this scale, and it fails in *both* halves:
   - the answering habit does not transfer (0% exact, entity-echo ~0%), and
   - the bare-prompt knowledge itself is damaged by SFT (100% -> 21-44%).
2. **The born liar is real, but it is a *habit*, not eloquence.** Before
   SFT the model answers a question with 0/24 `a: ` responses (it emits
   more questions / corpus text). After the page-config SFT (16 demos), a
   question about a creature that exists nowhere gets an `a: `-marked
   response **67.5% of the time (per-seed 54.2-83.3%)** with length-matched
   fake names — and there is **never a refusal (structurally impossible)
   and never an echo of the fake name (0/1,860 probes across all configs)**.
   A *fully well-formed* fake answer happens 25.4% of the time; the rest is
   an `a: ` marker followed by confidently garbled text. The demo must use
   a pinned seed + probes validated below.

---

## B3. BEFORE: raw base model on the same probes

- All 90 real questions, prompt `q: <question>\n`, greedy: **exact answers
  0/90, answer-format 0/90**; 15/90 continue with another question, the
  rest continue with corpus-style fact lines or garble. (The page's hero
  "question begets question" uses continuation directly after `?` at T=0.8
  — that behaviour is 200/200, validated in REPORT-base.md item 2. After a
  *newline* the base mostly starts fact-like lines; still zero answers.)
- 18 fake-entity questions (dragon/robot/ghost/troll/yeti/wisp), greedy:
  **answer-format 0/18**; 360 samples @ T=0.8: **answer-format 0/360**.
- 24 page-set fake questions (FAKE2 below), greedy: **`a: ` marker 0/24,
  answer-format 0/24.**
- Bare completions of fake entities are already suggestive nearest-neighbour
  retrieval: `the wisp is ` -> `gold.`, `the wisp lives in the ` -> `nest.`,
  `the wisp eats ` -> `jam.` — exactly the **wasp**'s three facts;
  `the yeti is ` -> `black.`; `the dragon is ` -> `gres.` (garble).

## B1. Held-out entities (generalization)

SFT demos cover only N entities (all 3 relations each -> 3N demos); quiz
the remaining 30-N entities on all 3 relations. 10 seeds per N.

| config | demos | train exact per seed (300 steps) | held-out exact per seed |
|---|---|---|---|
| N=6  | 18 | 1.0, 1.0, 1.0, .944, 1.0, .944, 1.0, 1.0, 1.0, 1.0 | **0.0 x 10** (0/720 total) |
| N=10 | 30 | .967, .933, .70, .933, .833, .933, .967, .667, .867, .90 | **0.0 x 10** (0/600 total) |
| N=16 | 48 | .229, .292, .333, .50, .417, .313, .438, .438, .417, .292 | **0.0 x 10** (0/420 total) |
| N=16, 900 steps (offline, 1.49e9 MACs) | 48 | .875, .979, .896, .938, 1.0, .958, .896, .938, .896, .938 | **0.0 x 10** |

Per attribute type: colour 0, home 0, food 0 in every run (all zeros; see
`held_by_relation` in results_B.json). The 900-step row kills the
"it's just undertrained" objection: train recovers to 87.5-100% and
held-out stays exactly 0.

**What it says instead** (held-out questions, means over 10 seeds):

| metric | N=6 | N=10 | N=16 |
|---|---|---|---|
| answer-shaped line (`a: the X <verb> Y.`) | .270 (.181-.389) | .365 (.233-.500) | .410 (.310-.500) |
| sentence form matches question type | .204 | .295 | .329 |
| **echoes the queried entity** | **.000** | **.002 (1/600)** | **.000** |
| names a demoed entity instead | .168 | .235 | .212 |
| states a TRUE fact about a *different* animal | .135 | .170 | .119 |
| correct attribute for the queried entity | .008 (0-.028) | .008 (0-.033) | .031 (0-.048) |

Chance baselines for the attribute-correct row: uniform guess = colour
1/10 = .100, home 1/19 = .053, food 1/22 = .045; modal-value guess = .167
/ .133 / .133. Measured .008-.031 is **at or below chance** — SFT is not
retrieving the held-out fact even at attribute level.

Verbatim examples (N=6, seed 0): `what does the fox eat?` ->
`a: the bee eats crobs.`; `what colour is the fox?` ->
`a: the aot is gre woefs ai es what doas the w`.

**The alignment tax (new, page-worthy):** after SFT, bare-prompt recall
(`the owl is ` etc.) on the held-out entities collapses from the base's
100% to **29.2-44.4% (N=6), 21.7-43.3% (N=10), 21.4-33.3% (N=16)** per
seed. For the page config (16 random pairs, below): all-90-facts bare
recall drops to **37-47/90**, and the damage is uneven — colour survives
best (18-26/30), home worst (3-11/30); the 16 demoed pairs themselves stay
at 11-15/16. SFT overwrites the storyteller while installing the answerer.

**pairs16 (the page's live-demo config, 16 random (entity,relation) demos):**
train exact **16/16 on all 10 seeds** (final answer-line loss .0007-.019);
held-out pairs where the *entity appeared in another demo*: **0.0 x 10**;
held-out pairs with never-demoed entities: **0.0 x 10**. Demoing
(fox, colour) does not even help (fox, food). The quiz must use exactly
the demonstrated (entity, relation) pairs.

**Mechanism** (from REPORT-base.md / heldout_diagnostics.json, consistent
with everything above): answering requires copying the entity name from
the question into the answer; a fixed-window MLP without attention cannot
learn positional copying from <= 90 demos. Entity-echo ~0% here is that
mechanism measured directly.

## B2. The born liar (hallucination)

Fake entities never seen in any corpus. Two sets:
- hard set (task-specified): dragon, robot, ghost, troll, yeti, wisp
- **FAKE2 (page-ready, length-matched 3-5 chars like every real name):
  yeti, wisp, dodo, puma, lynx, imp, elf, orc**

**Name length is the whole game.** Real names are 3-5 chars; `dragon` (6)
is longer than any real name and its questions almost never trigger the
answer reflex. `a: `-marker rate by fake entity, pairs16 config, 10 seeds:
dragon 0/30, robot 1/30, ghost 0/30, troll 4/30 — vs **yeti 19/30, wisp
21/30**. The same split holds in every other config (dragon 0-1/30
everywhere). **Do not use dragon/robot/ghost on the page; use the FAKE2
names.**

**Page config (pairs16, 16 demos, 10 seeds), FAKE2 probes (24 per seed):**

| metric | before SFT | after SFT: mean (min-max per seed) |
|---|---|---|
| emits `a: ` marker | 0/24 | **.675 (.542-.833)** |
| fully well-formed fake answer | 0/24 | **.254 (.083-.333)** |
| `a: ` but garbled content | — | .421 |
| no marker (other text/garble) | 24/24 (all non-answer) | .325 |
| made-up but *real-vocabulary* attribute, right type | 0 | .125 (.042-.208) |
| states a true fact about a different real animal | 0 | .075 (0-.208) |
| names some real entity | 0 | .125 (0-.292) |
| **echoes the fake name** | 0 | **.000** (0/240; 0/1,860 across all configs) |
| refusal / "I don't know" | impossible | **impossible — no such behaviour exists in its world** |

**The reflex scales with demos (at fixed 4.956e8-MAC budget) but the quiz
dies:** `a: `-marker mean .675 (16 demos) -> .825 (30) -> .908 (48) ->
**.950 (90 demos)**; well-formed .254 -> .358 -> .367 -> .421; but train
exact collapses 1.00 -> .60-.97 -> .29-.50 -> **.02-.09**. 16 demos is the
right page config (perfect quiz + clear liar); 30 demos (one per entity,
train .60-.97) is the aggressive option only if the quiz is restricted to
verified demos.

**Best single probes** (pairs16, across 10 seeds — marker / well-formed):
`what does the puma eat?` **10/10, 6/10**; `where does the elf live?`
9/10, 5/10; `what does the wisp eat?` 9/10, 5/10; `where does the dodo
live?` 8/10, 5/10. Worst: `what does the imp eat?` 1/10, `what does the
orc eat?` 2/10 — probe choice matters as much as name choice.

Verbatim caption material (pairs16 + FAKE2, real outputs):
- seed 1: `what colour is the wisp?` -> **`a: the wasp is gold.`** — and
  seed 9: `what does the wisp eat?` -> **`a: the wasp eats jam.`**
  (nearest-neighbour hallucination: answers about the *wasp*, with the
  wasp's true facts)
- seed 2: `where does the dodo live?` -> **`a: the brag lives in the
  pond.`** (invents an animal *and* houses it)
- seed 0: `what does the lynx eat?` -> `a: the mole eats worms.` (true
  fact, wrong animal — confident deflection)
- seed 4: `what colour is the orc?` -> `a: the ant is black.`
- garbled-marker mode (the other ~42%): seed 0, `what colour is the
  yeti?` -> `a: the wren erts arolduewtaue re ts in the sn`

## Reproduce

    cd src/sft-lab
    python3 expB_generalization.py            # base,subsets,pairs16,liar2 (~4 s)
    python3 expB_generalization.py --stages liar_scale,offline

Stages checkpoint `results_B.json` incrementally. Config ids: subsets
301/302/303 (+10 for 900-step), pairs16 304, liar_scale 401/402/403;
demo-set selection rng = `default_rng([cfg_id, seed])`, training rng =
`default_rng(cfg_id*1000 + seed + 500)`.

## Caveats — framings the data does NOT support

- **"SFT unlocks knowledge the base already had" — NOT supported.**
  Held-out exact = 0.0% in all 40 runs; attribute-level accuracy
  (.008-.031) is at/below uniform chance (.045-.100). Do not caption the
  held-out section as "far above chance"; it is a clean negative.
- **"The knowledge is still in there, only the format is missing" — NOT
  supported after SFT.** Bare-prompt recall on held-out entities falls to
  21-44% (base: 100%). The honest framing: the *base* knew it (100%
  before); SFT both fails to transfer the habit and partially overwrites
  the knowledge (alignment tax).
- **"After SFT it confidently invents facts about the dragon" — NOT
  supported for 6-letter names.** dragon/robot/ghost mostly yield garble
  with no `a: ` marker (dragon: 0/30 marker in the page config). The liar
  demo requires length-matched names (FAKE2) and, for a clean single
  moment, a pinned seed + one of the validated probes above.
- **"Every hallucination is fluent" — NOT supported.** Fully well-formed
  fake answers are 25.4% (per-seed 8.3-33.3%); 42.1% are `a: ` + garble.
  The robust page claim is the *habit*: tries-to-answer 0/24 before vs
  67.5% (min 54.2%) after; use "always answers, never refuses, never
  admits the animal doesn't exist" (echo 0/1,860) rather than "always
  fluent lies".
- **"It makes up an attribute" needs care per probe:** 7.5% of probes
  recite a *true* fact about a different real animal (wisp->wasp is
  retrieval-flavoured, not pure invention). Captions derived from a pinned
  seed must quote that seed's actual line (all lines stored in
  results_B.json).
- The N=16 subset rows at 300 steps are undertrained (train .23-.50);
  held-out zero is nevertheless not budget-limited (900-step check:
  train .88-1.0, held-out still 0.0).
- Do not ship demos=90 in the reader budget (train exact .02-.09).

## Files

- `expB_generalization.py` — all stages (base / subsets / pairs16 / liar2
  / liar_scale / offline), pure numpy, seeded
- `results_B.json` — every number above, per run, incl. all 1,860 liar
  probe lines and all held-out example lines
