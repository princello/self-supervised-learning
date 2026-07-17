# Part 3 blueprint — "The Machine That Learns to Answer" (supervised fine-tuning)

## Place in the series
Pretraining (Part 1, The Machine That Teaches Itself) taught the model to talk.
RL (the companion page, The Machine That Learns From Consequences) teaches it judgment.
This page is the missing bridge: how an autocomplete becomes an assistant.
Distillation (Part 2) reappears as a cameo: SFT on a teacher's outputs IS distillation.

## Through-line (the mantra, restated by every section)
**"Teach the habit, not the facts."**
- Hero: it has the facts; it lacks the habit (a question gets continued, not answered).
- Copybook: a handful of demonstrations install the habit — same loss as pretraining,
  different data. Nothing else changes.
- The habit, dissected: fresh questions get the FORMAT (60.8% `a: ` marker) and none of
  the facts (0 exact, at/below chance) — SFT installed a habit, full stop. What looked
  like taught knowledge is memorized pairs. (LIMA/elicitation belongs to real-scale
  models with attention — L3/L4 only, clearly separated from what our toy shows.)
- The born liar: the habit fires even with nothing behind it — 0%→67.5% tries-to-answer,
  never refuses; hallucination is a well-trained habit with no library check.
- The two dials: the fine-tune itself eats half the library in its first 25 steps
  (front-loaded tax); the damage dial is lr, not steps; overtraining only grows parroting.
- Handoff: a habit can only be as good as the copybook — judgment needs consequences →
  link to the RL page.

## The one real object
A real char-level language model living in the page:
- Base model pretrained OFFLINE on a tiny storybook corpus (~25–40 creatures, each with a
  colour, a home, a food; plus unanswered quiz-style question lists). Weights embedded
  (fp16 base64), corpus embedded IN FULL and readable on the page ("its entire universe,
  ~8 KB of text").
- SFT happens LIVE in the browser: plain JS SGD+momentum on the embedded weights,
  rAF-animated, watchable loss + samples morphing mid-training.
- Reader's own typed question is a first-class citizen (constrained to the tiny vocab,
  with preset chips as the low-effort path).
All demo configs and quoted numbers come from src/sft-validated-configs.md (workflow
wf_94210f22-2f5) — do not deviate without re-validating.

## Sections (each at 4 depths, Part-1/2 architecture) — REVISED to the measured stories
All numbers/harnesses from src/sft-validated-configs.md; its "Stories the data does NOT
support" list is BINDING on every caption at every depth.
1. **Hero — "Ask it a question."** Base model + question chips + constrained free typing
   (lowercase vocab only; markers are lowercase `q:`/`a:`). BET before first sample:
   it answers / it asks more questions / it parrots you. Reveal: more questions
   (T=0.8, continue after `?` with NO trailing newline — the validated hero harness;
   never offer toad/home). Corpus viewer ("its entire universe, 8,352 bytes").
   Honesty note: toy model, toy world, real mechanism. Mantra under hero.
2. **Fluent ≠ helpful** — pretrain → SFT → RL positioning; links to Part 1 + RL page.
3. **The copybook → DEMO: Fine-tune it live.** The 16 demonstrations shown as a real
   copybook. Press = live SGD (300×24 @ lr .015, pinned JS seed): watchable loss fall +
   train-quiz score climbing + a LIBRARY gauge quietly bleeding 100%→~51% (front-loaded
   tax, checkpointed during the same run — debrief after, don't spoil). Before/after on
   the reader's hero question (after-beat = greedy from `q: …?\n`).
   Sub-bet, the ladder: at the SAME compute, is 64 demonstrations better than 16?
   NO — 4/4 and 16/16 ship, 64 → ~26% (under-trained). A compute lesson, framed honestly.
4. **DEMO: The habit, dissected.** What did it actually learn? Quiz beats: demoed pairs
   16/16; un-demoed pairs (same entities!) **0** — at/below chance; yet the FORMAT
   transfers (`a: ` marker on 60.8% of fresh questions). And the base-recall collapse on
   held-out entities (100% → 21–44%). The honest core: SFT installed a habit; the "facts"
   it answers are memorized pairs; at this scale it cannot copy an entity name (no
   attention — L3/L4 mechanism note), and real LLMs differ exactly there (LIMA at L3/L4:
   with attention, the habit unlocks the library; our toy dissects habit from library).
5. **DEMO: The born liar.** Length-matched fake creatures (puma/elf/wisp/dodo — NEVER
   dragon: 0/30). Before: 0% answer-shaped. After: 67.5% try-to-answer, refusal
   structurally impossible; only ~25% fluent (42% garbled) — caption honesty: "it always
   TRIES to answer", and check the wisp→wasp retrieval case before calling anything
   "made up". Hallucination's origin; why refusals must be trained → RL handoff.
6. **DEMO: The two dials.** (a) Masking A/B, both arms live at 300×24 lr .02; UI ALWAYS
   appends the trailing newline; demoable = accuracy 16/16 vs 88.1% + format bleed
   (self-quizzing 15.3% vs 0.6%); echo is NOT demoable (prose only). (b) The damage dial
   is lr, not steps: live lr rungs .015/.02/.05 → library retention ~60%/~48%/~13%;
   steps rungs 1×/3×/10× live (10× with time warning), 30× quoted from offline
   (parroting 8.1%→14.3%, pin the JS-equivalent of seeds 3/5/1; facts FLAT 1×→30× —
   never claim "keep training, forget more").
7. **The real thing** — scale prose: InstructGPT (~13k demos), LIMA (1,000), Alpaca
   (52k GPT-generated = SFT on a teacher's outputs = Part 2 in a chat template),
   behavior-cloning ceiling + exposure bias (L4), SFT as π_ref for RLHF, chat templates/
   EOS, why real models DO generalize (attention/capacity) while our toy cannot.
8. **Coda** — mantra, series nav: Part 1, Part 2, RL companion page.

## Interaction rules (non-negotiable, from explainer-series-principles)
Watchably-live rAF training (never instant blit); guess-first bets with graded reveals;
no spoilers in pre-demo prose at L1/L2 (debrief after); kid-gate ALL chrome; buttons do
exactly what labels say; .win chips only after results; honest rigging disclosures at
every depth; reduced-motion = chunked setTimeout(0), never frozen; window.__*Bench hooks
on every demo; captions DERIVED from measured state, never hardcoded to the happy seed.

## Visual identity (series member, own accent)
- Same paper/ink design system, fonts pipeline (/*__FONTS__*/), four-depth dial,
  data-tip tooltips, LSTR string table (EN-only launch, i18n-ready like Part 2).
- Accent: pencil-and-copybook OCHRE (light ~#A8641C family, dark ~#E2A45C) — distinct
  from Part 1 cobalt, Part 2 verdigris, RL teal. Highlight: graphite slate for the BASE
  model; accent = the fine-tuned model; dashed pink = held-out/never-taught (Part 2's
  withheld convention); copybook cards = ruled-paper texture.
- Favicon: ✏️. Built page: supervised-fine-tuning.html. Title: "The Machine That Learns
  to Answer". New artifact URL (Part 3), label sft-v1.

## Bench hooks (planned)
__sftBaseBench() → base fact-completion "N/M"; __sftHeroSample(q,seed); __sftTrainBench(n,seed)
→ before/after accuracy; __heldoutBench(seed); __liarBench(seed); __overBench(seed) →
perplexity+parroting ladder; __sftState() for caption checks.

## Cross-link updates at ship time
index landing page (Part 3 card replaces "in the still"; RL page added as companion card),
README, Part 1 footer (add Part 3), Part 2 footer/coda (add Part 3), memory files.
