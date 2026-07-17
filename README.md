# The Machine That Teaches Itself

**The Machine That Teaches Itself** is an interactive educational series exploring
how modern AI comes to know things, hand them on, and finally answer you —
self-supervised learning, knowledge distillation, and supervised fine-tuning.

**Live:** https://princello.github.io/self-supervised-learning/

## Core concept

One game, played three ways: **hide something, guess it, check yourself.** Part 1
plays that game against the world — no teacher, no answer key, just the trick
behind GPT, CLIP, and every modern foundation model. Part 2 plays the same game
against a copy of yourself: a trained model pouring what it knows into a smaller
model, no new data required. Part 3 plays it one last time with a copybook of
sixteen perfect answers — the same next-token loss, pointed at demonstrations,
turning an autocomplete into an assistant (and paying a measurable price for it).

## Parts

1. **[Self-supervised learning](https://princello.github.io/self-supervised-learning/self-supervised-learning.html)**
   (EN / 中文 / ES) — cloze (masked-LM) game, an MAE inpainting toy, a real InfoNCE
   contrastive-learning playground (gradient descent you can watch, including a
   "collapse" button), and a live linear-probe payoff demo.
2. **[Knowledge distillation & self-distillation](https://princello.github.io/self-supervised-learning/knowledge-distillation.html)**
   (EN) — a draw-a-digit hero read live by a real embedded 3,610-parameter teacher
   network, a hard-vs-soft-label training race, Hinton's missing-digit experiment
   as a temperature ladder, and a "photocopier" chain of models trained on their
   own copies.
3. **[Supervised fine-tuning](https://princello.github.io/self-supervised-learning/supervised-fine-tuning.html)**
   (EN) — a real 23,319-parameter character-level language model pretrained on an
   8 KB storybook corpus you can read in full: ask it questions (it only asks more
   questions back), fine-tune it live on a 16-demonstration copybook, dissect what
   the fine-tune actually taught, catch it answering about animals that don't
   exist, and measure the alignment tax with the loss-masking and learning-rate
   dials.

The parts cross-link, so the series is browsable end to end, and a companion page
on reinforcement learning — [The Machine That Learns From
Consequences](https://claude.ai/code/artifact/62ebc2d2-cfd5-47b1-995c-e46edfae94c1)
— picks up where the copybook stops: learning from judgment instead of examples.

## Technical approach

The distinguishing feature is a commitment to **real computation, never a canned
animation.** Every live demo runs its own math in the browser as you watch:
finite-difference gradient descent, seeded SGD on real embedded networks (a digit
teacher in Part 2, a character-level language model in Part 3), closed-form
softmax/temperature arithmetic. Every quoted number is validated offline first
(10-seed experiment suites live in `src/sft-lab/` and `src/*-validated-configs.md`),
and the pages' captions are derived from what the browser actually computes at
press time, not from a script. Where a demo is
deliberately simplified or its setup favors a particular outcome, the page says
so explicitly, at every reading depth.

Each page is readable at four depths — Explorer, Student, Engineer, Researcher —
selectable with a dial in the top bar, or by deep-linking with a URL hash
(`#d2`, `#d3-zh`, …).

## Structure

```
src/            source HTML bodies (fonts spliced in at build time via /*__FONTS__*/)
fonts/          base64-encoded @font-face CSS (Bricolage Grotesque, Atkinson Hyperlegible)
build.py        splices fonts/fonts.css into each src/*-body.html, writes to repo root
index.html, self-supervised-learning.html, knowledge-distillation.html,
supervised-fine-tuning.html
                built, fully standalone pages — also what GitHub Pages serves
```

Every built page is a single self-contained HTML file — no external requests, no
build step required to *read* it. Open any of them directly in a browser and it
works completely offline, fonts and all.

## Building

```
python3 build.py
```

Regenerates the built pages at the repo root. No dependencies beyond Python 3's
standard library. (Part 3 additionally has `src/sft-lab/bake_page.py`, which
splices the validated model weights and JS engine into `src/sft-body.html` —
only needed when the lab assets change.)

## Licensing

Content and code are distributed under [CC BY 4.0](LICENSE), permitting
educational reuse with attribution.
