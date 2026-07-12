# The Machine That Teaches Itself

**The Machine That Teaches Itself** is an interactive educational series exploring
how modern AI learns without a human handing it answers — self-supervised learning,
and the strange loop of a model teaching a copy of itself.

**Live:** https://princello.github.io/self-supervised-learning/

## Core concept

One game, played two ways: **hide something, guess it, check yourself.** Part 1
plays that game against the world — no teacher, no answer key, just the trick
behind GPT, CLIP, and every modern foundation model. Part 2 plays the same game
against a copy of yourself: a trained model pouring what it knows into a smaller
model, or into a fresh copy of its own architecture, no new data required. As the
content puts it, distillation and self-supervision are one idea wearing two costumes.

## Parts

1. **[Self-supervised learning](https://princello.github.io/self-supervised-learning/self-supervised-learning.html)**
   (EN / 中文 / ES) — cloze (masked-LM) game, an MAE inpainting toy, a real InfoNCE
   contrastive-learning playground (gradient descent you can watch, including a
   "collapse" button), and a live linear-probe payoff demo.
2. **[Knowledge distillation & self-distillation](https://princello.github.io/self-supervised-learning/knowledge-distillation.html)**
   (EN) — a softmax-temperature "dark knowledge" hero, a live hard-vs-soft-label
   training race with guess-first bets, an exact self-distillation "photocopier"
   (Mobahi–Farajtabar–Bartlett, with a real collapse-generation guessing game),
   and a born-again "twin experiment."

Part 2 links back to Part 1 and vice versa, so the series is browsable end to end.
Part 3 (the preference-optimization arc: SFT, RLHF, DPO) is in the still.

## Technical approach

The distinguishing feature is a commitment to **real computation, never a canned
animation.** Every live demo runs its own math in the browser as you watch:
finite-difference gradient descent, exact kernel ridge regression via
eigendecomposition, closed-form softmax/temperature arithmetic. Where a demo is
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
index.html, self-supervised-learning.html, knowledge-distillation.html
                built, fully standalone pages — also what GitHub Pages serves
```

Every built page is a single self-contained HTML file — no external requests, no
build step required to *read* it. Open any of them directly in a browser and it
works completely offline, fonts and all.

## Building

```
python3 build.py
```

Regenerates `index.html`, `self-supervised-learning.html`, and
`knowledge-distillation.html` at the repo root. No dependencies beyond Python 3's
standard library.

## Licensing

Content and code are distributed under [CC BY 4.0](LICENSE), permitting
educational reuse with attribution.
