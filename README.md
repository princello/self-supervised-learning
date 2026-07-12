# The Machine That Teaches Itself — an explainer series

Interactive, single-file HTML explainers on how modern AI learns, readable at four
depths (Explorer / Student / Engineer / Researcher) with live in-browser demos —
no build tooling required to read them.

## Parts

1. **Self-supervised learning** — `src/ssl-body.html` → `dist/self-supervised-learning.html`
   EN / 中文 / ES. Demos: cloze (masked-LM) game, MAE inpainting toy, a real
   InfoNCE contrastive-learning playground (gradient descent you can watch,
   including a "collapse" button), and a live linear-probe payoff demo.
2. **Knowledge distillation & self-distillation** — `src/distillation-body.html` → `dist/knowledge-distillation.html`
   English only (for now). Demos: a softmax-temperature "dark knowledge" hero,
   a live hard-vs-soft-label training race with guess-first bets, an exact
   self-distillation "photocopier" (Mobahi–Farajtabar–Bartlett, with a real
   collapse-generation guessing game), and a born-again "twin experiment."

Part 2 links back to Part 1 and vice versa, so the series is browsable end to end.

## Structure

```
src/            source HTML bodies (fonts spliced in at build time via /*__FONTS__*/)
fonts/          base64-encoded @font-face CSS (Bricolage Grotesque, Atkinson Hyperlegible)
dist/           built, fully standalone pages — open directly in a browser, works offline
build.py        splices fonts/fonts.css into each src/*-body.html, writes dist/
```

## Building

```
python3 build.py
```

Regenerates everything in `dist/`. No dependencies beyond Python 3's standard library.

## Reading

Open any file in `dist/` directly in a browser — everything (demos, fonts, styling)
is inlined, so it works completely offline. No server needed.

Each page has a four-position depth dial (top right) and, on the SSL page, a
language dial. Reading depth and language are reflected in the URL hash
(`#d2`, `#d3-zh`, …) so links can deep-link to a specific depth.

## Design notes

Every live demo is real computation running in the browser — finite-difference
gradient descent, exact kernel ridge regression with eigendecomposition, etc. —
not a canned animation. Where a demo is deliberately simplified or its setup
favors a particular outcome, the page says so explicitly at every reading depth.
