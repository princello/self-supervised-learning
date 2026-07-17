"""Bake sft-assets.json — the single data blob the Part 3 page embeds.

Everything here is reconstructed with the validated lab code paths:
  - demoA16 / demoA4 / demoA64: exp-A's own sampling (expA_dynamics.run_sft
    seeds the rng and draws demo pairs FIRST — replicated verbatim below).
  - pairsB16: exp-B's build_pairs16_model selection (default_rng([304, seed])).
  - weights_b64: verbatim from base_weights_fp16.b64.json (no re-encode).

Writes ../sft-assets.json (i.e. src/sft-assets.json) and validates it loads.

Usage: python3 make_assets.py
"""

import json
import os

import numpy as np

import model as M
from eval_base import question_text
from gen_corpus import ENTITIES, RELATIONS, canonical_answer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "..", "sft-assets.json")

ALL_PAIRS = [(n, rel) for n in ENTITIES for rel in RELATIONS]  # 90 pairs

FACT_PROMPTS = {
    "colour": "the {n} is ",
    "home": "the {n} lives in the ",
    "food": "the {n} eats ",
}

# exp-B page-ready fakes (length-matched 3-5 chars; see expB FAKE2)
FAKE2 = ["yeti", "wisp", "dodo", "puma", "lynx", "imp", "elf", "orc"]

# hero chip questions: demoed pairs from demoA16, never toad/home (the base
# accidentally answers it). Each chip is verified by js/parity.mjs phase h
# (T=0.8 no-trailing-newline harness: >=95% question-or-new-sentence,
# 0 accidental correct answers per chip).
HERO_CHIPS = [
    ("wolf", "colour"),
    ("whale", "food"),
    ("swan", "home"),
]
# validated pinned probes (sft-validated-configs.md section E)
PINNED_PROBES = [
    {"fake": "puma", "relation": "food", "question": "what does the puma eat?"},
    {"fake": "elf", "relation": "home", "question": "where does the elf live?"},
    {"fake": "wisp", "relation": "food", "question": "what does the wisp eat?"},
]


def qa(pair):
    n, r = pair
    return {"q": question_text(n, r), "a": canonical_answer(n, r)}


def expA_demo_pairs(seed, n_demos):
    """EXACT replica of expA_dynamics.run_sft's demo draw (rng first draws
    the demo pairs; minibatch sampling uses the same stream afterwards)."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ALL_PAIRS), n_demos, replace=False)
    return [ALL_PAIRS[i] for i in sel]


def expB_pairs16(seed):
    """EXACT replica of expB_generalization.build_pairs16_model's draw."""
    sel_rng = np.random.default_rng([304, seed])
    sel = sel_rng.choice(len(ALL_PAIRS), 16, replace=False)
    return [ALL_PAIRS[i] for i in sel]


def main():
    with open(os.path.join(HERE, "base_weights_fp16.b64.json")) as f:
        w = json.load(f)
    with open(os.path.join(HERE, "corpus.txt")) as f:
        corpus = f.read()

    facts90 = []
    for n, rel in ALL_PAIRS:
        bare = FACT_PROMPTS[rel].format(n=n)
        sentence = canonical_answer(n, rel)
        assert sentence.startswith(bare)
        facts90.append({
            "entity": n,
            "relation": rel,
            "question": question_text(n, rel),
            "bare_prompt": bare,
            "answer_sentence": sentence,
            # exact-match target of the greedy bare-prompt completion
            # (eval harness want_tpl "{a}." — e.g. "red.", "forest.")
            "attribute": sentence[len(bare):],
        })

    demoA16_pairs = expA_demo_pairs(0, 16)
    demoA4_pairs = expA_demo_pairs(0, 4)
    demoA64_pairs = expA_demo_pairs(0, 64)
    pairsB16_pairs = expB_pairs16(0)

    fake_probes = [{"fake": fk, "relation": rel,
                    "question": question_text(fk, rel)}
                   for fk in FAKE2 for rel in RELATIONS]

    assets = {
        "vocab": list(M.VOCAB),                      # 31 chars, index order
        "K": 26,
        "dims": {"V": M.V, "d": 8, "H": 96, "K": 26},
        "param_order": M.PARAM_ORDER,
        "shapes": w["shapes"],
        "weights_b64": w["b64"],                     # verbatim fp16 LE blob
        "weights_sha256_b64": w["sha256_b64"],
        "corpus": corpus,
        "facts90": facts90,
        "demoA16": [qa(p) for p in demoA16_pairs],   # exp-A shipping, seed 0
        "demoA4": [qa(p) for p in demoA4_pairs],     # ladder rung, seed 0
        "demoA64": [qa(p) for p in demoA64_pairs],   # ladder rung, seed 0
        "pairsB16": [qa(p) for p in pairsB16_pairs],  # exp-B pairs16, seed 0
        "fakeEntities": FAKE2,
        "fakeProbes": fake_probes,                   # FAKE2 x 3 relations = 24
        "pinnedProbes": PINNED_PROBES,
        # question string -> full correct answer sentence (all 90; the hero
        # uses this to detect accidental correct answers in continuations)
        "answerFor": {question_text(n, r): canonical_answer(n, r)
                      for n, r in ALL_PAIRS},
        "heroChips": [question_text(n, r) for n, r in HERO_CHIPS],
        "excludedHero": [question_text("toad", "home")],
        "shipping": {"lr": 0.015, "steps": 300, "batch": 24, "momentum": 0.9,
                     "lrMaskDemo": 0.02, "lrDamage": 0.05,
                     "checkpoints": [0, 10, 25, 50, 75, 100, 150, 200, 250,
                                     300]},
        # pinned JS seeds (js/JS-SEED-REPORT.md) — the page reads these from
        # SFT_ASSETS.pinnedSeeds; never hard-code them in the page JS
        "pinnedSeeds": {"shipping": 7, "liar": 1, "overtraining": 4},
        # base accidentally answers 'where does the toad live?' — never use
        # it in a before beat (sft-validated-configs.md section A)
        "excludedHeroProbes": [{"entity": "toad", "relation": "home",
                                "question": question_text("toad", "home")}],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(assets, f)

    # validate + size report
    with open(OUT_PATH) as f:
        back = json.load(f)
    assert back["weights_b64"] == w["b64"]
    assert len(back["vocab"]) == 31 and "".join(back["vocab"]) == M.VOCAB
    assert len(back["facts90"]) == 90
    assert len(back["demoA16"]) == 16 and len(back["demoA4"]) == 4
    assert len(back["demoA64"]) == 64 and len(back["pairsB16"]) == 16
    assert len(back["fakeProbes"]) == 24
    assert len(back["answerFor"]) == 90
    assert len(back["heroChips"]) == 3
    assert back["pinnedSeeds"] == {"shipping": 7, "liar": 1,
                                   "overtraining": 4}
    assert question_text("toad", "home") not in back["heroChips"]
    assert all(q in back["answerFor"] for q in back["heroChips"])
    for f in back["facts90"]:
        assert f["bare_prompt"] + f["attribute"] == f["answer_sentence"]
    demo_qs = {d["q"] for d in back["demoA16"]}
    assert all(q in demo_qs for q in back["heroChips"]), \
        "hero chips must be demoed pairs (the after-model must answer them)"
    size = os.path.getsize(OUT_PATH)
    print(f"sft-assets.json: {size} bytes "
          f"(b64 {len(w['b64'])}, corpus {len(corpus.encode())})")
    print("demoA16 pairs:", [f"{n}/{r}" for n, r in demoA16_pairs])
    print("demoA4 pairs:", [f"{n}/{r}" for n, r in demoA4_pairs])
    print("pairsB16 pairs:", [f"{n}/{r}" for n, r in pairsB16_pairs])


if __name__ == "__main__":
    main()
