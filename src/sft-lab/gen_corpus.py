"""Generate the micro-world corpus: a tiny storybook universe of 30 creatures,
each with a colour, a home, and a food.

Design constraints (see REPORT-base.md):
- Vocabulary: lowercase a-z, space, period, question mark, newline (31 chars
  incl. the reserved colon, which never appears here).
- Fact sentences in 6 templates (2 per relation) so facts repeat across forms.
- Quiz pages: runs of questions that are NEVER answered — this is what makes a
  base model continue a question with another question.
- Entity names <= 5 chars and home words <= 6 chars so that the full entity
  name always fits inside the model's K=26 char context window even in the
  longest template ("the snail rests in the garde|n" needs exactly 24 chars).
- Total <= ~10 KB (ships on the page in full).

Deterministic: rng seed 0.
"""

import numpy as np

# name: (colour, home, food)
ENTITIES = {
    "fox": ("red", "forest", "mice"),
    "owl": ("grey", "tree", "mice"),
    "frog": ("green", "pond", "flies"),
    "mole": ("brown", "burrow", "worms"),
    "whale": ("blue", "sea", "plankton"),
    "bee": ("gold", "meadow", "nectar"),
    "crow": ("black", "tree", "seeds"),
    "hare": ("white", "hill", "clover"),
    "wolf": ("grey", "cave", "deer"),
    "duck": ("green", "pond", "weeds"),
    "moth": ("grey", "lamp", "wool"),
    "snail": ("brown", "garden", "leaves"),
    "crab": ("pink", "reef", "worms"),
    "toad": ("tan", "marsh", "grubs"),
    "wren": ("brown", "hedge", "grubs"),
    "newt": ("black", "pond", "worms"),
    "vole": ("grey", "meadow", "roots"),
    "pike": ("green", "river", "fish"),
    "swan": ("white", "lake", "weeds"),
    "gull": ("white", "cliff", "fish"),
    "lark": ("tan", "sky", "bugs"),
    "hen": ("red", "barn", "corn"),
    "ant": ("black", "hill", "crumbs"),
    "wasp": ("gold", "nest", "jam"),
    "bat": ("black", "cave", "figs"),
    "koi": ("gold", "pond", "moss"),
    "jay": ("blue", "tree", "nuts"),
    "ram": ("white", "hill", "grass"),
    "eel": ("black", "river", "worms"),
    "seal": ("grey", "sea", "fish"),
}

RELATIONS = ["colour", "home", "food"]


def fact_sentences(name, canonical_only=False):
    c, h, f = ENTITIES[name]
    canonical = [
        f"the {name} is {c}.",
        f"the {name} lives in the {h}.",
        f"the {name} eats {f}.",
    ]
    if canonical_only:
        return canonical
    variants = [
        f"the {name} looks {c}.",
        f"the {name} rests in the {h}.",
        f"the {name} likes {f}.",
    ]
    return canonical + variants


def questions_for(name):
    return [
        f"what colour is the {name}?",
        f"where does the {name} live?",
        f"what does the {name} eat?",
    ]


def canonical_answer(name, relation):
    c, h, f = ENTITIES[name]
    return {
        "colour": f"the {name} is {c}.",
        "home": f"the {name} lives in the {h}.",
        "food": f"the {name} eats {f}.",
    }[relation]


def all_answer_forms(name, relation):
    """Every sentence that states the queried attribute (used to detect
    'accidentally answered the question' in eval)."""
    c, h, f = ENTITIES[name]
    return {
        "colour": [f"the {name} is {c}.", f"the {name} looks {c}."],
        "home": [f"the {name} lives in the {h}.", f"the {name} rests in the {h}."],
        "food": [f"the {name} eats {f}.", f"the {name} likes {f}."],
    }[relation]


def build_corpus(seed=0):
    rng = np.random.default_rng(seed)
    names = list(ENTITIES)

    # Fact pool: every fact in both templates once, plus the canonical form a
    # second time  ->  each fact appears 3x in text (2 canonical + 1 variant).
    facts = []
    for n in names:
        facts.extend(fact_sentences(n))
        facts.extend(fact_sentences(n, canonical_only=True))
    facts = [facts[i] for i in rng.permutation(len(facts))]

    # Quiz pool: every question exactly once, shuffled, in blocks of 5.
    qs = [q for n in names for q in questions_for(n)]
    qs = [qs[i] for i in rng.permutation(len(qs))]
    blocks = [" ".join(qs[i : i + 5]) for i in range(0, len(qs), 5)]

    # Interleave: one quiz line after every 15 fact lines.
    lines = []
    bi = 0
    for i, s in enumerate(facts):
        lines.append(s)
        if (i + 1) % 15 == 0 and bi < len(blocks):
            lines.append(blocks[bi])
            bi += 1
    lines.extend(blocks[bi:])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import os

    from model import VOCAB

    corpus = build_corpus(seed=0)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "corpus.txt")
    with open(path, "w") as f:
        f.write(corpus)

    used = sorted(set(corpus))
    assert all(c in VOCAB for c in used), "corpus uses chars outside VOCAB"
    assert ":" not in corpus, "colon must not appear in the base corpus"
    n_facts = sum(1 for l in corpus.splitlines() if "?" not in l)
    n_q = corpus.count("?")
    print(f"corpus bytes: {len(corpus.encode())}")
    print(f"fact lines: {n_facts}   questions: {n_q}")
    print(f"chars used ({len(used)}): {''.join(used)!r}")
    print(f"entities: {len(ENTITIES)}")
