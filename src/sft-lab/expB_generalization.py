"""Experiment B — the two money demos, validated offline.

B1  HELD-OUT ENTITIES (generalization): SFT on demos covering only a subset of
    entities (6 / 10 / 16 of 30, all 3 attribute types each), dynamics-style
    config (SGD+momentum 0.9, lr 0.02, 300 steps x batch 24, loss on the
    answer line only, incl. the 'a: ' marker). Quiz the REMAINING entities.
    10 seeds per config. Hypothesis under test: base already knows the facts,
    SFT only installs the answering habit -> held-out accuracy far above
    chance. Measured honestly, per seed, per attribute type, plus:
      - does the model still KNOW the held-out facts after SFT
        (bare-prompt completion 'the owl is ' etc.)?
      - what does it actually say (right format? right entity? whose fact?)

B2  THE BORN LIAR (hallucination): after SFT, ask about entities that exist
    nowhere in any corpus (dragon, robot, ghost, troll, yeti, wisp).
    Measure answer-format rate, made-up-attribute rate, refusal/garbage rate,
    fake-name echo rate. Headline config = pairs16 (the page's live-demo
    config: 16 random (entity, relation) demos), 10 seeds; subset configs
    reported as robustness.

B3  BEFORE: identical probes against the raw base model.

Pure numpy, explicit seeds, fully reproducible. Writes partial results to
results_B.json after every stage (checkpoint-friendly).

Usage:  python3 expB_generalization.py [--stages base,subsets,pairs16,offline]
"""

import argparse
import json
import os
import re
import time

import numpy as np

import model as M
from gen_corpus import ENTITIES, RELATIONS
from eval_base import question_text, sft_demo_text, build_sft_dataset, FACT_PROMPTS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results_B.json")

SEEDS = list(range(10))
STEPS, BATCH, LR, MOM = 300, 24, 0.02, 0.9

# entities that exist nowhere in any corpus; all chars in-vocab (a-z)
FAKE_ENTITIES = ["dragon", "robot", "ghost", "troll", "yeti", "wisp"]

# page-ready fake set: 3-5 letter names matching real-name length statistics
# (real names are 3-5 chars; 'dragon' (6) is longer than any real name and
# measurably kills the answer reflex — see results)
FAKE2 = ["yeti", "wisp", "dodo", "puma", "lynx", "imp", "elf", "orc"]

REL_IDX = {"colour": 0, "home": 1, "food": 2}
ATTR_VALUES = {rel: {ENTITIES[n][i] for n in ENTITIES} for rel, i in REL_IDX.items()}

# every well-formed fact-sentence shape the corpus taught, in answer form
ANSWER_FORMS = [
    (re.compile(r"^a: the ([a-z]+) (?:is|looks) ([a-z]+)\.$"), "colour"),
    (re.compile(r"^a: the ([a-z]+) (?:lives|rests) in the ([a-z]+)\.$"), "home"),
    (re.compile(r"^a: the ([a-z]+) (?:eats|likes) ([a-z]+)\.$"), "food"),
]


def true_attr(entity, relation):
    return ENTITIES[entity][REL_IDX[relation]]


def parse_answer(line):
    """If line is a well-formed 'a: <fact sentence>.', return its parts."""
    for rx, rel in ANSWER_FORMS:
        m = rx.match(line)
        if m:
            return {"entity": m.group(1), "relation": rel, "attr": m.group(2)}
    return None


def canonical_want(entity, relation):
    from gen_corpus import canonical_answer
    return f"a: {canonical_answer(entity, relation)}"


def load_base():
    return M.load_json(os.path.join(HERE, "base_weights.json"))


def answer_line(p, cfg, question):
    """Greedy answer to 'q: <question>\\n' (the page's post-SFT decode)."""
    return M.sample(p, cfg, f"q: {question}\n", 45, greedy=True, stop="\n").strip()


def sft_run(base_p, cfg, demo_pairs, train_seed, steps=STEPS, batch=BATCH, lr=LR):
    """One SFT run from the shipping base weights. Returns tuned params."""
    p = {k: v.copy() for k, v in base_p.items()}
    demos = [sft_demo_text(n, r) for n, r in demo_pairs]
    X, y = build_sft_dataset(demos, cfg["K"])
    rng = np.random.default_rng(train_seed)
    step_fn = M.make_sgd(p, lr, momentum=MOM)
    final_loss = None
    for _ in range(steps):
        idx = rng.integers(0, len(y), batch)
        final_loss, g = M.loss_and_grads(p, X[idx], y[idx])
        step_fn(g)
    macs = steps * batch * 3 * M.macs_per_position(cfg)
    return p, {"n_demos": len(demos), "n_answer_positions": int(len(y)),
               "final_batch_loss": round(float(final_loss), 4), "macs": macs}


def classify_qa(p, cfg, entity, relation, demoed_entities):
    """Ask one question, classify the greedy output line."""
    line = answer_line(p, cfg, question_text(entity, relation))
    want = canonical_want(entity, relation)
    ans = parse_answer(line)
    out = {
        "entity": entity, "relation": relation, "line": line,
        "exact": line == want,
        "answer_format": ans is not None,
        "form_matches_question": bool(ans and ans["relation"] == relation),
        "entity_echo": bool(ans and ans["entity"] == entity),
        "attr_correct": bool(ans and ans["relation"] == relation
                             and ans["attr"] == true_attr(entity, relation)),
        "named_real_entity": bool(ans and ans["entity"] in ENTITIES),
        "named_demoed_entity": bool(ans and ans["entity"] in demoed_entities),
        "stated_fact_true": bool(ans and ans["entity"] in ENTITIES
                                 and ans["attr"] == true_attr(ans["entity"], ans["relation"])),
        "fabricated_known_attr": bool(ans and ans["relation"] == relation
                                      and ans["attr"] in ATTR_VALUES[relation]),
    }
    return out


def knowledge_probe(p, cfg, entities):
    """Bare-prompt fact completion (the base model's own skill) on a set of
    entities: does the model still KNOW these facts after SFT?"""
    n_ok, by_rel, rows = 0, {r: 0 for r in RELATIONS}, []
    for n in entities:
        for rel in RELATIONS:
            tpl, want_tpl = FACT_PROMPTS[rel]
            got = M.sample(p, cfg, tpl.format(n=n), 16, greedy=True, stop=".")
            want = want_tpl.format(a=true_attr(n, rel))
            ok = got == want
            n_ok += ok
            by_rel[rel] += ok
            rows.append({"entity": n, "relation": rel, "got": got, "ok": bool(ok)})
    total = len(entities) * 3
    return {"n_ok": n_ok, "n_total": total,
            "by_relation": {r: f"{by_rel[r]}/{len(entities)}" for r in RELATIONS},
            "fails": [r for r in rows if not r["ok"]]}


def liar_probe(p, cfg, fakes=FAKE_ENTITIES):
    """Ask the 3 question types about each fake entity; classify greedily."""
    rows = []
    for fake in fakes:
        for rel in RELATIONS:
            line = answer_line(p, cfg, question_text(fake, rel))
            ans = parse_answer(line)
            rows.append({
                "fake": fake, "relation": rel, "line": line,
                "starts_a_marker": line.startswith("a:"),
                "answer_format": ans is not None,
                "form_matches_question": bool(ans and ans["relation"] == rel),
                "fabricated_known_attr": bool(ans and ans["relation"] == rel
                                              and ans["attr"] in ATTR_VALUES[rel]),
                "echoes_fake_name": bool(ans and ans["entity"] == fake),
                "named_real_entity": bool(ans and ans["entity"] in ENTITIES),
                "stated_fact_true": bool(ans and ans["entity"] in ENTITIES
                                         and ans["attr"] == true_attr(ans["entity"], ans["relation"])),
            })
    return rows


def rate(rows, key):
    return sum(r[key] for r in rows) / max(len(rows), 1)


def summarize_rows(rows, keys):
    return {k: {"n": sum(r[k] for r in rows), "of": len(rows),
                "rate": round(rate(rows, k), 4)} for k in keys}


HELD_KEYS = ["exact", "answer_format", "form_matches_question", "entity_echo",
             "attr_correct", "named_real_entity", "named_demoed_entity",
             "stated_fact_true", "fabricated_known_attr"]
LIAR_KEYS = ["starts_a_marker", "answer_format", "form_matches_question",
             "fabricated_known_attr", "echoes_fake_name", "named_real_entity",
             "stated_fact_true"]


def chance_baselines():
    """What 'guessing an attribute of the right type' would score."""
    out = {}
    for rel, i in REL_IDX.items():
        vals = [ENTITIES[n][i] for n in ENTITIES]
        uniq = sorted(set(vals))
        freq = {v: vals.count(v) for v in uniq}
        modal = max(freq.values())
        out[rel] = {
            "n_distinct_values": len(uniq),
            "uniform_guess_acc": round(1 / len(uniq), 4),
            "modal_value_acc": round(modal / len(vals), 4),
            "modal_value": max(freq, key=freq.get),
        }
    return out


# ------------------------------------------------------------------ stages
def checkpoint(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, RESULTS_PATH)


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"config": {"steps": STEPS, "batch": BATCH, "lr": LR, "momentum": MOM,
                       "loss": "answer line only, incl. 'a: ' marker",
                       "decode": "greedy, stop at newline",
                       "seeds": SEEDS, "fake_entities": FAKE_ENTITIES},
            "chance_baselines": chance_baselines()}


def stage_base(results):
    """BEFORE probes: raw base model, same questions, greedy + sampled."""
    p, cfg, _ = load_base()
    # (a) all 90 real questions, greedy 'q: <q>\n' continuation
    rows = []
    for n in ENTITIES:
        for rel in RELATIONS:
            line = answer_line(p, cfg, question_text(n, rel))
            ans = parse_answer(line)
            rows.append({"entity": n, "relation": rel, "line": line,
                         "exact": line == canonical_want(n, rel),
                         "answer_format": ans is not None,
                         "is_question": "?" in line})
    real = {"n": len(rows),
            "exact_answers": sum(r["exact"] for r in rows),
            "answer_format": sum(r["answer_format"] for r in rows),
            "question_continuation": sum(r["is_question"] for r in rows),
            "examples": [{"q": question_text(r["entity"], r["relation"]),
                          "line": r["line"]} for r in rows[:6]]}
    # (b) fake-entity questions, greedy
    fake_rows = liar_probe(p, cfg)
    for r in fake_rows:
        r["is_question"] = "?" in r["line"]
    fake_greedy = {"rows": fake_rows, "summary": summarize_rows(fake_rows, LIAR_KEYS),
                   "question_continuation": sum(r["is_question"] for r in fake_rows)}
    # (c) fake-entity questions, 20 samples @ T=0.8 each (hero-style decode)
    rng = np.random.default_rng(202)
    n_ans, n_q, n_total, ex = 0, 0, 0, []
    for fake in FAKE_ENTITIES:
        for rel in RELATIONS:
            for _ in range(20):
                cont = M.sample(p, cfg, f"q: {question_text(fake, rel)}\n", 45,
                                rng=rng, temp=0.8, stop="\n").strip()
                n_total += 1
                n_ans += parse_answer(cont) is not None
                n_q += "?" in cont
                if len(ex) < 6:
                    ex.append({"q": question_text(fake, rel), "cont": cont})
    fake_sampled = {"n": n_total, "temp": 0.8, "seed": 202,
                    "answer_format": n_ans, "question_like": n_q, "examples": ex}
    # (d) what does the base model COMPLETE for a fake entity? 'the dragon is '
    completions = []
    for fake in FAKE_ENTITIES:
        for rel in RELATIONS:
            tpl, _ = FACT_PROMPTS[rel]
            got = M.sample(p, cfg, tpl.format(n=fake), 24, greedy=True, stop=".")
            completions.append({"prompt": tpl.format(n=fake), "got": got})
    results["before_base"] = {"real_questions_greedy": real,
                              "fake_questions_greedy": fake_greedy,
                              "fake_questions_sampled": fake_sampled,
                              "fake_bare_completions": completions}
    checkpoint(results)
    print(f"[base] real q: exact {real['exact_answers']}/90, "
          f"answer-format {real['answer_format']}/90, "
          f"question-cont {real['question_continuation']}/90")
    print(f"[base] fake q greedy: answer-format "
          f"{fake_greedy['summary']['answer_format']['n']}/18, "
          f"question-cont {fake_greedy['question_continuation']}/18; "
          f"sampled(360@T.8): answer-format {n_ans}, question-like {n_q}")


def run_subset_config(base_p, cfg, n_ent, seed, cfg_id, steps=STEPS):
    """One held-out-entities run: demo n_ent entities (all 3 relations)."""
    sel_rng = np.random.default_rng([cfg_id, seed])
    names = list(ENTITIES)
    demoed = [names[i] for i in sel_rng.choice(len(names), n_ent, replace=False)]
    held = [n for n in names if n not in demoed]
    demo_pairs = [(n, rel) for n in demoed for rel in RELATIONS]
    p, info = sft_run(base_p, cfg, demo_pairs,
                      train_seed=cfg_id * 1000 + seed + 500, steps=steps)
    train_rows = [classify_qa(p, cfg, n, r, set(demoed)) for n, r in demo_pairs]
    held_rows = [classify_qa(p, cfg, n, r, set(demoed))
                 for n in held for r in RELATIONS]
    know = knowledge_probe(p, cfg, held)
    liar = liar_probe(p, cfg)
    by_rel = {}
    for rel in RELATIONS:
        rs = [r for r in held_rows if r["relation"] == rel]
        by_rel[rel] = {"exact": sum(r["exact"] for r in rs), "of": len(rs)}
    return {
        "seed": seed, "demoed_entities": demoed, **info,
        "train": summarize_rows(train_rows, ["exact"]),
        "held": summarize_rows(held_rows, HELD_KEYS),
        "held_by_relation": by_rel,
        "held_knowledge_after_sft": {"n_ok": know["n_ok"], "n_total": know["n_total"],
                                     "by_relation": know["by_relation"]},
        "held_examples": [{"q": question_text(r["entity"], r["relation"]),
                           "line": r["line"]} for r in held_rows[:6]],
        "liar": {"summary": summarize_rows(liar, LIAR_KEYS),
                 "lines": [{"q": question_text(r["fake"], r["relation"]),
                            "line": r["line"]} for r in liar]},
    }


def stage_subsets(results, steps=STEPS, tag="subsets", sizes=(6, 10, 16)):
    base_p, cfg, _ = load_base()
    out = results.setdefault(tag, {})
    for n_ent in sizes:
        key = f"entities_{n_ent}"
        cfg_id = {6: 301, 10: 302, 16: 303}[n_ent] + (0 if steps == STEPS else 10)
        runs = []
        t0 = time.time()
        for seed in SEEDS:
            runs.append(run_subset_config(base_p, cfg, n_ent, seed, cfg_id,
                                          steps=steps))
        out[key] = {"n_entities_demoed": n_ent, "n_demos": n_ent * 3,
                    "steps": steps, "cfg_id": cfg_id, "runs": runs,
                    "seconds": round(time.time() - t0, 1)}
        # aggregate across seeds
        agg = {}
        for k in ["exact", "answer_format", "form_matches_question",
                  "attr_correct", "entity_echo",
                  "named_demoed_entity", "stated_fact_true"]:
            vals = [r["held"][k]["rate"] for r in runs]
            agg[k] = {"per_seed": vals, "mean": round(float(np.mean(vals)), 4),
                      "min": min(vals), "max": max(vals)}
        agg["train_exact"] = {"per_seed": [r["train"]["exact"]["rate"] for r in runs]}
        agg["knowledge_after"] = {"per_seed": [
            round(r["held_knowledge_after_sft"]["n_ok"]
                  / r["held_knowledge_after_sft"]["n_total"], 4) for r in runs]}
        out[key]["aggregate"] = agg
        checkpoint(results)
        print(f"[{tag}/{key}] steps={steps} "
              f"train exact per seed: {agg['train_exact']['per_seed']}")
        print(f"    held exact per seed: {agg['exact']['per_seed']} "
              f"(mean {agg['exact']['mean']:.4f})")
        print(f"    held answer-format mean {agg['answer_format']['mean']:.4f}, "
              f"attr-correct mean {agg['attr_correct']['mean']:.4f}, "
              f"knowledge-after per seed {agg['knowledge_after']['per_seed']}")


def build_pairs16_model(base_p, cfg, seed):
    """Deterministically rebuild the page-config SFT model for a seed."""
    sel_rng = np.random.default_rng([304, seed])
    pairs = [(n, rel) for n in ENTITIES for rel in RELATIONS]
    sel = sel_rng.choice(len(pairs), 16, replace=False)
    train_pairs = [pairs[i] for i in sel]
    p, info = sft_run(base_p, cfg, train_pairs, train_seed=304000 + seed + 500)
    return p, info, train_pairs, pairs


def stage_pairs16(results):
    """The page's live-demo config: 16 random (entity, relation) demos.
    Used for the born-liar headline + seen-entity/unseen-pair split."""
    base_p, cfg, _ = load_base()
    runs = []
    for seed in SEEDS:
        p, info, train_pairs, pairs = build_pairs16_model(base_p, cfg, seed)
        demoed_entities = {n for n, _ in train_pairs}
        train_rows = [classify_qa(p, cfg, n, r, demoed_entities)
                      for n, r in train_pairs]
        rest = [pr for pr in pairs if pr not in train_pairs]
        rest_rows = [classify_qa(p, cfg, n, r, demoed_entities) for n, r in rest]
        seen_e = [r for r in rest_rows if r["entity"] in demoed_entities]
        unseen_e = [r for r in rest_rows if r["entity"] not in demoed_entities]
        liar = liar_probe(p, cfg)
        # alignment tax: bare-prompt fact recall over ALL 90 facts after SFT,
        # split into the 16 demoed pairs vs the 74 un-demoed pairs
        know = knowledge_probe(p, cfg, list(ENTITIES))
        fail_pairs = {(r["entity"], r["relation"]) for r in know["fails"]}
        demoed_ok = sum((n, r) not in fail_pairs for n, r in train_pairs)
        undemoed_ok = sum((n, r) not in fail_pairs for n, r in rest)
        runs.append({
            "seed": seed, **info,
            "knowledge_after_all90": {"n_ok": know["n_ok"], "n_total": 90,
                                      "by_relation": know["by_relation"],
                                      "demoed_pairs_ok": f"{demoed_ok}/16",
                                      "undemoed_pairs_ok": f"{undemoed_ok}/74"},
            "train_exact": summarize_rows(train_rows, ["exact"])["exact"],
            "heldout_pairs_seen_entity": summarize_rows(seen_e, HELD_KEYS),
            "heldout_pairs_unseen_entity": summarize_rows(unseen_e, HELD_KEYS),
            "liar": {"summary": summarize_rows(liar, LIAR_KEYS),
                     "lines": [{"q": question_text(r["fake"], r["relation"]),
                                "line": r["line"]} for r in liar]},
        })
        checkpoint(results | {"pairs16": {"runs": runs}})
    agg = {
        "train_exact_per_seed": [r["train_exact"]["rate"] for r in runs],
        "held_seen_entity_exact_per_seed":
            [r["heldout_pairs_seen_entity"]["exact"]["rate"] for r in runs],
        "held_unseen_entity_exact_per_seed":
            [r["heldout_pairs_unseen_entity"]["exact"]["rate"] for r in runs],
        "knowledge_after_all90_per_seed":
            [r["knowledge_after_all90"]["n_ok"] for r in runs],
    }
    for k in LIAR_KEYS:
        vals = [r["liar"]["summary"][k]["rate"] for r in runs]
        agg[f"liar_{k}"] = {"per_seed": vals,
                            "mean": round(float(np.mean(vals)), 4),
                            "min": min(vals), "max": max(vals)}
    results["pairs16"] = {"n_demos": 16, "steps": STEPS, "runs": runs,
                          "aggregate": agg}
    checkpoint(results)
    print(f"[pairs16] train exact per seed: {agg['train_exact_per_seed']}")
    print(f"    liar answer-format mean {agg['liar_answer_format']['mean']:.4f} "
          f"(min {agg['liar_answer_format']['min']}, "
          f"max {agg['liar_answer_format']['max']})")
    print(f"    liar fabricated-attr mean "
          f"{agg['liar_fabricated_known_attr']['mean']:.4f}, "
          f"echoes-fake-name mean {agg['liar_echoes_fake_name']['mean']:.4f}")


def stage_liar2(results):
    """Page-ready fake names (length-matched to real names) on the exact
    pairs16 models, plus the same probes on the raw base (before)."""
    base_p, cfg, _ = load_base()
    before = liar_probe(base_p, cfg, FAKE2)
    runs = []
    for seed in SEEDS:
        p, info, train_pairs, _ = build_pairs16_model(base_p, cfg, seed)
        rows = liar_probe(p, cfg, FAKE2)
        runs.append({"seed": seed,
                     "summary": summarize_rows(rows, LIAR_KEYS),
                     "lines": [{"q": question_text(r["fake"], r["relation"]),
                                "line": r["line"],
                                "answer_format": r["answer_format"],
                                "fabricated_known_attr": r["fabricated_known_attr"]}
                               for r in rows]})
    # per (fake, relation) stats across seeds — which probe should the page use?
    per_probe = {}
    for seed_i, run in enumerate(runs):
        for r in run["lines"]:
            key = r["q"]
            d = per_probe.setdefault(key, {"marker": 0, "answer_format": 0,
                                           "fabricated": 0, "of": 0, "lines": []})
            d["of"] += 1
            d["marker"] += r["line"].startswith("a:")
            d["answer_format"] += r["answer_format"]
            d["fabricated"] += r["fabricated_known_attr"]
            if r["answer_format"]:
                d["lines"].append({"seed": seed_i, "line": r["line"]})
    agg = {}
    for k in LIAR_KEYS:
        vals = [r["summary"][k]["rate"] for r in runs]
        agg[k] = {"per_seed": vals, "mean": round(float(np.mean(vals)), 4),
                  "min": min(vals), "max": max(vals)}
    results["pairs16_liar2"] = {
        "fakes": FAKE2, "n_probes_per_seed": len(FAKE2) * 3,
        "before_base": {"summary": summarize_rows(before, LIAR_KEYS),
                        "lines": [{"q": question_text(r["fake"], r["relation"]),
                                   "line": r["line"]} for r in before]},
        "runs": runs, "aggregate": agg, "per_probe": per_probe,
    }
    checkpoint(results)
    print(f"[liar2] fakes={FAKE2}")
    print(f"    before: marker {summarize_rows(before, LIAR_KEYS)['starts_a_marker']['n']}"
          f"/{len(before)}, answer-format "
          f"{summarize_rows(before, LIAR_KEYS)['answer_format']['n']}/{len(before)}")
    for k in LIAR_KEYS:
        print(f"    after {k}: mean {agg[k]['mean']:.4f} "
              f"min {agg[k]['min']} max {agg[k]['max']}")
    best = sorted(per_probe.items(), key=lambda kv: -kv[1]["fabricated"])[:6]
    for q, d in best:
        print(f"    probe {q!r}: marker {d['marker']}/10, "
              f"well-formed {d['answer_format']}/10, fabricated {d['fabricated']}/10")


def stage_liar_scale(results):
    """Does the answer-at-all-costs reflex strengthen with demo count?
    Same budget (300 steps x batch 24) with 30 / 48 / 90 demos; FAKE2 probes.
    Also tracks demoed-question accuracy and the bare-recall tax."""
    base_p, cfg, _ = load_base()
    pairs = [(n, rel) for n in ENTITIES for rel in RELATIONS]
    out = {}
    for cfg_id, n_demos in [(401, 30), (402, 48), (403, 90)]:
        runs = []
        for seed in SEEDS:
            sel_rng = np.random.default_rng([cfg_id, seed])
            if n_demos == 30:
                rels = sel_rng.integers(0, 3, len(ENTITIES))
                train_pairs = [(n, RELATIONS[rels[i]])
                               for i, n in enumerate(ENTITIES)]
            elif n_demos == 90:
                train_pairs = pairs
            else:
                sel = sel_rng.choice(len(pairs), n_demos, replace=False)
                train_pairs = [pairs[i] for i in sel]
            p, info = sft_run(base_p, cfg, train_pairs,
                              train_seed=cfg_id * 1000 + seed + 500)
            demoed_entities = {n for n, _ in train_pairs}
            train_rows = [classify_qa(p, cfg, n, r, demoed_entities)
                          for n, r in train_pairs]
            liar = liar_probe(p, cfg, FAKE2)
            know = knowledge_probe(p, cfg, list(ENTITIES))
            runs.append({
                "seed": seed, **info,
                "train_exact": summarize_rows(train_rows, ["exact"])["exact"],
                "knowledge_after_all90": know["n_ok"],
                "liar": {"summary": summarize_rows(liar, LIAR_KEYS),
                         "lines": [{"q": question_text(r["fake"], r["relation"]),
                                    "line": r["line"]} for r in liar]},
            })
        agg = {"train_exact_per_seed":
                   [r["train_exact"]["rate"] for r in runs],
               "knowledge_after_all90_per_seed":
                   [r["knowledge_after_all90"] for r in runs]}
        for k in LIAR_KEYS:
            vals = [r["liar"]["summary"][k]["rate"] for r in runs]
            agg[f"liar_{k}"] = {"per_seed": vals,
                                "mean": round(float(np.mean(vals)), 4),
                                "min": min(vals), "max": max(vals)}
        out[f"demos_{n_demos}"] = {"n_demos": n_demos, "cfg_id": cfg_id,
                                   "steps": STEPS, "runs": runs, "aggregate": agg}
        results["liar_scale"] = out
        checkpoint(results)
        print(f"[liar_scale/demos_{n_demos}] train exact per seed: "
              f"{agg['train_exact_per_seed']}")
        print(f"    liar marker mean {agg['liar_starts_a_marker']['mean']:.4f}, "
              f"well-formed mean {agg['liar_answer_format']['mean']:.4f}, "
              f"fabricated mean {agg['liar_fabricated_known_attr']['mean']:.4f}")
        print(f"    knowledge after (of 90): "
              f"{agg['knowledge_after_all90_per_seed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="base,subsets,pairs16,liar2")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    results = load_results()
    t0 = time.time()
    if "base" in stages:
        stage_base(results)
    if "subsets" in stages:
        stage_subsets(results)
    if "pairs16" in stages:
        stage_pairs16(results)
    if "liar2" in stages:
        stage_liar2(results)
    if "liar_scale" in stages:
        stage_liar_scale(results)
    if "offline" in stages:
        # offline-only budget check: does full convergence unlock held-out?
        stage_subsets(results, steps=900, tag="subsets_offline_900steps",
                      sizes=(16,))
    print(f"total {time.time() - t0:.1f}s -> {RESULTS_PATH}")
