"""Acceptance-checklist evaluation for the base model (+ SFT smoke test).

Items:
  1. fact completion (greedy) for every entity x relation      -> results/eval_facts.json
  2. question continuation behaviour, 200 samples @ temp 0.8   -> results/eval_questions.json
  3. fluency samples from scratch                              -> results/eval_fluency.json
  4. SFT smoke test (16 demos, answer-chars-only loss)         -> results/sft_smoke.json
  5. determinism (retrain twice, compare sha256)               -> results/determinism.json

Usage: python3 eval_base.py [--items 1,2,3,4,5] [--weights base_weights.json]
"""

import argparse
import json
import os

import numpy as np

import model as M
from gen_corpus import ENTITIES, RELATIONS, all_answer_forms, canonical_answer, questions_for

HERE = os.path.dirname(os.path.abspath(__file__))


def load(weights="base_weights.json"):
    return M.load_json(os.path.join(HERE, weights))


def save_result(name, obj):
    with open(os.path.join(HERE, "results", name), "w") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------- item 1
FACT_PROMPTS = {
    "colour": ("the {n} is ", "{a}."),
    "home": ("the {n} lives in the ", "{a}."),
    "food": ("the {n} eats ", "{a}."),
}


def eval_facts(p, cfg, verbose=True):
    """Greedy-complete every entity x relation; correct iff the completion up
    to the first period is exactly the right attribute."""
    rows, n_ok = [], 0
    for n, (c, h, f) in ENTITIES.items():
        attrs = {"colour": c, "home": h, "food": f}
        for rel in RELATIONS:
            tpl, want_tpl = FACT_PROMPTS[rel]
            prompt = tpl.format(n=n)
            want = want_tpl.format(a=attrs[rel])
            got = M.sample(p, cfg, prompt, 16, greedy=True, stop=".")
            ok = got == want
            n_ok += ok
            rows.append({"entity": n, "relation": rel, "prompt": prompt,
                         "got": got, "want": want, "ok": bool(ok)})
    acc = n_ok / len(rows)
    by_rel = {}
    for rel in RELATIONS:
        rs = [r for r in rows if r["relation"] == rel]
        by_rel[rel] = sum(r["ok"] for r in rs) / len(rs)
    out = {"accuracy": acc, "n_ok": n_ok, "n_total": len(rows), "by_relation": by_rel,
           "failures": [r for r in rows if not r["ok"]], "rows": rows}
    if verbose:
        print(f"[1] fact completion: {n_ok}/{len(rows)} = {acc:.1%}  by_rel={by_rel}")
        for r in out["failures"]:
            print(f"    FAIL {r['prompt']!r} -> {r['got']!r} (want {r['want']!r})")
    return out


# ---------------------------------------------------------------- item 2
def first_segment(text):
    """Continuation text up to and including the first terminator (. or ?)."""
    for i, ch in enumerate(text):
        if ch in ".?":
            return text[: i + 1].strip(), ch
    return text.strip(), None


def load_corpus_sentences():
    sents = set()
    with open(os.path.join(HERE, "corpus.txt")) as f:
        for line in f:
            for s in line.strip().split("? "):
                s = s.strip()
                if s:
                    sents.add(s if s.endswith((".", "?")) else s + "?")
    return sents


def classify_continuation(cont, name, relation, corpus_sentences):
    seg, term = first_segment(cont)
    if term == "?":
        return "question", seg
    if term == ".":
        if seg in all_answer_forms(name, relation):
            return "answered_question", seg
        if seg in corpus_sentences:
            return "corpus_sentence", seg
        # well-formed-looking declarative not present verbatim in corpus
        return "other_declarative", seg
    return "unterminated", seg


def eval_questions(p, cfg, seed=42, n_questions=10, samples_per_q=20, temp=0.8,
                   verbose=True):
    rng = np.random.default_rng(seed)
    corpus_sentences = load_corpus_sentences()
    # pick n_questions (entity, relation) pairs spread across the whole world
    pairs = [(n, rel) for n in ENTITIES for rel in RELATIONS]
    pick = [pairs[i] for i in rng.choice(len(pairs), n_questions, replace=False)]
    q_text = {"colour": "what colour is the {n}?", "home": "where does the {n} live?",
              "food": "what does the {n} eat?"}
    counts = {"question": 0, "answered_question": 0, "corpus_sentence": 0,
              "other_declarative": 0, "unterminated": 0}
    per_q, examples = [], []
    for name, rel in pick:
        q = q_text[rel].format(n=name)
        qc = dict.fromkeys(counts, 0)
        for _ in range(samples_per_q):
            cont = M.sample(p, cfg, q, 90, rng=rng, temp=temp)
            cat, seg = classify_continuation(cont, name, rel, corpus_sentences)
            counts[cat] += 1
            qc[cat] += 1
            if len(examples) < 12:
                examples.append({"q": q, "cont": cont[:70], "cat": cat})
        per_q.append({"question": q, **qc})
    total = n_questions * samples_per_q
    ok_frac = (counts["question"] + counts["corpus_sentence"]
               + counts["other_declarative"]) / total
    ans_frac = counts["answered_question"] / total
    out = {"n_samples": total, "temp": temp, "counts": counts,
           "question_or_new_sentence_frac": ok_frac,
           "accidental_answer_frac": ans_frac,
           "question_frac": counts["question"] / total,
           "per_question": per_q, "examples": examples}
    if verbose:
        print(f"[2] question continuation ({total} samples @ T={temp}): {counts}")
        print(f"    question-or-new-sentence: {ok_frac:.1%}   accidental answer: {ans_frac:.1%}")
    return out


def eval_q_prefix(p, cfg, seed=43, n=50, temp=0.8, verbose=True):
    """Same hero question but prefixed 'q: ' (colon is unseen in base corpus)."""
    rng = np.random.default_rng(seed)
    prompt = "q: what colour is the fox?"
    corpus_sentences = load_corpus_sentences()
    conts = [M.sample(p, cfg, prompt, 90, rng=rng, temp=temp) for _ in range(n)]
    cats = {"question": 0, "answered_question": 0, "corpus_sentence": 0,
            "other_declarative": 0, "unterminated": 0}
    for c in conts:
        cat, _ = classify_continuation(c, "fox", "colour", corpus_sentences)
        cats[cat] += 1
    out = {"prompt": prompt, "n": n, "temp": temp, "counts": cats,
           "examples": [c[:70] for c in conts[:8]]}
    if verbose:
        print(f"[2b] 'q: ' prefix ({n} samples): {cats}")
    return out


# ---------------------------------------------------------------- item 3
def eval_fluency(p, cfg, seed=7, n_samples=5, length=300, temp=0.8, verbose=True):
    rng = np.random.default_rng(seed)
    corpus_sentences = load_corpus_sentences()
    samples, in_corpus, n_sent = [], 0, 0
    for _ in range(n_samples):
        s = M.sample(p, cfg, "\n", length, rng=rng, temp=temp)
        samples.append(s)
        text = s.replace("\n", " ")
        segs, cur = [], ""
        for ch in text:
            cur += ch
            if ch in ".?":
                segs.append(cur.strip())
                cur = ""
        for seg in segs:
            n_sent += 1
            in_corpus += seg in corpus_sentences
    frac = in_corpus / max(n_sent, 1)
    out = {"temp": temp, "verbatim_corpus_sentence_frac": frac,
           "n_sentences": n_sent, "n_verbatim": in_corpus, "samples": samples}
    if verbose:
        print(f"[3] fluency: {in_corpus}/{n_sent} = {frac:.1%} generated sentences are verbatim corpus sentences")
        print("    sample:", samples[0][:160].replace("\n", " / "))
    return out


# ---------------------------------------------------------------- item 4
def question_text(name, relation):
    return {"colour": f"what colour is the {name}?",
            "home": f"where does the {name} live?",
            "food": f"what does the {name} eat?"}[relation]


def sft_demo_text(name, relation):
    return f"q: {question_text(name, relation)}\na: {canonical_answer(name, relation)}\n"


def build_sft_dataset(demos, K):
    """Positions with loss = every char of the answer line 'a: ...\\n'
    (so the model also learns to emit the 'a: ' marker)."""
    Xs, ys = [], []
    for d in demos:
        ids = M.encode(d)
        padded = np.concatenate([np.full(K, M.PAD, np.int64), ids])
        a_start = d.index("\na: ") + 1  # index of 'a'
        for t in range(a_start, len(ids)):
            Xs.append(padded[t : t + K])
            ys.append(ids[t])
    return np.stack(Xs), np.array(ys)


def eval_sft_answers(p, cfg, pairs, verbose=False, tag=""):
    n_ok, rows = 0, []
    for name, rel in pairs:
        prompt = f"q: {question_text(name, rel)}\n"
        got = M.sample(p, cfg, prompt, 45, greedy=True, stop="\n")
        want = f"a: {canonical_answer(name, rel)}"
        ok = got.strip() == want
        n_ok += ok
        rows.append({"q": question_text(name, rel), "got": got.strip(),
                     "want": want, "ok": bool(ok)})
    if verbose:
        for r in rows:
            mark = "ok " if r["ok"] else "BAD"
            print(f"    {tag} {mark} {r['q']!r} -> {r['got']!r}")
    return n_ok, rows


def sft_smoke(seed=0, steps=300, batch=24, lr=0.02, verbose=True,
              weights="base_weights.json", n_train=16, n_held=8):
    p, cfg, meta = load(weights)
    rng = np.random.default_rng(seed + 100)
    pairs = [(n, rel) for n in ENTITIES for rel in RELATIONS]
    sel = rng.choice(len(pairs), n_train + n_held, replace=False)
    train_pairs = [pairs[i] for i in sel[:n_train]]
    held_pairs = [pairs[i] for i in sel[n_train:]]
    demos = [sft_demo_text(n, r) for n, r in train_pairs]
    X, y = build_sft_dataset(demos, cfg["K"])

    before_train, _ = eval_sft_answers(p, cfg, train_pairs)
    before_held, _ = eval_sft_answers(p, cfg, held_pairs)

    step_fn = M.make_sgd(p, lr, momentum=0.9)
    losses = []
    for s in range(steps):
        idx = rng.integers(0, len(y), batch)
        loss, g = M.loss_and_grads(p, X[idx], y[idx])
        step_fn(g)
        if s % 50 == 0 or s == steps - 1:
            losses.append({"step": s, "loss": round(loss, 4)})

    after_train, rows_t = eval_sft_answers(p, cfg, train_pairs, verbose, "train")
    after_held, rows_h = eval_sft_answers(p, cfg, held_pairs, verbose, "held")
    macs = steps * batch * 3 * M.macs_per_position(cfg)
    out = {
        "seed": seed, "steps": steps, "batch": batch, "lr": lr, "momentum": 0.9,
        "n_demos": len(demos), "n_held": len(held_pairs),
        "n_answer_positions": int(len(y)),
        "loss_curve": losses,
        "answer_rate_before": {"train": before_train / len(train_pairs),
                               "held": before_held / max(len(held_pairs), 1)},
        "answer_rate_after": {"train": after_train / len(train_pairs),
                              "held": after_held / max(len(held_pairs), 1)},
        "macs_fwd_bwd_total": macs, "macs_budget": 5e8,
        "rows_train": rows_t, "rows_held": rows_h,
    }
    if verbose:
        print(f"[4] SFT smoke: {len(demos)} demos, {steps} steps x batch {batch}, lr {lr}")
        print(f"    exact-answer rate before: train {before_train}/{len(train_pairs)}, held-out {before_held}/{len(held_pairs)}")
        print(f"    exact-answer rate after:  train {after_train}/{len(train_pairs)}, held-out {after_held}/{len(held_pairs)}")
        print(f"    loss first->last: {losses[0]['loss']} -> {losses[-1]['loss']}")
        print(f"    MACs (fwd+bwd approx 3x fwd): {macs:.3g} (budget 5e8)")
    return out


# ---------------------------------------------------------------- item 5
def determinism_check(verbose=True):
    from train_base import train
    hashes = []
    for run in range(2):
        p, cfg, meta, _ = train(seed=0, quiet=True)
        hashes.append(M.params_hash(p))
    same = hashes[0] == hashes[1]
    out = {"seed": 0, "hashes": hashes, "identical": same}
    if verbose:
        print(f"[5] determinism: run1 {hashes[0][:16]}... run2 {hashes[1][:16]}... identical={same}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="1,2,3,4,5")
    ap.add_argument("--weights", default="base_weights.json")
    args = ap.parse_args()
    items = set(args.items.split(","))
    p, cfg, meta = load(args.weights)
    print(f"model: {M.n_params(p)} params, K={cfg['K']} d={cfg['d']} H={cfg['H']} V={cfg['V']}")
    if "1" in items:
        save_result("eval_facts.json", eval_facts(p, cfg))
    if "2" in items:
        save_result("eval_questions.json", eval_questions(p, cfg))
        save_result("eval_q_prefix.json", eval_q_prefix(p, cfg))
    if "3" in items:
        save_result("eval_fluency.json", eval_fluency(p, cfg))
    if "4" in items:
        save_result("sft_smoke.json", sft_smoke(weights=args.weights))
    if "5" in items:
        save_result("determinism.json", determinism_check())
