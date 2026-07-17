"""Experiment C — the "two dials" demos for Part 3 (SFT explainer).

C1. LOSS MASKING A/B: identical SFT runs (16 demos, 300 steps x batch 24,
    lr 0.02, momentum 0.9) except which positions carry loss:
      arm "masked"   — loss on the answer line only (incl. 'a: ' marker);
                       this is the shipping config from eval_base.sft_smoke.
      arm "unmasked" — loss on ALL chars of each demo (prompt + answer).
    10 seeds. Measures answer accuracy, question-echo rate on fresh
    questions (greedy + T=0.8), a no-trailing-newline prompt probe,
    base-corpus loss (split into fact lines vs question lines), base fact
    recall (90 prompts, split by demoed/un-demoed), from-scratch generation
    drift, a step-trajectory (checkpoints 50..300), and verbatim transcripts.
    Trajectory evals are greedy-only, so they never touch the training rng
    stream: the 300-step endpoint stays bit-identical to eval_base.sft_smoke.

C2. OVERTRAINING / ALIGNMENT TAX: shipping (masked) config trained past the
    normal 300 steps: fine rungs inside the normal run (25..300 steps) plus
    overtraining rungs at 3x/10x/30x = 900/3000/9000 steps. One continuous
    run per seed, 10 seeds, all rung evals greedy/deterministic. At each
    rung: base-corpus loss/ppl (+line split), base fact recall (90 prompts,
    +split), trained-pair QA accuracy (16), held-out QA accuracy (8 + all 74
    un-demoed), parroting rate (fresh answers verbatim equal to a copybook
    answer line), question-echo rate, distinct-output collapse, answer-line
    loss. A supplement repeats the 4 headline rungs at lr 0.05 to test
    whether the overtraining curve shape is lr-specific.

Pure numpy, explicit seeds, everything deterministic. Results are written
incrementally to results_C.json (checkpointed after every unit of work).

Usage: python3 expC_masking_overtraining.py [--exp 1|2|2b|all] [--seeds 10]
"""

import argparse
import json
import os
import time

import numpy as np

import model as M
from eval_base import build_sft_dataset, sft_demo_text, question_text
from gen_corpus import ENTITIES, RELATIONS, all_answer_forms, canonical_answer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results_C.json")

# ------------------------------------------------------------ shipping config
STEPS_1X = 300
BATCH = 24
LR = 0.02
MOMENTUM = 0.9
N_TRAIN = 16
N_HELD = 8
TRAJ_CHECKPOINTS = [50, 100, 150, 200, 300]           # C1 (within normal run)
RUNGS = [25, 50, 75, 100, 150, 200, 300, 900, 3000, 9000]   # C2
HEADLINE_RUNGS = {300: 1, 900: 3, 3000: 10, 9000: 30}       # steps -> mult
WEIGHTS = "base_weights.json"

ALL_PAIRS = [(n, r) for n in ENTITIES for r in RELATIONS]

FACT_PROMPTS = {
    "colour": ("the {n} is ", "{a}."),
    "home": ("the {n} lives in the ", "{a}."),
    "food": ("the {n} eats ", "{a}."),
}

_CORPUS_CACHE = {}


def load_base():
    return M.load_json(os.path.join(HERE, WEIGHTS))


def select_pairs(rng):
    """Identical selection scheme (and rng consumption) to eval_base.sft_smoke."""
    sel = rng.choice(len(ALL_PAIRS), N_TRAIN + N_HELD, replace=False)
    train_pairs = [ALL_PAIRS[i] for i in sel[:N_TRAIN]]
    held_pairs = [ALL_PAIRS[i] for i in sel[N_TRAIN:]]
    return train_pairs, held_pairs


def build_sft_dataset_all(demos, K):
    """Unmasked variant: EVERY char of the demo is a loss position
    (prompt chars + markers + answer), contexts left-padded with newline."""
    Xs, ys = [], []
    for d in demos:
        ids = M.encode(d)
        padded = np.concatenate([np.full(K, M.PAD, np.int64), ids])
        for t in range(len(ids)):
            Xs.append(padded[t : t + K])
            ys.append(ids[t])
    return np.stack(Xs), np.array(ys)


def continue_sft(p, step_fn, X, y, rng, n_more):
    """SGD+momentum(0.9), identical to the planned JS loop."""
    for _ in range(n_more):
        idx = rng.integers(0, len(y), BATCH)
        _, g = M.loss_and_grads(p, X[idx], y[idx])
        step_fn(g)


def mean_ce(p, X, y):
    total, n = 0.0, 0
    for i in range(0, len(y), 1024):
        probs = M.forward(p, X[i : i + 1024])["probs"]
        yb = y[i : i + 1024]
        total += float(-np.log(probs[np.arange(len(yb)), yb] + 1e-12).sum())
        n += len(yb)
    return total / n


def corpus_data(K):
    if K not in _CORPUS_CACHE:
        with open(os.path.join(HERE, "corpus.txt")) as f:
            text = f.read()
        X, ids, _ = M.build_dataset(text, K)
        line_is_q = []
        for line in text.split("\n"):
            if line:
                line_is_q += [("?" in line)] * (len(line) + 1)
        line_is_q = np.array(line_is_q[: len(ids)], dtype=bool)
        _CORPUS_CACHE[K] = (X, ids, line_is_q)
    return _CORPUS_CACHE[K]


def base_corpus_loss(p, cfg):
    """Mean next-char cross-entropy (nats/char) over the full 8,352-char
    corpus with true full contexts — same metric as REPORT-base (0.01402).
    Also split into fact-line positions vs question-line positions."""
    X, ids, line_is_q = corpus_data(cfg["K"])
    tot = np.zeros(len(ids))
    for i in range(0, len(ids), 1024):
        pr = M.forward(p, X[i : i + 1024])["probs"]
        yb = ids[i : i + 1024]
        tot[i : i + 1024] = -np.log(pr[np.arange(len(yb)), yb] + 1e-12)
    return {"full": float(tot.mean()),
            "fact_lines": float(tot[~line_is_q].mean()),
            "question_lines": float(tot[line_is_q].mean())}


def eval_facts90(p, cfg, train_pairs=None):
    """Greedy base-style fact completion for every entity x relation (the
    'library'). Optional split by relationship to the demo set."""
    n_ok, fails, rows = 0, [], []
    for n, (c, h, f) in ENTITIES.items():
        attrs = {"colour": c, "home": h, "food": f}
        for rel in RELATIONS:
            tpl, want_tpl = FACT_PROMPTS[rel]
            prompt = tpl.format(n=n)
            want = want_tpl.format(a=attrs[rel])
            got = M.sample(p, cfg, prompt, 16, greedy=True, stop=".")
            ok = got == want
            n_ok += ok
            rows.append((n, rel, ok))
            if not ok:
                fails.append({"prompt": prompt, "got": got, "want": want})
    out = {"acc": n_ok / len(ALL_PAIRS), "n_ok": n_ok, "n_fails": len(fails),
           "fail_examples": fails[:6]}
    if train_pairs is not None:
        tp = set(train_pairs)
        te = {n for n, r in train_pairs}
        cats = {"demoed_pair": [0, 0], "demoed_entity_other_rel": [0, 0],
                "undemoed_entity": [0, 0]}
        for n, rel, ok in rows:
            if (n, rel) in tp:
                c = "demoed_pair"
            elif n in te:
                c = "demoed_entity_other_rel"
            else:
                c = "undemoed_entity"
            cats[c][0] += ok
            cats[c][1] += 1
        out["split"] = {k: {"ok": v[0], "n": v[1], "acc": v[0] / max(v[1], 1)}
                        for k, v in cats.items()}
    return out


def classify_line(line, name, rel, parrot_set):
    want = "a: " + canonical_answer(name, rel)
    if line == want:
        return "correct"
    if line in {"a: " + f for f in all_answer_forms(name, rel)}:
        return "correct_variant"
    if line in parrot_set:
        return "parrot"
    if line.startswith("a: "):
        return "wrong_answer"
    if "?" in line or line.startswith("q"):
        return "question_echo"
    return "other"


def qa_greedy(p, cfg, pairs, parrot_set):
    """Prompt 'q: <question>\\n', greedy to newline; classify each line."""
    rows = []
    for name, rel in pairs:
        prompt = f"q: {question_text(name, rel)}\n"
        got = M.sample(p, cfg, prompt, 45, greedy=True, stop="\n").strip()
        rows.append({"name": name, "rel": rel, "q": question_text(name, rel),
                     "got": got, "cat": classify_line(got, name, rel, parrot_set)})
    return rows


def cat_counts(rows):
    cats = ["correct", "correct_variant", "parrot", "wrong_answer",
            "question_echo", "other"]
    return {c: sum(r["cat"] == c for r in rows) for c in cats}


def qa_sampled_echo(p, cfg, pairs, parrot_set, seed, n_samples=2, temp=0.8):
    """T=0.8 sampled continuations of fresh questions; count echo/parrot."""
    rng = np.random.default_rng(seed)
    counts = {"correct": 0, "correct_variant": 0, "parrot": 0,
              "wrong_answer": 0, "question_echo": 0, "other": 0}
    total = 0
    for name, rel in pairs:
        prompt = f"q: {question_text(name, rel)}\n"
        for _ in range(n_samples):
            got = M.sample(p, cfg, prompt, 45, rng=rng, temp=temp, stop="\n").strip()
            counts[classify_line(got, name, rel, parrot_set)] += 1
            total += 1
    return counts, total


def probe_no_newline(p, cfg, pairs):
    """Prompt 'q: <question>' WITHOUT the trailing newline (greedy 60 chars).
    Which comes first in the continuation: an answer line or more question?"""
    counts = {"answer": 0, "question": 0, "other": 0}
    examples = []
    for name, rel in pairs:
        prompt = f"q: {question_text(name, rel)}"
        cont = M.sample(p, cfg, prompt, 60, greedy=True)
        marks = [(cont.find("a: "), "answer"), (cont.find("?"), "question"),
                 (cont.find("q: "), "question")]
        marks = [(i, c) for i, c in marks if i >= 0]
        cat = min(marks)[1] if marks else "other"
        counts[cat] += 1
        if len(examples) < 4:
            examples.append({"prompt": prompt, "cont": cont, "cat": cat})
    return counts, examples


def scratch_generation(p, cfg, seed, n_samples=3, length=300, temp=0.8):
    """Generate from a bare newline context; what fraction of the produced
    lines are q:/a: format (SFT bleed-through into free generation)?"""
    rng = np.random.default_rng(seed)
    lines_total, qa_lines, samples = 0, 0, []
    for _ in range(n_samples):
        s = M.sample(p, cfg, "\n", length, rng=rng, temp=temp)
        samples.append(s)
        for ln in s.split("\n"):
            ln = ln.strip()
            if ln:
                lines_total += 1
                if ln.startswith("q:") or ln.startswith("a:"):
                    qa_lines += 1
    return qa_lines / max(lines_total, 1), lines_total, samples


def transcripts(p, cfg, pairs, n=3, length=130):
    """Verbatim greedy multi-line transcripts from fresh-question prompts."""
    out = []
    for name, rel in pairs[:n]:
        prompt = f"q: {question_text(name, rel)}\n"
        cont = M.sample(p, cfg, prompt, length, greedy=True)
        out.append({"prompt": prompt, "continuation": cont})
    return out


def weight_distance(p, p_base):
    """L2 distance of all params from the base model, absolute and relative."""
    d2 = sum(float(((p[k] - p_base[k]) ** 2).sum()) for k in p)
    n2 = sum(float((p_base[k] ** 2).sum()) for k in p_base)
    return {"l2": float(np.sqrt(d2)), "rel_l2": float(np.sqrt(d2 / n2))}


def save_results(obj):
    obj["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_PATH, "w") as f:
        json.dump(obj, f, indent=1)


# ================================================================ C1: masking
def run_masking_arm(seed, arm):
    """One SFT run. arm in {'masked','unmasked'}. Returns full eval record."""
    p, cfg, _ = load_base()
    p_base = {k: v.copy() for k, v in p.items()}
    rng = np.random.default_rng(seed + 100)   # same stream as sft_smoke
    train_pairs, held_pairs = select_pairs(rng)
    demos = [sft_demo_text(n, r) for n, r in train_pairs]
    if arm == "masked":
        X, y = build_sft_dataset(demos, cfg["K"])
    else:
        X, y = build_sft_dataset_all(demos, cfg["K"])
    Xa, ya = build_sft_dataset(demos, cfg["K"])   # answer-line positions (metric)

    parrot_set = {"a: " + canonical_answer(n, r) for n, r in train_pairs}
    fresh_pairs = [pr for pr in ALL_PAIRS if pr not in set(train_pairs)]

    # train in segments; greedy trajectory evals don't touch the training rng
    step_fn = M.make_sgd(p, LR, momentum=MOMENTUM)
    traj, done, t0 = [], 0, time.time()
    for ck in TRAJ_CHECKPOINTS:
        continue_sft(p, step_fn, X, y, rng, ck - done)
        done = ck
        rows_t = qa_greedy(p, cfg, train_pairs, parrot_set)
        rows_f = qa_greedy(p, cfg, fresh_pairs, parrot_set)
        ct, cf = cat_counts(rows_t), cat_counts(rows_f)
        facts = eval_facts90(p, cfg)
        traj.append({
            "step": ck,
            "train16_acc": ct["correct"] / N_TRAIN,
            "fresh74_echo_rate": cf["question_echo"] / len(fresh_pairs),
            "fresh74_answer_formed_rate":
                (cf["correct"] + cf["correct_variant"] + cf["parrot"]
                 + cf["wrong_answer"]) / len(fresh_pairs),
            "facts90_acc": facts["acc"],
        })
    train_s = time.time() - t0

    rows_train = qa_greedy(p, cfg, train_pairs, parrot_set)
    rows_fresh = qa_greedy(p, cfg, fresh_pairs, parrot_set)
    cnt_train, cnt_fresh = cat_counts(rows_train), cat_counts(rows_fresh)
    samp_counts, samp_total = qa_sampled_echo(
        p, cfg, fresh_pairs, parrot_set, seed=seed * 1000 + 777)
    nn_counts, nn_examples = probe_no_newline(p, cfg, train_pairs + held_pairs)
    qa_frac, n_lines, scratch_samples = scratch_generation(p, cfg, seed * 1000 + 555)
    facts = eval_facts90(p, cfg, train_pairs)
    bl = base_corpus_loss(p, cfg)

    fresh_outputs = [r["got"] for r in rows_fresh]
    uniq, top = np.unique(fresh_outputs, return_counts=True)
    top_i = int(np.argmax(top))

    rec = {
        "seed": seed, "arm": arm,
        "steps": STEPS_1X, "batch": BATCH, "lr": LR, "momentum": MOMENTUM,
        "n_loss_positions": int(len(y)),
        "n_answer_positions": int(len(ya)),
        "train_pairs": [list(t) for t in train_pairs],
        "macs_fwd_bwd": STEPS_1X * BATCH * 3 * M.macs_per_position(cfg),
        "train_plus_traj_seconds": round(train_s, 2),
        "answer_line_loss_final": mean_ce(p, Xa, ya),
        "weight_distance_from_base": weight_distance(p, p_base),
        "base_corpus_loss": bl,
        "facts90": facts,
        "trajectory": traj,
        "train16": cnt_train,
        "train16_acc": cnt_train["correct"] / N_TRAIN,
        "fresh74": cnt_fresh,
        "fresh74_echo_rate": cnt_fresh["question_echo"] / len(fresh_pairs),
        "fresh74_parrot_rate": cnt_fresh["parrot"] / len(fresh_pairs),
        "fresh74_answer_formed_rate":
            (cnt_fresh["correct"] + cnt_fresh["correct_variant"]
             + cnt_fresh["parrot"] + cnt_fresh["wrong_answer"]) / len(fresh_pairs),
        "fresh74_distinct_outputs": int(len(uniq)),
        "fresh74_top_output": str(uniq[top_i]),
        "fresh74_top_share": int(top[top_i]) / len(fresh_pairs),
        "sampled_fresh": {"counts": samp_counts, "total": samp_total,
                          "echo_rate": samp_counts["question_echo"] / samp_total},
        "no_newline_probe": {"counts": nn_counts, "n": N_TRAIN + N_HELD,
                             "examples": nn_examples},
        "scratch_qa_line_frac": qa_frac,
        "scratch_n_lines": n_lines,
        "fresh_examples": [
            {"q": r["q"], "got": r["got"], "cat": r["cat"]} for r in rows_fresh[:5]],
    }
    if seed == 0:
        rec["transcripts"] = transcripts(p, cfg, held_pairs)
        rec["scratch_sample"] = scratch_samples[0]
    return rec


# =========================================================== C2: overtraining
def eval_rung(p, cfg, train_pairs, held_pairs, Xa, ya, seed, steps, lr,
              p_base=None):
    parrot_set = {"a: " + canonical_answer(n, r) for n, r in train_pairs}
    fresh_pairs = [pr for pr in ALL_PAIRS if pr not in set(train_pairs)]
    rows_train = qa_greedy(p, cfg, train_pairs, parrot_set)
    rows_held = qa_greedy(p, cfg, held_pairs, parrot_set)
    rows_fresh = qa_greedy(p, cfg, fresh_pairs, parrot_set)
    cnt_train, cnt_held, cnt_fresh = (cat_counts(rows_train),
                                      cat_counts(rows_held), cat_counts(rows_fresh))
    facts = eval_facts90(p, cfg, train_pairs)
    bl = base_corpus_loss(p, cfg)
    fresh_outputs = [r["got"] for r in rows_fresh]
    uniq, top = np.unique(fresh_outputs, return_counts=True)
    top_i = int(np.argmax(top))
    return {
        "seed": seed, "steps": steps, "lr": lr,
        "mult": HEADLINE_RUNGS.get(steps, round(steps / STEPS_1X, 3)),
        "weight_distance_from_base":
            weight_distance(p, p_base) if p_base is not None else None,
        "macs_fwd_bwd": steps * BATCH * 3 * M.macs_per_position(cfg),
        "answer_line_loss": mean_ce(p, Xa, ya),
        "base_corpus_loss": bl["full"],
        "base_corpus_loss_split": bl,
        "base_corpus_ppl_per_char": float(np.exp(bl["full"])),
        "facts90_acc": facts["acc"],
        "facts90_split": facts.get("split"),
        "n_fact_fails": facts["n_fails"],
        "fact_fail_examples": facts["fail_examples"][:4],
        "train16_acc": cnt_train["correct"] / N_TRAIN,
        "held8_acc": (cnt_held["correct"] + cnt_held["correct_variant"]) / N_HELD,
        "fresh74_acc": (cnt_fresh["correct"] + cnt_fresh["correct_variant"])
                       / len(fresh_pairs),
        "fresh74": cnt_fresh,
        "fresh74_parrot_rate": cnt_fresh["parrot"] / len(fresh_pairs),
        "fresh74_echo_rate": cnt_fresh["question_echo"] / len(fresh_pairs),
        "fresh74_distinct_outputs": int(len(uniq)),
        "fresh74_top_output": str(uniq[top_i]),
        "fresh74_top_share": int(top[top_i]) / len(fresh_pairs),
        "held_examples": [
            {"q": r["q"], "got": r["got"], "cat": r["cat"]} for r in rows_held[:4]],
    }


def run_overtraining(seed, lr=LR, rungs=RUNGS):
    p, cfg, _ = load_base()
    p_base = {k: v.copy() for k, v in p.items()}
    rng = np.random.default_rng(seed + 100)   # same stream as shipping config
    train_pairs, held_pairs = select_pairs(rng)
    demos = [sft_demo_text(n, r) for n, r in train_pairs]
    X, y = build_sft_dataset(demos, cfg["K"])   # masked (shipping) loss
    step_fn = M.make_sgd(p, lr, momentum=MOMENTUM)
    out, done = [], 0
    for steps in rungs:
        continue_sft(p, step_fn, X, y, rng, steps - done)
        done = steps
        out.append(eval_rung(p, cfg, train_pairs, held_pairs, X, y,
                             seed, steps, lr, p_base=p_base))
    return out


# ==================================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["1", "2", "2b", "all"])
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    p0, cfg, meta = load_base()
    results = {
        "meta": {
            "description": "Experiment C: loss masking A/B + overtraining",
            "base_weights": WEIGHTS,
            "base_weights_sha256": M.params_hash(p0),
            "params": M.n_params(p0),
            "macs_per_position_fwd": M.macs_per_position(cfg),
            "config": {"steps_1x": STEPS_1X, "batch": BATCH, "lr": LR,
                       "momentum": MOMENTUM, "n_train": N_TRAIN,
                       "n_held": N_HELD, "rungs": RUNGS,
                       "traj_checkpoints": TRAJ_CHECKPOINTS, "seeds": seeds},
        },
        "masking": [], "overtraining": [], "overtraining_lr05": [],
    }
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            old = json.load(f)
        for k in ("masking", "overtraining", "overtraining_lr05"):
            results[k] = old.get(k, [])

    if args.exp in ("1", "all"):
        results["masking"] = []
        for seed in seeds:
            for arm in ("masked", "unmasked"):
                t0 = time.time()
                rec = run_masking_arm(seed, arm)
                rec["wall_seconds"] = round(time.time() - t0, 1)
                results["masking"].append(rec)
                save_results(results)
                print(f"[C1] seed {seed} {arm:9s}: train16 {rec['train16_acc']:.0%} "
                      f"echo(fresh,greedy) {rec['fresh74_echo_rate']:.1%} "
                      f"parrot {rec['fresh74_parrot_rate']:.1%} "
                      f"facts90 {rec['facts90']['acc']:.1%} "
                      f"base_loss {rec['base_corpus_loss']['full']:.4f} "
                      f"ans_loss {rec['answer_line_loss_final']:.4f}", flush=True)

    if args.exp in ("2", "all"):
        results["overtraining"] = []
        for seed in seeds:
            t0 = time.time()
            rungs = run_overtraining(seed)
            for r in rungs:
                r["wall_seconds_run"] = round(time.time() - t0, 1)
            results["overtraining"].extend(rungs)
            save_results(results)
            for r in rungs:
                if r["steps"] in HEADLINE_RUNGS:
                    print(f"[C2] seed {seed} {r['mult']:>4}x ({r['steps']:4d}): "
                          f"base_loss {r['base_corpus_loss']:.4f} "
                          f"facts90 {r['facts90_acc']:.1%} "
                          f"train16 {r['train16_acc']:.0%} "
                          f"held8 {r['held8_acc']:.0%} "
                          f"parrot {r['fresh74_parrot_rate']:.1%} "
                          f"top_share {r['fresh74_top_share']:.1%}", flush=True)

    if args.exp in ("2b", "all"):
        results["overtraining_lr05"] = []
        for seed in seeds:
            rungs = run_overtraining(seed, lr=0.05,
                                     rungs=[300, 900, 3000, 9000])
            results["overtraining_lr05"].extend(rungs)
            save_results(results)
            for r in rungs:
                print(f"[C2b lr.05] seed {seed} {r['mult']:>2}x: "
                      f"base_loss {r['base_corpus_loss']:.4f} "
                      f"facts90 {r['facts90_acc']:.1%} "
                      f"train16 {r['train16_acc']:.0%} "
                      f"parrot {r['fresh74_parrot_rate']:.1%}", flush=True)

    save_results(results)
    print(f"\nresults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
