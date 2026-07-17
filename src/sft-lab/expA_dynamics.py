"""Experiment A — SFT dynamics: the "watch it learn to answer" demo and the
demonstration-count ladder.

Implements SFT *exactly* as the page will (and exactly as the validated smoke
test in eval_base.py):
  - demonstrations  "q: <question>\na: <full-sentence answer>\n"
    (lowercase q:/a: — uppercase is outside the 31-char vocab, see
    REPORT-base.md deviations)
  - cross-entropy on the ANSWER LINE chars only (loss masking on; includes the
    "a: " marker chars and the trailing newline) via eval_base.build_sft_dataset
  - plain SGD + momentum(0.9) via model.make_sgd (the exact JS loop)
  - starts from the shipping base_weights.json every run

Measured things:
  1. BEFORE numbers: base-model behaviour on all 90 questions with the same
     greedy harness (exact answers, answer-format rate, question-continuation
     rate) + base fact retention (90 completion prompts).
  2. Demo-count ladder N = 4 / 16 / 64 (N distinct entity-relation pairs),
     same step budget (300 steps x batch 24, lr 0.02), 10 seeds each.
     Per seed: trained-question exact accuracy, answer-format rate on FRESH
     (un-demoed) questions, fresh exact rate (held-out honesty), fact
     retention, full answer-set loss at checkpoints, per-step minibatch loss
     (loss-curve shape / watchability), train-accuracy at checkpoints.
  3. Config grid inside the browser budget (steps x batch <= 7,264 so that
     steps*batch*3*22,944 MACs <= 5e8): lr sweep and steps/batch shape sweep
     at N=16, plus lr spot-checks at N=4 and N=64. 10 seeds each.

Seeding: each run uses rng = np.random.default_rng(seed); the rng first draws
the N demo pairs (choice without replacement over the 90 (entity,relation)
pairs), then drives minibatch sampling. Base weights are fixed (shipping seed-0
weights) — the SFT seed controls demo sampling + minibatch shuffling only.
Everything is deterministic given (seed, N, steps, batch, lr).

Usage:
  python3 expA_dynamics.py --selftest          # batched-greedy parity check
  python3 expA_dynamics.py --phase before
  python3 expA_dynamics.py --phase ladder
  python3 expA_dynamics.py --phase grid
  python3 expA_dynamics.py --phase all         # everything, checkpointed

Results are merged into results_A.json after every config (checkpointed so a
crash loses at most one config).
"""

import argparse
import json
import os
import re
import time

import numpy as np

import model as M
from eval_base import (build_sft_dataset, first_segment,
                       load_corpus_sentences, question_text, sft_demo_text)
from gen_corpus import ENTITIES, RELATIONS, all_answer_forms, canonical_answer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results_A.json")
BASE_WEIGHTS = os.path.join(HERE, "base_weights.json")

ALL_PAIRS = [(n, rel) for n in ENTITIES for rel in RELATIONS]  # 90 pairs

# An "answer-shaped" line: a: + any well-formed fact sentence (either template
# family), regardless of whether the fact is true. Used for the FORMAT metric.
FORMAT_RE = re.compile(
    r"^a: the [a-z]+ (?:is|looks) [a-z]+\.$"
    r"|^a: the [a-z]+ (?:lives|rests) in the [a-z]+\.$"
    r"|^a: the [a-z]+ (?:eats|likes) [a-z]+\.$"
)

FACT_PROMPTS = {
    "colour": ("the {n} is ", "{a}."),
    "home": ("the {n} lives in the ", "{a}."),
    "food": ("the {n} eats ", "{a}."),
}


# ---------------------------------------------------------------- generation
def batched_greedy(p, cfg, prompts, n_chars=45, stop="\n"):
    """Greedy-decode continuations for many prompts at once (identical output
    to M.sample(..., greedy=True, stop=...) per prompt; see --selftest)."""
    K = cfg["K"]
    ctx = np.stack([M.context_ids(pr, K) for pr in prompts])
    outs = [[] for _ in prompts]
    done = np.zeros(len(prompts), bool)
    for _ in range(n_chars):
        probs = M.forward(p, ctx)["probs"]
        nxt = probs.argmax(1).astype(np.int64)
        for i in range(len(prompts)):
            if not done[i]:
                outs[i].append(int(nxt[i]))
                if M.VOCAB[int(nxt[i])] in stop:
                    done[i] = True
        ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
        if done.all():
            break
    return [M.decode(o) for o in outs]


def selftest():
    p, cfg, _ = M.load_json(BASE_WEIGHTS)
    prompts = ["q: what colour is the fox?\n", "q: where does the mole live?\n",
               "the fox is ", "the snail lives in the "]
    batched_nl = batched_greedy(p, cfg, prompts[:2], 45, "\n")
    for i in range(2):
        single = M.sample(p, cfg, prompts[i], 45, greedy=True, stop="\n")
        assert batched_nl[i] == single, (i, batched_nl[i], single)
    batched_dot = batched_greedy(p, cfg, prompts[2:], 16, ".")
    for j in range(2):
        single = M.sample(p, cfg, prompts[2 + j], 16, greedy=True, stop=".")
        assert batched_dot[j] == single, (j, batched_dot[j], single)
    print("selftest OK: batched greedy identical to model.sample")


# ---------------------------------------------------------------- evaluation
def eval_question_set(p, cfg, pairs):
    """Greedy-answer every (entity, relation) in pairs from the page prompt
    'q: <question>\\n'. Returns per-question rows + aggregate counts."""
    prompts = [f"q: {question_text(n, r)}\n" for n, r in pairs]
    gots = batched_greedy(p, cfg, prompts, 45, "\n")
    rows, n_exact, n_format, n_question, n_marker = [], 0, 0, 0, 0
    for (n, r), got in zip(pairs, gots):
        got = got.strip()
        want = f"a: {canonical_answer(n, r)}"
        exact = got == want
        fmt = bool(FORMAT_RE.match(got))
        isq = got.endswith("?")
        marker = got.startswith("a:")
        n_exact += exact
        n_format += fmt
        n_question += isq
        n_marker += marker
        rows.append({"entity": n, "relation": r, "got": got, "want": want,
                     "exact": bool(exact), "format_ok": fmt, "question": isq,
                     "a_marker": bool(marker)})
    return {"n": len(pairs), "exact": n_exact, "format": n_format,
            "question": n_question, "a_marker": n_marker, "rows": rows}


def eval_fact_retention(p, cfg):
    """Same greedy fact-completion harness as eval_base item 1 (batched)."""
    prompts, wants = [], []
    for n, (c, h, f) in ENTITIES.items():
        attrs = {"colour": c, "home": h, "food": f}
        for rel in RELATIONS:
            tpl, want_tpl = FACT_PROMPTS[rel]
            prompts.append(tpl.format(n=n))
            wants.append(want_tpl.format(a=attrs[rel]))
    gots = batched_greedy(p, cfg, prompts, 16, ".")
    n_ok = sum(g == w for g, w in zip(gots, wants))
    fails = [{"prompt": pr, "got": g, "want": w}
             for pr, g, w in zip(prompts, gots, wants) if g != w]
    return {"n_ok": n_ok, "n_total": len(prompts), "failures": fails[:12]}


def full_answer_loss(p, X, y):
    probs = M.forward(p, X)["probs"][np.arange(len(y)), y]
    return float(-np.mean(np.log(probs + 1e-12)))


# ---------------------------------------------------------------- SFT run
_CORPUS_CACHE = {}


def corpus_dataset(K):
    if K not in _CORPUS_CACHE:
        with open(os.path.join(HERE, "corpus.txt")) as f:
            text = f.read()
        Xc, ids, sent_off = M.build_dataset(text, K)
        _CORPUS_CACHE[K] = (Xc, ids, sent_off)
    return _CORPUS_CACHE[K]


def run_sft(seed, n_demos, steps, batch, lr, checkpoints=None,
            record_minibatch=False, n_examples=3,
            replay_frac=0.0, replay_fresh_p=0.0):
    """One full SFT run from the shipping base weights. Returns a record with
    every measured number. Deterministic given all arguments.

    replay_frac > 0: that fraction of every minibatch is plain next-char
    corpus examples (the forgetting mitigation); replay_fresh_p applies
    fresh-start masking (as in base training) to that share of replay rows."""
    p, cfg, _ = M.load_json(BASE_WEIGHTS)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ALL_PAIRS), n_demos, replace=False)
    train_pairs = [ALL_PAIRS[i] for i in sel]
    fresh_pairs = [pr for i, pr in enumerate(ALL_PAIRS) if i not in set(sel)]
    demos = [sft_demo_text(n, r) for n, r in train_pairs]
    X, y = build_sft_dataset(demos, cfg["K"])

    n_rep = int(round(batch * replay_frac))
    n_demo_rows = batch - n_rep
    if n_rep:
        Xc, ids_c, sent_off_c = corpus_dataset(cfg["K"])

    cps = sorted(set(checkpoints or []))
    loss_cp, acc_cp, mb_losses = [], [], []

    def take_checkpoint(step_no):
        l = full_answer_loss(p, X, y)
        ev = eval_question_set(p, cfg, train_pairs)
        loss_cp.append([step_no, round(l, 5)])
        acc_cp.append([step_no, ev["exact"], len(train_pairs)])

    if cps:
        take_checkpoint(0)
    step_fn = M.make_sgd(p, lr, momentum=0.9)
    t0 = time.time()
    for s in range(steps):
        idx = rng.integers(0, len(y), n_demo_rows)
        Xb, yb = X[idx], y[idx]
        if n_rep:
            ridx = rng.integers(0, len(ids_c), n_rep)
            flags = rng.random(n_rep) < replay_fresh_p
            Xr = M.fresh_start_mask(Xc[ridx], sent_off_c[ridx], flags,
                                    cfg["K"])
            Xb = np.concatenate([Xb, Xr])
            yb = np.concatenate([yb, ids_c[ridx]])
        loss, g = M.loss_and_grads(p, Xb, yb)
        step_fn(g)
        if record_minibatch:
            mb_losses.append(round(loss, 4))
        if (s + 1) in cps:
            take_checkpoint(s + 1)
    train_time = time.time() - t0

    ev_train = eval_question_set(p, cfg, train_pairs)
    ev_fresh = eval_question_set(p, cfg, fresh_pairs)
    facts = eval_fact_retention(p, cfg)
    rec = {
        "seed": seed, "n_demos": n_demos, "steps": steps, "batch": batch,
        "lr": lr, "momentum": 0.9,
        "replay_frac": replay_frac, "replay_fresh_p": replay_fresh_p,
        "n_answer_positions": int(len(y)),
        "macs_sft": int(steps * batch * 3 * M.macs_per_position(cfg)),
        "train_exact": ev_train["exact"], "train_n": ev_train["n"],
        "fresh_exact": ev_fresh["exact"], "fresh_format": ev_fresh["format"],
        "fresh_a_marker": ev_fresh["a_marker"],
        "fresh_question": ev_fresh["question"], "fresh_n": ev_fresh["n"],
        "fact_retention": facts["n_ok"], "fact_n": facts["n_total"],
        "loss_full_after": round(full_answer_loss(p, X, y), 5),
        "train_time_s": round(train_time, 2),
        "train_fails": [r for r in ev_train["rows"] if not r["exact"]][:8],
        "fresh_examples": [
            {"q": r["entity"] + "/" + r["relation"], "got": r["got"],
             "want": r["want"]}
            for r in ev_fresh["rows"][:n_examples]],
        "fact_fails": facts["failures"][:6],
    }
    if cps:
        rec["loss_checkpoints"] = loss_cp
        rec["acc_checkpoints"] = acc_cp
    if record_minibatch:
        rec["minibatch_losses"] = mb_losses
    return rec


# ---------------------------------------------------------------- phases
def merge_results(update):
    obj = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            obj = json.load(f)
    obj.update(update)
    with open(RESULTS_PATH, "w") as f:
        json.dump(obj, f, indent=1)


def categorize_continuations(gots, pairs, corpus_sentences):
    """First-segment classification of raw continuations (greedy):
    question / accidental_answer (queried fact in ANY template, with or
    without 'a: ') / corpus_sentence (verbatim, not the queried fact) /
    other (garbled or non-corpus)."""
    counts = {"question": 0, "accidental_answer": 0, "corpus_sentence": 0,
              "other": 0}
    rows = []
    for (n, r), got in zip(pairs, gots):
        seg, term = first_segment(got.replace("a: ", "", 1)
                                  if got.strip().startswith("a: ") else got)
        if seg in all_answer_forms(n, r):
            cat = "accidental_answer"
        elif term == "?":
            cat = "question"
        elif seg in corpus_sentences:
            cat = "corpus_sentence"
        else:
            cat = "other"
        counts[cat] += 1
        rows.append({"q": n + "/" + r, "got": got.strip()[:60], "cat": cat})
    return counts, rows


def phase_before():
    p, cfg, _ = M.load_json(BASE_WEIGHTS)
    corpus_sentences = load_corpus_sentences()
    ev = eval_question_set(p, cfg, ALL_PAIRS)
    facts = eval_fact_retention(p, cfg)
    cat_main, rows_main = categorize_continuations(
        [r["got"] for r in ev["rows"]], ALL_PAIRS, corpus_sentences)

    # variant harnesses (greedy, 60 chars, no stop) for the hero framing:
    variants = {}
    for tag, mk in (("q_prefix_no_newline",
                     lambda n, r: f"q: {question_text(n, r)}"),
                    ("bare_question",
                     lambda n, r: question_text(n, r))):
        prompts = [mk(n, r) for n, r in ALL_PAIRS]
        gots = batched_greedy(p, cfg, prompts, 60, stop="")
        counts, rows = categorize_continuations(gots, ALL_PAIRS,
                                                corpus_sentences)
        variants[tag] = {"counts": counts, "examples": rows[:8]}

    out = {
        "harness": "greedy from 'q: <question>\\n', stop at newline "
                   "(identical to the after-SFT harness)",
        "n_questions": ev["n"],
        "exact_answers": ev["exact"],
        "answer_format": ev["format"],
        "continuation_categories": cat_main,
        "fact_completion": {"n_ok": facts["n_ok"], "n_total": facts["n_total"]},
        "example_continuations": rows_main[:12],
        "variant_harnesses": variants,
    }
    merge_results({"before": out})
    print(f"[before] exact {ev['exact']}/90  format {ev['format']}/90  "
          f"cats {cat_main}  facts {facts['n_ok']}/90")
    for tag, v in variants.items():
        print(f"[before/{tag}] {v['counts']}")
    return out


LADDER_CPS = [10, 25, 50, 75, 100, 150, 200, 250, 300]


def summarize(recs):
    def col(k):
        return [r[k] for r in recs]
    n = recs[0]["train_n"]
    fresh_n = recs[0]["fresh_n"]
    return {
        "n_seeds": len(recs),
        "train_exact_per_seed": col("train_exact"), "train_n": n,
        "train_acc_mean": round(float(np.mean(col("train_exact"))) / n, 4),
        "train_acc_min": round(min(col("train_exact")) / n, 4),
        "seeds_at_100pct": sum(1 for v in col("train_exact") if v == n),
        "fresh_format_per_seed": col("fresh_format"), "fresh_n": fresh_n,
        "fresh_format_mean": round(float(np.mean(col("fresh_format"))) / fresh_n, 4),
        "fresh_a_marker_per_seed": col("fresh_a_marker"),
        "fresh_a_marker_mean": round(float(np.mean(col("fresh_a_marker"))) / fresh_n, 4),
        "fresh_exact_per_seed": col("fresh_exact"),
        "fact_retention_per_seed": col("fact_retention"),
        "fact_n": recs[0]["fact_n"],
        "loss_after_per_seed": col("loss_full_after"),
        "train_time_s_mean": round(float(np.mean(col("train_time_s"))), 2),
    }


def phase_ladder(seeds=range(10), steps=300, batch=24, lr=0.02,
                 key="ladder"):
    ladder = {}
    for n_demos in (4, 16, 64):
        recs = []
        for seed in seeds:
            rec = run_sft(seed, n_demos, steps, batch, lr,
                          checkpoints=LADDER_CPS,
                          record_minibatch=(seed == 0))
            recs.append(rec)
            print(f"[ladder N={n_demos} seed={seed}] "
                  f"train {rec['train_exact']}/{rec['train_n']}  "
                  f"fresh-fmt {rec['fresh_format']}/{rec['fresh_n']}  "
                  f"fresh-exact {rec['fresh_exact']}  "
                  f"facts {rec['fact_retention']}/90  "
                  f"loss {rec['loss_full_after']}  {rec['train_time_s']}s")
        ladder[f"N{n_demos}"] = {
            "config": {"steps": steps, "batch": batch, "lr": lr,
                       "momentum": 0.9,
                       "macs_sft": recs[0]["macs_sft"]},
            "summary": summarize(recs),
            "runs": recs,
        }
        merge_results({key: ladder})
    return ladder


GRID = (
    # lr sweep at N=16, shipping shape
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.01},
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.02},
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.03},
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.05},
    # steps x batch shape sweep at lr 0.02 (same MACs: steps*batch = 7,200)
    {"n_demos": 16, "steps": 150, "batch": 48, "lr": 0.02},
    {"n_demos": 16, "steps": 450, "batch": 16, "lr": 0.02},
    {"n_demos": 16, "steps": 600, "batch": 12, "lr": 0.02},
    # lr spot-checks at the ladder ends
    {"n_demos": 4, "steps": 300, "batch": 24, "lr": 0.05},
    {"n_demos": 64, "steps": 300, "batch": 24, "lr": 0.05},
    {"n_demos": 64, "steps": 300, "batch": 24, "lr": 0.03},
)


def phase_grid(seeds=range(10), configs=GRID, key="grid"):
    grid = []
    for cfg_ in configs:
        recs = [run_sft(seed, **cfg_) for seed in seeds]
        entry = {"config": {**cfg_, "momentum": 0.9,
                            "macs_sft": recs[0]["macs_sft"]},
                 "summary": summarize(recs),
                 "train_fails_by_seed": {
                     str(r["seed"]): [f["entity"] + "/" + f["relation"]
                                      for f in r["train_fails"]]
                     for r in recs if r["train_fails"]}}
        grid.append(entry)
        s = entry["summary"]
        print(f"[{key} {cfg_}] train/seed {s['train_exact_per_seed']} "
              f"(mean {s['train_acc_mean']:.3f})  "
              f"fresh-fmt mean {s['fresh_format_mean']:.3f}  "
              f"facts/seed {s['fact_retention_per_seed']}")
        merge_results({key: grid})
    return grid


GRID2 = (
    # N=16 robustness candidates (all <= 5e8 MACs)
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.015},
    {"n_demos": 16, "steps": 200, "batch": 36, "lr": 0.02},
    {"n_demos": 16, "steps": 150, "batch": 48, "lr": 0.03},
    {"n_demos": 16, "steps": 100, "batch": 72, "lr": 0.02},
    {"n_demos": 16, "steps": 100, "batch": 72, "lr": 0.03},
    # N=64 in-budget attempts (bigger batch = less gradient noise)
    {"n_demos": 64, "steps": 150, "batch": 48, "lr": 0.02},
    {"n_demos": 64, "steps": 100, "batch": 72, "lr": 0.02},
    {"n_demos": 64, "steps": 100, "batch": 72, "lr": 0.03},
    {"n_demos": 64, "steps": 75, "batch": 96, "lr": 0.03},
    # replay mitigation at the smoke config (same MACs)
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.02,
     "replay_frac": 0.25, "replay_fresh_p": 0.0},
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.02,
     "replay_frac": 0.25, "replay_fresh_p": 0.5},
    {"n_demos": 16, "steps": 300, "batch": 24, "lr": 0.02,
     "replay_frac": 0.5, "replay_fresh_p": 0.5},
)

# out-of-budget N=64 reference points (offline only, labeled)
GRID64_REF = (
    {"n_demos": 64, "steps": 600, "batch": 48, "lr": 0.02},   # 2.0e9 MACs
    {"n_demos": 64, "steps": 1200, "batch": 48, "lr": 0.02},  # 4.0e9 MACs
)


def phase_meta():
    p, cfg, _ = M.load_json(BASE_WEIGHTS)
    merge_results({"meta": {
        "date": time.strftime("%Y-%m-%d"),
        "base_weights": "base_weights.json",
        "base_hash": M.params_hash(p),
        "params": M.n_params(p),
        "macs_per_position_fwd": M.macs_per_position(cfg),
        "macs_budget": 5e8,
        "macs_formula": "steps * batch * 3 * 22944 (fwd+bwd ~ 3x fwd)",
        "demo_format": "q: <question>\\na: <canonical answer sentence>\\n",
        "loss_mask": "answer line only, incl. 'a: ' marker and trailing \\n",
        "optimizer": "SGD + momentum 0.9 (model.make_sgd, JS-parity loop)",
        "seeding": "rng=default_rng(seed): demo choice then minibatch stream",
        "eval_prompt": "q: <question>\\n, greedy, stop at newline",
    }})


def phase_ship(seeds=range(10), n_demos=16, steps=300, batch=24, lr=0.02,
               **kw):
    """Re-run the chosen shipping config with full curves for every seed."""
    recs = []
    for seed in seeds:
        rec = run_sft(seed, n_demos, steps, batch, lr,
                      checkpoints=LADDER_CPS, record_minibatch=True, **kw)
        recs.append(rec)
        print(f"[ship seed={seed}] train {rec['train_exact']}/{rec['train_n']}"
              f"  fresh-fmt {rec['fresh_format']}/{rec['fresh_n']}  "
              f"facts {rec['fact_retention']}/90  loss {rec['loss_full_after']}")
    out = {"config": {"n_demos": n_demos, "steps": steps, "batch": batch,
                      "lr": lr, "momentum": 0.9, **kw,
                      "macs_sft": recs[0]["macs_sft"]},
           "summary": summarize(recs), "runs": recs}
    merge_results({"shipping": out})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["before", "ladder", "grid", "grid2", "grid64ref",
                             "ship", "all"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        raise SystemExit(0)
    t0 = time.time()
    phase_meta()
    seeds = range(args.seeds)
    if args.phase in ("before", "all"):
        phase_before()
    if args.phase in ("ladder", "all"):
        phase_ladder(seeds)
    if args.phase in ("grid", "all"):
        phase_grid(seeds)
    if args.phase in ("grid2", "all"):
        phase_grid(seeds, GRID2, "grid2")
    if args.phase in ("grid64ref", "all"):
        phase_grid(seeds, GRID64_REF, "grid64_ref")
    if args.phase in ("ship", "all"):
        # chosen shipping config: 16 demos, 300 steps x batch 24, lr 0.015
        # (16/16 on all 10 seeds in grid2; best fact retention of the
        # all-seeds-perfect configs; 4.958e8 MACs)
        phase_ship(seeds, n_demos=16, steps=300, batch=24, lr=0.015)
    print(f"done in {time.time() - t0:.1f}s -> {RESULTS_PATH}")
