"""Train the base char-LM on corpus.txt with SGD + momentum(0.9).

Deterministic: numpy.random.default_rng(seed) drives init and batch order.
Writes base_weights.json (fp32 weights + config + training meta + sha256).

Usage: python3 train_base.py [--seed 0] [--steps 150000] [--batch 256]
       [--lr 0.3] [--out base_weights.json] [--quiet]
"""

import argparse
import json
import os
import time

import numpy as np

import model as M

HERE = os.path.dirname(os.path.abspath(__file__))

# Architecture (see REPORT-base.md for the sizing argument)
K = 26   # context window: entity name always fully visible in every template
D = 8    # char embedding dim
H = 96   # hidden units
P_FRESH = 0.5     # examples whose pre-sentence context is wiped to newlines
P_SCRAMBLE = 0.0  # examples whose pre-sentence context is a random corpus window


def lr_at(step, steps, lr):
    """Step-decay schedule: full lr for 70%, x0.3 to 90%, x0.1 after."""
    frac = step / steps
    if frac < 0.7:
        return lr
    if frac < 0.9:
        return lr * 0.3
    return lr * 0.1


def train(seed=0, steps=150000, batch=256, lr=0.3, quiet=False, corpus=None):
    if corpus is None:
        with open(os.path.join(HERE, "corpus.txt")) as f:
            corpus = f.read()
    X, y, sent_off = M.build_dataset(corpus, K)
    p, cfg = M.init_params(seed, K, D, H)
    step_fn = M.make_sgd(p, lr, momentum=0.9)
    rng = np.random.default_rng(seed + 1)
    log = []
    t0 = time.time()
    col = np.arange(K)[None, :]
    # positions whose context window ends exactly after a sentence terminator
    # (so a scrambled prefix always looks like a completed previous sentence)
    boundary = np.array([i for i in range(len(corpus))
                         if corpus[i - 1] in ".?\n"], np.int64)
    for s in range(steps):
        idx = rng.integers(0, len(y), batch)
        u = rng.random(batch)
        # fresh-start: wipe pre-sentence context to newline padding
        Xb = M.fresh_start_mask(X[idx], sent_off[idx], u < P_FRESH, K)
        # scramble: replace pre-sentence context with the tail of an unrelated
        # boundary-aligned corpus window -> forces fact recall to key on the
        # current sentence only (this is what lets SFT'd answers survive a
        # question-filled context window) without teaching mid-word jumps
        scram = (u >= P_FRESH) & (u < P_FRESH + P_SCRAMBLE)
        if scram.any():
            rw = boundary[rng.integers(0, len(boundary), batch)]
            keep = np.minimum(sent_off[idx], K)
            # shift each random window so its terminator abuts sentence start
            src_col = np.minimum(col + keep[:, None], K - 1)
            rnd = X[rw][np.arange(batch)[:, None], src_col]
            wipe = scram[:, None] & (col < (K - keep)[:, None])
            Xb[wipe] = rnd[wipe]
        loss, g = M.loss_and_grads(p, Xb, y[idx])
        step_fn(g, lr_at(s, steps, lr))
        if s % 2000 == 0 or s == steps - 1:
            log.append({"step": s, "batch_loss": round(loss, 4)})
            if not quiet:
                print(f"step {s:6d}  batch_loss {loss:.4f}")
    # full-corpus loss, computed in chunks
    tot, n = 0.0, 0
    for i in range(0, len(y), 4096):
        c = M.forward(p, X[i : i + 4096])
        yy = y[i : i + 4096]
        tot += float(-np.log(c["probs"][np.arange(len(yy)), yy] + 1e-12).sum())
        n += len(yy)
    full_loss = tot / n
    meta = {
        "seed": seed, "steps": steps, "batch": batch, "lr": lr,
        "momentum": 0.9, "lr_schedule": "1.0x to 70%, 0.3x to 90%, 0.1x after",
        "p_fresh_start_aug": P_FRESH, "p_scramble_aug": P_SCRAMBLE,
        "full_corpus_loss_nats_per_char": round(full_loss, 5),
        "n_params": M.n_params(p),
        "corpus_bytes": len(corpus.encode()),
        "train_seconds": round(time.time() - t0, 1),
        "weights_sha256_fp32": M.params_hash(p),
    }
    if not quiet:
        print(f"full-corpus loss: {full_loss:.4f} nats/char")
        print(f"params: {meta['n_params']}  hash: {meta['weights_sha256_fp32'][:16]}...")
    return p, cfg, meta, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--out", default="base_weights.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    p, cfg, meta, log = train(args.seed, args.steps, args.batch, args.lr, args.quiet)
    out = os.path.join(HERE, args.out)
    M.save_json(out, p, cfg, meta)
    with open(os.path.join(HERE, "results", f"train_log_seed{args.seed}.json"), "w") as f:
        json.dump({"meta": meta, "log": log}, f, indent=1)
    print(f"saved {out}")
