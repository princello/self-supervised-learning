"""Shared model library for the sft-lab char-level language model.

Pure numpy. Architecture chosen for exact JS parity with the Part 2 engine:
    char embedding (V x d)  ->  concat last K chars (K*d)  ->
    dense tanh hidden (H)   ->  dense softmax over V, cross-entropy,
    trained with plain SGD + momentum(0.9).

No attention, no layernorm, no Adam. Everything seeded and reproducible.
"""

import hashlib
import json

import numpy as np

# 31-character vocabulary (<= 32 target). Colon is reserved for SFT-time
# "q: " / "a: " markers and NEVER appears in the base corpus.
VOCAB = "\n .?:abcdefghijklmnopqrstuvwxyz"
V = len(VOCAB)
C2I = {c: i for i, c in enumerate(VOCAB)}
PAD = C2I["\n"]  # left-padding char for short contexts

PARAM_ORDER = ["E", "W1", "b1", "W2", "b2"]


def encode(s):
    return np.array([C2I[c] for c in s], dtype=np.int64)


def decode(ids):
    return "".join(VOCAB[int(i)] for i in ids)


def init_params(seed, K, d, H):
    rng = np.random.default_rng(seed)
    p = {
        "E": (rng.standard_normal((V, d)) * 0.5).astype(np.float32),
        "W1": (rng.standard_normal((K * d, H)) / np.sqrt(K * d)).astype(np.float32),
        "b1": np.zeros(H, np.float32),
        "W2": (rng.standard_normal((H, V)) / np.sqrt(H)).astype(np.float32),
        "b2": np.zeros(V, np.float32),
    }
    cfg = {"K": K, "d": d, "H": H, "V": V, "vocab": VOCAB}
    return p, cfg


def n_params(p):
    return int(sum(a.size for a in p.values()))


def macs_per_position(cfg):
    """Forward multiply-accumulates per predicted position (dense matmuls)."""
    return cfg["K"] * cfg["d"] * cfg["H"] + cfg["H"] * cfg["V"]


def forward(p, X):
    """X: (B, K) int context windows. Returns cache with probs."""
    emb = p["E"][X].reshape(X.shape[0], -1)
    z1 = emb @ p["W1"] + p["b1"]
    h = np.tanh(z1)
    logits = h @ p["W2"] + p["b2"]
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    probs = e / e.sum(axis=1, keepdims=True)
    return {"emb": emb, "h": h, "probs": probs}


def loss_and_grads(p, X, y, weights=None):
    """Mean (or weight-averaged) cross-entropy loss + grads for all params."""
    cache = forward(p, X)
    B = X.shape[0]
    probs = cache["probs"]
    if weights is None:
        w = np.full(B, 1.0 / B, np.float32)
    else:
        w = (np.asarray(weights, np.float32) / float(np.sum(weights)))
    idx = np.arange(B)
    loss = float(-(np.log(probs[idx, y] + 1e-12) * w).sum())
    dlogits = probs * w[:, None]
    dlogits[idx, y] -= w
    g = {}
    g["W2"] = cache["h"].T @ dlogits
    g["b2"] = dlogits.sum(0)
    dh = dlogits @ p["W2"].T
    dz1 = dh * (1.0 - cache["h"] ** 2)
    g["W1"] = cache["emb"].T @ dz1
    g["b1"] = dz1.sum(0)
    demb = (dz1 @ p["W1"].T).reshape(X.shape[0], X.shape[1], -1)
    gE = np.zeros_like(p["E"])
    np.add.at(gE, X, demb.astype(np.float32))
    g["E"] = gE
    return loss, g


def make_sgd(p, lr, momentum=0.9):
    """Plain SGD + momentum(0.9), identical to the planned JS loop."""
    vel = {k: np.zeros_like(v) for k, v in p.items()}

    def step(g, lr_now=None):
        eta = lr if lr_now is None else lr_now
        for k in p:
            vel[k] = momentum * vel[k] - eta * g[k].astype(np.float32)
            p[k] += vel[k]

    return step


def build_dataset(text, K):
    """Every position in text is a training example: context = previous K
    chars (left-padded with newline), target = the char itself.

    Also returns sent_off: for each position, the number of context chars
    since the current sentence start (sentences begin after '\n' or '? ').
    Used for fresh-start augmentation so the model handles bare prompts."""
    ids = encode(text)
    padded = np.concatenate([np.full(K, PAD, np.int64), ids])
    X = np.lib.stride_tricks.sliding_window_view(padded, K)[: len(ids)].copy()
    sent_off = np.zeros(len(ids), np.int64)
    start = 0
    for i, ch in enumerate(text):
        sent_off[i] = i - start
        if ch == "\n" or (ch == " " and i > 0 and text[i - 1] == "?"):
            start = i + 1
    return X, ids, sent_off


def fresh_start_mask(X, sent_off, flags, K):
    """Return a copy of X where flagged rows have all context chars before the
    current sentence start replaced by PAD (newline) — i.e. the context looks
    exactly like a bare prompt at demo time."""
    Xm = X.copy()
    keep = np.minimum(sent_off, K)
    col = np.arange(K)[None, :]
    wipe = flags[:, None] & (col < (K - keep)[:, None])
    Xm[wipe] = PAD
    return Xm


def context_ids(text, K):
    ids = encode(text)
    if len(ids) >= K:
        return ids[-K:].copy()
    return np.concatenate([np.full(K - len(ids), PAD, np.int64), ids])


def sample(p, cfg, prompt, n_chars, rng=None, temp=1.0, greedy=False, stop=None):
    """Generate n_chars continuation of prompt. greedy=True ignores rng/temp.
    stop: optional set/str of chars that end generation (char is kept)."""
    K = cfg["K"]
    ctx = list(context_ids(prompt, K))
    out = []
    for _ in range(n_chars):
        probs = forward(p, np.array([ctx], dtype=np.int64))["probs"][0]
        if greedy:
            i = int(np.argmax(probs))
        else:
            logp = np.log(probs + 1e-12) / temp
            logp -= logp.max()
            q = np.exp(logp)
            q /= q.sum()
            i = int(rng.choice(V, p=q))
        out.append(i)
        ctx = ctx[1:] + [i]
        if stop is not None and VOCAB[i] in stop:
            break
    return decode(out)


def params_hash(p):
    h = hashlib.sha256()
    for k in PARAM_ORDER:
        h.update(np.ascontiguousarray(p[k].astype(np.float32)).tobytes())
    return h.hexdigest()


def save_json(path, p, cfg, meta=None):
    obj = {
        "config": cfg,
        "meta": meta or {},
        "param_order": PARAM_ORDER,
        "params": {k: p[k].astype(np.float32).tolist() for k in PARAM_ORDER},
    }
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        obj = json.load(f)
    p = {k: np.array(obj["params"][k], dtype=np.float32) for k in obj["param_order"]}
    return p, obj["config"], obj.get("meta", {})
